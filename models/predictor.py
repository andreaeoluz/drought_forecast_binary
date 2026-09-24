"""predictor_model.py - ConvLSTM model for binary drought classification.

Implements the priority improvements from the architecture review, without
adding any new input variable/feature:

  1. Partial & progressive transfer learning - the encoder can now be
     frozen/unfrozen one ConvLSTM layer at a time (see encoder.py's
     `layer_param_groups`/`freeze_layers`/`unfreeze_layers`), instead of
     only "all frozen" vs "all unfrozen". `unfreeze_stage()` below drives
     that progressively, layer by layer, from deepest (most task-specific)
     to shallowest (most general).
  2. Residual connections between the encoder and the predictor head - the
     predictor no longer has to overwrite the pretrained representation; it
     only has to learn an *adaptation* on top of it:
         H_out = H_encoder + f(H_encoder)
  3. Dual temporal attention (unchanged, already adaptive).
  4. A multiscale temporal module (fast + slow branches, see
     multiscale_temporal.py) run in parallel with the attention, so the
     model can represent both fast-onset and slow, persistent drought
     development without needing a longer history window `p`.
"""

import torch
import torch.nn as nn

from .encoder import ConvLSTMEncoder
from .attention import DualTemporalAttention
from .multiscale_temporal import MultiscaleTemporalModule
from .decoder import PredictionDecoder


class ResidualAdapter(nn.Module):
    """
    Lightweight residual adaptation block: H_out = H_in + f(H_in).

    Keeps the pretrained/fused representation intact by default (f is
    initialized close to zero via the final conv's zero-init) and lets the
    predictor learn only the adjustments needed for the drought task,
    instead of being forced to replace the whole representation.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(1, hidden_dim),
            nn.ELU(alpha=1.0, inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
        )
        # Zero-init the last conv so the block starts as an identity
        # mapping (H_out = H_in on the very first forward pass) and only
        # gradually learns a useful adaptation.
        nn.init.zeros_(self.block[-1].weight)
        if self.block[-1].bias is not None:
            nn.init.zeros_(self.block[-1].bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.block(h)


class ConvLSTMPredictor(nn.Module):
    """
    ConvLSTM Predictor for binary classification of extreme drought.

    The model combines:
    - A multi-layer ConvLSTM encoder (optionally pretrained via the autoencoder)
    - Dual temporal attention (local + global)
    - A multiscale temporal module (fast + slow branches)
    - A residual adapter between the fused representation and the decoder
    - A binary classification decoder
    - Partial/progressive transfer learning support with selective,
      layer-wise weight loading and freezing
    """

    def __init__(self, config: dict, prevalence: float = 0.025):
        """
        Initialize the predictor.

        Args:
            config: Dict with model configuration. New optional keys:
                - use_multiscale (bool, default True)
                - multiscale_fast_window (int, default 3)
                - multiscale_weight (float, default 0.3): how much the
                  multiscale context contributes to the fused latent,
                  mirroring the existing attention_weight of 0.3.
                - use_residual_adapter (bool, default True)
            prevalence: Expected positive-class prevalence (used to
                initialize the decoder's output bias).
        """
        super().__init__()

        self.config = config
        self.input_dim = config["input_dim"]
        self.hidden_dims = config["hidden_dims"]
        self.kernel_size = config.get("kernel_size", 3)
        self.output_dim = config.get("output_dim", 1)
        self.use_attention = config.get("use_attention", True)
        self.attention_dropout = config.get("attention_dropout", 0.3)
        self.attention_weight = config.get("attention_weight", 0.3)
        self.prevalence = prevalence

        self.use_multiscale = config.get("use_multiscale", True)
        self.multiscale_fast_window = config.get("multiscale_fast_window", 3)
        self.multiscale_weight = config.get("multiscale_weight", 0.3)

        self.use_residual_adapter = config.get("use_residual_adapter", True)

        # ====================================================================
        # ENCODER
        # ====================================================================
        self.encoder = ConvLSTMEncoder(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            kernel_size=self.kernel_size
        )

        # ====================================================================
        # TEMPORAL ATTENTION
        # ====================================================================
        if self.use_attention:
            self.attention = DualTemporalAttention(
                self.hidden_dims[-1],
                dropout=self.attention_dropout
            )
        else:
            self.attention = None

        # ====================================================================
        # MULTISCALE TEMPORAL MODULE
        # ====================================================================
        if self.use_multiscale:
            self.multiscale = MultiscaleTemporalModule(
                self.hidden_dims[-1],
                fast_window=self.multiscale_fast_window,
                dropout=min(0.2, self.attention_dropout),
            )
        else:
            self.multiscale = None

        # ====================================================================
        # RESIDUAL ADAPTER (encoder/attention/multiscale -> decoder)
        # ====================================================================
        if self.use_residual_adapter:
            self.residual_adapter = ResidualAdapter(self.hidden_dims[-1])
        else:
            self.residual_adapter = None

        # ====================================================================
        # CLASSIFICATION DECODER
        # ====================================================================
        self.decoder = PredictionDecoder(
            latent_dim=self.hidden_dims[-1],
            kernel_size=self.kernel_size,
            prevalence=prevalence
        )

    # ============================================================================
    # FORWARD
    # ============================================================================

    def _fuse_latent(self, latent: torch.Tensor, hidden_seq: torch.Tensor, mask: torch.Tensor = None):
        """Combine the encoder's final state with attention + multiscale
        contexts, then apply the residual adapter. Shared by forward() and
        encode_with_attention()."""
        info = {}

        if self.attention is not None:
            attn_context, attn_info = self.attention(hidden_seq, mask=mask)
            info.update(attn_info)
            latent = latent + attn_context * self.attention_weight

        if self.multiscale is not None:
            ms_context, ms_info = self.multiscale(hidden_seq, mask=mask)
            info.update(ms_info)
            latent = latent + ms_context * self.multiscale_weight

        if self.residual_adapter is not None:
            latent = self.residual_adapter(latent)

        return latent, info

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, return_info: bool = False):
        """
        Forward pass of the predictor.

        Args:
            x: Input [B, T, C, H, W].
            mask: Optional spatial validity mask ([H,W], [B,H,W] or
                [B,1,H,W]) so the attention/multiscale modules ignore
                invalid (ocean/no-data) pixels instead of mixing their
                zeroed-out values into the computation.
            return_info: If True, also return diagnostic info.

        Returns:
            logits: Classification logits [B, 1, H, W].
            info: Dict with diagnostics.
        """
        latent, hidden_seq = self.encoder(x)
        latent, info = self._fuse_latent(latent, hidden_seq, mask=mask)

        logits = self.decoder(latent)

        if return_info:
            return logits, info
        return logits, info

    # ============================================================================
    # TRANSFER LEARNING (LOADING)
    # ============================================================================

    def load_encoder_from_autoencoder(
        self,
        ae_checkpoint: dict,
        strict: bool = False,
        load_attention: bool = True
    ) -> None:
        """
        Load encoder and (optionally) attention weights from a pretrained autoencoder.

        Args:
            ae_checkpoint: Autoencoder checkpoint containing model_state_dict.
            strict: If True, require every weight to be loaded.
            load_attention: If True, also load the attention weights.

        Notes:
            - The encoder is always transferred when available.
            - The attention module is transferred only if load_attention=True.
            - The multiscale module and residual adapter are new components
              with no autoencoder counterpart, so they always start randomly
              initialized (the residual adapter is zero-initialized to
              behave as an identity mapping at first anyway - see
              ResidualAdapter).
            - Shape compatibility is verified automatically.
        """
        ae_state = ae_checkpoint.get("model_state_dict", ae_checkpoint)

        encoder_state = {}
        attention_state = {}

        # ====================================================================
        # 1. TRANSFER ENCODER WEIGHTS (always)
        # ====================================================================
        for key, value in ae_state.items():
            if key.startswith("encoder."):
                target_key = key.replace("encoder.", "")
                if target_key in self.encoder.state_dict():
                    if value.shape == self.encoder.state_dict()[target_key].shape:
                        encoder_state[target_key] = value

        if encoder_state:
            self.encoder.load_state_dict(encoder_state, strict=False)
            print(f"  ✅ Encoder: {len(encoder_state)}/{len(self.encoder.state_dict())} "
                  "weights transferred")
        else:
            print("  ⚠️ Encoder: no compatible weights found")

        # ====================================================================
        # 2. TRANSFER ATTENTION WEIGHTS (optional)
        # ====================================================================
        if load_attention and self.attention is not None:
            for key, value in ae_state.items():
                if key.startswith("attention."):
                    target_key = key.replace("attention.", "")
                    if target_key in self.attention.state_dict():
                        if value.shape == self.attention.state_dict()[target_key].shape:
                            attention_state[target_key] = value

            if attention_state:
                self.attention.load_state_dict(attention_state, strict=False)
                print(f"  ✅ Attention: {len(attention_state)}/{len(self.attention.state_dict())} "
                      "weights transferred")
            else:
                print("  ⚠️ Attention: no compatible weights found (kept random)")
        elif self.attention is not None and not load_attention:
            print("  ⚠️ Attention: kept random (load_attention=False)")

        if self.multiscale is not None:
            print("  ℹ️ Multiscale module: kept random (no autoencoder counterpart)")
        if self.residual_adapter is not None:
            print("  ℹ️ Residual adapter: zero-initialized identity (no autoencoder counterpart)")

    # ============================================================================
    # PARTIAL / PROGRESSIVE TRANSFER LEARNING (FREEZING)
    # ============================================================================

    def freeze_encoder(self, freeze: bool = True) -> None:
        """
        Freeze/unfreeze the WHOLE encoder at once (coarse, all-or-nothing
        control - kept for backward compatibility). For progressive,
        layer-by-layer control use `unfreeze_stage()` instead.

        Note:
            The attention and multiscale modules are NEVER frozen, so the
            model can keep adapting to the downstream task.
        """
        self.encoder.freeze_all(freeze)

        if self.attention is not None:
            for param in self.attention.parameters():
                param.requires_grad = True
        if self.multiscale is not None:
            for param in self.multiscale.parameters():
                param.requires_grad = True

    def num_unfreeze_stages(self) -> int:
        """Number of progressive-unfreeze stages available (one per encoder layer)."""
        return len(self.hidden_dims)

    def unfreeze_stage(self, stage: int) -> None:
        """
        Progressive/partial transfer learning (see improvement doc, item 1).

        Freezes the whole encoder, then unfreezes its layers one at a time
        starting from the DEEPEST (last, most task-specific) layer and
        working back toward the SHALLOWEST (first, most general) layer,
        which is unfrozen last. This way the general climate patterns
        learned during autoencoder pretraining are preserved the longest,
        while the task-specific layers adapt to the drought-prediction
        objective earliest.

        Args:
            stage: 0 = everything frozen (only attention/multiscale/decoder
                train); stage `i` unfreezes the `i` deepest encoder layers;
                stage `num_unfreeze_stages()` unfreezes the entire encoder
                (equivalent to `freeze_encoder(False)`).
        """
        n_layers = len(self.hidden_dims)
        stage = max(0, min(stage, n_layers))

        # Start fully frozen.
        self.encoder.freeze_all(True)

        # Unfreeze the `stage` deepest layers (indices n_layers-1 down to
        # n_layers-stage), i.e. the most task-specific ones first.
        unfreeze_indices = list(range(n_layers - stage, n_layers))
        self.encoder.unfreeze_layers(unfreeze_indices)

        # Once the shallowest ConvLSTM layer (index 0) is unfrozen, also
        # unfreeze the encoder's input_proj/input_norm, which sit even
        # earlier in the pipeline.
        if 0 in unfreeze_indices:
            for param in self.encoder.input_proj.parameters():
                param.requires_grad = True
            for param in self.encoder.input_norm.parameters():
                param.requires_grad = True

        if self.attention is not None:
            for param in self.attention.parameters():
                param.requires_grad = True
        if self.multiscale is not None:
            for param in self.multiscale.parameters():
                param.requires_grad = True

    def freeze_attention(self, freeze: bool = True) -> None:
        """
        Freeze/unfreeze the attention parameters.

        Note:
            Additional knob for fine-grained transfer-learning control.
            By default the attention module is never frozen.
        """
        if self.attention is not None:
            for param in self.attention.parameters():
                param.requires_grad = not freeze

    def get_trainable_params_count(self) -> dict:
        """
        Return the number of trainable parameters per component.

        Returns:
            Dict with parameter counts by component.
        """
        encoder_params = sum(
            p.numel() for p in self.encoder.parameters() if p.requires_grad
        )
        attention_params = (
            sum(p.numel() for p in self.attention.parameters() if p.requires_grad)
            if self.attention else 0
        )
        multiscale_params = (
            sum(p.numel() for p in self.multiscale.parameters() if p.requires_grad)
            if self.multiscale else 0
        )
        residual_params = (
            sum(p.numel() for p in self.residual_adapter.parameters() if p.requires_grad)
            if self.residual_adapter else 0
        )
        decoder_params = sum(
            p.numel() for p in self.decoder.parameters() if p.requires_grad
        )

        return {
            "encoder": encoder_params,
            "attention": attention_params,
            "multiscale": multiscale_params,
            "residual_adapter": residual_params,
            "decoder": decoder_params,
            "total": encoder_params + attention_params + multiscale_params
            + residual_params + decoder_params
        }

    def get_encoder_state_dict(self) -> dict:
        """Return the encoder weights for export."""
        return self.encoder.state_dict()

    def get_attention_state_dict(self) -> dict:
        """Return the attention weights for export (empty dict if none)."""
        if self.attention is None:
            return {}
        return self.attention.state_dict()

    def encode_with_attention(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Return the latent representation WITH attention/multiscale/residual
        fusion applied.

        This mirrors the representation used during autoencoder
        pretraining, for consistency between training phases.

        Args:
            x: Input [B, T, C, H, W].
            mask: Optional spatial validity mask, see forward().

        Returns:
            Fused latent representation [B, C, H, W].
        """
        latent, hidden_seq = self.encoder(x)
        latent, _ = self._fuse_latent(latent, hidden_seq, mask=mask)
        return latent

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the latent representation WITHOUT attention.

        Args:
            x: Input [B, T, C, H, W].

        Returns:
            Latent representation [B, C, H, W].
        """
        latent, _ = self.encoder(x)
        return latent
