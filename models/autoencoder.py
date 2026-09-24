"""autoencoder.py - ConvLSTM Autoencoder with full-sequence reconstruction.

Structure:
    Climate anomalies (input)
           |
    ConvLSTM Encoder
           |
    Dual Temporal Attention
           |
    Latent representation
           |
    Temporal Decoder
           |
    Reconstruction (output)
"""

import torch
import torch.nn as nn

from .encoder import ConvLSTMEncoder
from .attention import DualTemporalAttention


class ConvLSTMAutoencoder(nn.Module):
    """
    ConvLSTM Autoencoder for unsupervised pretraining via sequence reconstruction.

    Structure:
        Climate anomalies (B, T, C, H, W)
               |
        ConvLSTM Encoder
               |
        Dual Temporal Attention
               |
        Latent representation (B, C_latent, H, W)
               |
        Temporal Decoder
               |
        Reconstruction (B, T, C_out, H, W)
    """

    def __init__(self, config):
        """
        Args:
            config: Config object or dict.
        """
        super().__init__()

        # ================================================================
        # READ CONFIGURATION
        # ================================================================
        if hasattr(config, 'input_dim'):
            self.input_dim = config.input_dim
            self.hidden_dims = config.hidden_dims
            self.kernel_size = config.kernel_size
            self.output_dim = config.output_dim if hasattr(config, 'output_dim') else config.input_dim
            self.use_attention = config.use_attention if hasattr(config, 'use_attention') else True
            self.attention_dropout = config.attention_dropout if hasattr(config, 'attention_dropout') else 0.3
            self.attention_weight = config.attention_weight if hasattr(config, 'attention_weight') else 0.3
            self.reconstruct_full_sequence = config.reconstruct_full_sequence if hasattr(config, 'reconstruct_full_sequence') else True
            self.sequence_length = config.sequence_length if hasattr(config, 'sequence_length') else 12
            self.decay_rate = config.decay_rate if hasattr(config, 'decay_rate') else 0.3
            self.skip_weight = config.skip_weight if hasattr(config, 'skip_weight') else 0.1
            self.diversity_weight = config.diversity_weight if hasattr(config, 'diversity_weight') else 0.01
        else:
            self.input_dim = config.get('input_dim', 7)
            self.hidden_dims = config.get('hidden_dims', [32, 64, 128])
            self.kernel_size = config.get('kernel_size', 3)
            self.output_dim = config.get('output_dim', self.input_dim)
            self.use_attention = config.get('use_attention', True)
            self.attention_dropout = config.get('attention_dropout', 0.3)
            self.attention_weight = config.get('attention_weight', 0.3)
            self.reconstruct_full_sequence = config.get('reconstruct_full_sequence', True)
            self.sequence_length = config.get('sequence_length', 12)
            self.decay_rate = config.get('decay_rate', 0.3)
            self.skip_weight = config.get('skip_weight', 0.1)
            self.diversity_weight = config.get('diversity_weight', 0.01)

        # ================================================================
        # 1. CONVLSTM ENCODER
        # ================================================================
        self.encoder = ConvLSTMEncoder(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            kernel_size=self.kernel_size
        )

        # ================================================================
        # 2. DUAL TEMPORAL ATTENTION
        # ================================================================
        if self.use_attention:
            self.attention = DualTemporalAttention(
                self.hidden_dims[-1],
                dropout=self.attention_dropout
            )
        else:
            self.attention = None

        # ================================================================
        # 3. TEMPORAL DECODER
        # ================================================================
        self.temporal_decoder = nn.Sequential(
            # Project the latent representation.
            nn.Conv2d(self.hidden_dims[-1], self.hidden_dims[-1], kernel_size=1),
            nn.GroupNorm(1, self.hidden_dims[-1]),
            nn.ELU(alpha=1.0, inplace=True),

            # Expand into the full sequence.
            nn.Conv2d(
                self.hidden_dims[-1],
                self.sequence_length * self.output_dim,
                kernel_size=1
            ),
        )

        # Per-timestep refinement.
        self.temporal_refine = nn.Sequential(
            nn.Conv2d(self.output_dim, self.output_dim, kernel_size=3, padding=1),
            nn.GroupNorm(1, self.output_dim),
            nn.ELU(alpha=1.0, inplace=True),
            nn.Conv2d(self.output_dim, self.output_dim, kernel_size=1),
        )

        # ================================================================
        # SKIP PROJECTION (residual connections)
        # ================================================================
        self.skip_proj = nn.Conv2d(self.input_dim, self.output_dim, kernel_size=1)

        # ================================================================
        # TEMPORAL WEIGHTS (non-trainable buffer)
        # ================================================================
        self._init_temporal_weights()

    def _init_temporal_weights(self) -> None:
        """Initialize the exponentially-decayed temporal reconstruction weights."""
        T = self.sequence_length
        time_idx = torch.arange(T, dtype=torch.float32)

        # More recent timesteps get more weight.
        weights = torch.exp(-self.decay_rate * (T - 1 - time_idx))
        weights = weights / weights.sum()

        self.register_buffer("temporal_weights", weights.view(1, T, 1, 1, 1))

    def _reconstruct_sequence(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct the full temporal sequence from the latent representation.

        Args:
            latent: Latent representation [B, C_latent, H, W].

        Returns:
            recon_seq: Reconstructed sequence [B, T, C_out, H, W].
        """
        B, C, H, W = latent.shape

        out = self.temporal_decoder(latent)  # [B, T * C_out, H, W]
        out = out.view(B, self.sequence_length, self.output_dim, H, W)  # [B, T, C_out, H, W]

        B, T, C, H, W = out.shape
        out_flat = out.view(B * T, C, H, W)  # [B*T, C_out, H, W]
        out_flat = self.temporal_refine(out_flat)
        out = out_flat.view(B, T, C, H, W)  # [B, T, C_out, H, W]

        return out

    def _apply_skip_to_sequence(
        self,
        x: torch.Tensor,
        recon_seq: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply the input-to-output skip connection at every timestep.

        Args:
            x: Original input [B, T, C_in, H, W].
            recon_seq: Reconstructed sequence [B, T, C_out, H, W].

        Returns:
            Reconstructed sequence with the skip connection applied.
        """
        B, T, C, H, W = x.shape

        skip_flat = x.view(B * T, C, H, W)
        skip_flat = self.skip_proj(skip_flat)
        skip = skip_flat.view(B, T, self.output_dim, H, W)

        return recon_seq + self.skip_weight * skip

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, return_info: bool = False):
        """
        Forward pass of the autoencoder.

        Structure:
            Climate anomalies (input)
                   |
            ConvLSTM Encoder
                   |
            Dual Temporal Attention
                   |
            Latent representation
                   |
            Temporal Decoder
                   |
            Reconstruction (output)

        Args:
            x: Input [B, T, C, H, W].
            mask: Optional spatial validity mask ([H,W], [B,H,W] or
                [B,1,H,W]) so the attention module ignores invalid
                (ocean/no-data) pixels.
            return_info: If True, also return diagnostic info.

        Returns:
            recon: Reconstruction [B, T, C_out, H, W].
            info: Dict with diagnostics (if return_info=True).
        """
        B, T, C, H, W = x.shape

        # ================================================================
        # 1. CONVLSTM ENCODER
        # ================================================================
        latent, hidden_seq = self.encoder(x)
        info = {"latent": latent}

        # ================================================================
        # 2. DUAL TEMPORAL ATTENTION
        # ================================================================
        if self.attention is not None:
            context, attn_info = self.attention(hidden_seq, mask=mask)
            info.update(attn_info)

            latent = latent + self.attention_weight * context
            info["latent_with_attention"] = latent

        # ================================================================
        # 3. TEMPORAL DECODER
        # ================================================================
        if self.reconstruct_full_sequence:
            recon_seq = self._reconstruct_sequence(latent)
            recon_seq = self._apply_skip_to_sequence(x, recon_seq)

            info["recon_seq"] = recon_seq
            info["recon_loss_type"] = "sequence"

            recon = recon_seq
        else:
            # Single-timestep reconstruction (last input timestep only).
            recon = self.decoder(latent)
            last_input = x[:, -1]
            skip = self.skip_proj(last_input)
            recon = recon + self.skip_weight * skip

        if return_info:
            return recon, info
        return recon

    def compute_loss(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        variable_weights: torch.Tensor = None
    ) -> torch.Tensor:
        """Compute the temporally-weighted reconstruction loss plus the diversity penalty."""
        recon, info = self.forward(x, mask=mask, return_info=True)

        # ================================================================
        # 1. RECONSTRUCTION LOSS
        # ================================================================
        if self.reconstruct_full_sequence and "recon_seq" in info:
            target_seq = torch.nan_to_num(x, nan=0.0)
            recon_loss = self._compute_temporal_loss(
                info["recon_seq"],
                target_seq,
                mask,
                variable_weights
            )
        else:
            target = torch.nan_to_num(x[:, -1], nan=0.0)
            recon = torch.nan_to_num(recon, nan=0.0)

            beta = 0.1
            diff = torch.abs(recon - target)
            recon_loss = torch.where(
                diff < beta,
                0.5 * diff ** 2 / beta,
                diff - 0.5 * beta
            )

            if variable_weights is not None:
                recon_loss = recon_loss * variable_weights.view(1, -1, 1, 1)

            if mask is not None:
                if mask.dim() == 2:
                    mask = mask.unsqueeze(0).unsqueeze(0)
                recon_loss = recon_loss * mask
                valid_pixels = mask.sum()
                if valid_pixels > 0:
                    recon_loss = recon_loss.sum() / valid_pixels
                else:
                    recon_loss = torch.tensor(0.0, device=x.device)
            else:
                recon_loss = recon_loss.mean()

        # ================================================================
        # 2. DIVERSITY PENALTY
        # ================================================================
        diversity_loss = torch.tensor(0.0, device=x.device)

        if 'latent_with_attention' in info:
            latent = info['latent_with_attention']
            diversity_loss = self._compute_diversity_loss(latent)

        # ================================================================
        # 3. TOTAL LOSS
        # ================================================================
        loss = recon_loss + self.diversity_weight * diversity_loss

        return loss

    def _compute_temporal_loss(
        self,
        pred_seq: torch.Tensor,
        target_seq: torch.Tensor,
        mask: torch.Tensor = None,
        variable_weights: torch.Tensor = None
    ) -> torch.Tensor:
        """Compute the Smooth L1 reconstruction loss with temporal weighting."""
        B, T, C, H, W = pred_seq.shape

        diff = torch.abs(pred_seq - target_seq)
        beta = 0.1
        loss = torch.where(
            diff < beta,
            0.5 * diff ** 2 / beta,
            diff - 0.5 * beta
        )

        if variable_weights is not None:
            loss = loss * variable_weights.view(1, 1, C, 1, 1)

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 4:
                mask = mask.unsqueeze(2)

            mask = mask.to(loss.dtype)

            if mask.shape[2] == 1:
                mask = mask.expand(-1, -1, C, -1, -1)

            loss = loss * mask
            denominator = mask.sum().clamp_min(1.0)
        else:
            denominator = torch.tensor(loss.numel(), device=loss.device, dtype=loss.dtype)

        loss_per_time = loss.sum(dim=(0, 2, 3, 4)) / denominator

        temporal_weights = self.temporal_weights.squeeze()
        temporal_weights = temporal_weights / temporal_weights.sum()

        return torch.sum(loss_per_time * temporal_weights)

    def _compute_diversity_loss(self, latent: torch.Tensor) -> torch.Tensor:
        """Diversity penalty based on cross-channel correlation of the latent representation."""
        B, C, H, W = latent.shape

        # Reshape to [B*H*W, C].
        z = latent.permute(0, 2, 3, 1).reshape(-1, C)

        # Center and standardize each channel.
        z = z - z.mean(dim=0, keepdim=True)
        std = z.std(dim=0, keepdim=True).clamp_min(1e-6)
        z = z / std

        # Cross-channel correlation matrix.
        corr = (z.T @ z) / z.shape[0]

        # Penalize off-diagonal correlations.
        off_diag = corr - torch.eye(C, device=latent.device)

        return (off_diag ** 2).mean()

    # ================================================================
    # AUXILIARY METHODS
    # ================================================================

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the latent representation WITHOUT attention."""
        latent, _ = self.encoder(x)
        return latent

    def encode_with_attention(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Return the latent representation WITH attention."""
        latent, hidden_seq = self.encoder(x)
        if self.attention is not None:
            context, _ = self.attention(hidden_seq, mask=mask)
            latent = latent + self.attention_weight * context
        return latent

    def get_encoder_state_dict(self) -> dict:
        """Return the encoder weights for transfer learning."""
        encoder_state = {}
        for key, value in self.encoder.state_dict().items():
            encoder_state[f"encoder.{key}"] = value
        return encoder_state

    def get_attention_state_dict(self) -> dict:
        """Return the attention weights for transfer learning."""
        if self.attention is None:
            return {}

        attention_state = {}
        for key, value in self.attention.state_dict().items():
            attention_state[f"attention.{key}"] = value
        return attention_state
