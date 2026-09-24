"""attention.py - Dual Temporal Attention with spatial mask support."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DualTemporalAttention(nn.Module):
    """
    Dual Temporal Attention for modeling climate memory.

    Combines local (short-term) and global (long-term) attention through
    an adaptive gate. The spatial mask is applied to ignore invalid pixels
    (ocean, no data).
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.3):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout
        proj_dim = max(16, hidden_dim // 4)

        self.dropout = nn.Dropout2d(dropout)

        self.local_attn = nn.Sequential(
            nn.Conv2d(hidden_dim, proj_dim, kernel_size=1),
            nn.ELU(alpha=1.0, inplace=True),
            self.dropout,
            nn.Conv2d(proj_dim, 1, kernel_size=1)
        )

        self.global_attn = nn.Sequential(
            nn.Conv2d(hidden_dim, proj_dim, kernel_size=1),
            nn.ELU(alpha=1.0, inplace=True),
            self.dropout,
            nn.Conv2d(proj_dim, 1, kernel_size=1)
        )

        # Learnable decay rate, applied through a sigmoid reparameterization.
        self.decay_logit = nn.Parameter(torch.tensor(0.0))

        # GroupNorm(1) behaves as InstanceNorm, which works with batch_size=1.
        # NOTE: no spatial pooling here (deliberately) - the gate must stay
        # per-pixel. An AdaptiveAvgPool2d(1) used to sit here, which collapsed
        # H,W before the gate was computed, making it a single scalar per
        # (batch, timestep) broadcast over the whole region instead of an
        # actual per-pixel "adaptive" gate - and, since that pooling had no
        # mask awareness, the collapsed scalar was diluted by invalid/masked
        # pixels in proportion to how much of the region was invalid.
        self.gate = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1),
            nn.GroupNorm(1, hidden_dim),
            nn.ELU(alpha=1.0, inplace=True),
            self.dropout,
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.out_proj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.norm = nn.GroupNorm(1, hidden_dim)

        # Learnable residual connection to the last hidden state.
        self.skip_weight = nn.Parameter(torch.tensor(0.1))

    @property
    def decay(self) -> torch.Tensor:
        """Decay rate constrained to [0.05, 0.95] to avoid extreme values."""
        return torch.sigmoid(self.decay_logit) * 0.9 + 0.05

    def _exponential_pool(
        self,
        hidden_seq: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Exponentially-decayed temporal pooling.

        Args:
            hidden_seq: Sequence of hidden states [B, T, C, H, W].
            mask: Spatial/temporal mask [B, T, 1, H, W] or None.

        Returns:
            Temporally-weighted context [B, C, H, W].
        """
        B, T, C, H, W = hidden_seq.shape
        decay_val = self.decay
        time_idx = torch.arange(T, device=hidden_seq.device).float()
        weights = torch.exp(-decay_val * (T - 1 - time_idx))
        weights = weights / (weights.sum() + 1e-8)
        weights = weights.view(1, T, 1, 1, 1)

        if mask is not None:
            # Normalize the mask shape to [B, T, 1, H, W].
            if mask.dim() == 2:  # [H, W]
                mask = mask.unsqueeze(0).unsqueeze(0).unsqueeze(2)
            elif mask.dim() == 3:  # [B, H, W] or [T, H, W]
                if mask.shape[0] == B:  # [B, H, W]
                    mask = mask.unsqueeze(1).unsqueeze(2)
                else:  # [T, H, W]
                    mask = mask.unsqueeze(0).unsqueeze(2)
            elif mask.dim() == 4:  # [B, T, H, W] or [B, 1, H, W]
                if mask.shape[1] == T:  # [B, T, H, W]
                    mask = mask.unsqueeze(2)  # [B, T, 1, H, W]
                else:  # [B, 1, H, W]
                    mask = mask.unsqueeze(2)  # [B, 1, 1, H, W]

            weights = weights * mask
            # Renormalize so invalid pixels don't influence the pooled context.
            weights_sum = weights.sum(dim=1, keepdim=True) + 1e-8
            weights = weights / weights_sum

        return (hidden_seq * weights).sum(dim=1)

    def forward(self, hidden_seq: torch.Tensor, mask: torch.Tensor = None):
        """
        Forward pass of the dual temporal attention.

        Args:
            hidden_seq: Sequence of hidden states [B, T, C, H, W].
            mask: Spatial mask [H, W] or [B, H, W] or [B, T, H, W].

        Returns:
            output: Attention context [B, C, H, W].
            info: Dict with diagnostics (weights, gate, decay).
        """
        B, T, C, H, W = hidden_seq.shape

        # ================================================================
        # 1. NORMALIZE THE MASK SHAPE
        # ================================================================
        mask_expanded = None
        if mask is not None:
            if mask.dim() == 2:  # [H, W]
                mask_expanded = mask.unsqueeze(0).unsqueeze(0).unsqueeze(2)
            elif mask.dim() == 3:  # [B, H, W] or [T, H, W]
                if mask.shape[0] == B:  # [B, H, W]
                    mask_expanded = mask.unsqueeze(1).unsqueeze(2)
                else:  # [T, H, W]
                    mask_expanded = mask.unsqueeze(0).unsqueeze(2)
            elif mask.dim() == 4:  # [B, T, H, W] or [B, 1, H, W]
                if mask.shape[1] == T:  # [B, T, H, W]
                    mask_expanded = mask.unsqueeze(2)  # [B, T, 1, H, W]
                else:  # [B, 1, H, W]
                    mask_expanded = mask.unsqueeze(2)  # [B, 1, 1, H, W]

            hidden_seq = hidden_seq * mask_expanded

        # ================================================================
        # 2. LOCAL AND GLOBAL ATTENTION SCORES
        # ================================================================
        local_scores = []
        global_scores = []

        for t in range(T):
            h_t = hidden_seq[:, t]
            local_scores.append(self.local_attn(h_t))

            if mask_expanded is not None:
                mask_t = mask_expanded[:, :t + 1]  # [B, t+1, 1, H, W]
            else:
                mask_t = None

            context = self._exponential_pool(hidden_seq[:, :t + 1], mask_t)
            global_scores.append(self.global_attn(context))

        local_attn = torch.stack(local_scores, dim=1)   # [B, T, 1, H, W]
        global_attn = torch.stack(global_scores, dim=1)  # [B, T, 1, H, W]

        # ================================================================
        # 3. ADAPTIVE GATE
        # ================================================================
        h_mean = hidden_seq.mean(dim=1)        # [B, C, H, W]
        h_std = hidden_seq.std(dim=1)          # [B, C, H, W]
        gate_input = torch.cat([h_mean, h_std], dim=1)  # [B, 2C, H, W]
        gate = self.gate(gate_input)           # [B, 1, H, W]
        gate = gate.unsqueeze(1).expand(-1, T, -1, -1, -1)  # [B, T, 1, H, W]

        # ================================================================
        # 4. COMBINE ATTENTION COMPONENTS
        # ================================================================
        combined_attn = gate * local_attn + (1 - gate) * global_attn
        combined_attn = F.softmax(combined_attn, dim=1)
        attended = (combined_attn * hidden_seq).sum(dim=1)  # [B, C, H, W]

        # ================================================================
        # 5. PROJECTION AND RESIDUAL
        # ================================================================
        output = self.out_proj(attended)
        output = self.norm(output)

        # Residual connection to the last hidden state.
        output = output + self.skip_weight * hidden_seq[:, -1]

        return output, {
            "weights": combined_attn.squeeze(2),
            "gate": gate.squeeze(2),
            "decay": self.decay.item()
        }
