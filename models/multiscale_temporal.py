"""multiscale_temporal.py - Multiscale temporal module (fast + slow branches).

Motivation (see improvement doc, item 2 - "Módulo temporal multiescala"):
there is no single optimal history length `p` for every region/threshold,
because droughts develop at different speeds in different places. Instead
of forcing the model to capture every timescale through one exponential
pooling operator (as DualTemporalAttention does), this module runs two
parallel, independently-parameterized branches over the same hidden-state
sequence:

    - a FAST branch, which only looks at the most recent `fast_window`
      timesteps and is meant to react quickly to sudden onset drought;
    - a SLOW branch, which pools over the *entire* window with a learnable,
      slowly-decaying weighting and is meant to capture persistence /
      gradual drought evolution.

The two contexts are fused through a learnable, per-pixel gate (not a
fixed average), so the network can decide - region by region, pixel by
pixel - how much weight fast vs. slow developments should get.

This module is deliberately independent from DualTemporalAttention: it
does not replace it, it supplies a second, differently-biased summary of
the same sequence that the Predictor fuses in addition to the attention
context.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiscaleTemporalModule(nn.Module):
    """Two-branch (fast/slow) temporal summarizer with a learned pixelwise gate."""

    def __init__(self, hidden_dim: int, fast_window: int = 3, dropout: float = 0.2):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.fast_window = fast_window
        proj_dim = max(16, hidden_dim // 4)

        # --------------------------------------------------------------
        # FAST BRANCH: only the last `fast_window` timesteps, combined
        # with per-step learned scores (like a tiny local attention) so
        # a sudden, sharp change in the last month or two can dominate.
        # --------------------------------------------------------------
        self.fast_score = nn.Sequential(
            nn.Conv2d(hidden_dim, proj_dim, kernel_size=1),
            nn.ELU(alpha=1.0, inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(proj_dim, 1, kernel_size=1),
        )

        # --------------------------------------------------------------
        # SLOW BRANCH: exponential pooling over the *whole* sequence with
        # its own learnable decay, independent from the one used inside
        # DualTemporalAttention, biased toward a much slower decay so it
        # actually captures persistence rather than duplicating the fast
        # branch. Initialized near 0 (slow decay) via the sigmoid
        # reparameterization below.
        self.slow_decay_logit = nn.Parameter(torch.tensor(-2.0))

        self.slow_proj = nn.Sequential(
            nn.Conv2d(hidden_dim, proj_dim, kernel_size=1),
            nn.ELU(alpha=1.0, inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(proj_dim, hidden_dim, kernel_size=1),
        )

        # --------------------------------------------------------------
        # FUSION GATE: per-pixel, learned from both branches (GroupNorm(1)
        # so it still works with batch_size=1, same as the rest of the
        # model).
        # --------------------------------------------------------------
        self.fuse_gate = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1),
            nn.GroupNorm(1, hidden_dim),
            nn.ELU(alpha=1.0, inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        self.out_norm = nn.GroupNorm(1, hidden_dim)

    @property
    def slow_decay(self) -> torch.Tensor:
        """Slow-branch decay rate, constrained to [0.01, 0.30] - much
        slower than DualTemporalAttention's [0.05, 0.95] global decay, so
        this branch stays biased toward persistence."""
        return torch.sigmoid(self.slow_decay_logit) * 0.29 + 0.01

    def _mask_seq(self, hidden_seq: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Zero out invalid pixels in-place-safe fashion (returns a new tensor)."""
        if mask is None:
            return hidden_seq
        B, T, C, H, W = hidden_seq.shape
        if mask.dim() == 2:  # [H, W]
            m = mask.unsqueeze(0).unsqueeze(0).unsqueeze(2)
        elif mask.dim() == 3:  # [B, H, W] or [T, H, W]
            m = mask.unsqueeze(1).unsqueeze(2) if mask.shape[0] == B else mask.unsqueeze(0).unsqueeze(2)
        elif mask.dim() == 4:  # [B, T, H, W] or [B, 1, H, W]
            m = mask.unsqueeze(2)
        else:
            m = mask
        return hidden_seq * m

    def forward(self, hidden_seq: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            hidden_seq: [B, T, C, H, W] encoder hidden states.
            mask: optional spatial validity mask, same conventions as
                DualTemporalAttention.

        Returns:
            context: [B, C, H, W] fused multiscale context.
            info: dict with diagnostics (fast/slow gate, slow decay).
        """
        hidden_seq = self._mask_seq(hidden_seq, mask)
        B, T, C, H, W = hidden_seq.shape

        # ---- FAST BRANCH ----
        k = min(self.fast_window, T)
        recent = hidden_seq[:, T - k:]  # [B, k, C, H, W]
        scores = torch.stack(
            [self.fast_score(recent[:, t]) for t in range(k)], dim=1
        )  # [B, k, 1, H, W]
        weights = F.softmax(scores, dim=1)
        fast_context = (recent * weights).sum(dim=1)  # [B, C, H, W]

        # ---- SLOW BRANCH ----
        decay_val = self.slow_decay
        time_idx = torch.arange(T, device=hidden_seq.device).float()
        slow_weights = torch.exp(-decay_val * (T - 1 - time_idx))
        slow_weights = slow_weights / (slow_weights.sum() + 1e-8)
        slow_weights = slow_weights.view(1, T, 1, 1, 1)
        slow_pooled = (hidden_seq * slow_weights).sum(dim=1)  # [B, C, H, W]
        slow_context = self.slow_proj(slow_pooled)

        # ---- FUSION ----
        gate = self.fuse_gate(torch.cat([fast_context, slow_context], dim=1))  # [B, 1, H, W]
        fused = gate * fast_context + (1 - gate) * slow_context
        fused = self.out_norm(fused)

        return fused, {
            "multiscale_gate": gate.squeeze(1),
            "slow_decay": self.slow_decay.item(),
            "fast_window": k,
        }
