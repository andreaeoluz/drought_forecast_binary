"""encoder.py - Multi-layer ConvLSTM encoder with spatial mask support."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .convlstm_cell import ConvLSTMCell


class ConvLSTMEncoder(nn.Module):
    """Multi-layer ConvLSTM encoder with spatial mask support."""

    def __init__(self, input_dim: int, hidden_dims: list, kernel_size: int = 3):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.kernel_size = kernel_size

        self.layers = nn.ModuleList()
        for i, hdim in enumerate(hidden_dims):
            in_dim = input_dim if i == 0 else hidden_dims[i - 1]
            self.layers.append(ConvLSTMCell(in_dim, hdim, kernel_size))

        self.input_proj = nn.Conv2d(input_dim, input_dim, kernel_size=1)
        self.input_norm = nn.GroupNorm(1, input_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        Forward pass of the encoder.

        Args:
            x: Input [B, T, C, H, W].
            mask: Spatial mask [H, W] or [B, H, W].

        Returns:
            latent: Final state [B, C, H, W].
            hidden_seq: Sequence of hidden states [B, T, C, H, W].
        """
        B, T, C, H, W = x.shape
        device = x.device

        if mask is not None:
            if mask.dim() == 2:  # [H, W]
                mask_expanded = mask.unsqueeze(0).unsqueeze(0).unsqueeze(2)  # [1, 1, 1, H, W]
            elif mask.dim() == 3:  # [B, H, W]
                mask_expanded = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, H, W]
            else:
                mask_expanded = mask

            x = x * mask_expanded

        states = [layer.init_hidden(B, (H, W), device) for layer in self.layers]
        hidden_states = []

        for t in range(T):
            inp = x[:, t]
            inp = self.input_proj(inp)
            inp = self.input_norm(inp)
            inp = F.elu(inp, alpha=1.0)

            for i, layer in enumerate(self.layers):
                h_next, c_next = layer(inp, states[i])
                states[i] = (h_next, c_next)
                inp = h_next

            hidden_states.append(h_next)

        hidden_seq = torch.stack(hidden_states, dim=1)
        latent = states[-1][0]

        return latent, hidden_seq

    # ========================================================================
    # PARTIAL / PROGRESSIVE TRANSFER LEARNING SUPPORT
    #
    # Layer 0 processes the raw input and learns the most general climate
    # patterns (see improvement doc, item 1 - "Transfer Learning parcial e
    # progressivo"); later layers are progressively more task-specific.
    # These helpers let a caller freeze/unfreeze *individual* ConvLSTM
    # layers instead of the whole encoder at once, so training can start
    # with everything frozen, unfreeze the deepest (most task-specific)
    # layer first, and only unfreeze layer 0 - the most general one - last
    # and with the smallest learning rate.
    # ========================================================================

    def freeze_layers(self, layer_indices) -> None:
        """Freeze the given encoder layer indices (0 = first/most general)."""
        for i in layer_indices:
            for param in self.layers[i].parameters():
                param.requires_grad = False

    def unfreeze_layers(self, layer_indices) -> None:
        """Unfreeze the given encoder layer indices."""
        for i in layer_indices:
            for param in self.layers[i].parameters():
                param.requires_grad = True

    def freeze_all(self, freeze: bool = True) -> None:
        """Freeze/unfreeze every ConvLSTM layer (input_proj/input_norm too)."""
        for param in self.parameters():
            param.requires_grad = not freeze

    def layer_param_groups(self) -> list:
        """
        Return one parameter list per ConvLSTM layer, ordered shallow (0,
        most general) to deep (-1, most task-specific), plus a final group
        for input_proj/input_norm. Used by the trainer to build the
        progressive-unfreeze optimizer groups with per-layer learning rates.
        """
        groups = [list(layer.parameters()) for layer in self.layers]
        groups.append(list(self.input_proj.parameters()) + list(self.input_norm.parameters()))
        return groups
