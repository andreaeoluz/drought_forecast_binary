"""encoder.py - Encoder ConvLSTM multicamada com suporte a máscara"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .convlstm_cell import ConvLSTMCell


class ConvLSTMEncoder(nn.Module):
    """
    Encoder ConvLSTM multicamada com suporte a máscara espacial.
    """
    
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
        Forward pass do encoder.
        
        Args:
            x: Entrada [B, T, C, H, W]
            mask: Máscara espacial [H, W] ou [B, H, W]
        
        Returns:
            latent: Estado final [B, C, H, W]
            hidden_seq: Sequência de estados [B, T, C, H, W]
        """
        B, T, C, H, W = x.shape
        device = x.device
        
        # ✅ Aplicar máscara se disponível
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