"""convlstm_cell.py - Célula ConvLSTM"""

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """
    Célula ConvLSTM para processamento espaço-temporal.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3, bias: bool = True):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        
        padding = kernel_size // 2
        
        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=padding,
            bias=bias
        )
        
        nn.init.orthogonal_(self.conv.weight)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)
            self.conv.bias.data[self.hidden_dim:2*self.hidden_dim].fill_(1.0)
    
    def forward(self, x: torch.Tensor, state: tuple):
        h_cur, c_cur = state
        combined = torch.cat([x, h_cur], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.split(gates, self.hidden_dim, dim=1)
        
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next
    
    def init_hidden(self, batch_size: int, spatial_size: tuple, device: torch.device):
        H, W = spatial_size
        h = torch.zeros(batch_size, self.hidden_dim, H, W, device=device)
        c = torch.zeros(batch_size, self.hidden_dim, H, W, device=device)
        return h, c