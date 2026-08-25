"""attention.py - Atenção Temporal Dual com suporte a máscara espacial"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DualTemporalAttention(nn.Module):
    """
    Atenção Temporal Dual para modelagem de memória climática.
    
    Combina atenção local (curto prazo) e global (longo prazo) com gate adaptativo.
    A máscara espacial é aplicada para ignorar pixels inválidos (oceano, sem dados).
    """
    
    def __init__(self, hidden_dim: int, dropout: float = 0.3):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout
        proj_dim = max(16, hidden_dim // 4)
        
        # Dropout para regularização
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
        
        # Decaimento com regularização (via logit)
        self.decay_logit = nn.Parameter(torch.tensor(0.0))
        
        # Gate - usando GroupNorm em vez de BatchNorm (funciona com batch_size=1)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1),
            nn.GroupNorm(1, hidden_dim),  # GroupNorm(1) = InstanceNorm, funciona com batch=1
            nn.ELU(alpha=1.0, inplace=True),
            self.dropout,
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.out_proj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.norm = nn.GroupNorm(1, hidden_dim)  # InstanceNorm equivalente
        
        # Skip connection residual
        self.skip_weight = nn.Parameter(torch.tensor(0.1))
    
    @property
    def decay(self) -> torch.Tensor:
        """Decaimento entre 0.05 e 0.95 para evitar extremos."""
        return torch.sigmoid(self.decay_logit) * 0.9 + 0.05
    
    def _exponential_pool(
        self, 
        hidden_seq: torch.Tensor, 
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Pooling temporal exponencial com decaimento controlado.
        
        Args:
            hidden_seq: Sequência de estados ocultos [B, T, C, H, W]
            mask: Máscara espacial/temporal [B, T, 1, H, W] ou None
        
        Returns:
            Contexto temporal ponderado [B, C, H, W]
        """
        B, T, C, H, W = hidden_seq.shape
        decay_val = self.decay
        time_idx = torch.arange(T, device=hidden_seq.device).float()
        weights = torch.exp(-decay_val * (T - 1 - time_idx))
        weights = weights / (weights.sum() + 1e-8)
        weights = weights.view(1, T, 1, 1, 1)
        
        # ✅ Aplicar máscara no pool (se disponível)
        if mask is not None:
            # Garantir que a máscara tenha o formato correto [B, T, 1, H, W]
            if mask.dim() == 2:  # [H, W]
                mask = mask.unsqueeze(0).unsqueeze(0).unsqueeze(2)
            elif mask.dim() == 3:  # [B, H, W] ou [T, H, W]
                if mask.shape[0] == B:  # [B, H, W]
                    mask = mask.unsqueeze(1).unsqueeze(2)
                else:  # [T, H, W]
                    mask = mask.unsqueeze(0).unsqueeze(2)
            elif mask.dim() == 4:  # [B, T, H, W] ou [B, 1, H, W]
                if mask.shape[1] == T:  # [B, T, H, W]
                    mask = mask.unsqueeze(2)  # [B, T, 1, H, W]
                else:  # [B, 1, H, W]
                    mask = mask.unsqueeze(2)  # [B, 1, 1, H, W]
            
            # Aplicar máscara aos pesos
            weights = weights * mask
            # Renormalizar para evitar que pixels inválidos influenciem
            weights_sum = weights.sum(dim=1, keepdim=True) + 1e-8
            weights = weights / weights_sum
        
        return (hidden_seq * weights).sum(dim=1)
    
    def forward(self, hidden_seq: torch.Tensor, mask: torch.Tensor = None):
        """
        Forward pass da atenção temporal dual.
        
        Args:
            hidden_seq: Sequência de estados ocultos [B, T, C, H, W]
            mask: Máscara espacial [H, W] ou [B, H, W] ou [B, T, H, W]
        
        Returns:
            output: Contexto atencional [B, C, H, W]
            info: Dicionário com informações (weights, gate, decay)
        """
        B, T, C, H, W = hidden_seq.shape
        
        # ================================================================
        # 1. PREPARAR MÁSCARA
        # ================================================================
        mask_expanded = None
        if mask is not None:
            # Garantir formato [B, T, 1, H, W]
            if mask.dim() == 2:  # [H, W]
                mask_expanded = mask.unsqueeze(0).unsqueeze(0).unsqueeze(2)
            elif mask.dim() == 3:  # [B, H, W] ou [T, H, W]
                if mask.shape[0] == B:  # [B, H, W]
                    mask_expanded = mask.unsqueeze(1).unsqueeze(2)
                else:  # [T, H, W]
                    mask_expanded = mask.unsqueeze(0).unsqueeze(2)
            elif mask.dim() == 4:  # [B, T, H, W] ou [B, 1, H, W]
                if mask.shape[1] == T:  # [B, T, H, W]
                    mask_expanded = mask.unsqueeze(2)  # [B, T, 1, H, W]
                else:  # [B, 1, H, W]
                    mask_expanded = mask.unsqueeze(2)  # [B, 1, 1, H, W]
            
            # Aplicar máscara ao hidden_seq (zerar pixels inválidos)
            hidden_seq = hidden_seq * mask_expanded
        
        # ================================================================
        # 2. CALCULAR ATENÇÃO LOCAL E GLOBAL
        # ================================================================
        local_scores = []
        global_scores = []
        
        for t in range(T):
            # Atenção local: apenas o instante atual
            h_t = hidden_seq[:, t]
            local_scores.append(self.local_attn(h_t))
            
            # Atenção global: contexto temporal até o instante atual
            # ✅ Passar máscara para o pool (apenas timesteps até t)
            if mask_expanded is not None:
                mask_t = mask_expanded[:, :t+1]  # [B, t+1, 1, H, W]
            else:
                mask_t = None
            
            context = self._exponential_pool(hidden_seq[:, :t+1], mask_t)
            global_scores.append(self.global_attn(context))
        
        local_attn = torch.stack(local_scores, dim=1)   # [B, T, 1, H, W]
        global_attn = torch.stack(global_scores, dim=1)  # [B, T, 1, H, W]
        
        # ================================================================
        # 3. GATE ADAPTATIVO
        # ================================================================
        h_mean = hidden_seq.mean(dim=1)        # [B, C, H, W]
        h_std = hidden_seq.std(dim=1)          # [B, C, H, W]
        gate_input = torch.cat([h_mean, h_std], dim=1)  # [B, 2C, H, W]
        gate = self.gate(gate_input)           # [B, 1, H, W]
        gate = gate.unsqueeze(1).expand(-1, T, -1, -1, -1)  # [B, T, 1, H, W]
        
        # ================================================================
        # 4. COMBINAR ATENÇÕES
        # ================================================================
        combined_attn = gate * local_attn + (1 - gate) * global_attn
        combined_attn = F.softmax(combined_attn, dim=1)
        attended = (combined_attn * hidden_seq).sum(dim=1)  # [B, C, H, W]
        
        # ================================================================
        # 5. PROJEÇÃO E RESIDUAL
        # ================================================================
        output = self.out_proj(attended)
        output = self.norm(output)
        
        # Residual connection com o último estado oculto
        output = output + self.skip_weight * hidden_seq[:, -1]
        
        return output, {
            "weights": combined_attn.squeeze(2),
            "gate": gate.squeeze(2),
            "decay": self.decay.item()
        }