"""autoencoder.py - Modelo ConvLSTM Autoencoder com reconstrução de sequência"""

import torch
import torch.nn as nn

from .encoder import ConvLSTMEncoder
from .attention import DualTemporalAttention
from .decoder import ReconstructionDecoder


class ConvLSTMAutoencoder(nn.Module):
    """
    ConvLSTM Autoencoder para pré-treinamento com reconstrução de sequência.
    
    O modelo aprende uma representação compacta de séries temporais climáticas
    através da reconstrução da sequência completa, com:
    - Encoder ConvLSTM multicamada
    - Atenção temporal dual (local + global)
    - Decodificador temporal com skip connections
    - Ponderação temporal exponencial
    - Penalidade de diversidade para evitar colapso do latente
    """
    
    def __init__(self, config: dict):
        super().__init__()
        
        # Configurações principais
        self.config = config
        self.input_dim = config["input_dim"]
        self.hidden_dims = config["hidden_dims"]
        self.kernel_size = config.get("kernel_size", 3)
        self.output_dim = config.get("output_dim", self.input_dim)
        self.use_attention = config.get("use_attention", True)
        self.attention_dropout = config.get("attention_dropout", 0.3)
        
        # Configurações de reconstrução temporal
        self.reconstruct_full_sequence = config.get("reconstruct_full_sequence", True)
        self.sequence_length = config.get("sequence_length", 12)
        self.decay_rate = config.get("decay_rate", 0.6)
        self.skip_weight = config.get("skip_weight", 0.1)
        
        # ✅ Peso da penalidade de diversidade (configurável)
        self.diversity_weight = config.get("diversity_weight", 0.05)
        
        # ====================================================================
        # ENCODER
        # ====================================================================
        self.encoder = ConvLSTMEncoder(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            kernel_size=self.kernel_size
        )
        
        # ====================================================================
        # ATENÇÃO TEMPORAL
        # ====================================================================
        if self.use_attention:
            self.attention = DualTemporalAttention(
                self.hidden_dims[-1],
                dropout=self.attention_dropout
            )
        else:
            self.attention = None
        
        # ====================================================================
        # DECODER
        # ====================================================================
        self.decoder = ReconstructionDecoder(
            latent_dim=self.hidden_dims[-1],
            output_dim=self.output_dim,
            kernel_size=self.kernel_size
        )
        
        # Skip projection para conexões residuais
        self.skip_proj = nn.Conv2d(self.input_dim, self.output_dim, kernel_size=1)
        
        # ====================================================================
        # DECODIFICADOR TEMPORAL (reconstrução da sequência completa)
        # ====================================================================
        if self.reconstruct_full_sequence:
            # Projeção temporal: latent → sequência completa
            self.temporal_decoder = nn.Sequential(
                nn.Conv2d(self.hidden_dims[-1], self.hidden_dims[-1], kernel_size=1),
                nn.GroupNorm(1, self.hidden_dims[-1]),
                nn.ELU(alpha=1.0, inplace=True),
                nn.Conv2d(
                    self.hidden_dims[-1],
                    self.sequence_length * self.output_dim,
                    kernel_size=1
                ),
            )
            
            # Refinamento por instante
            self.temporal_refine = nn.Sequential(
                nn.Conv2d(self.output_dim, self.output_dim, kernel_size=3, padding=1),
                nn.GroupNorm(1, self.output_dim),
                nn.ELU(alpha=1.0, inplace=True),
                nn.Conv2d(self.output_dim, self.output_dim, kernel_size=1),
            )
            
            # Pesos temporais (buffer não treinável)
            self._init_temporal_weights()
    
    def _init_temporal_weights(self) -> None:
        """Inicializa pesos temporais com decaimento exponencial."""
        T = self.sequence_length
        time_idx = torch.arange(T, dtype=torch.float)
        weights = torch.exp(-self.decay_rate * (T - 1 - time_idx))
        weights = weights / (weights.sum() + 1e-8)
        weights = weights * T
        self.register_buffer("temporal_weights", weights.view(1, T, 1, 1, 1))
    
    def _reconstruct_sequence(self, latent: torch.Tensor) -> torch.Tensor:
        """Reconstrói toda a sequência temporal a partir do latente."""
        B, C, H, W = latent.shape
        
        out = self.temporal_decoder(latent)
        out = out.view(B, self.sequence_length, self.output_dim, H, W)
        
        B, T, C, H, W = out.shape
        out_flat = out.view(B * T, C, H, W)
        out_flat = self.temporal_refine(out_flat)
        out = out_flat.view(B, T, C, H, W)
        
        return out
    
    def _apply_skip_to_sequence(
        self,
        x: torch.Tensor,
        recon_seq: torch.Tensor
    ) -> torch.Tensor:
        """Aplica skip connections a cada instante da sequência."""
        B, T, C, H, W = x.shape
        
        skip_flat = x.view(B * T, C, H, W)
        skip_flat = self.skip_proj(skip_flat)
        skip = skip_flat.view(B, T, self.output_dim, H, W)
        
        return recon_seq + self.skip_weight * skip
    
    def _compute_temporal_loss(
        self,
        pred_seq: torch.Tensor,
        target_seq: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """Calcula perda com ponderação temporal e normalização correta da máscara."""
        B, T, C, H, W = pred_seq.shape
        
        diff = torch.abs(pred_seq - target_seq)
        beta = 0.1
        loss_per_element = torch.where(
            diff < beta,
            0.5 * diff ** 2 / beta,
            diff - 0.5 * beta
        )
        
        if mask is not None:
            while mask.dim() < 5:
                mask = mask.unsqueeze(0) if mask.dim() == 2 else mask.unsqueeze(1)
            
            if mask.shape[2] == 1:
                mask = mask.expand(-1, -1, C, -1, -1)
            
            masked_loss = loss_per_element * mask
            valid_pixels = mask.sum()
            
            if valid_pixels > 0:
                loss = masked_loss.sum() / valid_pixels
            else:
                return torch.tensor(0.0, device=pred_seq.device)
        else:
            loss = loss_per_element.mean()
        
        return loss
    
    def forward(
        self,
        x: torch.Tensor,
        return_info: bool = False
    ):
        """
        Forward pass do autoencoder.
        
        Args:
            x: Entrada [B, T, C, H, W]
            return_info: Se True, retorna informações adicionais
        
        Returns:
            recon: Reconstrução do último instante [B, C_out, H, W]
            info: Dicionário com informações (se return_info=True)
        """
        B, T, C, H, W = x.shape
        
        # ====================================================================
        # ENCODER
        # ====================================================================
        latent, hidden_seq = self.encoder(x)
        info = {"latent": latent}
        
        # ====================================================================
        # ATENÇÃO TEMPORAL
        # ====================================================================
        if self.attention is not None:
            context, attn_info = self.attention(hidden_seq)
            info.update(attn_info)
            latent = latent + context * 0.3
            info["latent_with_attention"] = latent
        
        # ====================================================================
        # RECONSTRUÇÃO
        # ====================================================================
        if self.reconstruct_full_sequence:
            recon_seq = self._reconstruct_sequence(latent)
            recon_seq = self._apply_skip_to_sequence(x, recon_seq)
            recon = recon_seq[:, -1]
            info["recon_seq"] = recon_seq
            info["recon_loss_type"] = "sequence"
        else:
            recon = self.decoder(latent)
            last_input = x[:, -1]
            skip = self.skip_proj(last_input)
            recon = recon + skip
        
        if return_info:
            return recon, info
        return recon, info
    
    def compute_loss(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        variable_weights: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Calcula a perda de reconstrução com ponderação temporal, por variável,
        e penalidade de diversidade para evitar colapso do latente.
        """
        recon, info = self.forward(x, return_info=True)
        
        # ================================================================
        # 1. PERDA DE RECONSTRUÇÃO
        # ================================================================
        if self.reconstruct_full_sequence and "recon_seq" in info:
            target_seq = torch.nan_to_num(x, nan=0.0)
            recon_loss = self._compute_temporal_loss(info["recon_seq"], target_seq, mask)
            
            if variable_weights is not None:
                B, T, C, H, W = info["recon_seq"].shape
                diff = torch.abs(info["recon_seq"] - target_seq)
                beta = 0.1
                loss_per_element = torch.where(
                    diff < beta,
                    0.5 * diff ** 2 / beta,
                    diff - 0.5 * beta
                )
                
                var_weights = variable_weights.view(1, 1, -1, 1, 1)
                loss_per_element = loss_per_element * var_weights
                
                if mask is not None:
                    if mask.dim() == 2:
                        mask = mask.unsqueeze(0).unsqueeze(0).unsqueeze(2)
                    elif mask.dim() == 3:
                        mask = mask.unsqueeze(1).unsqueeze(2)
                    elif mask.dim() == 4:
                        mask = mask.unsqueeze(2)
                    
                    loss_per_element = loss_per_element * mask
                    valid_pixels = mask.sum()
                    if valid_pixels > 0:
                        loss_per_element = loss_per_element * (loss_per_element.numel() / valid_pixels)
                    else:
                        return torch.tensor(0.0, device=x.device)
                
                loss_per_time = loss_per_element.mean(dim=(0, 2, 3, 4))
                temporal_weights = self.temporal_weights.squeeze()
                recon_loss = (loss_per_time * temporal_weights).sum()
                recon_loss = recon_loss / temporal_weights.sum()
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
        # 2. PENALIDADE DE DIVERSIDADE
        # ================================================================
        diversity_loss = torch.tensor(0.0, device=x.device)
        
        # ✅ USAR latent_with_attention (latente real usado na reconstrução)
        if 'latent_with_attention' in info:
            latent = info['latent_with_attention']  # [B, C, H, W]
            
            # Calcular variância por canal
            var_per_channel = latent.var(dim=(0, 2, 3))  # [C]
            mean_var = var_per_channel.mean()
            
            # ✅ Penalidade exponencial (mais suave que 1/var)
            # Quanto menor a variância, maior a penalidade
            diversity_loss = torch.exp(-mean_var * 10.0)
            
            # ✅ Log para depuração (será impresso durante o treino)
            if torch.rand(1).item() < 0.01:  # 1% das iterações
                print(f"  📊 Recon Loss: {recon_loss.item():.6f}")
                print(f"  📊 Div Loss: {diversity_loss.item():.6f}")
                print(f"  📊 Mean Var: {mean_var.item():.6f}")
            
            # ✅ Peso da penalidade (configurável)
            loss = recon_loss + self.diversity_weight * diversity_loss
        else:
            loss = recon_loss
        
        return loss
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Retorna representação latente SEM atenção."""
        latent, _ = self.encoder(x)
        return latent
    
    def encode_with_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Retorna representação latente COM atenção."""
        latent, hidden_seq = self.encoder(x)
        if self.attention is not None:
            context, _ = self.attention(hidden_seq)
            latent = latent + context * 0.3
        return latent
    
    def get_encoder_state_dict(self) -> dict:
        """Retorna os pesos do encoder para transfer learning."""
        encoder_state = {}
        for key, value in self.encoder.state_dict().items():
            encoder_state[f"encoder.{key}"] = value
        return encoder_state
    
    def get_attention_state_dict(self) -> dict:
        """Retorna os pesos da atenção para transfer learning."""
        if self.attention is None:
            return {}
        
        attention_state = {}
        for key, value in self.attention.state_dict().items():
            attention_state[f"attention.{key}"] = value
        return attention_state