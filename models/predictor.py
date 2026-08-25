"""predictor.py - Modelo ConvLSTM para classificação de seca"""

import torch
import torch.nn as nn

from .encoder import ConvLSTMEncoder
from .attention import DualTemporalAttention
from .decoder import PredictionDecoder


class ConvLSTMPredictor(nn.Module):
    """
    ConvLSTM Predictor para classificação binária de seca extrema.
    
    O modelo utiliza:
    - Encoder ConvLSTM multicamada (pode ser pré-treinado via autoencoder)
    - Atenção temporal dual (local + global)
    - Decodificador para classificação binária
    - Suporte para transfer learning com carregamento seletivo de pesos
    """
    
    def __init__(self, config: dict, prevalence: float = 0.025):
        """
        Inicializa o predictor.
        
        Args:
            config: Dicionário com configurações do modelo
            prevalence: Prevalência esperada da classe positiva (para inicialização do bias)
        """
        super().__init__()

        self.config = config
        self.input_dim = config["input_dim"]
        self.hidden_dims = config["hidden_dims"]
        self.kernel_size = config.get("kernel_size", 3)
        self.output_dim = config.get("output_dim", 1)
        self.use_attention = config.get("use_attention", True)
        self.attention_dropout = config.get("attention_dropout", 0.3)
        self.prevalence = prevalence

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
        # DECODER PARA CLASSIFICAÇÃO
        # ====================================================================
        self.decoder = PredictionDecoder(
            latent_dim=self.hidden_dims[-1],
            kernel_size=self.kernel_size,
            prevalence=prevalence
        )

    def forward(self, x: torch.Tensor, return_info: bool = False):
        """
        Forward pass do predictor.
        
        Args:
            x: Entrada [B, T, C, H, W]
            return_info: Se True, retorna informações adicionais
        
        Returns:
            logits: Logits de classificação [B, 1, H, W]
            info: Dicionário com informações (se return_info=True)
        """
        # Encoder
        latent, hidden_seq = self.encoder(x)
        info = {}

        # Atenção temporal
        if self.attention is not None:
            context, attn_info = self.attention(hidden_seq)
            info.update(attn_info)
            # Combinação: estado final + contexto atencional
            latent = latent + context * 0.3

        # Decoder para classificação
        logits = self.decoder(latent)

        if return_info:
            return logits, info
        return logits, info

    def load_encoder_from_autoencoder(
        self,
        ae_checkpoint: dict,
        strict: bool = False,
        load_attention: bool = True
    ) -> None:
        """
        Carrega pesos do encoder e (opcionalmente) da atenção a partir de autoencoder pré-treinado.
        
        Args:
            ae_checkpoint: Checkpoint do autoencoder contendo model_state_dict
            strict: Se True, exige que todos os pesos sejam carregados
            load_attention: Se True, carrega também os pesos da atenção
        
        Nota:
            - O encoder é sempre transferido (quando disponível)
            - A atenção é transferida apenas se load_attention=True
            - A compatibilidade de shapes é verificada automaticamente
        """
        ae_state = ae_checkpoint.get("model_state_dict", ae_checkpoint)

        encoder_state = {}
        attention_state = {}

        # ====================================================================
        # 1. TRANSFERIR PESOS DO ENCODER (sempre)
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
                  f"pesos transferidos")
        else:
            print(f"  ⚠️ Encoder: nenhum peso compatível encontrado")

        # ====================================================================
        # 2. TRANSFERIR PESOS DA ATENÇÃO (opcional)
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
                      f"pesos transferidos")
            else:
                print(f"  ⚠️ Attention: nenhum peso compatível encontrado (mantida aleatória)")
        elif self.attention is not None and not load_attention:
            print(f"  ⚠️ Attention: mantida aleatória (load_attention=False)")

    def freeze_encoder(self, freeze: bool = True) -> None:
        """
        Congela/descongela os parâmetros do encoder.
        
        Args:
            freeze: Se True, congela o encoder; se False, descongela
        
        Nota:
            A atenção NUNCA é congelada, permanecendo sempre treinável.
            Isso permite que o modelo se adapte à tarefa downstream.
        """
        # Congelar/descongelar encoder
        for param in self.encoder.parameters():
            param.requires_grad = not freeze
        
        # Atenção permanece sempre treinável
        if self.attention is not None:
            for param in self.attention.parameters():
                param.requires_grad = True

    def freeze_attention(self, freeze: bool = True) -> None:
        """
        Congela/descongela os parâmetros da atenção.
        
        Args:
            freeze: Se True, congela a atenção; se False, descongela
        
        Nota:
            Método adicional para controle fino do transfer learning.
            Por padrão, a atenção NUNCA é congelada.
        """
        if self.attention is not None:
            for param in self.attention.parameters():
                param.requires_grad = not freeze

    def get_trainable_params_count(self) -> dict:
        """
        Retorna a contagem de parâmetros treináveis por componente.
        
        Returns:
            Dicionário com contagem de parâmetros por componente
        """
        encoder_params = sum(
            p.numel() for p in self.encoder.parameters() if p.requires_grad
        )
        attention_params = (
            sum(p.numel() for p in self.attention.parameters() if p.requires_grad)
            if self.attention else 0
        )
        decoder_params = sum(
            p.numel() for p in self.decoder.parameters() if p.requires_grad
        )

        return {
            "encoder": encoder_params,
            "attention": attention_params,
            "decoder": decoder_params,
            "total": encoder_params + attention_params + decoder_params
        }

    def get_encoder_state_dict(self) -> dict:
        """
        Retorna os pesos do encoder para exportação.
        
        Returns:
            Dicionário com os pesos do encoder
        """
        return self.encoder.state_dict()

    def get_attention_state_dict(self) -> dict:
        """
        Retorna os pesos da atenção para exportação.
        
        Returns:
            Dicionário com os pesos da atenção (ou vazio se não houver)
        """
        if self.attention is None:
            return {}
        return self.attention.state_dict()

    def encode_with_attention(self, x: torch.Tensor) -> torch.Tensor:
        """
        Retorna representação latente COM atenção.
        
        Esta é a representação equivalente à usada no autoencoder durante o pré-treino,
        permitindo consistência entre as fases de treinamento.
        
        Args:
            x: Entrada [B, T, C, H, W]
        
        Returns:
            Latente com atenção [B, C, H, W]
        """
        latent, hidden_seq = self.encoder(x)
        if self.attention is not None:
            context, _ = self.attention(hidden_seq)
            latent = latent + context * 0.3
        return latent

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Retorna representação latente SEM atenção.
        
        Args:
            x: Entrada [B, T, C, H, W]
        
        Returns:
            Latente puro [B, C, H, W]
        """
        latent, _ = self.encoder(x)
        return latent