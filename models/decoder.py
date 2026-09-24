"""decoder.py - Decoders for the autoencoder and the predictor."""

import numpy as np
import torch
import torch.nn as nn


class ReconstructionDecoder(nn.Module):
    """Decoder for reconstructing climate variables from a latent representation."""

    def __init__(self, latent_dim: int, output_dim: int, kernel_size: int = 3):
        super().__init__()

        padding = kernel_size // 2

        hidden1 = max(64, latent_dim * 2)
        hidden2 = max(32, hidden1 // 2)
        hidden3 = max(16, hidden2 // 2)

        self.net = nn.Sequential(
            nn.Conv2d(latent_dim, hidden1, kernel_size, padding=padding),
            nn.GroupNorm(1, hidden1),
            nn.ELU(alpha=1.0, inplace=True),
            nn.Dropout2d(0.1),

            nn.Conv2d(hidden1, hidden2, kernel_size, padding=padding),
            nn.GroupNorm(1, hidden2),
            nn.ELU(alpha=1.0, inplace=True),
            nn.Dropout2d(0.1),

            nn.Conv2d(hidden2, hidden3, kernel_size, padding=padding),
            nn.GroupNorm(1, hidden3),
            nn.ELU(alpha=1.0, inplace=True),

            nn.Conv2d(hidden3, output_dim, kernel_size=1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor):
        return self.net(z)


class PredictionDecoder(nn.Module):
    """Decoder for binary drought classification."""

    def __init__(self, latent_dim: int, kernel_size: int = 3, prevalence: float = 0.025):
        super().__init__()

        self.prevalence = prevalence
        padding = kernel_size // 2

        hidden1 = max(64, latent_dim * 2)
        hidden2 = max(32, hidden1 // 2)
        hidden3 = max(16, hidden2 // 2)

        self.net = nn.Sequential(
            nn.Conv2d(latent_dim, hidden1, kernel_size, padding=padding),
            nn.GroupNorm(1, hidden1),
            nn.ELU(alpha=1.0, inplace=True),
            nn.Dropout2d(0.2),

            nn.Conv2d(hidden1, hidden2, kernel_size, padding=padding),
            nn.GroupNorm(1, hidden2),
            nn.ELU(alpha=1.0, inplace=True),
            nn.Dropout2d(0.2),

            nn.Conv2d(hidden2, hidden3, kernel_size, padding=padding),
            nn.GroupNorm(1, hidden3),
            nn.ELU(alpha=1.0, inplace=True),

            nn.Conv2d(hidden3, 1, kernel_size=1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Initialize the last layer's bias from the expected class
        # prevalence, so initial predictions are roughly calibrated to the
        # drought base rate before training begins.
        last_conv = self.net[-1]
        nn.init.normal_(last_conv.weight, mean=0.0, std=0.01)

        prevalence = self.prevalence
        adjusted_prevalence = max(prevalence * 2, 0.05)
        bias_init = np.log(adjusted_prevalence / (1 - adjusted_prevalence))
        nn.init.constant_(last_conv.bias, bias_init)

    def forward(self, z: torch.Tensor):
        return self.net(z)
