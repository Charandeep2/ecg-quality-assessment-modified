"""
PyTorch Deep Tensor Neural Network (T-DNN) architecture and loss functions.
Shared across pretraining, fine-tuning, cross-validation, and model export.
"""

import sys
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def get_device() -> torch.device:
    """Return CUDA device if available (e.g. RTX 3050), else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DeepTensorResNet(nn.Module):
    """
    Deep Residual MLP with Squeeze-and-Excitation Attention for 
    multilinear tensor singular values and physical ECG characteristics.
    """
    def __init__(self, in_features: int = 31, latent_dim: int = 128):
        super().__init__()
        self.in_features = in_features
        self.latent_dim = latent_dim

        # 1. Input Projection Layer
        self.projector = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        # 2. Residual Block with Mode Attention
        self.res1 = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
        )

        # Squeeze-and-Excitation Attention over modes
        self.se_attention = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 256),
            nn.Sigmoid(),
        )

        # 3. Latent Representation Bottleneck
        self.latent_layer = nn.Sequential(
            nn.Linear(256, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.GELU(),
        )

        # 4. Pretext Reconstruction Decoder Head (Self-Supervised Pretraining)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Linear(256, in_features),
        )

        # 5. Supervised Quality Classification Head (Fine-Tuning)
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2),  # [0: Acceptable, 1: Unacceptable]
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Extract latent representation vector."""
        h = self.projector(x)
        residual = self.res1(h)
        attn = self.se_attention(residual)
        h = h + (residual * attn)
        return self.latent_layer(h)

    def forward_pretrain(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct masked / noisy input tensor features."""
        z = self.encode(x)
        return self.decoder(z)

    def forward_classify(self, x: torch.Tensor) -> torch.Tensor:
        """Predict binary ECG quality logits."""
        z = self.encode(x)
        return self.classifier(z)


class FocalLoss(nn.Module):
    """
    Focal Loss for severe class imbalance (773 Acceptable vs 225 Unacceptable).
    Down-weights easy clean examples and focuses gradients on corrupted records.
    """
    def __init__(self, alpha=None, gamma=2.0, device=None):
        super().__init__()
        self.device = device or get_device()
        if alpha is None:
            alpha = torch.tensor([1.0, 3.43])  # 773 / 225 ratio
        self.alpha = alpha.to(self.device)
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=self.alpha, reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()
