import torch
import torch.nn as nn
import torch.nn.functional as F


class GeoCorrectionNet(nn.Module):
    """Predict a lightweight dense offset field for geometric correction."""

    def __init__(self, in_channels=3, base_channels=16, max_delta=0.08):
        super().__init__()
        self.max_delta = float(max_delta)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(base_channels * 4, 2, kernel_size=3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        features = self.encoder(x)
        delta = self.head(features)
        delta = F.interpolate(delta, size=x.shape[-2:], mode='bilinear', align_corners=True)
        return torch.tanh(delta) * self.max_delta
