"""
ELEC 475: Lab 3 - Correction Model (Starter Template)
A small CNN that predicts the offset from a patch centre to the true nose.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CorrectionNet(nn.Module):
    """
    CorrectionNet for (3, 64, 64) ImageNet-normalized patches.

    Spatial dimension arithmetic (each block halves the size with 2x2 max-pooling):
        Input        : (B, 3, 64, 64)
        Block 1      : (B, 32, 32, 32)
        Block 2      : (B, 64, 16, 16)
        Block 3      : (B, 128, 8, 8)
        Block 4      : (B, 128, 4, 4)
        Flatten      : 128 * 4 * 4 = 2,048 features
        FC1          : 2,048 -> 256
        FC2          : 256 -> 2, the normalized offset (du, dv) / HALF_PATCH
    """
    def __init__(self):
        super(CorrectionNet, self).__init__()

        self.block1 = self._conv_block(3, 32)
        self.block2 = self._conv_block(32, 64)
        self.block3 = self._conv_block(64, 128)
        self.block4 = self._conv_block(128, 128)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 2)

    @staticmethod
    def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class ResidualConvBlock(nn.Module):
    """Two-convolution residual block followed by 2x2 downsampling."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))
        return F.relu(self.pool(x + residual), inplace=True)


class CorrectionNetModified(nn.Module):
    """
    Residual-block variant of CorrectionNet with dropout before its output layer.
    It uses the same input (B, 3, 64, 64) and output (B, 2) normalized offsets.
    """
    def __init__(self):
        super(CorrectionNetModified, self).__init__()
        self.block1 = ResidualConvBlock(3, 32)
        self.block2 = ResidualConvBlock(32, 64)
        self.block3 = ResidualConvBlock(64, 128)
        self.block4 = ResidualConvBlock(128, 128)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.dropout = nn.Dropout(p=0.25)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


if __name__ == "__main__":
    from torchinfo import summary
    for model in [CorrectionNet(), CorrectionNetModified()]:
        print(f"Testing {type(model).__name__} Architecture:")
        try:
            summary(model, input_size=(1, 3, 64, 64))
        except Exception as e:
            print(f"Model not fully implemented yet: {e}")
