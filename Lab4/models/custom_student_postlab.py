import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


def conv_bn_hardswish(in_channels, out_channels, kernel_size=1):
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=kernel_size // 2,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels),
        nn.Hardswish(inplace=True),
    )


class SeparableContext(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.Hardswish(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Hardswish(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class CustomStudentNet(nn.Module):
    """MobileNetV3-Small encoder with a compact multi-scale stride-16 decoder."""

    low_channels: int = 24
    high_channels: int = 48

    def __init__(self, num_classes=21, pretrained=True):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        features = mobilenet_v3_small(weights=weights).features
        self.low = features[:4]
        self.mid = features[4:7]
        self.high = features[7:9]

        context_channels = 32
        self.context_input = conv_bn_hardswish(self.high_channels, context_channels)
        self.context_branches = nn.ModuleList(
            [SeparableContext(context_channels, rate) for rate in (1, 2, 4)]
        )
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.high_channels, context_channels, kernel_size=1, bias=False),
            nn.Hardswish(inplace=True),
        )
        self.context_fuse = conv_bn_hardswish(context_channels * 5, 64)
        self.low_project = conv_bn_hardswish(self.low_channels, 24)
        self.decoder = SeparableContext(88, dilation=1)
        self.classifier = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(88, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.Hardswish(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1),
        )

    def forward(self, x):
        output_size = x.shape[-2:]
        low = self.low(x)
        mid = self.mid(low)
        high = self.high(mid)

        context_input = self.context_input(high)
        context = [context_input]
        context.extend(branch(context_input) for branch in self.context_branches)
        pooled = self.global_context(high)
        context.append(
            F.interpolate(
                pooled,
                size=high.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        )
        context = self.context_fuse(torch.cat(context, dim=1))
        context = F.interpolate(
            context, size=low.shape[-2:], mode="bilinear", align_corners=False
        )

        decoded = torch.cat((context, self.low_project(low)), dim=1)
        logits = self.classifier(self.decoder(decoded))
        logits = F.interpolate(
            logits, size=output_size, mode="bilinear", align_corners=False
        )
        return {"out": logits, "low": low, "mid": mid, "high": high}
