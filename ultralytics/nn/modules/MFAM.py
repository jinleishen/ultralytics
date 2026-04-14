"""MASF-YOLO: Multi-scale Feature Aggregation Module (MFAM).

Registered as PKIModule_2 in model configs.
Reference: arXiv:2504.18136
"""

import torch.nn as nn
from typing import Optional, Sequence

from .conv import Conv, autopad


class ConvModule(nn.Module):
    """Conv + optional BN + optional activation."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, norm_cfg=None, act_cfg=None):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False)
        self.norm = (
            nn.BatchNorm2d(out_channels, momentum=norm_cfg.get("momentum", 0.03), eps=norm_cfg.get("eps", 0.001))
            if norm_cfg is not None else None
        )
        if act_cfg is not None:
            self.activation = {"ReLU": nn.ReLU(inplace=True), "SiLU": nn.SiLU(inplace=True)}.get(act_cfg["type"])
        else:
            self.activation = None

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class PKIModule_2(nn.Module):
    """Multi-scale Feature Aggregation Module (MFAM).

    Uses parallel multi-scale depthwise convolutions (3×3, 5×5, decomposed 7×7, 9×9) and feature fusion
    to improve detection accuracy for small objects.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        kernel_sizes: Sequence[int] = (3, 5, 7, 9),
        dilations: Sequence[int] = (1, 1, 1, 1),
        expansion: float = 1.0,
        add_identity: bool = True,
        norm_cfg: Optional[dict] = dict(type="BN", momentum=0.03, eps=0.001),
        act_cfg: Optional[dict] = dict(type="SiLU"),
    ):
        super().__init__()
        hidden_channels = int(in_channels * expansion)

        self.pre_conv = ConvModule(in_channels, hidden_channels, 1, 1, 0, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)

        # Small kernels: 3×3, 5×5
        self.dw_conv = ConvModule(hidden_channels, hidden_channels, kernel_sizes[0], 1,
                                  autopad(kernel_sizes[0], None, dilations[0]), dilations[0],
                                  groups=hidden_channels, norm_cfg=None, act_cfg=None)
        self.dw_conv1 = ConvModule(hidden_channels, hidden_channels, kernel_sizes[1], 1,
                                   autopad(kernel_sizes[1], None, dilations[1]), dilations[1],
                                   groups=hidden_channels, norm_cfg=None, act_cfg=None)

        # Decomposed large kernels: 7×7 → (1×7)+(7×1), 9×9 → (1×9)+(9×1)
        self.dw_conv2_w = ConvModule(hidden_channels, hidden_channels, kernel_size=(1, kernel_sizes[2]),
                                     padding=(0, kernel_sizes[2] // 2), groups=hidden_channels, norm_cfg=None, act_cfg=None)
        self.dw_conv2_h = ConvModule(hidden_channels, hidden_channels, kernel_size=(kernel_sizes[2], 1),
                                     padding=(kernel_sizes[2] // 2, 0), groups=hidden_channels, norm_cfg=None, act_cfg=None)
        self.dw_conv3_w = ConvModule(hidden_channels, hidden_channels, kernel_size=(1, kernel_sizes[3]),
                                     padding=(0, kernel_sizes[3] // 2), groups=hidden_channels, norm_cfg=None, act_cfg=None)
        self.dw_conv3_h = ConvModule(hidden_channels, hidden_channels, kernel_size=(kernel_sizes[3], 1),
                                     padding=(kernel_sizes[3] // 2, 0), groups=hidden_channels, norm_cfg=None, act_cfg=None)

        self.pw_conv = ConvModule(hidden_channels, hidden_channels, 1, 1, 0, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.add_identity = add_identity and in_channels == out_channels
        self.post_conv = ConvModule(hidden_channels, out_channels, 1, 1, 0, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x):
        x = self.pre_conv(x)
        y = x
        x = x + self.dw_conv(x) + self.dw_conv1(x) + self.dw_conv2_h(self.dw_conv2_w(x)) + self.dw_conv3_h(self.dw_conv3_w(x))
        x = self.pw_conv(x)
        if self.add_identity:
            x = x + y
        x = self.post_conv(x)
        return x
