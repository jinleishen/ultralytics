"""YOLO11s-UAV: Space-to-Depth Residual Convolution (S2DResConv) module.

Reference: YOLO11s-UAV: An Advanced Algorithm for Small Object Detection in UAV Aerial Imagery
Journal of Imaging, 2026, DOI: 10.3390/jimaging12020069
"""

import torch
import torch.nn as nn
from .conv import Conv


class DWR(nn.Module):
    """Dilated-Width Residual block with multi-scale dilated convolutions."""

    def __init__(self, dim):
        super().__init__()
        self.conv_3x3 = Conv(dim, dim // 2, 3)
        self.conv_3x3_d1 = Conv(dim // 2, dim, 3, d=1)
        self.conv_3x3_d3 = Conv(dim // 2, dim // 2, 3, d=3)
        self.conv_3x3_d5 = Conv(dim // 2, dim // 2, 3, d=5)
        self.conv_1x1 = Conv(dim * 2, dim, k=1)

    def forward(self, x):
        conv_3x3 = self.conv_3x3(x)
        x1 = self.conv_3x3_d1(conv_3x3)
        x2 = self.conv_3x3_d3(conv_3x3)
        x3 = self.conv_3x3_d5(conv_3x3)
        x_out = self.conv_1x1(torch.cat([x1, x2, x3], dim=1))
        return x_out + x


class DWRSeg_Conv(nn.Module):
    """DWR-enhanced segmentation convolution."""

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, groups=1, dilation=1):
        super().__init__()
        self.conv = Conv(in_channels, out_channels, k=1)
        self.dcnv3 = DWR(out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.dcnv3(x)
        return self.gelu(self.bn(x))


class S2DResConv(nn.Module):
    """Space-to-Depth Residual Convolution block.

    Applies space-to-depth transformation (pixel unshuffle) to preserve spatial info,
    followed by DWR-enhanced convolution with multi-scale dilated receptive fields.
    """

    def __init__(self, c1, c2, k=1, s=1, p=None, g=4, act=True):
        super().__init__()
        self.conv = DWRSeg_Conv(in_channels=c1 * 4, out_channels=c2, kernel_size=k, stride=s, groups=g, dilation=1)

    def forward(self, x):
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
