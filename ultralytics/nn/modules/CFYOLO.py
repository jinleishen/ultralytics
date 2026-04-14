"""CF-YOLO: Split-block Attention Module (SBAM) and Multi-branch Downsampling (MBD2).

Reference: CF-YOLO: a cross-layer feature fusion YOLO framework for small object detection
Journal of Electronic Imaging, 2025, DOI: 10.1117/1.JEI.34.5.053006
"""

import torch
import torch.nn as nn

from .block import Attention
from .conv import Conv, DWConv


class ChannelAttention2(nn.Module):
    """Channel Attention Module with dual-path pooling."""

    def __init__(self, c1, mid_reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(c1, c1 // mid_reduction, kernel_size=1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(c1 // mid_reduction, c1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention2(nn.Module):
    """Spatial Attention Module with avg+max pooling."""

    def __init__(self, kernel=3):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel, padding=kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = self.conv1(torch.cat([avg_out, max_out], dim=1))
        return self.sigmoid(out)


class SPDConv(nn.Module):
    """Space-to-Depth Convolution: rearranges spatial pixels into depth then applies conv."""

    def __init__(self, input_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(4 * input_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        spatial_output = torch.cat([
            x[..., ::2, ::2],
            x[..., 1::2, ::2],
            x[..., ::2, 1::2],
            x[..., 1::2, 1::2],
        ], 1)
        return self.conv(spatial_output)


class SRAM10(nn.Module):
    """Split-block Attention Module (SBAM) from CF-YOLO.

    Splits feature maps spatially (2x2) for channel attention, then splits channels (4 groups)
    for multi-scale spatial attention with 3/5/7/9 convolutions. Uses learnable weights to
    combine split vs whole-map attention. Includes self-attention at the start.
    """

    def __init__(self, c1, reduction=4, kernel_size=3):
        super().__init__()
        self.channel_attention = ChannelAttention2(c1, reduction)
        self.spatial_attention = SpatialAttention2(kernel_size)

        c_mid = c1 // 4
        c_mid2 = c1 - c_mid * 3
        self.conv1 = nn.Conv2d(c_mid, c_mid, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(c_mid, c_mid, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(c_mid, c_mid, kernel_size=7, padding=3)
        self.conv4 = nn.Conv2d(c_mid2, c_mid2, kernel_size=9, padding=4)
        self.attn = Attention(c1, attn_ratio=0.5, num_heads=c1 // 64)

        self.weight_split_channel = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.weight_overall_channel = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.weight_split_spatial = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.weight_overall_spatial = nn.Parameter(torch.tensor(0.5), requires_grad=True)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x + self.attn(x)

        h_split = H // 2
        w_split = W // 2

        parts_channel = [
            x[:, :, 0:h_split, 0:w_split],
            x[:, :, 0:h_split, w_split:W],
            x[:, :, h_split:H, 0:w_split],
            x[:, :, h_split:H, w_split:W],
        ]

        attended_parts = []
        for part in parts_channel:
            channel_attention = self.channel_attention(part)
            attended_parts.append(part * channel_attention)

        top_row = torch.cat(attended_parts[:2], dim=3)
        bottom_row = torch.cat(attended_parts[2:], dim=3)
        x_split = torch.cat([top_row, bottom_row], dim=2)

        overall_channel_attention = self.channel_attention(x)
        combined = self.weight_split_channel * x_split + self.weight_overall_channel * (x * overall_channel_attention)

        # Channel-split spatial attention with multi-scale convolutions
        c_split = C // 4
        parts_spatial = [
            combined[:, :c_split, :, :],
            combined[:, c_split:2 * c_split, :, :],
            combined[:, 2 * c_split:3 * c_split, :, :],
            combined[:, 3 * c_split:, :, :],
        ]

        convs = [self.conv1, self.conv2, self.conv3, self.conv4]
        attended_parts_spatial = []
        for i, part in enumerate(parts_spatial):
            part = convs[i](part)
            spatial_attention = self.spatial_attention(part)
            attended_parts_spatial.append(part * spatial_attention)

        combined_split = torch.cat(attended_parts_spatial, dim=1)
        overall_spatial_attention = self.spatial_attention(combined)
        output = self.weight_split_spatial * combined_split + self.weight_overall_spatial * (combined * overall_spatial_attention)

        return output


class MBD2(nn.Module):
    """Multi-Branch Downsampling module from CF-YOLO.

    Preserves high-resolution shallow information through 4 diverse downsampling paths:
    MaxPool, DWConv, Conv, and SPDConv, then fuses via 1x1 convolution.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.intermediate_channels = in_channels // 2

        self.reduce_channels = Conv(in_channels, self.intermediate_channels, k=1, act=True)
        self.branch1 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.branch2 = DWConv(self.intermediate_channels, self.intermediate_channels, k=3, s=2, act=True)
        self.branch3 = Conv(self.intermediate_channels, self.intermediate_channels, k=3, s=2, act=True)
        self.branch4 = SPDConv(self.intermediate_channels, 4 * self.intermediate_channels)
        self.final_conv = Conv(self.intermediate_channels * 7, out_channels, k=1, act=True)

    def forward(self, x):
        x = self.reduce_channels(x)
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        out = torch.cat((out1, out2, out3, out4), dim=1)
        return self.final_conv(out)
