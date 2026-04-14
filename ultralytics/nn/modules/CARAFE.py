"""YOLO11s-UAV: Content-Aware ReAssembly of FEatures (CARAFE) upsampling.

Reference: YOLO11s-UAV: An Advanced Algorithm for Small Object Detection in UAV Aerial Imagery
Journal of Imaging, 2026, DOI: 10.3390/jimaging12020069
"""

import torch
import torch.nn as nn
from .conv import Conv


class CARAFE(nn.Module):
    """Content-Aware ReAssembly of FEatures (CARAFE) upsampling module.

    Generates content-aware upsampling kernels for each spatial location,
    producing higher quality upsampled features than bilinear/nearest methods.
    Reference: https://arxiv.org/abs/1905.02188
    """

    def __init__(self, c, k_enc=3, k_up=5, c_mid=64, scale=2):
        super().__init__()
        self.scale = scale
        self.comp = Conv(c, c_mid)
        self.enc = Conv(c_mid, (scale * k_up) ** 2, k=k_enc, act=False)
        self.pix_shf = nn.PixelShuffle(scale)
        self.upsmp = nn.Upsample(scale_factor=scale, mode='nearest')
        self.unfold = nn.Unfold(kernel_size=k_up, dilation=scale, padding=k_up // 2 * scale)

    def forward(self, X):
        b, c, h, w = X.size()
        h_, w_ = h * self.scale, w * self.scale

        W = self.comp(X)
        W = self.enc(W)
        W = self.pix_shf(W)
        W = torch.softmax(W, dim=1)

        X = self.upsmp(X)
        X = self.unfold(X)
        X = X.view(b, c, -1, h_, w_)

        X = torch.einsum('bkhw,bckhw->bchw', [W, X])
        return X
