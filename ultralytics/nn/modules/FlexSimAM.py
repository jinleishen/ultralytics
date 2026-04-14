"""YOLO11s-UAV: FlexSimAM (Flexible SimAM Attention) module.

Reference: YOLO11s-UAV: An Advanced Algorithm for Small Object Detection in UAV Aerial Imagery
Journal of Imaging, 2026, DOI: 10.3390/jimaging12020069
"""

import torch
import torch.nn as nn
from .conv import Conv
from .block import C2f, C3, Bottleneck


class simam_module(nn.Module):
    """Simplified Attention Module (SimAM) — lightweight parameter-free attention."""

    def __init__(self, channels=None, e_lambda=1e-4):
        super().__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.activaton(y)


class Bottleneck_simam(nn.Module):
    """Bottleneck block with SimAM attention replacing second convolution."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = simam_module(c_)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3k_simam(C3):
    """C3k with SimAM-enhanced bottlenecks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck_simam(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class FlexSimAM(C2f):
    """Flexible CSP Bottleneck with optional SimAM attention blocks.

    Switches between standard Bottleneck and SimAM-enhanced C3k blocks
    based on the c3k parameter.
    """

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k_simam(self.c, self.c, 2, shortcut, g) if c3k
            else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )
