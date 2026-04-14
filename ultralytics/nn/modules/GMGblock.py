import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ultralytics.nn.modules.conv import LightConv

__all__ = ['GMGblock', 'C3k2_GMG']

def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

def gaussian_kernel(kernel_size, sigma=1.0):
    kernel = torch.zeros(kernel_size, kernel_size)
    center = (kernel_size - 1) / 2
    for i in range(kernel_size):
        distance = (i - center) ** 2
        kernel[i] = math.exp(-distance / (2 * sigma** 2))
    kernel /= kernel.sum()
    return kernel

class GMGblock(nn.Module):

    def __init__(self, c1, c2, k, s, sigma=1.0):
        super().__init__()
        self.branch_ch = c2 // 8
        self.k = k
        self.s = s

        p_original = [(k, 0, 1, 0), (0, k, 0, 1), (0, 1, k, 0), (1, 0, 0, k)]
        self.pad_original = [nn.ZeroPad2d(p) for p in p_original]
        p_mirror = [(0, 1, k, 0), (k, 0, 1, 0), (1, 0, 0, k), (0, k, 0, 1)]
        self.pad_mirror = [nn.ZeroPad2d(p) for p in p_mirror]

        self.cw_original = Conv(c1, self.branch_ch, (1, k), s=s, p=0)
        self.ch_original = Conv(c1, self.branch_ch, (k, 1), s=s, p=0)
        self.cw_mirror = Conv(c1, self.branch_ch, (1, k), s=s, p=0)
        self.ch_mirror = Conv(c1, self.branch_ch, (k, 1), s=s, p=0)

        # 拼接后高斯增强模块
        self.gaussian_conv = nn.Conv2d(c2, c2, (k, k), 1, padding=autopad(k), bias=False)  # 与拼接后通道匹配
        self.gaussian_bn = nn.BatchNorm2d(c2)
        self.gaussian_act = nn.SiLU()
        self._init_gaussian_kernel(sigma, k)  # 初始化高斯核

        # 动态融合权重（原始拼接特征 + 高斯增强特征）
        self.att_weight = nn.Parameter(torch.tensor([0.5, 0.5]))  # 控制两者权重
        self.softmax = nn.Softmax(dim=0)

        # 最终融合卷积（保持原1x1融合）
        self.cat = Conv(c2, c2, 2, s=1, p=0)

    def _init_gaussian_kernel(self, sigma, kernel_size):
        kernel = gaussian_kernel(kernel_size, sigma).view(1, 1, kernel_size, kernel_size)
        out_ch, in_ch = self.gaussian_conv.weight.shape[:2]
        self.gaussian_conv.weight.data = kernel.repeat(out_ch, in_ch, 1, 1)

    def forward(self, x):
        # 1. 提取8分支特征并对齐
        # 原始4分支
        yw0 = self.cw_original(self.pad_original[0](x))
        yw1 = self.cw_original(self.pad_original[1](x))
        yh0 = self.ch_original(self.pad_original[2](x))
        yh1 = self.ch_original(self.pad_original[3](x))

        # 镜面对称4分支
        yw2 = self.cw_mirror(self.pad_mirror[0](x))
        yw3 = self.cw_mirror(self.pad_mirror[1](x))
        yh2 = self.ch_mirror(self.pad_mirror[2](x))
        yh3 = self.ch_mirror(self.pad_mirror[3](x))

        # 对齐所有分支尺寸
        target_h, target_w = yw0.shape[2], yw0.shape[3]
        def align_feat(feat):
            return F.interpolate(feat, (target_h, target_w), mode='nearest')
        yw1, yh0, yh1, yw2, yw3, yh2, yh3 = map(align_feat, [yw1, yh0, yh1, yw2, yw3, yh2, yh3])

        # 2. 拼接8分支特征
        all_feats = [yw0, yw1, yh0, yh1, yw2, yw3, yh2, yh3]
        concat_feat = torch.cat(all_feats, dim=1)  # 通道数：c2

        # 3. 高斯增强特征
        gaussian_feat = self.gaussian_act(self.gaussian_bn(self.gaussian_conv(concat_feat)))

        # 4. 动态融合（原始拼接特征 + 高斯增强特征）
        weights = self.softmax(self.att_weight)
        fused_feat = weights[0] * concat_feat + weights[1] * gaussian_feat

        # 5. 最终融合输出
        return self.cat(fused_feat)

# 以下模块保持不变（确保兼容性）
class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C3(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

class C3k(C3):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

class C2f_GMGblock(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.att = GMGblock(c2, c2, 3, 1)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.att(self.cv2(torch.cat(y, 1)))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.att(self.cv2(torch.cat(y, 1)))

class C3k2_GMG(C2f_GMGblock):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g)
            for _ in range(n)
        )