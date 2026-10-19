import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import fft2, ifft2, fftshift, ifftshift

try:
    from .arch_util import SimpleGate, LayerNorm2d
except:
    from arch_util import SimpleGate, LayerNorm2d


# ---------------------------------------------------------------------------
# Attention blocks used in the encoder middle stage (DAM)
# ---------------------------------------------------------------------------

class BasicConv1(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, gelu=False, bn=False, bias=True):
        super(BasicConv1, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.gelu = nn.GELU() if gelu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.gelu is not None:
            x = self.gelu(x)
        return x


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self, channel):
        super(SpatialGate, self).__init__()
        kernel_size = 3
        self.compress = ChannelPool()
        self.spatial = BasicConv1(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, gelu=False)
        self.dw1 = nn.Sequential(
            BasicConv1(channel, channel, 5, stride=1, dilation=2, padding=4, groups=channel),
            BasicConv1(channel, channel, 7, stride=1, dilation=3, padding=9, groups=channel)
        )
        self.dw2 = BasicConv1(channel, channel, kernel_size, stride=1, padding=1, groups=channel)

    def forward(self, x):
        out = self.compress(x)
        out = self.spatial(out)
        out = self.dw1(x) * out + self.dw2(x)
        return out


class LocalAttention(nn.Module):
    def __init__(self, channel, p) -> None:
        super().__init__()
        self.channel = channel
        self.num_patch = 2 ** p
        self.a = nn.Parameter(torch.zeros(channel, 1, 1))
        self.b = nn.Parameter(torch.ones(channel, 1, 1))

    def forward(self, x):
        out = x - torch.mean(x, dim=(2, 3), keepdim=True)
        return self.a * out * x + self.b * x


class DAM(nn.Module):
    def __init__(self, channel) -> None:
        super().__init__()
        pyramid = 1
        self.spatial_gate = SpatialGate(channel)
        layers = [LocalAttention(channel, p=i) for i in range(pyramid - 1, -1, -1)]
        self.local_attention = nn.Sequential(*layers)
        self.a = nn.Parameter(torch.zeros(channel, 1, 1))
        self.b = nn.Parameter(torch.ones(channel, 1, 1))

    def forward(self, x):
        out = self.spatial_gate(x)
        out = self.local_attention(out)
        return self.a * out + self.b * x


# ---------------------------------------------------------------------------
# Encoder block
# ---------------------------------------------------------------------------

class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1, groups=1, bias=True),
        )

        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        x = y + x * self.gamma

        return x


# ---------------------------------------------------------------------------
# Decoder blocks
# ---------------------------------------------------------------------------

class Branch(nn.Module):
    def __init__(self, c, DW_Expand, dilation=1):
        super().__init__()
        self.dw_channel = DW_Expand * c
        self.branch = nn.Sequential(
            nn.Conv2d(in_channels=self.dw_channel, out_channels=self.dw_channel, kernel_size=3,
                      padding=dilation, stride=1, groups=self.dw_channel, bias=True, dilation=dilation)
        )

    def forward(self, input):
        return self.branch(input)


class TransBlockHF(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.dw_conv2 = nn.Conv2d(channels, channels, 5, padding=2, groups=channels)
        self.init_conv = nn.Conv2d(channels, channels, 1)
        self.dw_conv1 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.conv1 = nn.Conv2d(channels, channels, 1)
        self.conv2 = nn.Conv2d(channels, channels, 1)
        self.final_conv = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        x = self.init_conv(x)

        att1 = torch.sigmoid(self.dw_conv1(x))
        val1 = self.conv1(x)
        x_hat1 = att1 * val1

        att2 = torch.sigmoid(self.dw_conv2(x))
        val2 = self.conv2(x_hat1)
        x_hat2 = att2 * val2

        out = self.final_conv(x_hat2)
        return out


class FreqTransBlock(nn.Module):
    def __init__(self, c, dilations=None, reduce=False, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()

        self.trans_blockhf = TransBlockHF(c)
        self.reduce = reduce
        self.branches = nn.ModuleList()
        for dilation in dilations:
            self.branches.append(Branch(c, DW_Expand=1, dilation=dilation))

        assert len(dilations) == len(self.branches)

        ffn_channel = FFN_Expand * c

        self.sg = SimpleGate()

        self.conv4 = nn.Conv2d(in_channels=ffn_channel, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.conv_reduce = nn.Conv2d(in_channels=ffn_channel, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm2 = LayerNorm2d(2 * c)
        self.norm_lh = LayerNorm2d(2 * c)

        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta_low = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, ffn_channel, 1, 1)), requires_grad=True)

    def forward(self, inp):
        if isinstance(inp, (list, tuple)):
            low, high = inp
            lh = torch.cat([low, high], dim=1)
            lh = self.norm_lh(lh)
            low, high = torch.chunk(lh, 2, dim=1)
        else:
            inp = self.norm_lh(inp)
            low, high = torch.chunk(inp, 2, dim=1)

        l = 0
        for branch in self.branches:
            l += branch(low)

        h = self.trans_blockhf(high)

        h = h * self.beta + high
        l = low + l * self.beta_low

        y = torch.cat([l, h], dim=1)

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        out = y + x * self.gamma

        if self.reduce:
            out = self.conv_reduce(out)

        return out


# ---------------------------------------------------------------------------
# Frequency decomposition and harmonization
# ---------------------------------------------------------------------------

class SpectralDecomposition(nn.Module):
    def __init__(self, channels, cutoff_ratio=0.12):
        super().__init__()
        self.cutoff_ratio = cutoff_ratio
        self.conv = nn.Conv2d(channels, channels, 1, bias=True)

    def forward(self, x):
        B, C, H, W = x.shape

        f_shift = fftshift(fft2(x, dim=(-2, -1)), dim=(-2, -1))

        cy, cx = H // 2, W // 2
        ys = torch.arange(H, device=x.device) - cy
        xs = torch.arange(W, device=x.device) - cx
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        dist = torch.sqrt(grid_x ** 2 + grid_y ** 2)

        diag = (H ** 2 + W ** 2) ** 0.5
        r0 = self.cutoff_ratio * diag
        sigma = 0.15 * r0
        k = 2.5

        mask = torch.zeros_like(dist)
        low = dist <= (r0 - k * sigma)
        high = dist >= (r0 + k * sigma)
        mid = (~low) & (~high)

        mask[high] = 1.0
        d = dist[mid] - r0
        mask[mid] = 1.0 - torch.exp(-(d ** 2) / (2 * sigma ** 2))

        high_freq = torch.real(ifft2(
            ifftshift(f_shift * mask.unsqueeze(0).unsqueeze(0), dim=(-2, -1)),
            dim=(-2, -1)
        ))

        low_freq = torch.real(ifft2(
            ifftshift(f_shift * (1 - mask).unsqueeze(0).unsqueeze(0), dim=(-2, -1)),
            dim=(-2, -1)
        ))

        return torch.sigmoid(self.conv(low_freq)), high_freq


class SpectralHarmonization(nn.Module):
    def __init__(self, cutoff_ratio=0.12):
        super().__init__()
        self.cutoff_ratio = cutoff_ratio

    def forward(self, inp, lowhigh):
        x = inp
        low_skip, high_skip = lowhigh[0], lowhigh[1]

        B, C, H, W = x.shape

        f_shift = fftshift(fft2(x, dim=(-2, -1)), dim=(-2, -1))

        cy, cx = H // 2, W // 2
        ys = torch.arange(H, device=x.device) - cy
        xs = torch.arange(W, device=x.device) - cx
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        dist = torch.sqrt(grid_x ** 2 + grid_y ** 2)

        diag = (H ** 2 + W ** 2) ** 0.5
        r0 = self.cutoff_ratio * diag
        sigma = 0.15 * r0
        k = 2.5

        mask = torch.zeros_like(dist)
        low = dist <= (r0 - k * sigma)
        high = dist >= (r0 + k * sigma)
        mid = (~low) & (~high)

        mask[high] = 1.0
        d = dist[mid] - r0
        mask[mid] = 1.0 - torch.exp(-(d ** 2) / (2 * sigma ** 2))

        high_freq = torch.real(ifft2(
            ifftshift(f_shift * mask.unsqueeze(0).unsqueeze(0), dim=(-2, -1)),
            dim=(-2, -1)
        ))

        low_freq = torch.real(ifft2(
            ifftshift(f_shift * (1 - mask).unsqueeze(0).unsqueeze(0), dim=(-2, -1)),
            dim=(-2, -1)
        ))

        low_final = low_skip + low_freq
        high_final = high_skip + high_freq

        return [low_final, high_final]
