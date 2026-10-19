import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .arch_model import (FreqTransBlock, SpectralHarmonization,
                              SpectralDecomposition, NAFBlock, DAM)
    from .arch_util import Local_Base
except:
    from arch_model import (FreqTransBlock, SpectralHarmonization,
                             SpectralDecomposition, NAFBlock, DAM)
    from arch_util import Local_Base


class FReSHIR(nn.Module):

    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[], cutoff=[], dilations=None):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1, bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.multi_sides = nn.ModuleList()
        self.divide = nn.ModuleList()
        self.concatenate = nn.ModuleList()

        levels = len(enc_blk_nums)
        self.scales = [2 ** i for i in range(levels)]

        chan = width
        enc_channels = []

        for i, num in enumerate(enc_blk_nums):
            enc_channels.append(chan)

            self.divide.append(SpectralDecomposition(chan, cutoff_ratio=cutoff[i]))

            self.encoders.append(
                nn.Sequential(*[NAFBlock(chan) for _ in range(num)])
            )
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan = chan * 2

        self.middle_blks = nn.Sequential(
            *[DAM(chan) for _ in range(middle_blk_num)]
        )

        for i, num in enumerate(dec_blk_nums):
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            cut = cutoff[len(cutoff) - 1 - i]

            self.decoders.append(
                nn.Sequential(
                    *[FreqTransBlock(chan, dilations, reduce=(j == num - 1)) for j in range(num)]
                )
            )

            self.concatenate.append(SpectralHarmonization(cutoff_ratio=cut))

        for ch in enc_channels:
            self.multi_sides.append(
                nn.Conv2d(in_channels=ch, out_channels=img_channel, kernel_size=3, stride=1, padding=1)
            )

        self.padder_size = 2 ** len(self.encoders)

        self.side_out = nn.Conv2d(in_channels=width * 2 ** len(self.encoders), out_channels=img_channel,
                                  kernel_size=3, stride=1, padding=1)

    def forward(self, inp, side_loss=False, multi_side=False):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []
        enc_outs = []
        for encoder, down, divide, side in zip(self.encoders, self.downs, self.divide, self.multi_sides):
            low, high = divide(x)

            low = encoder(low)
            encs.append([low, high])
            x = low + high
            if multi_side:
                enc_outs.append(side(high))
            x = down(x)

        for middle in self.middle_blks:
            x = middle(x)
        if side_loss:
            side_out = self.side_out(x)

        for decoder, up, enc_skip, concatenate in zip(self.decoders, self.ups, encs[::-1], self.concatenate):
            x = up(x)
            x = concatenate(x, enc_skip)
            x = decoder(x)

        x = self.ending(x)
        x = x + inp
        output_dict = {'output': x[:, :, :H, :W]}

        if side_loss:
            return {'output': x[:, :, :H, :W], 'out_side': side_out}

        elif multi_side:
            output_dict['out_side'] = [out[:, :, :H // scale, :W // scale] for out, scale in zip(enc_outs, self.scales)]
            return output_dict

        else:
            return {'output': x[:, :, :H, :W]}

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), value=0)
        return x


class FReSHIRLocal(Local_Base, FReSHIR):
    def __init__(self, *args, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        FReSHIR.__init__(self, *args, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)


if __name__ == '__main__':

    from ptflops import get_model_complexity_info

    net = FReSHIR(img_channel=3, width=24,
                  middle_blk_num=1, enc_blk_nums=[4, 6, 6, 8], dec_blk_nums=[8, 6, 6, 4],
                  cutoff=[0.4, 0.32, 0.23, 0.12], dilations=[1, 4, 9])
    inp = torch.randn(1, 3, 256, 256)

    macs, params = get_model_complexity_info(net, input_res=(3, 256, 256), print_per_layer_stat=False, verbose=False)

    print(macs, params)
