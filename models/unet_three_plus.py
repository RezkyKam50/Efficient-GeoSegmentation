import torch.nn.functional as F
import torch
import torch.nn as nn

from models.networks.GhostNet import GhostModule, GhostBottleneck
from models.networks.PRCNPTN import PRCNPTNLayer


class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch, scheme="ghost"):
        super(conv_block, self).__init__()
        self.scheme = scheme

        if scheme == "ghost":
            self.conv = nn.Sequential(
                GhostModule(in_ch, out_ch, kernel_size=1, ratio=2, relu=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                GhostModule(out_ch, out_ch, kernel_size=1, ratio=2, relu=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        elif scheme == "double_cnn":
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        elif scheme == "prc":
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                PRCNPTNLayer(
                    inch=out_ch,
                    outch=out_ch,
                    G=10,
                    CMP=2,
                    kernel_size=3,
                    padding=1
                )
            )
            self.shortcut = (
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 1, bias=False),
                    nn.BatchNorm2d(out_ch)
                )
                if in_ch != out_ch else nn.Identity()
            )

    def forward(self, x):
        if self.scheme == "prc":
            identity = self.shortcut(x)   
            x = self.conv(x)
            x = x + identity
            x = F.relu(x)
        else:
            x = self.conv(x)
        return x


class encoder_block(nn.Module):
    def __init__(self, in_c, out_c, scheme):
        super().__init__()
        self.c1 = nn.Sequential(
            conv_block(in_c, out_c, scheme=scheme),
            conv_block(out_c, out_c, scheme=scheme)
        )
        self.p1 = nn.MaxPool2d((2, 2))

    def forward(self, x):
        x = self.c1(x)
        p = self.p1(x)
        return x, p


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.conv(x)


class UNet3Plus(nn.Module):
    def __init__(self, cfg, n_channels=None, n_classes=None, deep_sup=False, scheme="ghost"):
        super().__init__()

        self._cfg = cfg
        n_channels = cfg.MODEL.IN_CHANNELS if n_channels is None else n_channels
        n_classes = cfg.MODEL.OUT_CHANNELS if n_classes is None else n_classes
        self.deep_sup = deep_sup

        if hasattr(cfg.MODEL, 'TOPOLOGY'):
            topology = cfg.MODEL.TOPOLOGY
            f1, f2, f3, f4, f5 = topology
 

        self.e1 = encoder_block(n_channels, f1, scheme=scheme)
        self.e2 = encoder_block(f1, f2, scheme=scheme)
        self.e3 = encoder_block(f2, f3, scheme=scheme)
        self.e4 = encoder_block(f3, f4, scheme=scheme)

        self.e5 = nn.Sequential(
            conv_block(f4, f5, scheme=scheme),
            conv_block(f5, f5, scheme=scheme)
        )

        self.reduction_channels = f1

        self.e1_d4 = conv_block(f1, self.reduction_channels, scheme=scheme)
        self.e2_d4 = conv_block(f2, self.reduction_channels, scheme=scheme)
        self.e3_d4 = conv_block(f3, self.reduction_channels, scheme=scheme)
        self.e4_d4 = conv_block(f4, self.reduction_channels, scheme=scheme)
        self.e5_d4 = conv_block(f5, self.reduction_channels, scheme=scheme)
        self.d4 = conv_block(self.reduction_channels * 5, self.reduction_channels, scheme=scheme)

        self.e1_d3 = conv_block(f1, self.reduction_channels, scheme=scheme)
        self.e2_d3 = conv_block(f2, self.reduction_channels, scheme=scheme)
        self.e3_d3 = conv_block(f3, self.reduction_channels, scheme=scheme)
        self.e4_d3 = conv_block(self.reduction_channels, self.reduction_channels, scheme=scheme)
        self.e5_d3 = conv_block(f5, self.reduction_channels, scheme=scheme)
        self.d3 = conv_block(self.reduction_channels * 5, self.reduction_channels, scheme=scheme)

        self.e1_d2 = conv_block(f1, self.reduction_channels, scheme=scheme)
        self.e2_d2 = conv_block(f2, self.reduction_channels, scheme=scheme)
        self.e3_d2 = conv_block(self.reduction_channels, self.reduction_channels, scheme=scheme)
        self.e4_d2 = conv_block(self.reduction_channels, self.reduction_channels, scheme=scheme)
        self.e5_d2 = conv_block(f5, self.reduction_channels, scheme=scheme)
        self.d2 = conv_block(self.reduction_channels * 5, self.reduction_channels, scheme=scheme)

        self.e1_d1 = conv_block(f1, self.reduction_channels, scheme=scheme)
        self.e2_d1 = conv_block(self.reduction_channels, self.reduction_channels, scheme=scheme)
        self.e3_d1 = conv_block(self.reduction_channels, self.reduction_channels, scheme=scheme)
        self.e4_d1 = conv_block(self.reduction_channels, self.reduction_channels, scheme=scheme)
        self.e5_d1 = conv_block(f5, self.reduction_channels, scheme=scheme)
        self.d1 = conv_block(self.reduction_channels * 5, self.reduction_channels, scheme=scheme)

        if deep_sup == True:
            self.y1 = nn.Conv2d(self.reduction_channels, n_classes, kernel_size=3, padding=1)
            self.y2 = nn.Conv2d(self.reduction_channels, n_classes, kernel_size=3, padding=1)
            self.y3 = nn.Conv2d(self.reduction_channels, n_classes, kernel_size=3, padding=1)
            self.y4 = nn.Conv2d(self.reduction_channels, n_classes, kernel_size=3, padding=1)
            self.y5 = nn.Conv2d(f5, n_classes, kernel_size=3, padding=1)
        else:
            self.y1 = nn.Conv2d(self.reduction_channels, n_classes, kernel_size=3, padding=1)

    def encode(self, inputs):
        e1, p1 = self.e1(inputs)
        e2, p2 = self.e2(p1)
        e3, p3 = self.e3(p2)
        e4, p4 = self.e4(p3)
        e5 = self.e5(p4)
        return [e1, e2, e3, e4, e5]

    def decode(self, skips):
        e1, e2, e3, e4, e5 = skips

        # Decoder level d4
        e1_d4 = F.max_pool2d(e1, kernel_size=8, stride=8)
        e1_d4 = self.e1_d4(e1_d4)

        e2_d4 = F.max_pool2d(e2, kernel_size=4, stride=4)
        e2_d4 = self.e2_d4(e2_d4)

        e3_d4 = F.max_pool2d(e3, kernel_size=2, stride=2)
        e3_d4 = self.e3_d4(e3_d4)

        e4_d4 = self.e4_d4(e4)

        e5_d4 = F.interpolate(e5, scale_factor=2, mode="bilinear", align_corners=True)
        e5_d4 = self.e5_d4(e5_d4)

        d4 = torch.cat([e1_d4, e2_d4, e3_d4, e4_d4, e5_d4], dim=1)
        d4 = self.d4(d4)

        # Decoder level d3
        e1_d3 = F.max_pool2d(e1, kernel_size=4, stride=4)
        e1_d3 = self.e1_d3(e1_d3)

        e2_d3 = F.max_pool2d(e2, kernel_size=2, stride=2)
        e2_d3 = self.e2_d3(e2_d3)

        e3_d3 = self.e3_d3(e3)

        e4_d3 = F.interpolate(d4, scale_factor=2, mode="bilinear", align_corners=True)
        e4_d3 = self.e4_d3(e4_d3)

        e5_d3 = F.interpolate(e5, scale_factor=4, mode="bilinear", align_corners=True)
        e5_d3 = self.e5_d3(e5_d3)

        d3 = torch.cat([e1_d3, e2_d3, e3_d3, e4_d3, e5_d3], dim=1)
        d3 = self.d3(d3)

        # Decoder level d2
        e1_d2 = F.max_pool2d(e1, kernel_size=2, stride=2)
        e1_d2 = self.e1_d2(e1_d2)

        e2_d2 = self.e2_d2(e2)

        e3_d2 = F.interpolate(d3, scale_factor=2, mode="bilinear", align_corners=True)
        e3_d2 = self.e3_d2(e3_d2)

        e4_d2 = F.interpolate(d4, scale_factor=4, mode="bilinear", align_corners=True)
        e4_d2 = self.e4_d2(e4_d2)

        e5_d2 = F.interpolate(e5, scale_factor=8, mode="bilinear", align_corners=True)
        e5_d2 = self.e5_d2(e5_d2)

        d2 = torch.cat([e1_d2, e2_d2, e3_d2, e4_d2, e5_d2], dim=1)
        d2 = self.d2(d2)

        # Decoder level d1
        e1_d1 = self.e1_d1(e1)

        e2_d1 = F.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=True)
        e2_d1 = self.e2_d1(e2_d1)

        e3_d1 = F.interpolate(d3, scale_factor=4, mode="bilinear", align_corners=True)
        e3_d1 = self.e3_d1(e3_d1)

        e4_d1 = F.interpolate(d4, scale_factor=8, mode="bilinear", align_corners=True)
        e4_d1 = self.e4_d1(e4_d1)

        e5_d1 = F.interpolate(e5, scale_factor=16, mode="bilinear", align_corners=True)
        e5_d1 = self.e5_d1(e5_d1)

        d1 = torch.cat([e1_d1, e2_d1, e3_d1, e4_d1, e5_d1], dim=1)
        d1 = self.d1(d1)

        if self.deep_sup == True:
            y1 = self.y1(d1)
            y2 = F.interpolate(self.y2(d2), scale_factor=2, mode="bilinear", align_corners=True)
            y3 = F.interpolate(self.y3(d3), scale_factor=4, mode="bilinear", align_corners=True)
            y4 = F.interpolate(self.y4(d4), scale_factor=8, mode="bilinear", align_corners=True)
            y5 = F.interpolate(self.y5(e5), scale_factor=16, mode="bilinear", align_corners=True)
            return (y1, y2, y3, y4, y5)
        else:
            y1 = self.y1(d1)
            return y1

    def forward(self, sar=None, optical=None, dem=None, pw=None):
        
        skips = self.encode(optical)
        return self.decode(skips)