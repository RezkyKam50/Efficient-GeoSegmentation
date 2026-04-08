from collections import OrderedDict
import torch.nn.functional as F
import torch
import torch.nn as nn
import math

class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = torch.clamp(y, 0, 1)
        return x * y

def depthwise_conv(inp, oup, kernel_size=3, stride=1, relu=False):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, kernel_size//2, groups=inp, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True) if relu else nn.Sequential(),
    )

class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)
        return out[:,:self.oup,:,:]


class GhostBottleneck(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride):
        super(GhostBottleneck, self).__init__()
        assert stride in [1, 2]

        self.conv = nn.Sequential(
            # pw
            GhostModule(inp, hidden_dim, kernel_size=1, relu=True),
            # dw
            depthwise_conv(hidden_dim, hidden_dim, kernel_size, stride, relu=False) if stride==2 else nn.Sequential(),
            # Squeeze-and-Excite
            SELayer(hidden_dim),
            # pw-linear
            GhostModule(hidden_dim, oup, kernel_size=1, relu=False),
        )

        if stride == 1 and inp == oup:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                depthwise_conv(inp, inp, kernel_size, stride, relu=False),
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)

#Nested Unet

class Ghost_Unet(nn.Module):
    """
    Implementation of this paper:
    https://arxiv.org/pdf/1807.10165.pdf
    """
    def __init__(self, in_ch=1, out_ch=1):
        super(Ghost_Unet, self).__init__()

        n1 = 16
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.GBneck0_0 = GhostBottleneck(in_ch, filters[0], filters[0],3,1)
        self.GBneck1_0 = GhostBottleneck(filters[0], filters[1], filters[1],3,1)
        self.GBneck2_0 = GhostBottleneck(filters[1], filters[2], filters[2],3,1)
        self.GBneck3_0 = GhostBottleneck(filters[2], filters[3], filters[3],3,1)
        self.GBneck4_0 = GhostBottleneck(filters[3], filters[4],filters[4],3,1)

        self.GBneck0_1 = GhostBottleneck(filters[0] + filters[1], filters[0], filters[0],3,1)
        self.GBneck1_1 = GhostBottleneck(filters[1] + filters[2], filters[1], filters[1],3,1)
        self.GBneck2_1 = GhostBottleneck(filters[2] + filters[3], filters[2], filters[2],3,1)
        self.GBneck3_1 = GhostBottleneck(filters[3] + filters[4], filters[3], filters[3],3,1)

        self.GBneck0_2 = GhostBottleneck(filters[0]*2 + filters[1], filters[0], filters[0],3,1)
        self.GBneck1_2 = GhostBottleneck(filters[1]*2 + filters[2], filters[1], filters[1],3,1)
        self.GBneck2_2 = GhostBottleneck(filters[2]*2 + filters[3], filters[2], filters[2],3,1)

        self.GBneck0_3 = GhostBottleneck(filters[0]*3 + filters[1], filters[0], filters[0],3,1)
        self.GBneck1_3 = GhostBottleneck(filters[1]*3 + filters[2], filters[1], filters[1],3,1)

        self.GBneck0_4 = GhostBottleneck(filters[0]*4 + filters[1], filters[0], filters[0],3,1)

        self.final = nn.Conv2d(filters[0], out_ch, kernel_size=1)

    def forward(self, s1_img=None, s2_img=None, dem_img=None, water_occur=None):
        x = s2_img

        x0_0 = self.GBneck0_0(x)
        x1_0 = self.GBneck1_0(self.pool(x0_0))
        x0_1 = self.GBneck0_1(torch.cat([x0_0, self.Up(x1_0)], 1))

        x2_0 = self.GBneck2_0(self.pool(x1_0))
        x1_1 = self.GBneck1_1(torch.cat([x1_0, self.Up(x2_0)], 1))
        x0_2 = self.GBneck0_2(torch.cat([x0_0, x0_1, self.Up(x1_1)], 1))

        x3_0 = self.GBneck3_0(self.pool(x2_0))
        x2_1 = self.GBneck2_1(torch.cat([x2_0, self.Up(x3_0)], 1))
        x1_2 = self.GBneck1_2(torch.cat([x1_0, x1_1, self.Up(x2_1)], 1))
        x0_3 = self.GBneck0_3(torch.cat([x0_0, x0_1, x0_2, self.Up(x1_2)], 1))

        x4_0 = self.GBneck4_0(self.pool(x3_0))
        x3_1 = self.GBneck3_1(torch.cat([x3_0, self.Up(x4_0)], 1))
        x2_2 = self.GBneck2_2(torch.cat([x2_0, x2_1, self.Up(x3_1)], 1))
        x1_3 = self.GBneck1_3(torch.cat([x1_0, x1_1, x1_2, self.Up(x2_2)], 1))
        x0_4 = self.GBneck0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.Up(x1_3)], 1))

        output = self.final(x0_4)
        output = F.sigmoid(output)
        return output

