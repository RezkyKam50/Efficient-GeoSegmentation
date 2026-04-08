from collections import OrderedDict
import torch.nn.functional as F
import torch
import torch.nn as nn

from models.networks.GhostNet import GhostModule
from models.networks.PRCNPTN import PRCNPTNLayer
 

import functools
# Dual Stream Classical UNet

# Reference from DS_Unet https://github.com/SebastianHafner/DS_UNet/blob/master/utils/networks.py
class ENet(nn.Module):

    def __init__(self, cfg, backbone):
        super(ENet, self).__init__()
 
        out = cfg.MODEL.OUT_CHANNELS
        topology = cfg.MODEL.TOPOLOGY

        n_s2_bands = len(cfg.DATASET.SENTINEL2_BANDS)
        s2_in = n_s2_bands  
        self.s2_stream = UNet(cfg, n_channels=s2_in, n_classes=out, topology=topology, backbone=backbone)
        self.n_s2_bands = n_s2_bands
 
        out_dim = cfg.MODEL.TOPOLOGY[0] 
        self.out_conv = OutConv(out_dim, out)
    
    def forward(self, s1_img=None, s2_img=None, dem_img=None, water_occur=None):
        
        s2_feature = self.s2_stream(s2_img)     
        out = self.out_conv(s2_feature)
        return out


class UNet(nn.Module):
    def __init__(self, cfg, n_channels=None, n_classes=None, topology=None, backbone="double_cnn"):
        self._cfg = cfg
        n_channels = cfg.MODEL.IN_CHANNELS if n_channels is None else n_channels
        n_classes = cfg.MODEL.OUT_CHANNELS if n_classes is None else n_classes
        topology = cfg.MODEL.TOPOLOGY if topology is None else topology
        super(UNet, self).__init__()
        first_chan = topology[0]

        conv_factory = functools.partial(Conv, scheme=backbone)  

        self.inc = InConv(n_channels, first_chan, conv_factory)
        self.outc = OutConv(first_chan, n_classes)

        down_topo = topology
        down_dict = OrderedDict()
        n_layers = len(down_topo)
        up_topo = [first_chan]
        up_dict = OrderedDict()

        for idx in range(n_layers):
            is_not_last_layer = idx != n_layers - 1
            in_dim = down_topo[idx]
            out_dim = down_topo[idx + 1] if is_not_last_layer else down_topo[idx]
            layer = Down(in_dim, out_dim, conv_factory)  
            print(f'down{idx + 1}: in {in_dim}, out {out_dim}')
            down_dict[f'down{idx + 1}'] = layer
            up_topo.append(out_dim)
        self.down_seq = nn.ModuleDict(down_dict)

        for idx in reversed(range(n_layers)):
            is_not_last_layer = idx != 0
            x1_idx = idx
            x2_idx = idx - 1 if is_not_last_layer else idx
            in_dim = up_topo[x1_idx] * 2
            out_dim = up_topo[x2_idx]
            layer = Up(in_dim, out_dim, conv_factory)   
            print(f'up{idx + 1}: in {in_dim}, out {out_dim}')
            up_dict[f'up{idx + 1}'] = layer
        self.up_seq = nn.ModuleDict(up_dict)

    def encode(self, x1, x2=None, x3=None):
        if x2 is None and x3 is None:
            x = x1
        elif x3 is None:
            x = torch.cat((x1, x2), 1)
        else:
            x = torch.cat((x1, x2, x3), 1)

        x1 = self.inc(x)
        inputs = [x1]
        for layer in self.down_seq.values():
            out = layer(inputs[-1])
            inputs.append(out)

        inputs.reverse()            
        bottleneck = inputs[0] 
        skips = inputs[1:]              
        return bottleneck, skips

    def decode(self, bottleneck, skips):
        x1 = bottleneck
        for idx, layer in enumerate(self.up_seq.values()):
            x2 = skips[idx]
            x1 = layer(x1, x2)

        return x1

    def forward(self, x1, x2=None, x3=None):
        bottleneck, skips = self.encode(x1, x2, x3)
        return self.decode(bottleneck, skips)

 
class Conv(nn.Module):
    def __init__(self, in_ch, out_ch, scheme="single_cnn"):
        super(Conv, self).__init__()
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
        elif scheme == "single_cnn":
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
        elif scheme == "prc":
            self.proj = nn.Conv2d(in_ch, out_ch, 1, bias=False)
            self.bn_proj = nn.BatchNorm2d(out_ch)
            self.prc = PRCNPTNLayer(
                inch=out_ch,    
                outch=out_ch,
                G=10,
                CMP=2,
                kernel_size=3,
                padding=1
            )
 
            self.act2 = nn.ReLU(inplace=True)

    def forward(self, x):

        if self.scheme == "prc":
            x = self.proj(x)
            x = self.bn_proj(x)   
            x = self.prc(x)
            x = self.act2(x)
            return x
        else:
            return self.conv(x)


class InConv(nn.Module):
    def __init__(self, in_ch, out_ch, conv_block):
        super(InConv, self).__init__()
        self.conv = conv_block(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, conv_block):
        super(Down, self).__init__()

        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            conv_block(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, conv_block):
        super(Up, self).__init__()

        self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
        self.conv = conv_block(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Handle padding for 2D images
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [
            diffX // 2, diffX - diffX // 2,
            diffY // 2, diffY - diffY // 2,
        ])
            
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x