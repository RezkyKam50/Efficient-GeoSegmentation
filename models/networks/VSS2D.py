import torch
from torch import nn
import torch.nn.functional as F

from ..vmamba import VSSLayer

class VSSBlock(nn.Module):
    def __init__(self, in_ch, out_ch, depth):
        super().__init__()

        self.vss = VSSLayer(
            out_ch, 
            depth 
        )

    def forward(self, x):
        
        B, C, H, W = x.shape

        x = x.permute(0, 2, 3, 1) # B, H, W, C
        x = self.vss(x)
        x = x.permute(0, 3, 1, 2) # B, H->C, W->H, C->W 
        x = F.relu(x)
        return x