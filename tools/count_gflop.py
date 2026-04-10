import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import torch
from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    deeplabv3_mobilenet_v3_large
)
from models.deeplab import DeepLabWrapper
from models.UNet import UNet
from models.unet_three_plus import UNet3Plus
from models.attn_ghostnet import Ghost_Unet
from models.fast_scnn import FastSCNN
from models.vm_unet import VMUNet
from models.h_vmunet import H_vmunet
from models.networks.E_Net import ENet
from models.networks.config import Config_Unet, Config_Unet3P
from thop import profile

models = {
    "UNet_Sentinel2": UNet(
        in_channels=6,
        out_channels=2,
        unet_encoder_size=768
    ),
    "UNet3+_Sentinel2": UNet3Plus(
        cfg=Config_Unet3P,
        n_channels=6,
        n_classes=2,
        scheme="double_cnn"
    ),
    "DeeplabV3_Resnet50_Sentinel2": DeepLabWrapper(
        deeplabv3_resnet50(num_classes=2), in_channels=6
    ),
    "DeeplabV3_MobilenetV2Large_Sentinel2": DeepLabWrapper(
        deeplabv3_mobilenet_v3_large(num_classes=2), in_channels=6
    ),
    "Attention_GhostUNetPlusPlus_Sentinel2": Ghost_Unet(
        in_ch=6,
        out_ch=2
    ),
    "Fast_SCNN_Sentinel2": FastSCNN(
        in_channels=6,
        num_classes=2
    ),
    "VMUnet_Sentinel2": VMUNet(
        input_channels=6,
        num_classes=2
    ),
    "H_vmunet_Sentinel2": H_vmunet(
        input_channels=6,
        num_classes=2
    ),
    "UNet3+_Ghost_Sentinel2": UNet3Plus(
        cfg=Config_Unet3P,
        n_channels=6,
        n_classes=2,
        scheme="ghost"
    ),
}


dummy = torch.randn(1, 6, 224, 224, device="cuda")

results = {}

for name, model in models.items():
    model.to("cuda").eval()
 
    macs, params = profile(model, inputs=(dummy, dummy, dummy, dummy), verbose=False)
    gflops = macs * 2 / 1e9    
    results[name] = {
        "GFLOPs": round(gflops, 4),
        "MACs": int(macs),
        "Parameters_M": round(params / 1e6, 4),
        "Parameters": int(params),
    }
 
with open("model_gflops.json", "w") as f:
    json.dump(results, f, indent=2)
 