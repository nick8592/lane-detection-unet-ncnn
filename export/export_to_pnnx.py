import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from models.unet import UNet

model = UNet(in_channels=3, out_channels=1)
checkpoint = torch.load("checkpoints/exp_20250903_164448/weights/unet_final.pt")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

dummy_input = torch.randn(1, 3, 256, 256)
traced = torch.jit.trace(model, dummy_input)
traced.save("export/ncnn_models/unet_jit.pt")