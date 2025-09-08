import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
import pnnx
import torch
from models.unet import UNet
from models.unet_depthwise import UNetDepthwise
from models.unet_depthwise_nano import UNetDepthwiseNano
from models.unet_depthwise_small import UNetDepthwiseSmall

# Load export config
config_path = "config/export_config.yaml"
if not os.path.exists(config_path):
	raise FileNotFoundError(f"Export config file not found: {config_path}")
with open(config_path, "r") as f:
	config = yaml.safe_load(f)

model_type = config.get("model_type", "unet")
in_channels = config.get("in_channels", 3)
out_channels = config.get("out_channels", 1)
checkpoint_path = config["checkpoint_path"]
output_model_path = config["output_model_path"]
img_size = config.get("img_size", 256)
dummy_input_shape = [1, in_channels, img_size, img_size]

os.makedirs(os.path.dirname(output_model_path), exist_ok=True)

# Select model
if model_type == "unet":
	model = UNet(in_channels=in_channels, out_channels=out_channels)
elif model_type == "unet_depthwise":
	model = UNetDepthwise(in_channels=in_channels, out_channels=out_channels)
elif model_type == "unet_depthwise_small":
	model = UNetDepthwiseSmall(in_channels=in_channels, out_channels=out_channels)
elif model_type == "unet_depthwise_nano":
	model = UNetDepthwiseNano(in_channels=in_channels, out_channels=out_channels)
else:
	raise ValueError(f"Unknown model_type: {model_type}")

# Load checkpoint
checkpoint = torch.load(checkpoint_path, map_location="cpu")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Dummy input
dummy_input = torch.randn(*dummy_input_shape)
traced = torch.jit.trace(model, dummy_input)
traced.save(output_model_path)
print(f"Exported TorchScript model to {output_model_path}")

# Export to ncnn
opt_model = pnnx.export(model, output_model_path, dummy_input)
print(f"Exported ncnn model to {os.path.dirname(output_model_path)}")