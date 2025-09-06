
"""
Inference script for UNet lane segmentation on BDD100K dataset.
Loads config, model, checkpoint, and runs batch inference with reproducibility.
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from typing import Any, Dict, Tuple

import yaml
import torch
import numpy as np
from PIL import Image, ImageOps
import torchvision.transforms as T
from tqdm import tqdm
from models.unet import UNet
from utils.checkpoint import load_checkpoint

# --- Inference Function ---
def infer(
    model: torch.nn.Module,
    device: torch.device,
    image_path: str,
    transform: Any
) -> Tuple[Image.Image, Image.Image]:
    """
    Run inference and return both mask and overlaid image.
    """
    image = Image.open(image_path).convert("RGB")

    # Preprocess and predict
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        output = output.squeeze().cpu().numpy()
        output = np.uint8(output * 255)

    # Resize mask back to original image size
    orig_size = image.size  # (width, height)
    mask_img = Image.fromarray(output)
    mask_img = mask_img.resize(orig_size, resample=Image.BILINEAR)

    # Invert mask if needed (based on your setup)
    inverted_mask = ImageOps.invert(mask_img.convert("L"))

    # Create a color image (RGBA) for the mask overlay (green with alpha)
    color_mask = Image.new("RGBA", orig_size, color=(0, 255, 0, 0))  # transparent
    # Use inverted_mask as alpha channel, scale alpha to control transparency
    alpha = inverted_mask.point(lambda p: int(p * 0.8))  # lane line transparency

    # Put alpha channel into the green color mask
    color_mask.putalpha(alpha)

    # Convert original image to RGBA
    image_rgba = image.convert("RGBA")

    # Composite color mask only on masked areas over original image
    overlay_img = Image.alpha_composite(image_rgba, color_mask).convert("RGB")

    return mask_img, overlay_img

# --- Main Inference Loop ---
def main():
    """
    Main inference loop.
    Loads config, model, checkpoint, runs inference on all images in input dir, saves masks.
    """
    # Load config and save a copy with datetime postname in output directory
    config_path = "config/inference_config.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        config: Dict[str, Any] = yaml.safe_load(f)

    # Prepare output directory
    output_dir = config.get("inference_output_dir", None)
    if output_dir is None:
        print("Please specify 'inference_output_dir' in config/inference_config.yaml")
        return
    os.makedirs(output_dir, exist_ok=True)

    # Input dirs
    input_img_dir = config.get("inference_images_dir", None)
    if input_img_dir is None:
        print("Please specify 'inference_images_dir' in config/inference_config.yaml")
        return
    
    # Output dirs
    output_mask_dir = os.path.join(output_dir, f"inference_results", "masks")
    os.makedirs(output_mask_dir, exist_ok=True)
    output_overlay_dir = os.path.join(output_dir, f"inference_results", "overlay")
    os.makedirs(output_overlay_dir, exist_ok=True)

    # Save config copy
    config_save_path = os.path.join(output_dir, "config")
    os.makedirs(config_save_path, exist_ok=True)
    with open(f"{config_save_path}/inference_config.yaml", "w") as f:
        yaml.dump(config, f)
    
    # Set device and load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    model = UNet(in_channels=config["in_channels"], out_channels=config["out_channels"])

    # Load checkpoint
    checkpoint_path = config.get("inference_checkpoint", None)
    if checkpoint_path is None:
        print("Please specify 'inference_checkpoint' in config/inference_config.yaml")
        return
    load_checkpoint(checkpoint_path, device, model)
    model.to(device)
    model.eval()

    # Image transform
    infer_transform = T.Compose([
        T.Resize((config["img_size"], config["img_size"])),
        T.ToTensor(),
    ])
    
    image_files = [f for f in os.listdir(input_img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for img_name in tqdm(image_files, desc="Inference", unit="img"):
        img_path = os.path.join(input_img_dir, img_name)
        mask_img, overlay_img = infer(model, device, img_path, infer_transform)

        # Save mask
        mask_img.save(os.path.join(output_mask_dir, f"mask_{img_name}"))

        # Save overlay image
        overlay_img.save(os.path.join(output_overlay_dir, f"overlay_{img_name}"))

# --- Entrypoint ---
if __name__ == "__main__":
    main()
