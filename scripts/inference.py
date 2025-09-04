# --- Imports and Setup ---
import os
import sys

# Add project root to sys.path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm
from models.unet import UNet
from utils.checkpoint import load_checkpoint

# --- Inference Function ---
def infer(model, device, image_path, transform):
    """
    Run inference on a single image and return the predicted mask resized to original resolution.
    Args:
        model: Trained UNet model
        device: torch.device
        image_path: Path to input image
        transform: Preprocessing transform
    Returns:
        mask_img: PIL Image of predicted mask resized to original image size
    """
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()
        pred_mask = (pred_mask > 0.5).astype('uint8') * 255
    # Resize mask back to original image size
    orig_size = image.size  # (width, height)
    mask_img = Image.fromarray(pred_mask)
    mask_img = mask_img.resize(orig_size, resample=Image.NEAREST)
    return mask_img

# --- Main Inference Loop ---
def main():
    """
    Main inference loop.
    Loads config, model, checkpoint, runs inference on all images in input dir, saves masks.
    """
    # Load config
    with open("config/inference_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=config["in_channels"], out_channels=config["out_channels"])
    # Load checkpoint
    checkpoint_path = config.get("inference_checkpoint", None)
    if checkpoint_path is None:
        print("Please specify 'inference_checkpoint' in config/inference_config.yaml")
        return
    load_checkpoint(checkpoint_path, model)
    model.to(device)
    model.eval()
    # Image transform
    infer_transform = T.Compose([
        T.Resize((config["img_size"], config["img_size"])),
        T.ToTensor(),
    ])
    # Input/output dirs
    input_dir = config.get("inference_images_dir", None)
    output_dir = config.get("inference_output_dir", None)
    if input_dir is None:
        print("Please specify 'inference_images_dir' in config/inference_config.yaml")
        return
    if output_dir is None:
        print("Please specify 'inference_output_dir' in config/inference_config.yaml")
        return
    os.makedirs(output_dir, exist_ok=True)
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for img_name in tqdm(image_files, desc="Inference", unit="img"):
        img_path = os.path.join(input_dir, img_name)
        mask_img = infer(model, device, img_path, infer_transform)
        mask_img.save(os.path.join(output_dir, f"mask_{img_name}"))

# --- Entrypoint ---
if __name__ == "__main__":
    main()
