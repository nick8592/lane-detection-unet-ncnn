"""
Compare inference speed, FPS, IoU, F1-score, and Pixel Accuracy for UNet variants.
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import yaml
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm
from models.unet import UNet
from models.unet_depthwise import UNetDepthwise
from models.unet_depthwise_small import UNetDepthwiseSmall
from models.unet_depthwise_nano import UNetDepthwiseNano
from utils.checkpoint import load_checkpoint
from utils.metrics import compute_iou, compute_dice, pixel_accuracy, mean_absolute_error
try:
    from tabulate import tabulate
    TABULATE_AVAILABLE = True
except ImportError:
    TABULATE_AVAILABLE = False

# --- Model Configs ---
MODEL_CLASSES = {
    "unet": UNet,
    "unet_depthwise": UNetDepthwise,
    "unet_depthwise_small": UNetDepthwiseSmall,
    "unet_depthwise_nano": UNetDepthwiseNano,
}

# --- Inference Function ---

def infer(model, device, image_path, transform):
    """
    Run inference and return both mask and overlaid image.
    Matches the correct version from test.py.
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
    from PIL import ImageOps
    inverted_mask = ImageOps.invert(mask_img.convert("L"))

    # Create a color image (RGBA) for the mask overlay (green with alpha)
    color_mask = Image.new("RGBA", orig_size, color=(0, 255, 0, 0))  # transparent
    # Use inverted_mask as alpha channel, scale alpha to control transparency
    alpha = inverted_mask.point(lambda p: int(p * 0.8))  # lane line transparency
    color_mask.putalpha(alpha)
    image_rgba = image.convert("RGBA")
    overlay_img = Image.alpha_composite(image_rgba, color_mask).convert("RGB")

    return np.array(mask_img), overlay_img

# --- Main Comparison Script ---

def run_comparison(config, device_label, device):
    input_img_dir = config["inference_images_dir"]
    gt_mask_dir = config.get("inference_gt_mask_dir", None)
    checkpoint_paths = {
        "unet": config.get("unet_checkpoint"),
        "unet_depthwise": config.get("unet_depthwise_checkpoint"),
        "unet_depthwise_small": config.get("unet_depthwise_small_checkpoint"),
        "unet_depthwise_nano": config.get("unet_depthwise_nano_checkpoint"),
    }
    img_size = config["img_size"]
    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
    ])
    image_files = [f for f in os.listdir(input_img_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    num_images = config.get("num_images", None)
    if num_images is not None:
        image_files = image_files[:int(num_images)]
    results = {}
    metrics_available = True if gt_mask_dir else False
    for model_name, model_class in MODEL_CLASSES.items():
        print(f"\nEvaluating {model_name} on {device_label}...")
        checkpoint_path = checkpoint_paths[model_name]
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found for {model_name}, skipping.")
            continue
        model = model_class(in_channels=config["in_channels"], out_channels=config["out_channels"]).to(device)
        load_checkpoint(checkpoint_path, device, model)
        model.eval()
        times = []
        ious, f1s, accs, maes = [], [], [], []
        for img_name in tqdm(image_files, desc=f"{model_name} [{device_label}]", unit="img"):
            img_path = os.path.join(input_img_dir, img_name)
            start_time = time.time()
            pred_mask, _ = infer(model, device, img_path, transform)
            elapsed = (time.time() - start_time) * 1000  # ms
            times.append(elapsed)
            if gt_mask_dir:
                base_name = os.path.splitext(img_name)[0]
                # Try .png, .jpg, .jpeg for mask
                mask_exts = [".png", ".jpg", ".jpeg"]
                gt_path = None
                for ext in mask_exts:
                    candidate = os.path.join(gt_mask_dir, base_name + ext)
                    if os.path.exists(candidate):
                        gt_path = candidate
                        break
                if gt_path:
                    gt_mask = np.array(Image.open(gt_path).resize(pred_mask.shape[::-1], Image.NEAREST))
                    gt_mask = np.uint8(gt_mask > 0)
                    pred_tensor = torch.tensor(pred_mask).unsqueeze(0).unsqueeze(0)
                    gt_tensor = torch.tensor(gt_mask).unsqueeze(0).unsqueeze(0)
                    ious.append(compute_iou(pred_tensor, gt_tensor))
                    f1s.append(compute_dice(pred_tensor, gt_tensor))
                    accs.append(pixel_accuracy(pred_mask, gt_mask))
                    maes.append(mean_absolute_error(pred_mask, gt_mask))
        avg_time = sum(times) / len(times) if times else 0
        avg_fps = 1000.0 / avg_time if avg_time > 0 else 0
        avg_iou = np.mean(ious) if ious else None
        avg_f1 = np.mean(f1s) if f1s else None
        avg_acc = np.mean(accs) if accs else None
        avg_mae = np.mean(maes) if maes else None
        results[model_name] = {
            "avg_time_ms": avg_time,
            "avg_fps": avg_fps,
            "avg_iou": avg_iou,
            "avg_f1": avg_f1,
            "avg_acc": avg_acc,
            "avg_mae": avg_mae,
        }
    print(f"\n--- Comparison Results ({device_label}) ---")
    if not metrics_available:
        print("[Warning] Ground truth mask directory not provided or empty. IoU, F1, PixelAcc will show as N/A.")
    headers = ["Model", "Time(ms)", "FPS", "IoU", "F1", "PixelAcc", "MAE"]
    table = []
    for m, r in results.items():
        row = [
            m,
            f"{r['avg_time_ms']:.2f}",
            f"{r['avg_fps']:.2f}",
            f"{r['avg_iou']:.4f}" if r['avg_iou'] is not None else "N/A",
            f"{r['avg_f1']:.4f}" if r['avg_f1'] is not None else "N/A",
            f"{r['avg_acc']:.4f}" if r['avg_acc'] is not None else "N/A",
            f"{r['avg_mae']:.4f}" if r['avg_mae'] is not None else "N/A",
        ]
        table.append(row)
    if TABULATE_AVAILABLE:
        print(tabulate(table, headers=headers, tablefmt="github"))
    else:
        # Fallback to manual formatting
        print("\t".join(headers))
        for row in table:
            print("\t".join(row))

def main():
    config_path = "config/eval_config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Run on CUDA if available
    if torch.cuda.is_available():
        run_comparison(config, "CUDA", torch.device("cuda"))

    # Run on CPU
    run_comparison(config, "CPU", torch.device("cpu"))

if __name__ == "__main__":
    main()
