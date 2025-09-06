
"""
Training script for UNet lane segmentation on BDD100K dataset.
Handles config loading, checkpointing, logging, and reproducibility.
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from typing import Any, Dict

# Set random seed for reproducibility
import random
import numpy as np
import torch
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

import yaml
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from models.unet import UNet
from models.unet_depthwise import UNetDepthwise
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from datasets.bdd100k import BDD100KDataset
from utils.metrics import compute_iou, compute_dice
from utils.checkpoint import save_checkpoint

DEBUG = False  # Set to True for fast debugging

# --- Training Loop ---
def train_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    writer: Any = None
) -> float:
    """
    Train the model for one epoch.
    Logs images and metrics to TensorBoard.
    """
    model.train()
    running_loss = 0.0
    iou_score = 0.0
    dice_score = 0.0
    num_batches = 0
    progress_bar = tqdm(loader, desc=f"Training Epoch {epoch+1}/{total_epochs}", leave=False)
    for images, masks in progress_bar:
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        iou_score += compute_iou(outputs, masks)
        dice_score += compute_dice(outputs, masks)
        num_batches += 1
        progress_bar.set_postfix(loss=loss.item())
    epoch_loss = running_loss / len(loader.dataset)
    iou_score /= num_batches
    dice_score /= num_batches
    if writer is not None:
        writer.add_images("Train/Input", images, epoch)
        writer.add_images("Train/Mask", masks, epoch)
        writer.add_images("Train/Prediction", outputs, epoch)
        writer.add_scalar("Loss/Train", epoch_loss, epoch)
    return epoch_loss

# --- Validation Loop ---
def validate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    writer: Any = None
) -> float:
    """
    Validate the model for one epoch.
    Logs images and metrics to TensorBoard.
    """
    model.eval()
    val_loss = 0.0
    iou_score = 0.0
    dice_score = 0.0
    num_batches = 0
    progress_bar = tqdm(loader, desc=f"Validation Epoch {epoch+1}/{total_epochs}", leave=False)
    with torch.no_grad():
        for images, masks in progress_bar:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item() * images.size(0)
            iou_score += compute_iou(outputs, masks)
            dice_score += compute_dice(outputs, masks)
            num_batches += 1
            progress_bar.set_postfix(loss=loss.item())
    val_loss /= len(loader.dataset)
    iou_score /= num_batches
    dice_score /= num_batches
    if writer is not None:
        writer.add_images("Val/Input", images, epoch)
        writer.add_images("Val/Mask", masks, epoch)
        writer.add_images("Val/Prediction", outputs, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)
        writer.add_scalar("IoU/Val", iou_score, epoch)
        writer.add_scalar("Dice/Val", dice_score, epoch)
    print(f"Validation: Loss={val_loss:.4f}, IoU={iou_score:.4f}, Dice(F1)={dice_score:.4f}")
    return val_loss

# --- Main Training Function ---
def train(model, train_loader, val_loader, criterion, optimizer, device, config, exp_name, checkpoint_dir):
    """
    Main training loop.
    Handles warmup, checkpoint saving, and TensorBoard logging.
    """
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=config["warmup_epochs"])
    log_dir = f"runs/{exp_name}"
    writer = SummaryWriter(log_dir)
    best_val_loss = float('inf')
    best_model_path = None
    for epoch in range(config["epochs"]):
        print(f"Epoch [{epoch+1}/{config['epochs']}] Starting...")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, config["epochs"], writer=writer)
        val_loss = validate(model, val_loader, criterion, device, epoch, config["epochs"], writer=writer)
        print(f"Epoch [{epoch+1}/{config['epochs']}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        # Save model for each epoch if enabled
        if config.get("save_all_epochs", True):
            epoch_model_path = os.path.join(checkpoint_dir, f"unet_epoch_{epoch+1:03d}_val_{val_loss:.4f}.pt")
            save_checkpoint(model, optimizer, epoch+1, val_loss, epoch_model_path)
        # Save best model if enabled
        if config.get("save_best_model", True) and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(checkpoint_dir, f"unet_best.pt")
            save_checkpoint(model, optimizer, epoch+1, val_loss, best_model_path)
        if epoch < config["warmup_epochs"]:
            scheduler.step()
            print(f"Warm-up: Adjusted LR to {optimizer.param_groups[0]['lr']:.6f}")
    writer.close()

# --- Main Entrypoint ---
if __name__ == "__main__":
    """
    Load config, set up device, model, transforms, datasets, loaders, optimizer.
    Create experiment folder for checkpoints. Run training and save final model.
    """
    # Load config and save a copy with datetime postname in checkpoint folder
    config_path = "config/train_config.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        config: Dict[str, Any] = yaml.safe_load(f)

    exp_name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    checkpoint_dir = os.path.join("checkpoints", f"{exp_name}/weights")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save config copy
    config_save_path = os.path.join("checkpoints", exp_name, "config")
    os.makedirs(config_save_path, exist_ok=True)
    with open(f"{config_save_path}/train_config.yaml", "w") as f:
        yaml.dump(config, f)
    
    # Set device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_type = config.get("model_type", "unet")
    if model_type == "unet":
        model = UNet(in_channels=config["in_channels"], out_channels=config["out_channels"]).to(device)
    elif model_type == "unet_depthwise":
        model = UNetDepthwise(in_channels=config["in_channels"], out_channels=config["out_channels"]).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'unet' or 'unet_depthwise'.")

    # Transforms
    train_transform = T.Compose([
        T.Resize((config["img_size"], config["img_size"])),
        T.ToTensor(),
    ])
    val_transform = T.Compose([
        T.Resize((config["img_size"], config["img_size"])),
        T.ToTensor(),
    ])
    # Datasets and loaders
    train_dataset = BDD100KDataset(
        config["train_images_dir"], config["train_masks_dir"],
        transform=train_transform, debug=DEBUG
    )
    val_dataset = BDD100KDataset(
        config["val_images_dir"], config["val_masks_dir"],
        transform=val_transform, debug=DEBUG
    )
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"])
    train(model, train_loader, val_loader, criterion, optimizer, device, config, exp_name, checkpoint_dir)
    # Save final model
    final_model_path = os.path.join(checkpoint_dir, "unet_final.pt")
    save_checkpoint(model, optimizer, config["epochs"], None, final_model_path)
