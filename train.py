import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from models.unet import UNet
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from datasets.bdd100k import BDD100KDataset
from utils.metrics import compute_iou, compute_dice
from utils.checkpoint import save_checkpoint

DEBUG = False

def train_one_epoch(model, loader, criterion, optimizer, device, epoch, total_epochs, writer=None):
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
        preds = torch.sigmoid(outputs)
        iou_score += compute_iou(preds, masks)
        dice_score += compute_dice(preds, masks)
        num_batches += 1
        progress_bar.set_postfix(loss=loss.item())
    epoch_loss = running_loss / len(loader.dataset)
    iou_score /= num_batches
    dice_score /= num_batches
    if writer is not None:
        writer.add_images("Train/Input", images, epoch)
        writer.add_images("Train/Mask", masks, epoch)
        writer.add_images("Train/Prediction", torch.sigmoid(outputs), epoch)
        writer.add_scalar("Loss/Train", epoch_loss, epoch)
    return epoch_loss

def validate(model, loader, criterion, device, epoch, total_epochs, writer=None):
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
            preds = torch.sigmoid(outputs)
            iou_score += compute_iou(preds, masks)
            dice_score += compute_dice(preds, masks)
            num_batches += 1
            progress_bar.set_postfix(loss=loss.item())
    val_loss /= len(loader.dataset)
    iou_score /= num_batches
    dice_score /= num_batches
    if writer is not None:
        writer.add_images("Val/Input", images, epoch)
        writer.add_images("Val/Mask", masks, epoch)
        writer.add_images("Val/Prediction", torch.sigmoid(outputs), epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)
        writer.add_scalar("IoU/Val", iou_score, epoch)
        writer.add_scalar("Dice/Val", dice_score, epoch)
    print(f"Validation: Loss={val_loss:.4f}, IoU={iou_score:.4f}, Dice(F1)={dice_score:.4f}")
    return val_loss

def train(model, train_loader, val_loader, criterion, optimizer, device, config, checkpoint_dir):
    exp_name = os.path.basename(checkpoint_dir)
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

if __name__ == "__main__":
    with open("config/train_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=config["in_channels"], out_channels=config["out_channels"]).to(device)
    train_transform = T.Compose([
        T.Resize((config["img_size"], config["img_size"])),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
    ])
    val_transform = T.Compose([
        T.Resize((config["img_size"], config["img_size"])),
        T.ToTensor(),
    ])
    # Datasets and loaders
    train_dataset = BDD100KDataset(
        config["train_images_dir"], config["train_masks_dir"],
        transform=train_transform, mask_transform=train_transform,
        debug=DEBUG
    )
    val_dataset = BDD100KDataset(
        config["val_images_dir"], config["val_masks_dir"],
        transform=val_transform, mask_transform=val_transform,
        debug=DEBUG
    )
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"])
    exp_name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    checkpoint_dir = os.path.join("checkpoints", exp_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    train(model, train_loader, val_loader, criterion, optimizer, device, config, checkpoint_dir)
    # Save final model
    final_model_path = os.path.join(checkpoint_dir, "unet_final.pt")
    save_checkpoint(model, optimizer, config["epochs"], None, final_model_path)
