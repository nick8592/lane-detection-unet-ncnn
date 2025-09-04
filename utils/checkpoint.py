
"""
Checkpoint utility functions for saving and loading model state.
"""
import torch
from typing import Optional

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: Optional[float] = None,
    filename: Optional[str] = None
) -> None:
    """
    Save model and optimizer state to a checkpoint file.
    Args:
        model (torch.nn.Module): Model to save
        optimizer (torch.optim.Optimizer): Optimizer to save
        epoch (int): Current epoch
        val_loss (float, optional): Validation loss
        filename (str, optional): Path to save checkpoint
    """
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    if val_loss is not None:
        state['val_loss'] = val_loss
    if filename is not None:
        torch.save(state, filename)
    else:
        raise ValueError("filename must be provided to save checkpoint.")

def load_checkpoint(
    filename: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None
) -> dict:
    """
    Load model and optimizer state from a checkpoint file.
    Args:
        filename: Path to checkpoint file (str)
        model: PyTorch model (nn.Module)
        optimizer: PyTorch optimizer (optional)
    Returns:
        checkpoint (dict)
    """
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint
