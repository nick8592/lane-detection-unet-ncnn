import torch

def save_checkpoint(model, optimizer, epoch, val_loss=None, filename=None):
    """
    Save model and optimizer state to a checkpoint file.
    Args:
        model: PyTorch model (nn.Module)
        optimizer: PyTorch optimizer
        epoch: Current epoch (int)
        val_loss: Validation loss (float, optional)
        filename: Path to save checkpoint (str)
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

def load_checkpoint(filename, model, optimizer=None):
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
