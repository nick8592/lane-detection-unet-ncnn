"""
Metrics utility functions for segmentation evaluation.
"""
import numpy as np

def compute_iou(pred, target, threshold=0.5):
    """
    Compute Intersection over Union (IoU) for binary segmentation masks.
    Args:
        pred (Tensor): Predicted mask
        target (Tensor): Ground truth mask
        threshold (float): Threshold for binarization
    Returns:
        float: IoU score
    """
    pred = (pred > threshold).float()
    target = (target > threshold).float()
    intersection = (pred * target).sum(dim=(1,2,3))
    union = ((pred + target) > 0).float().sum(dim=(1,2,3))
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()

def compute_dice(pred, target, threshold=0.5):
    """
    Compute Dice coefficient (F1 score).
    Args:
        pred (Tensor): Predicted mask
        target (Tensor): Ground truth mask
        threshold (float): Threshold for binarization
    Returns:
        float: Dice score
    """
    pred = (pred > threshold).float()
    target = (target > threshold).float()
    intersection = (pred * target).sum(dim=(1,2,3))
    dice = (2 * intersection + 1e-6) / (pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3)) + 1e-6)
    return dice.mean().item()

def pixel_accuracy(pred, target, threshold=0.5):
    """
    Compute pixel accuracy for binary segmentation masks.
    Args:
        pred (ndarray or Tensor): Predicted mask
        target (ndarray or Tensor): Ground truth mask
        threshold (float): Threshold for binarization
    Returns:
        float: Pixel accuracy
    """
    if hasattr(pred, 'numpy'):
        pred = pred.cpu().numpy()
    if hasattr(target, 'numpy'):
        target = target.cpu().numpy()
    pred = (pred > threshold).astype(np.uint8)
    target = (target > threshold).astype(np.uint8)
    correct = (pred == target).sum()
    total = pred.size
    return correct / total

def mean_absolute_error(pred, target):
    """
    Compute Mean Absolute Error (MAE) between prediction and target masks.
    Args:
        pred (ndarray or Tensor): Predicted mask
        target (ndarray or Tensor): Ground truth mask
    Returns:
        float: MAE
    """
    if hasattr(pred, 'numpy'):
        pred = pred.cpu().numpy()
    if hasattr(target, 'numpy'):
        target = target.cpu().numpy()

    return np.mean(np.abs(pred.astype(np.float32) - target.astype(np.float32)))