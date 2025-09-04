"""
Metrics utility functions for segmentation evaluation.
"""

def compute_iou(pred, target, threshold=0.5):
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