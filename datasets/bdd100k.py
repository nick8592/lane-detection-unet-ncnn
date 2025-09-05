
"""
BDD100KDataset for lane segmentation.
Loads images and masks, applies transforms, and supports debug mode.
"""
import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class BDD100KDataset(Dataset):
    """Custom Dataset for BDD100K lane segmentation."""
    def __init__(self, images_dir: str, masks_dir: str, transform: T.Compose = None, debug: bool = False):
        """
        Args:
            images_dir (str): Path to images directory
            masks_dir (str): Path to masks directory
            transform (T.Compose, optional): Transform for images
            debug (bool, optional): Use subset for debugging
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.image_files = sorted(os.listdir(images_dir))
        self.mask_files = sorted(os.listdir(masks_dir))
        if debug:
            self.image_files = self.image_files[:5000]
            self.mask_files = self.mask_files[:5000]
        assert len(self.image_files) == len(self.mask_files), "Mismatch between images and masks"

    def __len__(self) -> int:
        """
        Returns:
            int: Number of samples
        """
        return len(self.image_files)

    def __getitem__(self, idx: int):
        """
        Args:
            idx (int): Index
        Returns:
            Tuple[Tensor, Tensor]: Image and mask tensors
        """
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        mask_path = os.path.join(self.masks_dir, self.mask_files[idx])
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        else:
            image = T.ToTensor()(image)
            mask = T.ToTensor()(mask)
        return image, mask
