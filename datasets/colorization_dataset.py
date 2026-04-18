"""
Colorization Dataset
====================
Custom PyTorch dataset that wraps STL-10 and converts images to Lab color space.

Pipeline:
  RGB image → Lab → L channel (input) + ab channels (target)
  Both normalized to [-1, 1] for stable training.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as T
from PIL import Image

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.lab_utils import rgb_to_lab, normalize_l, normalize_ab


class ColorizationDataset(Dataset):
    """
    Wraps an image dataset (STL-10) for the colorization pretext task.

    Each sample returns:
        L:  (1, H, W) tensor — normalized grayscale luminance ∈ [-1, 1]
        ab: (2, H, W) tensor — normalized chrominance ∈ [-1, 1]
    """

    def __init__(self, base_dataset, image_size=96, augment=True):
        """
        Args:
            base_dataset: A torchvision-style dataset yielding (PIL Image, label)
            image_size:   Target spatial resolution
            augment:      Whether to apply random horizontal flip
        """
        self.base_dataset = base_dataset
        self.image_size = image_size
        self.augment = augment

        # Build transforms
        transforms_list = [T.Resize((image_size, image_size))]
        if augment:
            transforms_list.append(T.RandomHorizontalFlip(p=0.5))
        self.transform = T.Compose(transforms_list)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, _ = self.base_dataset[idx]  # PIL Image, label (label ignored)

        # Apply spatial transforms
        image = self.transform(image)

        # Convert to numpy RGB
        image_rgb = np.array(image.convert("RGB"), dtype=np.uint8)

        # RGB → Lab
        image_lab = rgb_to_lab(image_rgb)

        # Split and normalize
        L = normalize_l(image_lab[:, :, 0])     # (H, W), [-1, 1]
        ab = normalize_ab(image_lab[:, :, 1:])   # (H, W, 2), [-1, 1]

        # To tensors: (C, H, W)
        L_tensor = torch.from_numpy(L).unsqueeze(0).float()            # (1, H, W)
        ab_tensor = torch.from_numpy(ab).permute(2, 0, 1).float()      # (2, H, W)

        return L_tensor, ab_tensor


class LabeledColorizationDataset(Dataset):
    """
    Wraps a labeled dataset for evaluation — returns L, ab, AND the label.
    Used for extracting labeled embeddings for classification/clustering.
    """

    def __init__(self, base_dataset, image_size=96):
        self.base_dataset = base_dataset
        self.image_size = image_size
        self.transform = T.Resize((image_size, image_size))

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]

        image = self.transform(image)
        image_rgb = np.array(image.convert("RGB"), dtype=np.uint8)

        image_lab = rgb_to_lab(image_rgb)

        L = normalize_l(image_lab[:, :, 0])
        ab = normalize_ab(image_lab[:, :, 1:])

        L_tensor = torch.from_numpy(L).unsqueeze(0).float()
        ab_tensor = torch.from_numpy(ab).permute(2, 0, 1).float()

        return L_tensor, ab_tensor, label


def get_stl10_colorization_loaders(data_dir="./data", image_size=96,
                                     batch_size=128, num_workers=4):
    """
    Create DataLoaders for the STL-10 unlabeled split (pretraining).

    Returns:
        train_loader: DataLoader for pretraining (100K unlabeled images)
    """
    # STL-10 unlabeled split
    stl10_unlabeled = torchvision.datasets.STL10(
        root=data_dir,
        split="unlabeled",
        download=True,
    )

    dataset = ColorizationDataset(stl10_unlabeled, image_size=image_size, augment=True)

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    print(f"[DATA] STL-10 Unlabeled: {len(dataset)} images, "
          f"batch_size={batch_size}, image_size={image_size}")

    return train_loader


def get_stl10_labeled_loaders(data_dir="./data", image_size=96,
                                batch_size=128, num_workers=4):
    """
    Create DataLoaders for the STL-10 labeled splits (evaluation).

    Returns:
        train_loader: DataLoader for labeled train split (5,000 images)
        test_loader:  DataLoader for labeled test split (8,000 images)
        class_names:  List of STL-10 class names
    """
    stl10_train = torchvision.datasets.STL10(
        root=data_dir, split="train", download=True)
    stl10_test = torchvision.datasets.STL10(
        root=data_dir, split="test", download=True)

    train_dataset = LabeledColorizationDataset(stl10_train, image_size=image_size)
    test_dataset = LabeledColorizationDataset(stl10_test, image_size=image_size)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    # STL-10 class names
    class_names = [
        "airplane", "bird", "car", "cat", "deer",
        "dog", "horse", "monkey", "ship", "truck"
    ]

    print(f"[DATA] STL-10 Labeled: train={len(train_dataset)}, "
          f"test={len(test_dataset)}")

    return train_loader, test_loader, class_names
