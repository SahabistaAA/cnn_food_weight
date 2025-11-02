"""
Step 3: Data Augmentation (PyTorch Implementation)
Implements various augmentation techniques to increase dataset diversity.
"""
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import cv2
from pathlib import Path
import pandas as pd
from typing import Tuple, List, Dict
import logging
import albumentations as alb
from albumentations.pytorch import ToTensorV2
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataAugmentation:
    """Handles data augmentation for training images."""

    def __init__(self, augmentation_params: Dict = None):
        """
        Initialize data augmentation.

        Args:
            augmentation_params: Dictionary of augmentation parameters
        """
        self.params = augmentation_params or config.AUGMENTATION_PARAMS

    def create_torch_transforms(self, augment: bool = True):
        """
        Create PyTorch transforms for augmentation.

        Args:
            augment: Whether to apply augmentations

        Returns:
            torchvision transforms
        """
        if augment:
            transform = T.Compose([
                T.ToPILImage(),
                T.RandomRotation(degrees=self.params.get('rotation_range', 20)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.2),
                T.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1
                ),
                T.RandomAffine(
                    degrees=0,
                    translate=(0.1, 0.1),
                    scale=(0.9, 1.1),
                    shear=10
                ),
                T.ToTensor(),
            ])
        else:
            transform = T.Compose([
                T.ToPILImage(),
                T.ToTensor(),
            ])

        logger.info("PyTorch transforms created")
        return transform

    def create_albumentations_pipeline(self) -> alb.Compose:
        """
        Create Albumentations augmentation pipeline.
        Albumentations provides more advanced augmentation options.

        Returns:
            Albumentations Compose object
        """
        transform = alb.Compose([
            alb.RandomRotate90(p=0.5),
            alb.HorizontalFlip(p=0.5),
            alb.VerticalFlip(p=0.2),
            alb.Transpose(p=0.3),

            # Optical distortions
            alb.OneOf([
                alb.OpticalDistortion(p=0.3),
                alb.GridDistortion(p=0.3),
                alb.ElasticTransform(p=0.3),
            ], p=0.3),

            # Color augmentations
            alb.OneOf([
                alb.CLAHE(clip_limit=2, p=0.5),
                alb.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                alb.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            ], p=0.5),

            # Blur and noise
            alb.OneOf([
                alb.MotionBlur(p=0.3),
                alb.MedianBlur(blur_limit=3, p=0.3),
                alb.GaussianBlur(p=0.3),
                alb.GaussNoise(p=0.3),
            ], p=0.3),

            # Shadow and lighting
            alb.OneOf([
                alb.RandomShadow(p=0.3),
                alb.RandomFog(p=0.2),
                alb.RandomSunFlare(p=0.2),
            ], p=0.2),

            # Geometric transformations
            alb.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=20,
                p=0.5
            ),

            # Quality degradation (simulates different camera conditions)
            alb.OneOf([
                alb.ImageCompression(quality_lower=75, quality_upper=100, p=0.3),
                alb.Downscale(scale_min=0.7, scale_max=0.9, p=0.3),
            ], p=0.2),
        ])

        logger.info("Albumentations pipeline created")
        return transform

    def augment_image(self, image: np.ndarray, method: str = 'albumentations') -> np.ndarray:
        """
        Apply augmentation to a single image.

        Args:
            image: Input image (numpy array, values in [0, 1] or [0, 255])
            method: Augmentation method ('torch' or 'albumentations')

        Returns:
            Augmented image
        """
        if method == 'albumentations':
            # Albumentations expects uint8
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)

            transform = self.create_albumentations_pipeline()
            augmented = transform(image=image)
            augmented_image = augmented['image'].astype(np.float32) / 255.0

        elif method == 'torch':
            # PyTorch expects values in [0, 1]
            if image.max() > 1.0:
                image = image.astype(np.float32) / 255.0

            transform = self.create_torch_transforms(augment=True)
            # Convert back to numpy after transformation
            augmented_tensor = transform((image * 255).astype(np.uint8))
            augmented_image = augmented_tensor.permute(1, 2, 0).numpy()

        else:
            raise ValueError(f"Unknown augmentation method: {method}")

        return augmented_image

    def augment_batch(self, images: np.ndarray, labels: np.ndarray,
                     augment_factor: int = 2, method: str = 'albumentations') -> Tuple[np.ndarray, np.ndarray]:
        """
        Augment a batch of images with their labels.

        Args:
            images: Array of images [N, H, W, C]
            labels: Array of labels [N, ...]
            augment_factor: Number of augmented versions per image
            method: Augmentation method

        Returns:
            Tuple of (augmented_images, augmented_labels)
        """
        augmented_images = []
        augmented_labels = []

        for img, label in zip(images, labels):
            # Keep original
            augmented_images.append(img)
            augmented_labels.append(label)

            # Generate augmented versions
            for _ in range(augment_factor - 1):
                aug_img = self.augment_image(img, method=method)
                augmented_images.append(aug_img)
                augmented_labels.append(label)

        return np.array(augmented_images), np.array(augmented_labels)

    def create_pytorch_dataset(self, images: np.ndarray, labels: np.ndarray,
                              augment: bool = True) -> 'AugmentedDataset':
        """
        Create PyTorch Dataset with augmentation.

        Args:
            images: Array of images
            labels: Array of labels
            augment: Whether to apply augmentation

        Returns:
            PyTorch Dataset
        """
        if augment:
            transform = self.create_albumentations_pipeline()
        else:
            transform = None

        dataset = AugmentedDataset(images, labels, transform=transform)

        logger.info(f"PyTorch Dataset created with {len(dataset)} samples, augment={augment}")

        return dataset

    def visualize_augmentations(self, image: np.ndarray, num_augmentations: int = 9,
                               method: str = 'albumentations') -> np.ndarray:
        """
        Generate multiple augmented versions for visualization.

        Args:
            image: Input image
            num_augmentations: Number of augmented versions to generate
            method: Augmentation method

        Returns:
            Grid of augmented images
        """
        augmented_images = [image]  # Include original

        for _ in range(num_augmentations - 1):
            aug_img = self.augment_image(image, method=method)
            augmented_images.append(aug_img)

        # Create grid
        grid_size = int(np.ceil(np.sqrt(num_augmentations)))
        h, w = image.shape[:2]

        grid = np.zeros((grid_size * h, grid_size * w, 3), dtype=np.float32)

        for idx, img in enumerate(augmented_images):
            row = idx // grid_size
            col = idx % grid_size
            grid[row*h:(row+1)*h, col*w:(col+1)*w] = img

        return grid

    def balance_dataset_with_augmentation(self, df: pd.DataFrame,
                                         target_samples_per_class: int = None) -> pd.DataFrame:
        """
        Balance dataset by augmenting underrepresented classes.

        Args:
            df: Input dataframe with image paths and labels
            target_samples_per_class: Target number of samples per class (default: max class count)

        Returns:
            Balanced dataframe with augmentation flags
        """
        # Count samples per class
        class_counts = df['food_category_id'].value_counts()

        if target_samples_per_class is None:
            target_samples_per_class = class_counts.max()

        logger.info(f"Balancing dataset to {target_samples_per_class} samples per class")

        # Calculate augmentation factor for each class
        augmentation_plan = {}
        for food_id, count in class_counts.items():
            if count < target_samples_per_class:
                aug_factor = target_samples_per_class // count
                augmentation_plan[food_id] = aug_factor
            else:
                augmentation_plan[food_id] = 1

        logger.info(f"Augmentation plan: {augmentation_plan}")

        df['augmentation_factor'] = df['food_category_id'].map(augmentation_plan)

        return df


class AugmentedDataset(Dataset):
    """PyTorch Dataset with augmentation support."""

    def __init__(self, images, labels, transform=None):
        """
        Initialize dataset.

        Args:
            images: numpy array of images [N, H, W, C]
            labels: numpy array of labels
            transform: albumentations transform
        """
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # Apply augmentation if specified
        if self.transform:
            # Albumentations expects uint8
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)

            augmented = self.transform(image=image)
            image = augmented['image'].astype(np.float32) / 255.0

        # Convert to PyTorch tensor and change format from HWC to CHW
        image = torch.from_numpy(image).permute(2, 0, 1).float()

        # Convert label to tensor
        if isinstance(label, np.ndarray):
            label = torch.from_numpy(label).float()
        else:
            label = torch.tensor(label, dtype=torch.float32)

        return image, label


def main():
    """Main function for testing."""
    # Create sample image
    sample_image = np.random.rand(224, 224, 3).astype(np.float32)

    # Initialize augmentation
    augmenter = DataAugmentation()

    # Test augmentation
    augmented = augmenter.augment_image(sample_image, method='albumentations')
    print(f"Original shape: {sample_image.shape}")
    print(f"Augmented shape: {augmented.shape}")

    # Test batch augmentation
    batch_images = np.random.rand(10, 224, 224, 3).astype(np.float32)
    batch_labels = np.random.randint(0, 10, size=(10,))

    aug_images, aug_labels = augmenter.augment_batch(
        batch_images, batch_labels, augment_factor=3
    )

    print(f"\nOriginal batch size: {len(batch_images)}")
    print(f"Augmented batch size: {len(aug_images)}")

    # Test PyTorch dataset
    dataset = augmenter.create_pytorch_dataset(
        batch_images, batch_labels, augment=True
    )

    print(f"\nPyTorch Dataset created: {dataset}")
    print(f"Dataset length: {len(dataset)}")

    # Test one sample
    img, lbl = dataset[0]
    print(f"Sample image shape: {img.shape}")  # Should be [C, H, W]
    print(f"Sample label: {lbl}")


if __name__ == "__main__":
    main()
