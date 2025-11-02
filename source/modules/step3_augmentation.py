"""
Step 3: Data Augmentation
Implements various augmentation techniques to increase dataset diversity.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from pathlib import Path
import pandas as pd
from typing import Tuple, List, Dict
import logging
import albumentations as alb
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

    def create_keras_generator(self) -> ImageDataGenerator:
        """
        Create Keras ImageDataGenerator for augmentation.

        Returns:
            ImageDataGenerator object
        """
        datagen = ImageDataGenerator(
            rotation_range=self.params.get('rotation_range', 20),
            width_shift_range=self.params.get('width_shift_range', 0.2),
            height_shift_range=self.params.get('height_shift_range', 0.2),
            shear_range=self.params.get('shear_range', 0.15),
            zoom_range=self.params.get('zoom_range', 0.15),
            horizontal_flip=self.params.get('horizontal_flip', True),
            fill_mode=self.params.get('fill_mode', 'nearest'),
            brightness_range=self.params.get('brightness_range', [0.8, 1.2])
        )

        logger.info("Keras ImageDataGenerator created")
        return datagen

    def create_albumentations_pipeline(self) -> alb.Compose:
        """
        Create Albumentations augmentation pipeline.
        Albumentations provides more advanced augmentation options.

        Returns:
            Albumentations Compose object
        """
        transform = alb.Compose([
            alb.RandomRotate90(p=0.5),
            alb.Flip(p=0.5),
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
            method: Augmentation method ('keras' or 'albumentations')

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

        elif method == 'keras':
            # Keras expects values in [0, 1]
            if image.max() > 1.0:
                image = image.astype(np.float32) / 255.0

            datagen = self.create_keras_generator()
            image_expanded = np.expand_dims(image, axis=0)

            # Generate augmented image
            aug_iter = datagen.flow(image_expanded, batch_size=1)
            augmented_image = next(aug_iter)[0]

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

    def create_tf_dataset(self, images: np.ndarray, labels: np.ndarray,
                         batch_size: int = 32, shuffle: bool = True,
                         augment: bool = True) -> tf.data.Dataset:
        """
        Create TensorFlow Dataset with augmentation.

        Args:
            images: Array of images
            labels: Array of labels
            batch_size: Batch size
            shuffle: Whether to shuffle data
            augment: Whether to apply augmentation

        Returns:
            tf.data.Dataset
        """
        def augment_fn(image, label):
            """TensorFlow function for augmentation."""
            if augment:
                # Random rotation
                image = tf.image.rot90(
                    image,
                    k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
                )

                # Random flip
                image = tf.image.random_flip_left_right(image)
                image = tf.image.random_flip_up_down(image)

                # Random brightness and contrast
                image = tf.image.random_brightness(image, max_delta=0.2)
                image = tf.image.random_contrast(image, lower=0.8, upper=1.2)

                # Random saturation and hue
                image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
                image = tf.image.random_hue(image, max_delta=0.1)

                # Clip values to [0, 1]
                image = tf.clip_by_value(image, 0.0, 1.0)

            return image, label

        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((images, labels))

        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(images))

        if augment:
            dataset = dataset.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)

        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        logger.info(f"TensorFlow Dataset created with batch_size={batch_size}, augment={augment}")

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

    # Test TensorFlow dataset
    dataset = augmenter.create_tf_dataset(
        batch_images, batch_labels, batch_size=4, augment=True
    )

    print(f"\nTensorFlow Dataset created: {dataset}")


if __name__ == "__main__":
    main()
