"""
Step 2: U-Net Image Segmentation
Implements U-Net for segmenting food from background.
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
from pathlib import Path
import pandas as pd
from typing import Tuple, List
import logging
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UNetSegmentation:
    """U-Net model for food image segmentation."""

    def __init__(self, img_size: int = config.SEGMENTATION_IMG_SIZE,
                 filters: List[int] = config.UNET_FILTERS):
        """
        Initialize U-Net segmentation model.

        Args:
            img_size: Input image size (square)
            filters: List of filter sizes for each layer
        """
        self.img_size = img_size
        self.filters = filters
        self.model = None

    def build_unet(self) -> keras.Model:
        """
        Build U-Net architecture.

        Returns:
            Keras model
        """
        inputs = layers.Input(shape=(self.img_size, self.img_size, 3))

        # Encoder (downsampling path)
        skip_connections = []
        x = inputs

        for i, num_filters in enumerate(self.filters[:-1]):
            # Convolutional block
            x = layers.Conv2D(num_filters, 3, padding='same', activation='relu')(x)
            x = layers.Conv2D(num_filters, 3, padding='same', activation='relu')(x)
            x = layers.BatchNormalization()(x)

            skip_connections.append(x)

            # Downsampling
            x = layers.MaxPooling2D(pool_size=(2, 2))(x)
            x = layers.Dropout(0.3)(x)

        # Bottleneck
        x = layers.Conv2D(self.filters[-1], 3, padding='same', activation='relu')(x)
        x = layers.Conv2D(self.filters[-1], 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)

        # Decoder (upsampling path)
        for i, num_filters in enumerate(reversed(self.filters[:-1])):
            # Upsampling
            x = layers.Conv2DTranspose(num_filters, 2, strides=2, padding='same')(x)

            # Concatenate with skip connection
            skip = skip_connections[-(i + 1)]
            x = layers.Concatenate()([x, skip])

            # Convolutional block
            x = layers.Conv2D(num_filters, 3, padding='same', activation='relu')(x)
            x = layers.Conv2D(num_filters, 3, padding='same', activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.3)(x)

        # Output layer (binary segmentation: food vs background)
        outputs = layers.Conv2D(1, 1, activation='sigmoid', padding='same')(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name='UNet_Segmentation')

        logger.info(f"U-Net model built with input size: {self.img_size}x{self.img_size}")

        return model

    def compile_model(self, learning_rate: float = 0.001):
        """Compile the U-Net model."""
        if self.model is None:
            self.model = self.build_unet()

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.MeanIoU(num_classes=2)]
        )

        logger.info("U-Net model compiled successfully")

    def dice_coefficient(self, y_true, y_pred, smooth=1):
        """
        Dice coefficient for evaluating segmentation quality.

        Args:
            y_true: Ground truth masks
            y_pred: Predicted masks
            smooth: Smoothing factor

        Returns:
            Dice coefficient
        """
        y_true_f = tf.keras.backend.flatten(y_true)
        y_pred_f = tf.keras.backend.flatten(y_pred)
        intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (
            tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth
        )

    def create_pseudo_masks(self, images: np.ndarray, method: str = 'otsu') -> np.ndarray:
        """
        Create pseudo ground truth masks using thresholding methods.
        Since we don't have ground truth masks, we use automated segmentation.

        Args:
            images: Array of images [N, H, W, C]
            method: Thresholding method ('otsu', 'adaptive', or 'grabcut')

        Returns:
            Binary masks [N, H, W, 1]
        """
        masks = []

        for img in images:
            if method == 'otsu':
                # Convert to grayscale
                gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

                # Apply Gaussian blur
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)

                # Otsu's thresholding
                _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            elif method == 'adaptive':
                gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
                mask = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, 11, 2
                )

            elif method == 'grabcut':
                img_uint8 = (img * 255).astype(np.uint8)

                # Initialize mask
                mask_gc = np.zeros(img_uint8.shape[:2], np.uint8)

                # Define rectangle for GrabCut (assume food is in center)
                h, w = img_uint8.shape[:2]
                rect = (int(w*0.1), int(h*0.1), int(w*0.8), int(h*0.8))

                # GrabCut algorithm
                bgd_model = np.zeros((1, 65), np.float64)
                fgd_model = np.zeros((1, 65), np.float64)

                try:
                    cv2.grabCut(img_uint8, mask_gc, rect, bgd_model, fgd_model, 5,
                               cv2.GC_INIT_WITH_RECT)
                    mask = np.where((mask_gc == 2) | (mask_gc == 0), 0, 255).astype(np.uint8)
                except:
                    # Fallback to Otsu if GrabCut fails
                    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
                    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Resize mask to match model input size
            mask_resized = cv2.resize(mask, (self.img_size, self.img_size))

            # Normalize to [0, 1]
            mask_normalized = mask_resized.astype(np.float32) / 255.0

            masks.append(mask_normalized)

        return np.array(masks)[..., np.newaxis]

    def load_and_preprocess_images(self, image_paths: List[str]) -> np.ndarray:
        """
        Load and preprocess images for segmentation.

        Args:
            image_paths: List of image file paths

        Returns:
            Preprocessed images array [N, H, W, C]
        """
        images = []

        for img_path in image_paths:
            # Read image
            img = cv2.imread(str(img_path))
            if img is None:
                logger.warning(f"Failed to load image: {img_path}")
                continue

            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Resize
            img_resized = cv2.resize(img, (self.img_size, self.img_size))

            # Normalize to [0, 1]
            img_normalized = img_resized.astype(np.float32) / 255.0

            images.append(img_normalized)

        return np.array(images)

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
              epochs: int = 30, batch_size: int = 16):
        """
        Train the U-Net model.

        Args:
            train_df: Training dataframe
            val_df: Validation dataframe
            epochs: Number of training epochs
            batch_size: Batch size
        """
        logger.info("Loading and preprocessing training images...")

        # Load before and after images
        train_before_imgs = self.load_and_preprocess_images(train_df['image_before_path'].tolist())
        train_after_imgs = self.load_and_preprocess_images(train_df['image_after_path'].tolist())

        val_before_imgs = self.load_and_preprocess_images(val_df['image_before_path'].tolist())
        val_after_imgs = self.load_and_preprocess_images(val_df['image_after_path'].tolist())

        # Combine before and after images for training
        train_images = np.concatenate([train_before_imgs, train_after_imgs], axis=0)
        val_images = np.concatenate([val_before_imgs, val_after_imgs], axis=0)

        logger.info(f"Creating pseudo ground truth masks...")

        # Create pseudo masks
        train_masks = self.create_pseudo_masks(train_images, method='otsu')
        val_masks = self.create_pseudo_masks(val_images, method='otsu')

        logger.info(f"Training images: {train_images.shape}")
        logger.info(f"Training masks: {train_masks.shape}")

        # Compile model if not already compiled
        if self.model is None:
            self.compile_model()

        # Callbacks
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                config.SEGMENTATION_MODEL_PATH,
                save_best_only=True,
                monitor='val_loss',
                verbose=1
            ),
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=config.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=config.REDUCE_LR_PATIENCE,
                verbose=1,
                min_lr=1e-7
            )
        ]

        # Train model
        logger.info("Starting U-Net training...")

        history = self.model.fit(
            train_images, train_masks,
            validation_data=(val_images, val_masks),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        logger.info("U-Net training completed!")

        return history

    def predict_masks(self, image_paths: List[str]) -> np.ndarray:
        """
        Predict segmentation masks for given images.

        Args:
            image_paths: List of image paths

        Returns:
            Predicted masks [N, H, W, 1]
        """
        images = self.load_and_preprocess_images(image_paths)
        masks = self.model.predict(images, verbose=0)
        return masks

    def apply_mask_to_image(self, image: np.ndarray, mask: np.ndarray,
                           threshold: float = 0.5) -> np.ndarray:
        """
        Apply segmentation mask to image.

        Args:
            image: Input image
            mask: Segmentation mask
            threshold: Threshold for binary mask

        Returns:
            Masked image
        """
        binary_mask = (mask > threshold).astype(np.float32)
        masked_image = image * binary_mask
        return masked_image

    def save_model(self, path: Path = config.SEGMENTATION_MODEL_PATH):
        """Save the trained model."""
        if self.model is not None:
            self.model.save(path)
            logger.info(f"Model saved to: {path}")

    def load_model(self, path: Path = config.SEGMENTATION_MODEL_PATH):
        """Load a trained model."""
        self.model = keras.models.load_model(path)
        logger.info(f"Model loaded from: {path}")


def main():
    """Main function for testing."""
    from step1_data_reader import FoodDataReader

    # Load data
    reader = FoodDataReader()
    data = reader.process()

    # Initialize segmentation model
    segmenter = UNetSegmentation()

    # Train model
    history = segmenter.train(
        data['train'].head(50),  # Use subset for testing
        data['val'].head(10),
        epochs=5,
        batch_size=8
    )

    print(f"\nTraining completed!")
    print(f"Final training loss: {history.history['loss'][-1]:.4f}")
    print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")


if __name__ == "__main__":
    main()
