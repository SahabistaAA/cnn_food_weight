# pylint: disable=no-member
"""
Step 2: U-Net Image Segmentation (PyTorch Implementation)
Implements U-Net for segmenting food from background.
"""
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import pandas as pd
from loguru import logger

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from source.config import config


class UNetModel(nn.Module):
    """U-Net architecture for image segmentation."""

    def __init__(self, in_channels=3, out_channels=1, filters=None):
        """
        Initialize U-Net model.

        Args:
            in_channels: Number of input channels (3 for RGB)
            out_channels: Number of output channels (1 for binary segmentation)
            filters: List of filter sizes for each layer
        """
        super(UNetModel, self).__init__()
        if filters is None:
            filters = [64, 128, 256, 512, 1024]
        self.filters = filters

        # Encoder (downsampling path)
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()

        in_ch = in_channels
        for num_filters in filters[:-1]:
            self.encoders.append(self._conv_block(in_ch, num_filters))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_ch = num_filters

        # Bottleneck
        self.bottleneck = self._conv_block(filters[-2], filters[-1])

        # Decoder (upsampling path)
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()

        for i in range(len(filters) - 1):
            idx = len(filters) - i - 1
            self.upconvs.append(
                nn.ConvTranspose2d(filters[idx], filters[idx-1], kernel_size=2, stride=2)
            )
            self.decoders.append(self._conv_block(filters[idx], filters[idx-1]))

        # Output layer
        self.output = nn.Conv2d(filters[0], out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def _conv_block(self, in_channels, out_channels):
        """Create a convolutional block with two conv layers."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3)
        )

    def forward(self, x):
        """Forward pass through U-Net."""
        # Encoder
        skip_connections = []
        for encoder, pool in zip(self.encoders, self.pools):
            x = encoder(x)
            skip_connections.append(x)
            x = pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        skip_connections = skip_connections[::-1]
        for i, (upconv, decoder) in enumerate(zip(self.upconvs, self.decoders)):
            x = upconv(x)
            skip = skip_connections[i]

            # Handle size mismatch
            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)

            x = torch.cat([x, skip], dim=1)
            x = decoder(x)

        # Output
        x = self.output(x)
        x = self.sigmoid(x)

        return x


class SegmentationDataset(Dataset):
    """PyTorch Dataset for segmentation."""

    def __init__(self, images, masks):
        """
        Initialize dataset.

        Args:
            images: numpy array of images [N, H, W, C]
            masks: numpy array of masks [N, H, W, 1]
        """
        self.images = images
        self.masks = masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]

        # Convert to PyTorch tensors and change format from HWC to CHW
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).permute(2, 0, 1).float()

        return image, mask


class UNetSegmentation:
    """U-Net model for food image segmentation."""

    def __init__(self, img_size=None, filters=None, device=None):
        """
        Initialize U-Net segmentation model.

        Args:
            img_size: Input image size (square)
            filters: List of filter sizes for each layer
            device: Device to use ('cuda' or 'cpu')
        """
        self.img_size = img_size or config.SEGMENTATION_IMG_SIZE
        self.filters = filters or config.UNET_FILTERS
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None

        logger.info(f"Using device: {self.device}")

    def build_unet(self) -> UNetModel:
        """
        Build U-Net architecture.

        Returns:
            PyTorch model
        """
        model = UNetModel(in_channels=3, out_channels=1, filters=self.filters)
        model = model.to(self.device)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        logger.info(f"U-Net model built with input size: {self.img_size}x{self.img_size}")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")

        return model

    def dice_coefficient(self, y_pred, y_true, smooth=1):
        """
        Dice coefficient for evaluating segmentation quality.

        Args:
            y_pred: Predicted masks
            y_true: Ground truth masks
            smooth: Smoothing factor

        Returns:
            Dice coefficient
        """
        y_pred_f = y_pred.flatten()
        y_true_f = y_true.flatten()
        intersection = (y_pred_f * y_true_f).sum()
        return (2. * intersection + smooth) / (y_pred_f.sum() + y_true_f.sum() + smooth)

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
                except cv2.error:  # pylint: disable=catching-non-exception
                    # Fallback to Otsu if GrabCut fails
                    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
                    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            else:
                raise ValueError(f"Unknown method: {method}")

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
              epochs: int = 30, batch_size: int = 16, learning_rate: float = 0.001):
        """
        Train the U-Net model.

        Args:
            train_df: Training dataframe
            val_df: Validation dataframe
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate

        Returns:
            Training history dictionary
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

        logger.info("Creating pseudo ground truth masks...")

        # Create pseudo masks
        train_masks = self.create_pseudo_masks(train_images, method='otsu')
        val_masks = self.create_pseudo_masks(val_images, method='otsu')

        logger.info(f"Training images: {train_images.shape}")
        logger.info(f"Training masks: {train_masks.shape}")

        # Create datasets and dataloaders
        train_dataset = SegmentationDataset(train_images, train_masks)
        val_dataset = SegmentationDataset(val_images, val_masks)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        # Build model if not already built
        if self.model is None:
            self.model = self.build_unet()

        # Loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=config.REDUCE_LR_PATIENCE
        )

        # Training history
        history = {
            'loss': [],
            'val_loss': [],
            'dice': [],
            'val_dice': []
        }

        best_val_loss = float('inf')
        patience_counter = 0

        logger.info("Starting U-Net training...")

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_dice = 0.0

            for _, (images, masks) in enumerate(train_loader):
                images = images.to(self.device)
                masks = masks.to(self.device)

                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, masks)

                # Backward pass
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_dice += self.dice_coefficient(outputs, masks).item()

            train_loss /= len(train_loader)
            train_dice /= len(train_loader)

            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_dice = 0.0

            with torch.no_grad():
                for images, masks in val_loader:
                    images = images.to(self.device)
                    masks = masks.to(self.device)

                    outputs = self.model(images)
                    loss = criterion(outputs, masks)

                    val_loss += loss.item()
                    val_dice += self.dice_coefficient(outputs, masks).item()

            val_loss /= len(val_loader)
            val_dice /= len(val_loader)

            # Update history
            history['loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['dice'].append(train_dice)
            history['val_dice'].append(val_dice)

            # Learning rate scheduling
            scheduler.step(val_loss)

            # Logging
            logger.info(
                f"Epoch [{epoch+1}/{epochs}] "
                f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}"
            )

            # Model checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model()
                patience_counter = 0
                logger.info(f"Model saved with val_loss: {val_loss:.4f}")
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                logger.info(f"Early stopping triggered after epoch {epoch+1}")
                break

        logger.info("U-Net training completed!")

        # Create history object similar to Keras
        class TrainingHistory:
            def __init__(self, history_dict):
                self.history = history_dict

        return TrainingHistory(history)

    def predict_masks(self, image_paths: List[str]) -> np.ndarray:
        """
        Predict segmentation masks for given images.

        Args:
            image_paths: List of image paths

        Returns:
            Predicted masks [N, H, W, 1]
        """
        images = self.load_and_preprocess_images(image_paths)

        # Convert to tensor and move to device
        images_tensor = torch.from_numpy(images).permute(0, 3, 1, 2).float().to(self.device)

        self.model.eval()
        with torch.no_grad():
            masks = self.model(images_tensor)

        # Convert back to numpy [N, H, W, 1]
        masks_np = masks.permute(0, 2, 3, 1).cpu().numpy()

        return masks_np

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

    def save_model(self, path=None):
        """Save the trained model."""
        if path is None:
            path = config.SEGMENTATION_MODEL_PATH
            
        if self.model is not None:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'img_size': self.img_size,
                'filters': self.filters
            }, path)
            logger.info(f"Model saved to: {path}")

    def load_model(self, path=None):
        """Load a trained model."""
        if path is None:
            path = config.SEGMENTATION_MODEL_PATH
            
        checkpoint = torch.load(path, map_location=self.device)

        if self.model is None:
            self.model = self.build_unet()

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

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

    logger.info("\nTraining completed!")
    logger.info(f"Final training loss: {history.history['loss'][-1]:.4f}")
    logger.info(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")


if __name__ == "__main__":
    main()