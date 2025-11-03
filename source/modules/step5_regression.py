"""
Step 5: Weight Prediction using CNN Regression (PyTorch Implementation)
Predicts the weight of leftover food using regression.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import timm
import cv2
from pathlib import Path
import pandas as pd
from typing import Tuple, List, Dict, Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from loguru import logger
import config


class DualInputRegressionModel(nn.Module):
    """Dual-input CNN regression model for weight prediction."""

    def __init__(self, pretrained: bool = True):
        """
        Initialize dual-input regression model.

        Args:
            pretrained: Whether to use ImageNet pretrained weights
        """
        super(DualInputRegressionModel, self).__init__()

        # Shared feature extractor (EfficientNet-B0)
        self.feature_extractor = timm.create_model(
            'efficientnet_b0',
            pretrained=pretrained,
            num_classes=0,  # Remove head
            global_pool='avg'
        )

        # Get number of features
        num_features = self.feature_extractor.num_features

        # After feature extractor processing
        self.feature_processing = nn.Sequential(
            nn.BatchNorm1d(num_features),
            nn.Dropout(0.3)
        )

        # Regression head (takes concatenated features from both images)
        self.regression_head = nn.Sequential(
            nn.Linear(num_features * 2, 512),  # *2 because we concatenate features
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)  # Single output (weight)
        )

    def forward(self, image_before, image_after):
        """
        Forward pass with dual inputs.

        Args:
            image_before: Before image tensor [B, C, H, W]
            image_after: After image tensor [B, C, H, W]

        Returns:
            Weight predictions [B, 1]
        """
        # Extract features from both images using shared extractor
        features_before = self.feature_extractor(image_before)
        features_after = self.feature_extractor(image_after)

        # Process features
        features_before = self.feature_processing(features_before)
        features_after = self.feature_processing(features_after)

        # Concatenate features
        combined_features = torch.cat([features_before, features_after], dim=1)

        # Predict weight
        weight = self.regression_head(combined_features)

        return weight


class SingleInputRegressionModel(nn.Module):
    """Single-input CNN regression model for weight prediction."""

    def __init__(self, pretrained: bool = True):
        """
        Initialize single-input regression model.

        Args:
            pretrained: Whether to use ImageNet pretrained weights
        """
        super(SingleInputRegressionModel, self).__init__()

        # Feature extractor (EfficientNet-B0)
        self.feature_extractor = timm.create_model(
            'efficientnet_b0',
            pretrained=pretrained,
            num_classes=0,
            global_pool='avg'
        )

        num_features = self.feature_extractor.num_features

        # Regression head
        self.regression_head = nn.Sequential(
            nn.BatchNorm1d(num_features),
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, image):
        """
        Forward pass with single input.

        Args:
            image: Image tensor [B, C, H, W]

        Returns:
            Weight predictions [B, 1]
        """
        features = self.feature_extractor(image)
        weight = self.regression_head(features)
        return weight


class RegressionDataset(Dataset):
    """PyTorch Dataset for weight regression."""

    def __init__(self, image_before_paths: Optional[List[str]],
                 image_after_paths: List[str], weights: np.ndarray,
                 img_size: int = 224, transform=None, dual_input: bool = True):
        """
        Initialize dataset.

        Args:
            image_before_paths: List of before-image paths (None if single input)
            image_after_paths: List of after-image paths
            weights: Weight targets
            img_size: Target image size
            transform: Optional albumentations transform
            dual_input: Whether to use dual input
        """
        self.image_before_paths = image_before_paths
        self.image_after_paths = image_after_paths
        self.weights = weights
        self.img_size = img_size
        self.transform = transform
        self.dual_input = dual_input

    def __len__(self):
        return len(self.weights)

    def __getitem__(self, idx):
        # Load after image
        img_after = cv2.imread(str(self.image_after_paths[idx]))
        if img_after is None:
            logger.warning(f"Failed to load image: {self.image_after_paths[idx]}")
            img_after = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        else:
            img_after = cv2.cvtColor(img_after, cv2.COLOR_BGR2RGB)

        # Resize
        img_after = cv2.resize(img_after, (self.img_size, self.img_size))

        if self.dual_input:
            # Load before image
            img_before = cv2.imread(str(self.image_before_paths[idx]))
            if img_before is None:
                logger.warning(f"Failed to load image: {self.image_before_paths[idx]}")
                img_before = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            else:
                img_before = cv2.cvtColor(img_before, cv2.COLOR_BGR2RGB)

            # Resize
            img_before = cv2.resize(img_before, (self.img_size, self.img_size))

            # Apply same augmentation to both images if specified
            if self.transform:
                augmented_before = self.transform(image=img_before)
                augmented_after = self.transform(image=img_after)
                img_before = augmented_before['image']
                img_after = augmented_after['image']

            # Normalize to [0, 1]
            img_before = img_before.astype(np.float32) / 255.0
            img_after = img_after.astype(np.float32) / 255.0

            # Convert to PyTorch tensors [C, H, W]
            img_before = torch.from_numpy(img_before).permute(2, 0, 1).float()
            img_after = torch.from_numpy(img_after).permute(2, 0, 1).float()

            # Get weight
            weight = torch.tensor(self.weights[idx], dtype=torch.float32)

            return img_before, img_after, weight

        else:
            # Single input
            if self.transform:
                augmented = self.transform(image=img_after)
                img_after = augmented['image']

            # Normalize to [0, 1]
            img_after = img_after.astype(np.float32) / 255.0

            # Convert to PyTorch tensor [C, H, W]
            img_after = torch.from_numpy(img_after).permute(2, 0, 1).float()

            # Get weight
            weight = torch.tensor(self.weights[idx], dtype=torch.float32)

            return img_after, weight


class WeightRegression:
    """CNN-based regression model for predicting food weight."""

    def __init__(self, img_size: int = config.IMG_HEIGHT,
                 use_dual_input: bool = True, device: str = None):
        """
        Initialize weight regression model.

        Args:
            img_size: Input image size
            use_dual_input: Whether to use both before and after images
            device: Device to use ('cuda' or 'cpu')
        """
        self.img_size = img_size
        self.use_dual_input = use_dual_input
        self.weight_stats = {}  # For normalization

        # Device configuration
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        logger.info(f"Using device: {self.device}")

        self.model = None

    def build_model(self, pretrained: bool = True):
        """Build regression model."""
        if self.use_dual_input:
            self.model = DualInputRegressionModel(pretrained=pretrained).to(self.device)
        else:
            self.model = SingleInputRegressionModel(pretrained=pretrained).to(self.device)

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        logger.info(f"Regression model built (dual_input={self.use_dual_input})")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")

    def normalize_weights(self, weights: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        Normalize weight values for better training.

        Args:
            weights: Array of weights
            fit: Whether to fit normalization parameters

        Returns:
            Normalized weights
        """
        if fit:
            self.weight_stats['mean'] = float(np.mean(weights))
            self.weight_stats['std'] = float(np.std(weights))
            logger.info(f"Weight statistics - Mean: {self.weight_stats['mean']:.2f}, "
                       f"Std: {self.weight_stats['std']:.2f}")

        normalized = (weights - self.weight_stats['mean']) / (self.weight_stats['std'] + 1e-7)
        return normalized

    def denormalize_weights(self, normalized_weights: np.ndarray) -> np.ndarray:
        """
        Denormalize weight predictions.

        Args:
            normalized_weights: Normalized weight values

        Returns:
            Original scale weights
        """
        return normalized_weights * self.weight_stats['std'] + self.weight_stats['mean']

    def create_dataloaders(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
                          batch_size: int = 32, use_augmentation: bool = True,
                          predict_difference: bool = False,
                          num_workers: int = 0) -> Tuple[DataLoader, DataLoader, np.ndarray, np.ndarray]:
        """
        Create PyTorch DataLoaders.

        Args:
            train_df: Training dataframe
            val_df: Validation dataframe
            batch_size: Batch size
            use_augmentation: Whether to use data augmentation
            predict_difference: Whether to predict weight difference
            num_workers: Number of data loading workers

        Returns:
            Tuple of (train_loader, val_loader, train_weights, val_weights)
        """
        # Prepare target weights
        if predict_difference:
            train_weights = train_df['weight_difference'].values
            val_weights = val_df['weight_difference'].values
            logger.info("Training to predict weight difference (leftover)")
        else:
            train_weights = train_df['Weight After Eaten (g)'].values
            val_weights = val_df['Weight After Eaten (g)'].values
            logger.info("Training to predict absolute weight after eating")

        # Normalize weights
        train_weights_norm = self.normalize_weights(train_weights, fit=True)
        val_weights_norm = self.normalize_weights(val_weights, fit=False)

        # Create augmentation pipeline if needed
        train_transform = None
        if use_augmentation:
            from step3_augmentation import DataAugmentation
            augmenter = DataAugmentation()
            train_transform = augmenter.create_albumentations_pipeline()

        # Create datasets
        if self.use_dual_input:
            train_dataset = RegressionDataset(
                train_df['image_before_path'].tolist(),
                train_df['image_after_path'].tolist(),
                train_weights_norm,
                img_size=self.img_size,
                transform=train_transform,
                dual_input=True
            )

            val_dataset = RegressionDataset(
                val_df['image_before_path'].tolist(),
                val_df['image_after_path'].tolist(),
                val_weights_norm,
                img_size=self.img_size,
                transform=None,
                dual_input=True
            )
        else:
            train_dataset = RegressionDataset(
                None,
                train_df['image_after_path'].tolist(),
                train_weights_norm,
                img_size=self.img_size,
                transform=train_transform,
                dual_input=False
            )

            val_dataset = RegressionDataset(
                None,
                val_df['image_after_path'].tolist(),
                val_weights_norm,
                img_size=self.img_size,
                transform=None,
                dual_input=False
            )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )

        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")

        return train_loader, val_loader, train_weights, val_weights

    def train_epoch(self, train_loader: DataLoader, criterion, optimizer, epoch: int, total_epochs: int):
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader
            criterion: Loss function
            optimizer: Optimizer
            epoch: Current epoch number
            total_epochs: Total number of epochs

        Returns:
            Dictionary with training metrics
        """
        self.model.train()

        running_loss = 0.0
        running_mae = 0.0
        running_mse = 0.0
        total_samples = 0

        for batch_idx, batch in enumerate(train_loader):
            if self.use_dual_input:
                img_before, img_after, weights = batch
                img_before = img_before.to(self.device)
                img_after = img_after.to(self.device)
                weights = weights.to(self.device).unsqueeze(1)

                # Forward pass
                optimizer.zero_grad()
                predictions = self.model(img_before, img_after)
            else:
                img_after, weights = batch
                img_after = img_after.to(self.device)
                weights = weights.to(self.device).unsqueeze(1)

                # Forward pass
                optimizer.zero_grad()
                predictions = self.model(img_after)

            # Calculate loss
            loss = criterion(predictions, weights)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Statistics
            batch_size = weights.size(0)
            running_loss += loss.item() * batch_size
            running_mae += torch.abs(predictions - weights).sum().item()
            running_mse += ((predictions - weights) ** 2).sum().item()
            total_samples += batch_size

            # Print progress
            if (batch_idx + 1) % 10 == 0:
                logger.info(
                    f'Epoch [{epoch+1}/{total_epochs}] '
                    f'Batch [{batch_idx+1}/{len(train_loader)}] '
                    f'Loss: {loss.item():.4f}'
                )

        epoch_loss = running_loss / total_samples
        epoch_mae = running_mae / total_samples
        epoch_mse = running_mse / total_samples
        epoch_rmse = np.sqrt(epoch_mse)

        return {
            'loss': epoch_loss,
            'mae': epoch_mae,
            'mse': epoch_mse,
            'rmse': epoch_rmse
        }

    def validate(self, val_loader: DataLoader, criterion):
        """
        Validate the model.

        Args:
            val_loader: Validation data loader
            criterion: Loss function

        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()

        running_loss = 0.0
        running_mae = 0.0
        running_mse = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                if self.use_dual_input:
                    img_before, img_after, weights = batch
                    img_before = img_before.to(self.device)
                    img_after = img_after.to(self.device)
                    weights = weights.to(self.device).unsqueeze(1)

                    predictions = self.model(img_before, img_after)
                else:
                    img_after, weights = batch
                    img_after = img_after.to(self.device)
                    weights = weights.to(self.device).unsqueeze(1)

                    predictions = self.model(img_after)

                # Calculate loss
                loss = criterion(predictions, weights)

                # Statistics
                batch_size = weights.size(0)
                running_loss += loss.item() * batch_size
                running_mae += torch.abs(predictions - weights).sum().item()
                running_mse += ((predictions - weights) ** 2).sum().item()
                total_samples += batch_size

        epoch_loss = running_loss / total_samples
        epoch_mae = running_mae / total_samples
        epoch_mse = running_mse / total_samples
        epoch_rmse = np.sqrt(epoch_mse)

        return {
            'loss': epoch_loss,
            'mae': epoch_mae,
            'mse': epoch_mse,
            'rmse': epoch_rmse
        }

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
              epochs: int = config.REGRESSION_EPOCHS,
              batch_size: int = config.REGRESSION_BATCH_SIZE,
              learning_rate: float = config.REGRESSION_LEARNING_RATE,
              use_augmentation: bool = True,
              predict_difference: bool = False,
              num_workers: int = 0):
        """
        Train the regression model.

        Args:
            train_df: Training dataframe
            val_df: Validation dataframe
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            use_augmentation: Whether to use data augmentation
            predict_difference: If True, predict weight difference instead of absolute weight
            num_workers: Number of data loading workers

        Returns:
            Training history dictionary
        """
        logger.info("Preparing training data for regression...")

        # Build model if not already built
        if self.model is None:
            self.build_model()

        # Create data loaders
        train_loader, val_loader, train_weights, val_weights = self.create_dataloaders(
            train_df, val_df,
            batch_size=batch_size,
            use_augmentation=use_augmentation,
            predict_difference=predict_difference,
            num_workers=num_workers
        )

        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5,
            patience=config.REDUCE_LR_PATIENCE, verbose=True
        )

        # History tracking
        history = {
            'loss': [],
            'mae': [],
            'mse': [],
            'rmse': [],
            'val_loss': [],
            'val_mae': [],
            'val_mse': [],
            'val_rmse': []
        }

        best_val_loss = float('inf')
        patience_counter = 0

        logger.info("Starting regression training...")

        for epoch in range(epochs):
            # Train
            train_metrics = self.train_epoch(
                train_loader, criterion, optimizer, epoch, epochs
            )

            # Validate
            val_metrics = self.validate(val_loader, criterion)

            # Update learning rate
            scheduler.step(val_metrics['loss'])

            # Denormalize MAE for logging (more interpretable)
            train_mae_denorm = train_metrics['mae'] * self.weight_stats['std']
            val_mae_denorm = val_metrics['mae'] * self.weight_stats['std']
            train_rmse_denorm = train_metrics['rmse'] * self.weight_stats['std']
            val_rmse_denorm = val_metrics['rmse'] * self.weight_stats['std']

            # Log metrics
            logger.info(
                f"Epoch [{epoch+1}/{epochs}] "
                f"Train Loss: {train_metrics['loss']:.4f} "
                f"Train MAE: {train_mae_denorm:.2f}g "
                f"Val Loss: {val_metrics['loss']:.4f} "
                f"Val MAE: {val_mae_denorm:.2f}g "
                f"Val RMSE: {val_rmse_denorm:.2f}g"
            )

            # Save history
            history['loss'].append(train_metrics['loss'])
            history['mae'].append(train_mae_denorm)
            history['mse'].append(train_metrics['mse'])
            history['rmse'].append(train_rmse_denorm)
            history['val_loss'].append(val_metrics['loss'])
            history['val_mae'].append(val_mae_denorm)
            history['val_mse'].append(val_metrics['mse'])
            history['val_rmse'].append(val_rmse_denorm)

            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                self.save_model(config.REGRESSION_MODEL_PATH)
                patience_counter = 0
                logger.info(f"Best model saved with val_loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break

        logger.info("Regression training completed!")

        # Create history object similar to Keras
        class History:
            def __init__(self, history_dict):
                self.history = history_dict

        return History(history)

    def predict(self, image_before_paths: List[str] = None,
               image_after_paths: List[str] = None, batch_size: int = 32) -> np.ndarray:
        """
        Predict weights for given images.

        Args:
            image_before_paths: List of before-image paths (if dual input)
            image_after_paths: List of after-image paths
            batch_size: Batch size for prediction

        Returns:
            Predicted weights
        """
        if self.model is None:
            raise ValueError("Model not loaded!")

        if self.use_dual_input:
            if image_before_paths is None or image_after_paths is None:
                raise ValueError("Both before and after images required for dual input model")

        # Create dataset
        dummy_weights = np.zeros(len(image_after_paths), dtype=np.float32)

        if self.use_dual_input:
            dataset = RegressionDataset(
                image_before_paths, image_after_paths, dummy_weights,
                img_size=self.img_size, transform=None, dual_input=True
            )
        else:
            dataset = RegressionDataset(
                None, image_after_paths, dummy_weights,
                img_size=self.img_size, transform=None, dual_input=False
            )

        # Create data loader
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=False
        )

        # Predict
        self.model.eval()
        all_predictions = []

        with torch.no_grad():
            for batch in loader:
                if self.use_dual_input:
                    img_before, img_after, _ = batch
                    img_before = img_before.to(self.device)
                    img_after = img_after.to(self.device)
                    predictions = self.model(img_before, img_after)
                else:
                    img_after, _ = batch
                    img_after = img_after.to(self.device)
                    predictions = self.model(img_after)

                all_predictions.append(predictions.cpu().numpy())

        predictions_norm = np.concatenate(all_predictions, axis=0).flatten()

        # Denormalize
        predictions = self.denormalize_weights(predictions_norm)

        return predictions

    def evaluate(self, test_df: pd.DataFrame, predict_difference: bool = False) -> Dict:
        """
        Evaluate model on test set.

        Args:
            test_df: Test dataframe
            predict_difference: Whether model predicts difference or absolute weight

        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Evaluating regression model...")

        # Predict
        if self.use_dual_input:
            predictions = self.predict(
                test_df['image_before_path'].tolist(),
                test_df['image_after_path'].tolist()
            )
        else:
            predictions = self.predict(
                image_after_paths=test_df['image_after_path'].tolist()
            )

        # Get ground truth
        if predict_difference:
            ground_truth = test_df['weight_difference'].values
        else:
            ground_truth = test_df['Weight After Eaten (g)'].values

        # Calculate metrics
        mae = mean_absolute_error(ground_truth, predictions)
        mse = mean_squared_error(ground_truth, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(ground_truth, predictions)

        # MAPE
        mape = np.mean(np.abs((ground_truth - predictions) / (ground_truth + 1e-7))) * 100

        metrics = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'mape': mape
        }

        logger.info(f"Test MAE: {mae:.2f}g")
        logger.info(f"Test RMSE: {rmse:.2f}g")
        logger.info(f"Test R²: {r2:.4f}")
        logger.info(f"Test MAPE: {mape:.2f}%")

        return metrics

    def save_model(self, path: Path = config.REGRESSION_MODEL_PATH):
        """Save the trained model."""
        if self.model is not None:
            # Ensure directory exists
            path.parent.mkdir(parents=True, exist_ok=True)

            # Save model state dict
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'use_dual_input': self.use_dual_input,
                'img_size': self.img_size,
                'weight_stats': self.weight_stats
            }, path)

            logger.info(f"Model saved to: {path}")

    def load_model(self, path: Path = config.REGRESSION_MODEL_PATH):
        """Load a trained model."""
        # Load checkpoint
        checkpoint = torch.load(path, map_location=self.device)

        # Build model with saved parameters
        self.use_dual_input = checkpoint['use_dual_input']
        self.img_size = checkpoint['img_size']
        self.weight_stats = checkpoint['weight_stats']

        self.build_model()

        # Load state dict
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        logger.info(f"Model loaded from: {path}")


def main():
    """Main function for testing."""
    from step1_data_reader import FoodDataReader

    # Load data
    reader = FoodDataReader()
    data = reader.process()

    # Initialize regression model
    regressor = WeightRegression(use_dual_input=True)

    # Train model
    history = regressor.train(
        data['train'].head(100),  # Use subset for testing
        data['val'].head(20),
        epochs=5,
        batch_size=16,
        use_augmentation=True,
        predict_difference=False
    )

    print(f"\nTraining completed!")
    print(f"Final training MAE: {history.history['mae'][-1]:.2f}g")
    print(f"Final validation MAE: {history.history['val_mae'][-1]:.2f}g")


if __name__ == "__main__":
    main()
