# pylint: disable=no-member
"""
Step 5: Weight Prediction using CNN Regression (PyTorch Implementation)
Predicts the weight of leftover food using regression WITH classification guidance.
UPDATED: Now uses classification predictions as additional input.
"""
import sys
from pathlib import Path
from typing import Tuple, List, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
import timm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from loguru import logger

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from source.config import config


class ClassConditionedDualInputRegressionModel(nn.Module):
    """Dual-input CNN regression model with classification guidance."""

    def __init__(self, num_classes: int, pretrained: bool = True):
        """
        Initialize class-conditioned regression model.

        Args:
            num_classes: Number of food categories
            pretrained: Whether to use ImageNet pretrained weights
        """
        super(ClassConditionedDualInputRegressionModel, self).__init__()

        # Shared feature extractor (EfficientNet-B0)
        self.feature_extractor = timm.create_model(
            'efficientnet_b0',
            pretrained=pretrained,
            num_classes=0,
            global_pool='avg'
        )

        num_features = self.feature_extractor.num_features

        # Feature processing
        self.feature_processing = nn.Sequential(
            nn.BatchNorm1d(num_features),
            nn.Dropout(0.3)
        )

        # Class embedding layer - converts class probabilities to useful representation
        self.class_embedding = nn.Sequential(
            nn.Linear(num_classes, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # Regression head (now includes class information)
        # Input: concatenated features from both images + class embedding
        self.regression_head = nn.Sequential(
            nn.Linear(num_features * 2 + 64, 512),
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

    def forward(self, image_before, image_after, class_probs):
        """
        Forward pass with dual inputs and class information.

        Args:
            image_before: Before image tensor [B, C, H, W]
            image_after: After image tensor [B, C, H, W]
            class_probs: Classification probabilities [B, num_classes]

        Returns:
            Weight predictions [B, 1]
        """
        # Extract features from both images
        features_before = self.feature_extractor(image_before)
        features_after = self.feature_extractor(image_after)

        # Process features
        features_before = self.feature_processing(features_before)
        features_after = self.feature_processing(features_after)

        # Process class information
        class_features = self.class_embedding(class_probs)

        # Concatenate all features: image features + class features
        combined_features = torch.cat([features_before, features_after, class_features], dim=1)

        # Predict weight
        weight = self.regression_head(combined_features)

        return weight


class ClassConditionedSingleInputRegressionModel(nn.Module):
    """Single-input CNN regression model with classification guidance."""

    def __init__(self, num_classes: int, pretrained: bool = True):
        """
        Initialize class-conditioned single-input regression model.

        Args:
            num_classes: Number of food categories
            pretrained: Whether to use ImageNet pretrained weights
        """
        super(ClassConditionedSingleInputRegressionModel, self).__init__()

        self.feature_extractor = timm.create_model(
            'efficientnet_b0',
            pretrained=pretrained,
            num_classes=0,
            global_pool='avg'
        )

        num_features = self.feature_extractor.num_features

        # Class embedding
        self.class_embedding = nn.Sequential(
            nn.Linear(num_classes, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # Regression head with class information
        self.regression_head = nn.Sequential(
            nn.BatchNorm1d(num_features + 64),
            nn.Dropout(0.3),
            nn.Linear(num_features + 64, 512),
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

    def forward(self, image, class_probs):
        """
        Forward pass with single input and class information.

        Args:
            image: Image tensor [B, C, H, W]
            class_probs: Classification probabilities [B, num_classes]

        Returns:
            Weight predictions [B, 1]
        """
        features = self.feature_extractor(image)
        class_features = self.class_embedding(class_probs)
        
        combined_features = torch.cat([features, class_features], dim=1)
        weight = self.regression_head(combined_features)
        
        return weight


class ClassConditionedRegressionDataset(Dataset):
    """PyTorch Dataset for weight regression with classification probabilities."""

    def __init__(self, image_before_paths: Optional[List[str]],
                 image_after_paths: List[str], 
                 weights: np.ndarray,
                 class_probs: np.ndarray,
                 img_size: int = 224, 
                 transform=None, 
                 dual_input: bool = True):
        """
        Initialize dataset with classification information.

        Args:
            image_before_paths: List of before-image paths
            image_after_paths: List of after-image paths
            weights: Weight targets
            class_probs: Classification probability distributions [N, num_classes]
            img_size: Target image size
            transform: Optional albumentations transform
            dual_input: Whether to use dual input
        """
        self.image_before_paths = image_before_paths
        self.image_after_paths = image_after_paths
        self.weights = weights
        self.class_probs = class_probs
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

        img_after = cv2.resize(img_after, (self.img_size, self.img_size))

        if self.dual_input:
            # Load before image
            img_before = cv2.imread(str(self.image_before_paths[idx]))
            if img_before is None:
                logger.warning(f"Failed to load image: {self.image_before_paths[idx]}")
                img_before = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            else:
                img_before = cv2.cvtColor(img_before, cv2.COLOR_BGR2RGB)

            img_before = cv2.resize(img_before, (self.img_size, self.img_size))

            # Apply augmentation
            if self.transform:
                augmented_before = self.transform(image=img_before)
                augmented_after = self.transform(image=img_after)
                img_before = augmented_before['image']
                img_after = augmented_after['image']

            # Normalize
            img_before = img_before.astype(np.float32) / 255.0
            img_after = img_after.astype(np.float32) / 255.0

            # Convert to tensors
            img_before = torch.from_numpy(img_before).permute(2, 0, 1).float()
            img_after = torch.from_numpy(img_after).permute(2, 0, 1).float()

            # Get weight and class probabilities
            weight = torch.tensor(self.weights[idx], dtype=torch.float32)
            class_prob = torch.from_numpy(self.class_probs[idx]).float()

            return img_before, img_after, class_prob, weight

        else:
            # Single input
            if self.transform:
                augmented = self.transform(image=img_after)
                img_after = augmented['image']

            img_after = img_after.astype(np.float32) / 255.0
            img_after = torch.from_numpy(img_after).permute(2, 0, 1).float()

            weight = torch.tensor(self.weights[idx], dtype=torch.float32)
            class_prob = torch.from_numpy(self.class_probs[idx]).float()

            return img_after, class_prob, weight


class WeightRegression:
    """Class-conditioned CNN regression model for predicting food weight."""

    def __init__(self, num_classes: int, img_size: int = config.IMG_HEIGHT,
                 use_dual_input: bool = True, device: str = None,
                 classification_model=None):
        """
        Initialize weight regression model with classification guidance.

        Args:
            num_classes: Number of food categories
            img_size: Input image size
            use_dual_input: Whether to use both before and after images
            device: Device to use ('cuda' or 'cpu')
            classification_model: Pre-trained classification model for predictions
        """
        self.num_classes = num_classes
        self.img_size = img_size
        self.use_dual_input = use_dual_input
        self.weight_stats = {}
        self.classification_model = classification_model

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        logger.info(f"Using device: {self.device}")
        logger.info(f"Class-conditioned regression with {num_classes} food categories")

        self.model = None

    def set_classification_model(self, classification_model):
        """Set the classification model for generating predictions."""
        self.classification_model = classification_model
        logger.info("Classification model set for regression")

    def build_model(self, pretrained: bool = True):
        """Build class-conditioned regression model."""
        if self.use_dual_input:
            self.model = ClassConditionedDualInputRegressionModel(
                num_classes=self.num_classes,
                pretrained=pretrained
            ).to(self.device)
        else:
            self.model = ClassConditionedSingleInputRegressionModel(
                num_classes=self.num_classes,
                pretrained=pretrained
            ).to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        logger.info(f"Class-conditioned regression model built (dual_input={self.use_dual_input})")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")

    def get_classification_predictions(self, image_paths: List[str], 
                                      batch_size: int = 32) -> np.ndarray:
        """
        Get classification predictions for images.

        Args:
            image_paths: List of image paths
            batch_size: Batch size for prediction

        Returns:
            Classification probabilities [N, num_classes]
        """
        if self.classification_model is None:
            raise ValueError("Classification model not set!")

        logger.info(f"Getting classification predictions for {len(image_paths)} images...")
        
        _, class_probs = self.classification_model.predict(
            image_paths, 
            batch_size=batch_size
        )
        
        return class_probs

    def normalize_weights(self, weights: np.ndarray, fit: bool = True) -> np.ndarray:
        """Normalize weight values."""
        if fit:
            self.weight_stats['mean'] = float(np.mean(weights))
            self.weight_stats['std'] = float(np.std(weights))
            logger.info(f"Weight statistics - Mean: {self.weight_stats['mean']:.2f}, "
                       f"Std: {self.weight_stats['std']:.2f}")

        normalized = (weights - self.weight_stats['mean']) / (self.weight_stats['std'] + 1e-7)
        return normalized

    def denormalize_weights(self, normalized_weights: np.ndarray) -> np.ndarray:
        """Denormalize weight predictions."""
        return normalized_weights * self.weight_stats['std'] + self.weight_stats['mean']

    def create_dataloaders(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
                          batch_size: int = 32, use_augmentation: bool = True,
                          predict_difference: bool = False,
                          num_workers: int = 0) -> Tuple[DataLoader, DataLoader, np.ndarray, np.ndarray]:
        """Create DataLoaders with classification predictions."""
        
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

        # Get classification predictions
        logger.info("Obtaining classification predictions for training data...")
        train_class_probs = self.get_classification_predictions(
            train_df['image_before_path'].tolist(),
            batch_size=batch_size
        )
        
        logger.info("Obtaining classification predictions for validation data...")
        val_class_probs = self.get_classification_predictions(
            val_df['image_before_path'].tolist(),
            batch_size=batch_size
        )

        # Create augmentation pipeline
        train_transform = None
        if use_augmentation:
            from step3_augmentation import DataAugmentation
            augmenter = DataAugmentation()
            train_transform = augmenter.create_albumentations_pipeline()

        # Create datasets
        if self.use_dual_input:
            train_dataset = ClassConditionedRegressionDataset(
                train_df['image_before_path'].tolist(),
                train_df['image_after_path'].tolist(),
                train_weights_norm,
                train_class_probs,
                img_size=self.img_size,
                transform=train_transform,
                dual_input=True
            )

            val_dataset = ClassConditionedRegressionDataset(
                val_df['image_before_path'].tolist(),
                val_df['image_after_path'].tolist(),
                val_weights_norm,
                val_class_probs,
                img_size=self.img_size,
                transform=None,
                dual_input=True
            )
        else:
            train_dataset = ClassConditionedRegressionDataset(
                None,
                train_df['image_after_path'].tolist(),
                train_weights_norm,
                train_class_probs,
                img_size=self.img_size,
                transform=train_transform,
                dual_input=False
            )

            val_dataset = ClassConditionedRegressionDataset(
                None,
                val_df['image_after_path'].tolist(),
                val_weights_norm,
                val_class_probs,
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

    def train_epoch(self, train_loader: DataLoader, criterion, optimizer, 
                   epoch: int, total_epochs: int):
        """Train for one epoch with class information."""
        self.model.train()

        running_loss = 0.0
        running_mae = 0.0
        running_mse = 0.0
        total_samples = 0

        for batch_idx, batch in enumerate(train_loader):
            if self.use_dual_input:
                img_before, img_after, class_probs, weights = batch
                img_before = img_before.to(self.device)
                img_after = img_after.to(self.device)
                class_probs = class_probs.to(self.device)
                weights = weights.to(self.device).unsqueeze(1)

                optimizer.zero_grad()
                predictions = self.model(img_before, img_after, class_probs)
            else:
                img_after, class_probs, weights = batch
                img_after = img_after.to(self.device)
                class_probs = class_probs.to(self.device)
                weights = weights.to(self.device).unsqueeze(1)

                optimizer.zero_grad()
                predictions = self.model(img_after, class_probs)

            loss = criterion(predictions, weights)
            loss.backward()
            optimizer.step()

            batch_size = weights.size(0)
            running_loss += loss.item() * batch_size
            running_mae += torch.abs(predictions - weights).sum().item()
            running_mse += ((predictions - weights) ** 2).sum().item()
            total_samples += batch_size

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
        """Validate the model."""
        self.model.eval()

        running_loss = 0.0
        running_mae = 0.0
        running_mse = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                if self.use_dual_input:
                    img_before, img_after, class_probs, weights = batch
                    img_before = img_before.to(self.device)
                    img_after = img_after.to(self.device)
                    class_probs = class_probs.to(self.device)
                    weights = weights.to(self.device).unsqueeze(1)

                    predictions = self.model(img_before, img_after, class_probs)
                else:
                    img_after, class_probs, weights = batch
                    img_after = img_after.to(self.device)
                    class_probs = class_probs.to(self.device)
                    weights = weights.to(self.device).unsqueeze(1)

                    predictions = self.model(img_after, class_probs)

                loss = criterion(predictions, weights)

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
        """Train the class-conditioned regression model."""
        
        if self.classification_model is None:
            raise ValueError("Classification model must be set before training!")

        logger.info("Preparing training data for class-conditioned regression...")

        if self.model is None:
            self.build_model()

        # Create data loaders (this will get classification predictions)
        train_loader, val_loader, _, _ = self.create_dataloaders(
            train_df, val_df,
            batch_size=batch_size,
            use_augmentation=use_augmentation,
            predict_difference=predict_difference,
            num_workers=num_workers
        )

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5,
            patience=config.REDUCE_LR_PATIENCE
        )

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

        logger.info("Starting class-conditioned regression training...")

        for epoch in range(epochs):
            train_metrics = self.train_epoch(
                train_loader, criterion, optimizer, epoch, epochs
            )

            val_metrics = self.validate(val_loader, criterion)

            scheduler.step(val_metrics['loss'])

            # Denormalize for logging
            train_mae_denorm = train_metrics['mae'] * self.weight_stats['std']
            val_mae_denorm = val_metrics['mae'] * self.weight_stats['std']
            train_rmse_denorm = train_metrics['rmse'] * self.weight_stats['std']
            val_rmse_denorm = val_metrics['rmse'] * self.weight_stats['std']

            logger.info(
                f"Epoch [{epoch+1}/{epochs}] "
                f"Train Loss: {train_metrics['loss']:.4f} "
                f"Train MAE: {train_mae_denorm:.2f}g "
                f"Val Loss: {val_metrics['loss']:.4f} "
                f"Val MAE: {val_mae_denorm:.2f}g "
                f"Val RMSE: {val_rmse_denorm:.2f}g"
            )

            history['loss'].append(train_metrics['loss'])
            history['mae'].append(train_mae_denorm)
            history['mse'].append(train_metrics['mse'])
            history['rmse'].append(train_rmse_denorm)
            history['val_loss'].append(val_metrics['loss'])
            history['val_mae'].append(val_mae_denorm)
            history['val_mse'].append(val_metrics['mse'])
            history['val_rmse'].append(val_rmse_denorm)

            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                self.save_model(config.REGRESSION_MODEL_PATH)
                patience_counter = 0
                logger.info(f"Best model saved with val_loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1

            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break

        logger.info("Class-conditioned regression training completed!")

        class History:
            def __init__(self, history_dict):
                self.history = history_dict

        return History(history)

    def predict(self, image_before_paths: List[str] = None,
               image_after_paths: List[str] = None, 
               batch_size: int = 32) -> np.ndarray:
        """Predict weights with classification guidance."""
        
        if self.model is None:
            raise ValueError("Model not loaded!")

        if self.classification_model is None:
            raise ValueError("Classification model not set!")

        if self.use_dual_input:
            if image_before_paths is None or image_after_paths is None:
                raise ValueError("Both before and after images required for dual input")

        # Get classification predictions
        logger.info("Getting classification predictions for weight estimation...")
        class_probs = self.get_classification_predictions(
            image_before_paths if self.use_dual_input else image_after_paths,
            batch_size=batch_size
        )

        # Create dataset
        dummy_weights = np.zeros(len(image_after_paths), dtype=np.float32)

        if self.use_dual_input:
            dataset = ClassConditionedRegressionDataset(
                image_before_paths, image_after_paths, dummy_weights, class_probs,
                img_size=self.img_size, transform=None, dual_input=True
            )
        else:
            dataset = ClassConditionedRegressionDataset(
                None, image_after_paths, dummy_weights, class_probs,
                img_size=self.img_size, transform=None, dual_input=False
            )

        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=False
        )

        self.model.eval()
        all_predictions = []

        with torch.no_grad():
            for batch in loader:
                if self.use_dual_input:
                    img_before, img_after, class_probs, _ = batch
                    img_before = img_before.to(self.device)
                    img_after = img_after.to(self.device)
                    class_probs = class_probs.to(self.device)
                    predictions = self.model(img_before, img_after, class_probs)
                else:
                    img_after, class_probs, _ = batch
                    img_after = img_after.to(self.device)
                    class_probs = class_probs.to(self.device)
                    predictions = self.model(img_after, class_probs)

                all_predictions.append(predictions.cpu().numpy())

        predictions_norm = np.concatenate(all_predictions, axis=0).flatten()
        predictions = self.denormalize_weights(predictions_norm)

        return predictions

    def evaluate(self, test_df: pd.DataFrame, predict_difference: bool = False) -> Dict:
        """Evaluate model on test set."""
        logger.info("Evaluating class-conditioned regression model...")

        if self.use_dual_input:
            predictions = self.predict(
                test_df['image_before_path'].tolist(),
                test_df['image_after_path'].tolist()
            )
        else:
            predictions = self.predict(
                image_after_paths=test_df['image_after_path'].tolist()
            )

        if predict_difference:
            ground_truth = test_df['weight_difference'].values
        else:
            ground_truth = test_df['Weight After Eaten (g)'].values

        mae = mean_absolute_error(ground_truth, predictions)
        mse = mean_squared_error(ground_truth, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(ground_truth, predictions)
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
            path.parent.mkdir(parents=True, exist_ok=True)

            torch.save({
                'model_state_dict': self.model.state_dict(),
                'num_classes': self.num_classes,
                'use_dual_input': self.use_dual_input,
                'img_size': self.img_size,
                'weight_stats': self.weight_stats
            }, path)

            logger.info(f"Class-conditioned model saved to: {path}")

    def load_model(self, path: Path = config.REGRESSION_MODEL_PATH):
        """Load a trained model."""
        checkpoint = torch.load(path, map_location=self.device)

        self.num_classes = checkpoint['num_classes']
        self.use_dual_input = checkpoint['use_dual_input']
        self.img_size = checkpoint['img_size']
        self.weight_stats = checkpoint['weight_stats']

        self.build_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        logger.info(f"Class-conditioned model loaded from: {path}")


def main():
    """Main function for testing."""
    from step1_data_reader import FoodDataReader
    from step4_classification import FoodClassification

    # Load data
    reader = FoodDataReader()
    data = reader.process()

    # Load or train classification model first
    num_classes = data['train']['food_category_id'].nunique()
    
    logger.info("Loading classification model...")
    classifier = FoodClassification(num_classes=num_classes)
    
    # Try to load existing model, or train a small one
    try:
        classifier.load_model()
    except Exception:
        logger.warning("No existing classification model found. Training a quick one...")
        classifier.train(
            data['train'].head(100),
            data['val'].head(20),
            epochs=2,
            batch_size=16,
            fine_tune=False
        )

    # Initialize class-conditioned regression model
    regressor = WeightRegression(
        num_classes=num_classes,
        use_dual_input=True,
        classification_model=classifier
    )

    # Train model
    history = regressor.train(
        data['train'].head(100),
        data['val'].head(20),
        epochs=5,
        batch_size=16,
        use_augmentation=True,
        predict_difference=False
    )

    logger.info("\nClass-conditioned training completed!")
    logger.info(f"Final training MAE: {history.history['mae'][-1]:.2f}g")
    logger.info(f"Final validation MAE: {history.history['val_mae'][-1]:.2f}g")


if __name__ == "__main__":
    main()