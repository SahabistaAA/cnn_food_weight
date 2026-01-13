# pylint: disable=no-member
"""
Step 4: Food Classification using EfficientNet (PyTorch Implementation)
Classifies food images into different food categories.
"""
import sys
from pathlib import Path
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
import joblib
from sklearn.preprocessing import LabelEncoder
import timm
from loguru import logger

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from source.config import config


class EfficientNetClassifier(nn.Module):
    """EfficientNet-B0 model for food classification."""

    def __init__(self, num_classes: int, pretrained: bool = True):
        """
        Initialize EfficientNet classifier.

        Args:
            num_classes: Number of food categories
            pretrained: Whether to use ImageNet pretrained weights
        """
        super(EfficientNetClassifier, self).__init__()

        # Load EfficientNet-B0 from timm (PyTorch Image Models)
        # num_classes=0 removes the classification head
        self.backbone = timm.create_model(
            'efficientnet_b0',
            pretrained=pretrained,
            num_classes=0,  # Remove head
            global_pool=''  # Remove global pooling
        )

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Get number of features from backbone
        num_features = self.backbone.num_features

        # Classification head
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(num_features),
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Logits [B, num_classes]
        """
        # Extract features
        features = self.backbone(x)

        # Global pooling
        features = self.global_pool(features)
        features = features.flatten(1)

        # Classification
        logits = self.classifier(features)

        return logits


class ClassificationDataset(Dataset):
    """PyTorch Dataset for food classification."""

    def __init__(self, image_paths: List[str], labels: np.ndarray,
                 img_size: int = 224, transform=None):
        """
        Initialize dataset.

        Args:
            image_paths: List of image file paths
            labels: Encoded labels array
            img_size: Target image size
            transform: Optional albumentations transform
        """
        self.image_paths = image_paths
        self.labels = labels
        self.img_size = img_size
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        img = cv2.imread(str(img_path))

        if img is None:
            logger.warning(f"Failed to load image: {img_path}")
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize
        img = cv2.resize(img, (self.img_size, self.img_size))

        # Apply augmentation if specified
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']

        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0

        # Convert to PyTorch tensor [C, H, W]
        img = torch.from_numpy(img).permute(2, 0, 1).float()

        # Get label
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return img, label


class FoodClassification:
    """EfficientNet-based food classification model."""

    def __init__(self, num_classes: int, img_size=None, pretrained: bool = True, device=None):
        """
        Initialize food classification model.

        Args:
            num_classes: Number of food categories
            img_size: Input image size
            pretrained: Whether to use ImageNet pretrained weights
            device: Device to use ('cuda' or 'cpu')
        """
        self.num_classes = num_classes
        self.img_size = img_size or config.IMG_HEIGHT
        self.pretrained = pretrained

        # Device configuration
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        logger.info(f"Using device: {self.device}")

        self.model = None
        self.label_encoder = LabelEncoder()

    def build_model(self):
        """Build EfficientNet model."""
        try:
            self.model = EfficientNetClassifier(
                num_classes=self.num_classes,
                pretrained=self.pretrained
            ).to(self.device)

            # Count parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

            logger.info(f"EfficientNet model built with {self.num_classes} classes")
            logger.info(f"Total parameters: {total_params:,}")
            logger.info(f"Trainable parameters: {trainable_params:,}")

        except Exception as e:
            logger.error(f"Failed to build model: {e}")
            raise

    def freeze_backbone(self):
        """Freeze backbone for transfer learning."""
        if self.model is None:
            raise ValueError("Model not built yet!")

        for param in self.model.backbone.parameters():
            param.requires_grad = False

        logger.info("Backbone frozen for transfer learning")

    def unfreeze_backbone(self):
        """
        Unfreeze layers of the backbone for fine-tuning.
        """
        if self.model is None:
            raise ValueError("Model not built yet!")

        # Unfreeze all backbone parameters
        for param in self.model.backbone.parameters():
            param.requires_grad = True

        logger.info("Backbone unfrozen for fine-tuning")

    def prepare_labels(self, food_category_ids: List[str]) -> np.ndarray:
        """
        Encode food category IDs to numerical labels.

        Args:
            food_category_ids: List of food category IDs (e.g., ['001', '002', ...])

        Returns:
            Encoded labels array
        """
        labels = self.label_encoder.fit_transform(food_category_ids)
        logger.info(f"Encoded {len(np.unique(labels))} unique food categories")
        return labels

    def create_dataloaders(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
                          batch_size: int = 32, use_augmentation: bool = True,
                          num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
        """
        Create PyTorch DataLoaders.

        Args:
            train_df: Training dataframe
            val_df: Validation dataframe
            batch_size: Batch size
            use_augmentation: Whether to use data augmentation
            num_workers: Number of data loading workers

        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Prepare labels
        train_labels = self.prepare_labels(train_df['food_category_id'].tolist())
        val_labels = self.label_encoder.transform(val_df['food_category_id'].tolist())

        # Create augmentation pipeline if needed
        train_transform = None
        if use_augmentation:
            from step3_augmentation import DataAugmentation
            augmenter = DataAugmentation()
            train_transform = augmenter.create_albumentations_pipeline()

        # Create datasets
        train_dataset = ClassificationDataset(
            train_df['image_before_path'].tolist(),
            train_labels,
            img_size=self.img_size,
            transform=train_transform
        )

        val_dataset = ClassificationDataset(
            val_df['image_before_path'].tolist(),
            val_labels,
            img_size=self.img_size,
            transform=None
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

        return train_loader, val_loader

    def train_epoch(self, train_loader: DataLoader, criterion, optimizer,
                   epoch: int, total_epochs: int):
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
        correct = 0
        total = 0
        top5_correct = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Top-5 accuracy
            _, top5_pred = outputs.topk(min(5, outputs.size(1)), 1, largest=True, sorted=True)
            top5_correct += top5_pred.eq(labels.view(-1, 1).expand_as(top5_pred)).sum().item()

            # Print progress
            if (batch_idx + 1) % 10 == 0:
                logger.info(
                    f'Epoch [{epoch+1}/{total_epochs}] '
                    f'Batch [{batch_idx+1}/{len(train_loader)}] '
                    f'Loss: {loss.item():.4f} '
                    f'Acc: {100.*correct/total:.2f}%'
                )

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        epoch_top5_acc = top5_correct / total

        return {
            'loss': epoch_loss,
            'accuracy': epoch_acc,
            'top5_accuracy': epoch_top5_acc
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
        correct = 0
        total = 0
        top5_correct = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.model(images)
                loss = criterion(outputs, labels)

                # Statistics
                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                # Top-5 accuracy
                _, top5_pred = outputs.topk(min(5, outputs.size(1)), 1, largest=True, sorted=True)
                top5_correct += top5_pred.eq(labels.view(-1, 1).expand_as(top5_pred)).sum().item()

        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_acc = correct / total
        epoch_top5_acc = top5_correct / total

        return {
            'loss': epoch_loss,
            'accuracy': epoch_acc,
            'top5_accuracy': epoch_top5_acc
        }

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
              epochs=None,
              batch_size=None,
              learning_rate=None,
              use_augmentation: bool = True,
              fine_tune: bool = True,
              fine_tune_epochs: int = 20,
              num_workers: int = 0):
        """
        Train the classification model.

        Args:
            train_df: Training dataframe
            val_df: Validation dataframe
            epochs: Number of training epochs for initial training
            batch_size: Batch size
            learning_rate: Initial learning rate
            use_augmentation: Whether to use data augmentation
            fine_tune: Whether to perform fine-tuning after initial training
            fine_tune_epochs: Number of epochs for fine-tuning
            num_workers: Number of data loading workers

        Returns:
            Training history dictionary
        """
        # Use config defaults if not provided
        if epochs is None:
            epochs = config.CLASSIFICATION_EPOCHS
        if batch_size is None:
            batch_size = config.CLASSIFICATION_BATCH_SIZE
        if learning_rate is None:
            learning_rate = config.CLASSIFICATION_LEARNING_RATE
            
        logger.info("Preparing training data...")

        # Build model if not already built
        if self.model is None:
            self.build_model()

        # Create data loaders
        train_loader, val_loader = self.create_dataloaders(
            train_df, val_df,
            batch_size=batch_size,
            use_augmentation=use_augmentation,
            num_workers=num_workers
        )

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()

        # History tracking
        history = {
            'loss': [],
            'accuracy': [],
            'top5_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_top5_accuracy': []
        }

        # Phase 1: Train with frozen backbone
        logger.info("Phase 1: Training with frozen backbone...")
        self.freeze_backbone()

        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=learning_rate
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5,
            patience=config.REDUCE_LR_PATIENCE
        )

        best_val_acc = 0.0
        patience_counter = 0

        for epoch in range(epochs):
            # Train
            train_metrics = self.train_epoch(
                train_loader, criterion, optimizer, epoch, epochs
            )

            # Validate
            val_metrics = self.validate(val_loader, criterion)

            # Update learning rate
            scheduler.step(val_metrics['loss'])

            # Log metrics
            logger.info(
                f"Epoch [{epoch+1}/{epochs}] "
                f"Train Loss: {train_metrics['loss']:.4f} "
                f"Train Acc: {train_metrics['accuracy']:.4f} "
                f"Val Loss: {val_metrics['loss']:.4f} "
                f"Val Acc: {val_metrics['accuracy']:.4f} "
                f"Val Top5 Acc: {val_metrics['top5_accuracy']:.4f}"
            )

            # Save history
            history['loss'].append(train_metrics['loss'])
            history['accuracy'].append(train_metrics['accuracy'])
            history['top5_accuracy'].append(train_metrics['top5_accuracy'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_accuracy'].append(val_metrics['accuracy'])
            history['val_top5_accuracy'].append(val_metrics['top5_accuracy'])

            # Save best model
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                self.save_model(config.CLASSIFICATION_MODEL_PATH)
                patience_counter = 0
                logger.info(f"Best model saved with val_acc: {best_val_acc:.4f}")
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break

        # Phase 2: Fine-tuning
        if fine_tune:
            logger.info("\nPhase 2: Fine-tuning with unfrozen backbone...")

            self.unfreeze_backbone()

            # Use lower learning rate for fine-tuning
            optimizer = optim.Adam(self.model.parameters(), lr=1e-5)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=3
            )

            best_val_acc = 0.0
            patience_counter = 0

            for epoch in range(fine_tune_epochs):
                # Train
                train_metrics = self.train_epoch(
                    train_loader, criterion, optimizer, epoch, fine_tune_epochs
                )

                # Validate
                val_metrics = self.validate(val_loader, criterion)

                # Update learning rate
                scheduler.step(val_metrics['loss'])

                # Log metrics
                logger.info(
                    f"Fine-tune Epoch [{epoch+1}/{fine_tune_epochs}] "
                    f"Train Loss: {train_metrics['loss']:.4f} "
                    f"Train Acc: {train_metrics['accuracy']:.4f} "
                    f"Val Loss: {val_metrics['loss']:.4f} "
                    f"Val Acc: {val_metrics['accuracy']:.4f}"
                )

                # Save history
                history['loss'].append(train_metrics['loss'])
                history['accuracy'].append(train_metrics['accuracy'])
                history['top5_accuracy'].append(train_metrics['top5_accuracy'])
                history['val_loss'].append(val_metrics['loss'])
                history['val_accuracy'].append(val_metrics['accuracy'])
                history['val_top5_accuracy'].append(val_metrics['top5_accuracy'])

                # Save best model
                if val_metrics['accuracy'] > best_val_acc:
                    best_val_acc = val_metrics['accuracy']
                    self.save_model(config.CLASSIFICATION_MODEL_PATH)
                    patience_counter = 0
                    logger.info(f"Best model saved with val_acc: {best_val_acc:.4f}")
                else:
                    patience_counter += 1

                # Early stopping
                if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                    logger.info(f"Early stopping triggered after {epoch+1} fine-tune epochs")
                    break

        logger.info("Classification training completed!")

        # Create history object similar to Keras
        class History:
            def __init__(self, history_dict):
                self.history = history_dict

        return History(history)

    def predict(self, image_paths: List[str], batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict food categories for given images.

        Args:
            image_paths: List of image paths
            batch_size: Batch size for prediction

        Returns:
            Tuple of (predicted_categories, prediction_probabilities)
        """
        if self.model is None:
            raise ValueError("Model not loaded!")

        # Create dataset
        dummy_labels = np.zeros(len(image_paths), dtype=np.int64)
        dataset = ClassificationDataset(
            image_paths, dummy_labels, img_size=self.img_size, transform=None
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
            for images, _ in loader:
                images = images.to(self.device)
                outputs = self.model(images)

                # Apply softmax to get probabilities
                probs = torch.softmax(outputs, dim=1)
                all_predictions.append(probs.cpu().numpy())

        predictions = np.concatenate(all_predictions, axis=0)
        predicted_labels = np.argmax(predictions, axis=1)
        predicted_categories = self.label_encoder.inverse_transform(predicted_labels)

        return predicted_categories, predictions

    def evaluate(self, test_df: pd.DataFrame, batch_size: int = 32):
        """
        Evaluate model on test set.

        Args:
            test_df: Test dataframe
            batch_size: Batch size for evaluation

        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Evaluating classification model...")

        # Prepare data
        test_labels = self.label_encoder.transform(test_df['food_category_id'].tolist())

        test_dataset = ClassificationDataset(
            test_df['image_before_path'].tolist(),
            test_labels,
            img_size=self.img_size,
            transform=None
        )

        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=False
        )

        # Evaluate
        criterion = nn.CrossEntropyLoss()
        test_metrics = self.validate(test_loader, criterion)

        logger.info(f"Test Loss: {test_metrics['loss']:.4f}")
        logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"Test Top-5 Accuracy: {test_metrics['top5_accuracy']:.4f}")

        return test_metrics

    def save_model(self, path=None):
        """Save the trained model."""
        if path is None:
            path = config.CLASSIFICATION_MODEL_PATH
            
        if self.model is not None:
            # Ensure directory exists
            path.parent.mkdir(parents=True, exist_ok=True)

            # Save model state dict
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'num_classes': self.num_classes,
                'img_size': self.img_size,
            }, path)

            logger.info(f"Model saved to: {path}")

            # Save label encoder
            encoder_path = path.parent / "label_encoder.pkl"
            joblib.dump(self.label_encoder, encoder_path)
            logger.info(f"Label encoder saved to: {encoder_path}")

    def load_model(self, path=None):
        """Load a trained model."""
        if path is None:
            path = config.CLASSIFICATION_MODEL_PATH
            
        # Load checkpoint
        checkpoint = torch.load(path, map_location=self.device)

        # Build model with saved parameters
        self.num_classes = checkpoint['num_classes']
        self.img_size = checkpoint['img_size']
        self.build_model()

        # Load state dict
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        logger.info(f"Model loaded from: {path}")

        # Load label encoder
        encoder_path = path.parent / "label_encoder.pkl"
        if encoder_path.exists():
            self.label_encoder = joblib.load(encoder_path)
            logger.info(f"Label encoder loaded from: {encoder_path}")


def main():
    """Main function for testing."""
    from step1_data_reader import FoodDataReader

    # Load data
    reader = FoodDataReader()
    data = reader.process()

    # Get number of unique food categories
    num_classes = data['train']['food_category_id'].nunique()
    logger.info(f"Number of food categories: {num_classes}")

    # Initialize classification model
    classifier = FoodClassification(num_classes=num_classes)

    # Train model
    history = classifier.train(
        data['train'].head(100),  # Use subset for testing
        data['val'].head(20),
        epochs=5,
        batch_size=16,
        use_augmentation=True,
        fine_tune=False
    )

    logger.info("\nTraining completed!")
    logger.info(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
    logger.info(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")


if __name__ == "__main__":
    main()