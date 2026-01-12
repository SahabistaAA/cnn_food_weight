"""
CNN (EfficientNet) implementation for food classification.
Adapted from source/modules/step4_classification.py
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import timm
import cv2
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from pathlib import Path
import optuna
from sklearn.preprocessing import LabelEncoder
from loguru import logger
import sys
import copy

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from source.config import config
from source.models.base_model import BaseFoodClassifier

class EfficientNetClassifier(nn.Module):
    """EfficientNet-B0 model for food classification."""

    def __init__(self, num_classes: int, pretrained: bool = True):
        super(EfficientNetClassifier, self).__init__()
        # Load EfficientNet-B0 from timm
        self.backbone = timm.create_model(
            config.EFFICIENTNET_MODEL,
            pretrained=pretrained,
            num_classes=0,
            global_pool=''
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        num_features = self.backbone.num_features
        
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
        features = self.backbone(x)
        features = self.global_pool(features)
        features = features.flatten(1)
        logits = self.classifier(features)
        return logits

class ClassificationDataset(Dataset):
    """PyTorch Dataset for food classification."""

    def __init__(self, image_paths: List[str], labels: np.ndarray = None,
                 img_size: int = 224, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.img_size = img_size
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = cv2.imread(str(img_path))

        if img is None:
            # logger.warning(f"Failed to load image: {img_path}")
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = cv2.resize(img, (self.img_size, self.img_size))

        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']

        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1).float()

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return img, label
        else:
            return img

class CNNClassifier(BaseFoodClassifier):
    """EfficientNet-based Classifier with Optuna optimization."""
    
    def __init__(self, num_classes: int, img_size: int = config.IMG_HEIGHT, 
                 device: str = None, **kwargs):
        super().__init__(num_classes, **kwargs)
        self.img_size = img_size
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        self.label_encoder = LabelEncoder()
        
        # Build model
        self.build_model()
        
    def build_model(self, pretrained: bool = True):
        self.model = EfficientNetClassifier(
            num_classes=self.num_classes,
            pretrained=pretrained
        ).to(self.device)
        
    def create_dataloaders(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                          batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
        # Fit encoder if not fitted (or assume pre-fitted/passed)
        # Here we fit on train data every time for simplicity in this isolated scope,
        # but in production we should save the encoder. 
        # Ideally, main.py ensures consistancy. 
        # But BaseFoodClassifier expects internal management.
        

        # Fit encoder on all available labels
        all_train_labels = train_df['food_category_id'].unique()
        all_val_labels = val_df['food_category_id'].unique()
        all_labels = np.unique(np.concatenate([all_train_labels, all_val_labels]))
        self.label_encoder.fit(all_labels)
        
        train_labels = self.label_encoder.transform(train_df['food_category_id'].tolist())
        val_labels = self.label_encoder.transform(val_df['food_category_id'].tolist())
        
        # Helper to get transform (simplified for now, reusing what main.py usually does optionally)
        # We'll skip complex augmentation import for speed unless requested, 
        # but to match existing code performance we should probably use it.
        # I'll stick to basic resizing in Dataset for now to keep this file self-contained
        # unless I import step3_augmentation.
        
        train_dataset = ClassificationDataset(
            train_df['image_before_path'].tolist(),
            train_labels,
            img_size=self.img_size
        )
        
        val_dataset = ClassificationDataset(
            val_df['image_before_path'].tolist(),
            val_labels,
            img_size=self.img_size
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY
        )
        
        return train_loader, val_loader

    def train_epoch(self, train_loader, criterion, optimizer):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        return running_loss / len(train_loader.dataset), correct / total

    def validate(self, val_loader, criterion):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
        return running_loss / len(val_loader.dataset), correct / total

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        epochs = kwargs.get('epochs', config.CLASSIFICATION_EPOCHS)
        batch_size = kwargs.get('batch_size', config.CLASSIFICATION_BATCH_SIZE)
        lr = kwargs.get('learning_rate', config.CLASSIFICATION_LEARNING_RATE)
        
        train_loader, val_loader = self.create_dataloaders(train_df, val_df, batch_size)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        best_val_acc = 0.0
        history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}
        
        logger.info(f"Training CNN for {epochs} epochs...")
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)
            val_loss, val_acc = self.validate(val_loader, criterion)
            
            history['accuracy'].append(train_acc)
            history['val_accuracy'].append(val_acc)
            history['loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                # Save best state internally if needed, or rely on caller to save
                
        return {'accuracy': best_val_acc, 'history': history}

    def predict(self, image_paths: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        dataset = ClassificationDataset(image_paths, img_size=self.img_size)
        loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=config.NUM_WORKERS)
        
        self.model.eval()
        all_probs = []
        
        with torch.no_grad():
            for images in loader:
                images = images.to(self.device)
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                all_probs.append(probs.cpu().numpy())
                
        probs = np.concatenate(all_probs)
        preds_idx = np.argmax(probs, axis=1)
        preds = self.label_encoder.inverse_transform(preds_idx)
        
        return preds, probs

    def evaluate(self, test_df: pd.DataFrame) -> Dict[str, float]:
        # Prepare data (reuse create_dataloaders logic partially)
        # Filter out unseen classes
        known_classes = set(self.label_encoder.classes_)
        unknown_mask = ~test_df['food_category_id'].isin(known_classes)
        if unknown_mask.any():
            unknown_count = unknown_mask.sum()
            logger.warning(f"Dropping {unknown_count} samples with unknown classes: {test_df[unknown_mask]['food_category_id'].unique()}")
            test_df = test_df[~unknown_mask].copy()
            
        if len(test_df) == 0:
            logger.warning("No test samples remaining after filtering unknown classes.")
            return {'accuracy': 0.0}

        test_labels = self.label_encoder.transform(test_df['food_category_id'].tolist())
        test_dataset = ClassificationDataset(
            test_df['image_before_path'].tolist(),
            test_labels,
            img_size=self.img_size
        )
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        criterion = nn.CrossEntropyLoss()
        test_loss, test_acc = self.validate(test_loader, criterion)
        
        logger.info(f"CNN Test Accuracy: {test_acc:.4f}")
        return {'accuracy': test_acc}

    def optimize(self, train_df: pd.DataFrame, val_df: pd.DataFrame, n_trials: int = 5) -> Dict[str, Any]:
        """
        Optimize CNN using Optuna. 
        Note: CNN training is slow, so we use fewer trials and epochs per trial by default.
        """
        logger.info("Starting CNN optimization...")
        
        def objective(trial):
            # Hyperparameters to tune
            lr = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
            batch_size = trial.suggest_categorical('batch_size', [16, 32])
            optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD'])
            
            # Re-init model for each trial
            self.build_model()
            
            criterion = nn.CrossEntropyLoss()
            if optimizer_name == 'Adam':
                optimizer = optim.Adam(self.model.parameters(), lr=lr)
            else:
                optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
                
            train_loader, val_loader = self.create_dataloaders(train_df, val_df, batch_size=batch_size)
            
            # Short training for optimization (e.g., 3-5 epochs)
            n_epochs = 3
            accuracy = 0.0
            
            for epoch in range(n_epochs):
                self.train_epoch(train_loader, criterion, optimizer)
                _, accuracy = self.validate(val_loader, criterion)
                
                trial.report(accuracy, epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
                    
            return accuracy

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        self.best_params = study.best_params
        logger.info(f"Best CNN params: {self.best_params}")
        
        # We don't automatically retrain fully here because it might take too long.
        # The caller should call train() with these params if desired.
        
        return self.best_params

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'num_classes': self.num_classes,
            'img_size': self.img_size,
            'label_encoder': self.label_encoder
        }, path)
        logger.info(f"CNN model saved to {path}")

    def load(self, path: Path):
        checkpoint = torch.load(path, map_location=self.device)
        self.num_classes = checkpoint['num_classes']
        self.img_size = checkpoint['img_size']
        self.label_encoder = checkpoint['label_encoder']
        
        self.build_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        logger.info(f"CNN model loaded from {path}")
