"""
CNN Implementation using Scratch-built EfficientNet-B0 (Non-Pre-trained).
Implements visualization capabilities for mathematical transparency.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import sys
from sklearn.preprocessing import LabelEncoder
from loguru import logger

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from source.config import config
from source.models.base_model import BaseFoodClassifier
from source.helpers.efficientnet_b0 import EfficientNetB0
from source.helpers.efficientnet_b0_visualization import EfficientNetMathTracer

class CNNScratchClassifier(BaseFoodClassifier):
    """
    CNN Classifier using the scratch-built EfficientNet-B0.
    Version: Non-Pre-trained (Random Initialization).
    """
    
    def __init__(self, num_classes: int, img_size: int = 224, 
                 device: str = None, use_visualization: bool = False, **kwargs):
        """
        Initialize the scratch CNN classifier.
        
        Args:
            num_classes: Number of food categories
            img_size: Input image size
            device: Device to use
            use_visualization: Whether to use the mathematical visualization version
        """
        super().__init__(num_classes, **kwargs)
        self.img_size = img_size
        self.use_visualization = use_visualization
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        self.label_encoder = LabelEncoder()
        
        # Build model and optionally attach tracer
        self.build_model()
        self.tracer = None
        if self.use_visualization:
            self.tracer = EfficientNetMathTracer(self.model)
            self.tracer.enable()
        
    def build_model(self):
        """Build the EfficientNet-B0 model from scratch with random initialization."""
        logger.info(f"Building EfficientNet-B0 from scratch (Random Initialization)")
        
        # Always use the standard model; visualization is now a wrapper/hook
        self.model = EfficientNetB0(num_classes=self.num_classes).to(self.device)
            
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Model built with {total_params:,} trainable parameters")

    def train_epoch(self, train_loader, criterion, optimizer, epoch_idx: int = 0):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Reset tracer for this epoch
        if self.tracer:
            self.tracer.reset_epoch(epoch_idx)
            # Ensure it's enabled for the first batch
            self.tracer.enable()
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass (hooks will capture math if enabled)
            outputs = self.model(images)
            
            # If visualization is active, print report after first batch and then disable to save perf
            if self.tracer and i == 0:
                self.tracer.print_report()
                self.tracer.disable()
            
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
        epochs = kwargs.get('epochs', 30)
        batch_size = kwargs.get('batch_size', 16)
        lr = kwargs.get('learning_rate', 0.001)
        
        # Prepare labels
        all_labels = np.unique(np.concatenate([
            train_df['food_category_id'].unique(),
            val_df['food_category_id'].unique()
        ]))
        self.label_encoder.fit(all_labels)
        
        train_labels = self.label_encoder.transform(train_df['food_category_id'].tolist())
        val_labels = self.label_encoder.transform(val_df['food_category_id'].tolist())
        
        # Datasets
        train_dataset = _ScratchDataset(train_df['image_before_path'].tolist(), train_labels, self.img_size)
        val_dataset = _ScratchDataset(val_df['image_before_path'].tolist(), val_labels, self.img_size)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        best_val_acc = 0.0
        history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            # Pass epoch index to train_epoch for visualization logging
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer, epoch_idx=epoch+1)
            val_loss, val_acc = self.validate(val_loader, criterion)
            
            history['accuracy'].append(train_acc)
            history['val_accuracy'].append(val_acc)
            history['loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
        
        return {'accuracy': best_val_acc, 'history': history}

    def predict(self, image_paths: List[str], visualize_math: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        dataset = _ScratchDataset(image_paths, img_size=self.img_size)
        loader = DataLoader(dataset, batch_size=1, shuffle=False) # Batch size 1 for visualization
        
        self.model.eval()
        all_probs = []
        
        # Manually trigger visualization if requested for prediction
        if visualize_math and self.tracer:
             self.tracer.enable()
             self.tracer.reset_epoch(0) # 0 for inference

        with torch.no_grad():
            for i, images in enumerate(loader):
                images = images.to(self.device)
                
                outputs = self.model(images)
                
                # Print viz for first sample if requested
                if visualize_math and self.tracer and i == 0:
                    print("Inference Visualization:")
                    self.tracer.print_report()
                    self.tracer.disable()

                probs = torch.softmax(outputs, dim=1)
                all_probs.append(probs.cpu().numpy())
                
        probs = np.concatenate(all_probs)
        preds_idx = np.argmax(probs, axis=1)
        preds = self.label_encoder.inverse_transform(preds_idx)
        
        return preds, probs

    def evaluate(self, test_df: pd.DataFrame) -> Dict[str, float]:
        """Evaluate the model on a test set."""
        # Preparation
        known_classes = set(self.label_encoder.classes_)
        unknown_mask = ~test_df['food_category_id'].isin(known_classes)
        if unknown_mask.any():
            test_df = test_df[~unknown_mask].copy()
            
        if len(test_df) == 0:
            return {'accuracy': 0.0}

        test_labels = self.label_encoder.transform(test_df['food_category_id'].tolist())
        test_dataset = _ScratchDataset(test_df['image_before_path'].tolist(), test_labels, self.img_size)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        
        criterion = nn.CrossEntropyLoss()
        _, test_acc = self.validate(test_loader, criterion)
        
        logger.info(f"Scratch CNN Test Accuracy: {test_acc:.4f}")
        return {'accuracy': test_acc}

    def optimize(self, train_df: pd.DataFrame, val_df: pd.DataFrame, n_trials: int = 5) -> Dict[str, Any]:
        """Simple optimization skeleton (Optuna can be added here)."""
        logger.info("Starting simple parameter search (Skeleton)...")
        # For now, just return default params or implement minimal Optuna logic
        self.best_params = {
            'learning_rate': 0.001,
            'batch_size': 16
        }
        return self.best_params

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'num_classes': self.num_classes,
            'img_size': self.img_size,
            'label_encoder': self.label_encoder,
            'use_visualization': self.use_visualization
        }, path)
        logger.info(f"Scratch CNN model saved to {path}")

    def load(self, path: Path):
        checkpoint = torch.load(path, map_location=self.device)
        self.num_classes = checkpoint['num_classes']
        self.img_size = checkpoint['img_size']
        self.label_encoder = checkpoint['label_encoder']
        self.use_visualization = checkpoint.get('use_visualization', False)
        
        self.build_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Re-attach tracer if needed
        if self.use_visualization:
            self.tracer = EfficientNetMathTracer(self.model)
            self.tracer.enable()
            
        logger.info(f"Scratch CNN model loaded from {path}")

class _ScratchDataset(Dataset):
    """Internal dataset helper for scratch models."""
    def __init__(self, image_paths: List[str], labels: np.ndarray = None, img_size: int = 224):
        self.image_paths = image_paths
        self.labels = labels
        self.img_size = img_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            img = cv2.imread(str(self.image_paths[idx]))
            if img is None:
                img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.img_size, self.img_size))
            img = img.astype(np.float32) / 255.0
            
            # Safe checking for dimensions just in case
            if len(img.shape) == 2:
                img = np.stack([img]*3, axis=-1)
                
            img = torch.from_numpy(img).permute(2, 0, 1).float()
            
            if self.labels is not None:
                return img, torch.tensor(self.labels[idx], dtype=torch.long)
            return img
        except Exception as e:
            # Fallback for broken images to keep training running
            logger.warning(f"Error loading {self.image_paths[idx]}: {e}")
            img = torch.zeros((3, self.img_size, self.img_size), dtype=torch.float32)
            if self.labels is not None:
                return img, torch.tensor(self.labels[idx], dtype=torch.long)
            return img

