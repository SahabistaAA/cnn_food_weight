"""
Abstract base class for food classification models.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
import optuna
from pathlib import Path

class BaseFoodClassifier(ABC):
    """Abstract base class for food classification models."""

    def __init__(self, num_classes: int, **kwargs):
        """
        Initialize the model.
        
        Args:
            num_classes: Number of food categories
            **kwargs: Additional model-specific arguments
        """
        self.num_classes = num_classes
        self.model = None
        self.best_params = {}
        
    @abstractmethod
    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            train_df: Training dataframe
            val_df: Validation dataframe
            **kwargs: Training arguments
            
        Returns:
            Dictionary containing training history/metrics
        """
        pass
    
    @abstractmethod
    def predict(self, image_paths: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict categories for images.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            Tuple of (predicted_categories, probabilities)
        """
        pass
    
    @abstractmethod
    def evaluate(self, test_df: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate model on test set.
        
        Args:
            test_df: Test dataframe
            
        Returns:
            Dictionary of evaluation metrics
        """
        pass
    
    @abstractmethod
    def optimize(self, train_df: pd.DataFrame, val_df: pd.DataFrame, n_trials: int = 20) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Optuna.
        
        Args:
            train_df: Training dataframe
            val_df: Validation dataframe
            n_trials: Number of optimization trials
            
        Returns:
            Dictionary of best parameters
        """
        pass
    
    @abstractmethod
    def save(self, path: Path):
        """Save the model to disk."""
        pass
    
    @abstractmethod
    def load(self, path: Path):
        """Load the model from disk."""
        pass
