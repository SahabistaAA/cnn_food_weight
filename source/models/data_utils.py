"""
Data loading utilities for classic Machine Learning models (SVM, RF, etc.).
"""
import cv2
import numpy as np
import pandas as pd
from typing import Tuple, List
from loguru import logger
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from source.config import config

def load_and_preprocess_images(image_paths: List[str], img_size: int = 64) -> np.ndarray:
    """
    Load images, resize, and flatten them for classic ML models.
    Using smaller image size (64x64) to keep feature vector size manageable (12,288 features)
    instead of 224x224 (150,528 features).
    
    Args:
        image_paths: List of file paths
        img_size: Target size for resizing
        
    Returns:
        Numpy array of shape (n_samples, n_features)
    """
    features = []
    
    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            logger.warning(f"Failed to load image: {path}")
            # Return zero vector if failed
            img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
        # Resize
        img = cv2.resize(img, (img_size, img_size))
        
        # Flatten and normalize
        img_flat = img.flatten().astype(np.float32) / 255.0
        features.append(img_flat)
        
    return np.array(features)

def prepare_sklearn_data(df: pd.DataFrame, img_size: int = 64) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare data for sklearn models.
    
    Args:
        df: DataFrame containing image paths and labels
        img_size: Image size
        
    Returns:
        X (features), y (labels)
    """
    X = load_and_preprocess_images(df['image_before_path'].tolist(), img_size)
    y = df['food_category_id'].values
    return X, y
