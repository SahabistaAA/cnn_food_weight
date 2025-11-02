"""
Configuration file for the food weight prediction pipeline (PyTorch).
"""
import os
from pathlib import Path
import torch

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
EXCEL_PATH = DATA_DIR / "data_original.xlsx"
IMAGES_BEFORE_DIR = DATA_DIR / "leftover_dataset" / "data_before"
IMAGES_AFTER_DIR = DATA_DIR / "leftover_dataset" / "data_after"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Ensure directories exist
MODELS_DIR.mkdir(exist_ok=True, parents=True)
OUTPUTS_DIR.mkdir(exist_ok=True, parents=True)

# Data split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Image parameters
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3

# Segmentation parameters (U-Net)
SEGMENTATION_IMG_SIZE = 256
UNET_FILTERS = [64, 128, 256, 512, 1024]
UNET_DROPOUT = 0.3

# Augmentation parameters
AUGMENTATION_PARAMS = {
    'rotation_range': 20,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'shear_range': 0.15,
    'zoom_range': 0.15,
    'horizontal_flip': True,
    'fill_mode': 'nearest',
    'brightness_range': [0.8, 1.2]
}

# Classification parameters (EfficientNet)
EFFICIENTNET_MODEL = 'efficientnet-b0'
NUM_CLASSES = None  # Will be set dynamically
CLASSIFICATION_BATCH_SIZE = 32
CLASSIFICATION_EPOCHS = 50
CLASSIFICATION_LEARNING_RATE = 0.001

# Regression parameters
REGRESSION_BATCH_SIZE = 32
REGRESSION_EPOCHS = 50
REGRESSION_LEARNING_RATE = 0.001

# Training parameters
EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 5
RANDOM_SEED = 42

# PyTorch device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# PyTorch DataLoader settings
NUM_WORKERS = 0  # Set to 0 for Windows, increase for Linux/Mac
PIN_MEMORY = True if torch.cuda.is_available() else False
USE_AMP = True  # Automatic Mixed Precision (for faster training on GPU)

# Model save paths (PyTorch uses .pth extension)
SEGMENTATION_MODEL_PATH = MODELS_DIR / "unet_segmentation.pth"
CLASSIFICATION_MODEL_PATH = MODELS_DIR / "efficientnet_classification.pth"
REGRESSION_MODEL_PATH = MODELS_DIR / "cnn_regression.pth"

# Outputs
TRAIN_VAL_TEST_SPLIT_PATH = OUTPUTS_DIR / "data_split.csv"
SEGMENTATION_RESULTS_DIR = OUTPUTS_DIR / "segmentation_results"
METRICS_DIR = OUTPUTS_DIR / "metrics"

# Create output directories
SEGMENTATION_RESULTS_DIR.mkdir(exist_ok=True, parents=True)
METRICS_DIR.mkdir(exist_ok=True, parents=True)
