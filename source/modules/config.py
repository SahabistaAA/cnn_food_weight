"""
Configuration file for the food weight prediction pipeline (PyTorch version).
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

# Device configuration (automatically select GPU if available)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 4  # For DataLoader (set to 0 on Windows if you have issues)
PIN_MEMORY = True if torch.cuda.is_available() else False

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

# Augmentation parameters (for Albumentations)
AUGMENTATION_PARAMS = {
    'rotation_limit': 20,
    'shift_limit': 0.2,
    'scale_limit': 0.15,
    'brightness_limit': 0.2,
    'contrast_limit': 0.2,
    'p': 0.8  # Probability of applying augmentation
}

# Classification parameters (EfficientNet via timm)
EFFICIENTNET_MODEL = 'efficientnet_b0'  # timm model name
NUM_CLASSES = None  # Will be set dynamically based on unique food categories
CLASSIFICATION_BATCH_SIZE = 32
CLASSIFICATION_EPOCHS = 50
CLASSIFICATION_LEARNING_RATE = 0.001
CLASSIFICATION_WEIGHT_DECAY = 1e-4

# Regression parameters
REGRESSION_BATCH_SIZE = 32
REGRESSION_EPOCHS = 50
REGRESSION_LEARNING_RATE = 0.001
REGRESSION_WEIGHT_DECAY = 1e-4

# Training parameters
EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 5
REDUCE_LR_FACTOR = 0.5
RANDOM_SEED = 42

# Mixed precision training (for faster training on modern GPUs)
USE_AMP = True if torch.cuda.is_available() else False

# Model save paths (PyTorch uses .pth or .pt extensions)
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

# Set random seeds for reproducibility
def set_seed(seed=RANDOM_SEED):
    """Set random seeds for reproducibility."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Make CUDA operations deterministic (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Print configuration info
def print_config_info():
    """Print configuration information."""
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"CUDA Available: Yes")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
    else:
        print(f"CUDA Available: No (using CPU)")
    print(f"Mixed Precision (AMP): {USE_AMP}")
