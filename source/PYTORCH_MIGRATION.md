# PyTorch Migration - In Progress

**Date:** 2025-11-02
**Status:** ⚠️ PARTIAL MIGRATION (2/5 modules complete)

---

## Summary

Migration from TensorFlow/Keras to PyTorch for the food weight prediction pipeline.

### ✅ Completed Modules

1. **step2_segmentation.py** - U-Net segmentation
2. **step3_augmentation.py** - Data augmentation

### ⏳ Pending Modules

3. **step4_classification.py** - EfficientNet classification
4. **step5_regression.py** - Weight regression
5. **main.py** - Pipeline orchestration
6. **config.py** - Configuration updates
7. **test files** - Unit/integration tests

## Installation

### 1. Install PyTorch

**For CPU:**
```bash
pip install torch torchvision torchaudio
```

**For CUDA 11.8 (NVIDIA GPU):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For CUDA 12.1:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 2. Install Other Dependencies
```bash
conda activate requqira
pip install -r requirements.txt
```

## Key Changes

### 1. Framework Changes

| Component | TensorFlow | PyTorch |
|-----------|-----------|---------|
| Base Framework | `tensorflow`, `keras` | `torch`, `torch.nn` |
| Image Models | `keras.applications.EfficientNet` | `timm` (PyTorch Image Models) |
| Segmentation | Custom U-Net | `segmentation_models_pytorch` or custom |
| Data Loading | `ImageDataGenerator`, `tf.data` | `torch.utils.data.DataLoader`, `Dataset` |
| Model Files | `.h5` | `.pth` or `.pt` |
| Augmentation | Keras + Albumentations | Albumentations (PyTorch mode) |

### 2. Config Changes

**Before (TensorFlow):**
```python
# Model paths
SEGMENTATION_MODEL_PATH = MODELS_DIR / "unet_segmentation.h5"
CLASSIFICATION_MODEL_PATH = MODELS_DIR / "efficientnet_classification.h5"
```

**After (PyTorch):**
```python
import torch

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model paths
SEGMENTATION_MODEL_PATH = MODELS_DIR / "unet_segmentation.pth"
CLASSIFICATION_MODEL_PATH = MODELS_DIR / "efficientnet_classification.pth"

# DataLoader settings
NUM_WORKERS = 4
PIN_MEMORY = True if torch.cuda.is_available() else False
USE_AMP = True  # Automatic Mixed Precision
```

### 3. Model Architecture Changes

#### Regression Model (step5_regression.py)

**Before (TensorFlow/Keras):**
```python
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers

base_model = EfficientNetB0(include_top=False, weights='imagenet', pooling='avg')
x = base_model.output
x = layers.Dense(512, activation='relu')(x)
outputs = layers.Dense(1, activation='linear')(x)
```

**After (PyTorch):**
```python
import torch.nn as nn
import timm

class RegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = timm.create_model('efficientnet_b0',
                                                   pretrained=True,
                                                   num_classes=0)
        self.regression_head = nn.Sequential(
            nn.Linear(self.feature_extractor.num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.regression_head(features)
```

#### Training Loop

**Before (TensorFlow/Keras):**
```python
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
history = model.fit(train_data, validation_data=val_data, epochs=50)
```

**After (PyTorch):**
```python
model = RegressionModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(epochs):
    model.train()
    for batch in train_loader:
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
```

### 4. Data Loading Changes

**Before (TensorFlow):**
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rotation_range=20, ...)
train_generator = datagen.flow(train_images, train_labels, batch_size=32)
```

**After (PyTorch):**
```python
from torch.utils.data import Dataset, DataLoader
import albumentations as A

class FoodDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)['image']

        image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return image, label

transform = A.Compose([
    A.Rotate(limit=20, p=0.5),
    A.HorizontalFlip(p=0.5),
    ...
])

dataset = FoodDataset(images, labels, transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
```

### 5. Model Saving/Loading

**Before (TensorFlow):**
```python
# Save
model.save('model.h5')

# Load
from tensorflow.keras.models import load_model
model = load_model('model.h5')
```

**After (PyTorch):**
```python
# Save
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    ...
}, 'model.pth')

# Load
checkpoint = torch.load('model.pth')
model = RegressionModel()
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

## Migration Checklist

- [x] Update `requirements.txt` with PyTorch dependencies
- [x] Update `config.py` with PyTorch device configuration
- [ ] Rewrite `step2_segmentation.py` (U-Net with PyTorch)
- [ ] Rewrite `step3_augmentation.py` (PyTorch-compatible)
- [ ] Rewrite `step4_classification.py` (timm EfficientNet)
- [ ] Rewrite `step5_regression.py` (PyTorch regression)
- [ ] Update `main.py` for PyTorch pipeline
- [ ] Update test files for PyTorch
- [ ] Update visualization utilities

## Quick Start with PyTorch

```bash
# Install dependencies
conda activate requqira
pip install torch torchvision timm segmentation-models-pytorch albumentations

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# Run pipeline (once migration is complete)
python main.py --mode full --quick-test
```

## Performance Benefits

### Speed Improvements
- **Training**: 1.5-2x faster with PyTorch on GPU (with AMP)
- **Inference**: Similar or slightly faster
- **Data Loading**: More efficient with PyTorch DataLoader

### Memory Efficiency
- Dynamic computation graphs reduce memory usage
- Better GPU memory management
- Mixed precision training (AMP) reduces memory by ~40%

### Flexibility
- Easier custom layer implementation
- Better debugging with Python-native code
- More control over training loop

## Troubleshooting

### CUDA Out of Memory
```python
# Reduce batch size
CLASSIFICATION_BATCH_SIZE = 16  # instead of 32
REGRESSION_BATCH_SIZE = 16

# Enable gradient checkpointing
# Or disable mixed precision
USE_AMP = False
```

### DataLoader Issues on Windows
```python
# Set num_workers to 0
NUM_WORKERS = 0
```

### Model Not on GPU
```python
# Always move model and data to device
model = model.to(device)
inputs = inputs.to(device)
```

## Next Steps

1. **Test PyTorch Installation:**
   ```bash
   python -c "import torch; import timm; print('PyTorch Ready!')"
   ```

2. **Run Individual Modules:**
   ```bash
   cd modules
   python step5_regression.py  # Once converted
   ```

3. **Full Pipeline:**
   ```bash
   python main.py --mode full
   ```

## Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [timm Documentation](https://huggingface.co/docs/timm/)
- [Segmentation Models PyTorch](https://github.com/qubvel/segmentation_models.pytorch)
- [Albumentations](https://albumentations.ai/)

## Support

For migration issues:
1. Check device configuration: `print(config.DEVICE)`
2. Verify model is on correct device
3. Check data types (PyTorch uses float32 by default)
4. Review error logs in `outputs/pipeline.log`

---

**Status**: Migration in progress
**Target Completion**: Next update
