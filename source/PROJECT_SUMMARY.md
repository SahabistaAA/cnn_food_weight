# Food Weight Prediction - Project Summary

## Overview

A complete end-to-end deep learning pipeline for predicting leftover food weight from images using Indonesian food dataset with 524 image pairs.

## Project Statistics

- **Total Python Files**: 16 files
- **Total Lines of Code**: ~3,500+ lines
- **Test Coverage**: Unit, Integration, and E2E tests
- **Models**: 3 deep learning models (U-Net, EfficientNet, CNN Regression)
- **Dataset Split**: 70% train / 15% validation / 15% test

## File Structure

```
cnn_food_weight/source/
│
├── main.py                          # Main pipeline orchestrator (350 lines)
├── requirements.txt                 # Python dependencies
├── README.md                        # Full documentation
├── QUICKSTART.md                    # Quick start guide
├── PROJECT_SUMMARY.md              # This file
├── utils.py                        # Visualization utilities (300 lines)
├── run_tests.py                    # Test runner (100 lines)
│
├── modules/                        # Core pipeline modules
│   ├── config.py                   # Configuration (80 lines)
│   ├── step1_data_reader.py       # Data loading (250 lines)
│   ├── step2_segmentation.py      # U-Net segmentation (350 lines)
│   ├── step3_augmentation.py      # Data augmentation (300 lines)
│   ├── step4_classification.py    # EfficientNet classification (400 lines)
│   └── step5_regression.py        # CNN regression (450 lines)
│
├── tests/                          # Comprehensive test suite
│   ├── __init__.py
│   ├── test_unit.py               # Unit tests (250 lines)
│   ├── test_pipeline.py           # Integration tests (200 lines)
│   └── test_e2e.py                # End-to-end tests (250 lines)
│
├── data/                           # Dataset directory
│   ├── data_original.xlsx         # Metadata (provided by user)
│   └── leftover_dataset/          # Image folders (provided by user)
│       ├── data_before/           # Pre-consumption images
│       └── data_after/            # Post-consumption images
│
├── models/                         # Saved models (generated)
│   ├── unet_segmentation.h5
│   ├── efficientnet_classification.h5
│   ├── cnn_regression.h5
│   └── label_encoder.pkl
│
└── outputs/                        # Results and logs (generated)
    ├── pipeline_results_*.json
    ├── data_split.csv
    ├── logs/
    └── metrics/
```

## Module Descriptions

### 1. Configuration (`config.py`)
- Centralized configuration management
- Path definitions
- Hyperparameters
- Model settings

### 2. Data Reader (`step1_data_reader.py`)
**Key Features**:
- Reads Excel metadata
- Validates image paths
- Extracts food category IDs from filenames
- Computes weight statistics
- Creates stratified train/val/test split (70/15/15)
- Saves split information for reproducibility

**Key Functions**:
- `read_excel()`: Load and validate Excel file
- `validate_image_paths()`: Check image existence
- `create_train_val_test_split()`: Stratified splitting
- `compute_weight_difference()`: Calculate consumption metrics

### 3. Segmentation (`step2_segmentation.py`)
**Architecture**: U-Net
- Encoder: 4 levels (64→128→256→512 filters)
- Bottleneck: 1024 filters
- Decoder: 4 levels with skip connections
- Output: Binary segmentation mask

**Key Features**:
- Creates pseudo ground truth masks using Otsu/GrabCut
- Batch normalization and dropout for regularization
- Dice coefficient metric
- Model checkpointing and early stopping

### 4. Augmentation (`step3_augmentation.py`)
**Techniques**:
- **Geometric**: Rotation, flip, shift, scale, shear
- **Color**: Brightness, contrast, hue, saturation
- **Quality**: Blur, noise, compression
- **Lighting**: Shadows, fog effects

**Implementations**:
- Keras ImageDataGenerator
- Albumentations (advanced)
- TensorFlow Dataset API

### 5. Classification (`step4_classification.py`)
**Architecture**: EfficientNet-B0
- Transfer learning from ImageNet
- Custom classification head
- Two-phase training:
  1. Frozen base (fast convergence)
  2. Fine-tuning (performance boost)

**Key Features**:
- Label encoding for food categories
- Top-1 and Top-5 accuracy metrics
- Learning rate scheduling
- TensorBoard logging

### 6. Regression (`step5_regression.py`)
**Architecture**: Dual-Input CNN
- Two EfficientNet-B0 feature extractors
- Concatenated features from before/after images
- Dense layers for regression
- Single output neuron (weight)

**Key Features**:
- Dual-input (before + after images)
- Weight normalization for stable training
- Multiple metrics: MAE, RMSE, R², MAPE
- Synchronized augmentation for image pairs

## Pipeline Flow

```
┌─────────────────────────────────────────────────────────────┐
│  STEP 1: Data Reading                                       │
│  - Load Excel (524 records)                                 │
│  - Validate images                                          │
│  - Split: Train (70%), Val (15%), Test (15%)              │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 2: Segmentation (U-Net)                               │
│  - Train on before/after images                            │
│  - Generate food segmentation masks                        │
│  - Save best model                                         │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 3: Augmentation Setup                                 │
│  - Configure augmentation pipeline                         │
│  - Apply to training data                                  │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 4: Classification (EfficientNet)                      │
│  - Classify food categories                                │
│  - Transfer learning + fine-tuning                         │
│  - Metrics: Accuracy, Top-5                                │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 5: Weight Regression (CNN)                            │
│  - Dual input: before + after                              │
│  - Predict leftover weight                                 │
│  - Metrics: MAE, RMSE, R²                                  │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│  Final Evaluation                                           │
│  - Test all models on test set                             │
│  - Generate comprehensive report                           │
│  - Save results to JSON                                    │
└─────────────────────────────────────────────────────────────┘
```

## Testing Strategy

### 1. Unit Tests (`test_unit.py`)
- Individual function testing
- Module isolation
- Edge case validation
- ~20 test cases

### 2. Integration Tests (`test_pipeline.py`)
- Module interaction testing
- Data flow validation
- Configuration testing
- Split consistency checks

### 3. End-to-End Tests (`test_e2e.py`)
- Complete pipeline execution
- Model training and evaluation
- Prediction functionality
- Result validation

## Key Technologies

| Technology | Purpose |
|------------|---------|
| TensorFlow/Keras | Deep learning framework |
| NumPy/Pandas | Data manipulation |
| OpenCV | Image processing |
| Albumentations | Advanced augmentation |
| scikit-learn | ML utilities & metrics |
| pytest | Testing framework |
| TensorBoard | Training visualization |

## Model Architecture Details

### U-Net Segmentation
```
Parameters: ~31M
Input: 256×256×3
Output: 256×256×1
Loss: Binary crossentropy
Optimizer: Adam (lr=0.001)
```

### EfficientNet Classification
```
Parameters: ~5.3M
Input: 224×224×3
Output: num_classes (softmax)
Loss: Sparse categorical crossentropy
Optimizer: Adam (lr=0.001 → 1e-5)
Training: 2-phase (frozen → fine-tuned)
```

### CNN Regression
```
Parameters: ~8M (dual EfficientNet)
Input: 2× 224×224×3
Output: 1 (linear)
Loss: MSE
Optimizer: Adam (lr=0.001)
Metrics: MAE, RMSE, R², MAPE
```

## Usage Commands

### Quick Start
```bash
# Full pipeline
python main.py --mode full

# Quick test (2 epochs)
python main.py --quick-test

# Custom epochs
python main.py --seg-epochs 30 --cls-epochs 50 --reg-epochs 50
```

### Testing
```bash
# All tests
python run_tests.py

# Specific suite
python run_tests.py --suite unit
python run_tests.py --suite pipeline
python run_tests.py --suite e2e
```

### Individual Modules
```bash
cd modules
python step1_data_reader.py
python step2_segmentation.py
python step3_augmentation.py
python step4_classification.py
python step5_regression.py
```

## Expected Performance

### Classification Metrics
- **Accuracy**: 70-90%
- **Top-5 Accuracy**: 85-95%
- **Training time**: 1-2 hours

### Regression Metrics
- **MAE**: 10-30 grams
- **RMSE**: 15-40 grams
- **R²**: 0.7-0.9
- **Training time**: 1-2 hours

*Performance varies based on dataset quality and hardware*

## Output Files

### Models
- `models/unet_segmentation.h5` (~120 MB)
- `models/efficientnet_classification.h5` (~20 MB)
- `models/cnn_regression.h5` (~32 MB)
- `models/label_encoder.pkl` (~1 KB)
- `models/weight_stats.json` (~1 KB)

### Results
- `outputs/pipeline_results_*.json` - Complete metrics
- `outputs/data_split.csv` - Train/val/test split
- `outputs/pipeline.log` - Execution logs
- `outputs/logs/` - TensorBoard logs

## Development Features

✅ Modular architecture (5 independent modules)
✅ Comprehensive testing (unit + integration + E2E)
✅ Configuration management
✅ Logging and monitoring
✅ Model persistence
✅ Result tracking
✅ Data validation
✅ Error handling
✅ Documentation (README + Quick Start)
✅ Visualization utilities
✅ Command-line interface

## Future Enhancements

- [ ] Web interface for inference
- [ ] Mobile app deployment
- [ ] Real-time video processing
- [ ] Model ensemble methods
- [ ] Attention mechanisms
- [ ] Multi-task learning
- [ ] Active learning loop
- [ ] Model compression

## Citation

```bibtex
@software{food_weight_prediction_2025,
  title={Food Weight Prediction System},
  author={Astronomi Research},
  year={2025},
  description={Deep learning pipeline for Indonesian food weight prediction}
}
```

## License

[To be specified by user]

---

**Status**: ✅ Complete and Ready for Use
**Last Updated**: 2025-11-02
**Version**: 1.0.0
