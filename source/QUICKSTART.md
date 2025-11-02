# Quick Start Guide

Get up and running with the Food Weight Prediction pipeline in minutes!

## Prerequisites

- Python 3.8 or higher
- Conda environment named `requqira`
- Dataset placed in `source/data/` directory

## Installation

### 1. Activate Conda Environment

```bash
conda activate requqira
```

### 2. Install Dependencies

```bash
cd cnn_food_weight/source
pip install -r requirements.txt
```

## Running the Pipeline

### Option 1: Full Pipeline (Recommended for First Run)

Run the complete pipeline with all 5 steps:

```bash
python main.py --mode full
```

This will:
- Load and preprocess data (70/15/15 train/val/test split)
- Train U-Net segmentation model
- Setup data augmentation
- Train EfficientNet classification model
- Train CNN regression model
- Evaluate all models on test set
- Save results and models

**Expected time**: 3-6 hours (depending on hardware and dataset size)

### Option 2: Quick Test

Test the pipeline with minimal epochs (for validation):

```bash
python main.py --mode full --quick-test
```

**Expected time**: 10-20 minutes

### Option 3: Custom Configuration

```bash
# Custom epochs for each component
python main.py --mode full --seg-epochs 30 --cls-epochs 50 --reg-epochs 50

# Force retrain even if models exist
python main.py --mode full --no-skip-existing
```

## Checking Your Setup

### Test Data Loading

```bash
cd modules
python step1_data_reader.py
```

Expected output:
```
Reading Excel file...
Successfully read XXX records
Valid image pairs: XXX/XXX
Data split - Train: XXX, Val: XXX, Test: XXX
```

### Run Unit Tests

```bash
cd ..
python run_tests.py --suite unit
```

### Run All Tests

```bash
python run_tests.py --suite all
```

## Directory Structure Check

Ensure your directory structure looks like this:

```
cnn_food_weight/source/
├── data/
│   ├── data_original.xlsx          ✓ Required
│   └── leftover_dataset/
│       ├── data_before/            ✓ Required
│       │   ├── 001/
│       │   ├── 002/
│       │   └── ...
│       └── data_after/             ✓ Required
│           ├── 001/
│           ├── 002/
│           └── ...
├── modules/                        ✓ Auto-created
├── models/                         ✓ Auto-created
├── outputs/                        ✓ Auto-created
├── tests/                          ✓ Provided
├── main.py                         ✓ Provided
└── requirements.txt                ✓ Provided
```

## Troubleshooting

### Issue: "No module named 'tensorflow'"

```bash
pip install tensorflow
```

### Issue: "Excel file not found"

Check that `data_original.xlsx` is in `source/data/`

### Issue: "Image files not found"

Verify image folder structure:
```bash
ls source/data/leftover_dataset/data_before/001/
```

Should show `.JPG` files.

### Issue: Out of Memory

Edit `modules/config.py`:
```python
CLASSIFICATION_BATCH_SIZE = 16  # Reduce from 32
REGRESSION_BATCH_SIZE = 16      # Reduce from 32
```

## Next Steps

### 1. View Results

```bash
# Check outputs directory
ls outputs/

# View latest results
python -c "import json; print(json.dumps(json.load(open('outputs/pipeline_results_XXXXXX.json')), indent=2))"
```

### 2. Visualize Training

```bash
# Start TensorBoard
tensorboard --logdir=outputs/logs
```

Then open: http://localhost:6006

### 3. Make Predictions

```python
from main import FoodWeightPredictionPipeline

# Load trained pipeline
pipeline = FoodWeightPredictionPipeline()
pipeline.step1_load_data()

# Load models
from modules.step5_regression import WeightRegression
regressor = WeightRegression(use_dual_input=True)
regressor.load_model()

# Predict
before_img = "path/to/before.jpg"
after_img = "path/to/after.jpg"
predicted_weight = regressor.predict([before_img], [after_img])
print(f"Predicted weight: {predicted_weight[0]:.2f}g")
```

### 4. Explore Individual Modules

Each module can be run independently:

```bash
cd modules

# Test data reading
python step1_data_reader.py

# Test segmentation
python step2_segmentation.py

# Test augmentation
python step3_augmentation.py

# Test classification
python step4_classification.py

# Test regression
python step5_regression.py
```

## Command Reference

### Main Pipeline

```bash
# Full pipeline
python main.py --mode full

# Training only
python main.py --mode train

# Evaluation only
python main.py --mode evaluate

# Quick test
python main.py --quick-test
```

### Testing

```bash
# All tests
python run_tests.py --suite all

# Unit tests only
python run_tests.py --suite unit

# Pipeline tests only
python run_tests.py --suite pipeline

# E2E tests only
python run_tests.py --suite e2e
```

## Performance Expectations

### Hardware Requirements

**Minimum**:
- CPU: 4 cores
- RAM: 8GB
- Storage: 10GB free

**Recommended**:
- CPU: 8+ cores
- RAM: 16GB+
- GPU: NVIDIA GPU with 6GB+ VRAM
- Storage: 20GB+ free

### Expected Metrics

**Classification** (food category):
- Accuracy: 70-90%
- Top-5 Accuracy: 85-95%

**Regression** (weight prediction):
- MAE: 10-30g
- RMSE: 15-40g
- R²: 0.7-0.9

*Actual performance depends on dataset quality and training time*

## Support

For issues or questions:
1. Check the full [README.md](README.md)
2. Review error messages in `outputs/pipeline.log`
3. Run diagnostic tests: `python run_tests.py`

## Summary of Key Files

| File | Purpose |
|------|---------|
| `main.py` | Run full pipeline |
| `modules/config.py` | Configuration settings |
| `modules/step1_data_reader.py` | Data loading |
| `modules/step2_segmentation.py` | U-Net segmentation |
| `modules/step3_augmentation.py` | Data augmentation |
| `modules/step4_classification.py` | EfficientNet classification |
| `modules/step5_regression.py` | Weight regression |
| `utils.py` | Visualization utilities |
| `run_tests.py` | Test runner |

---

**Ready to start?** Run: `python main.py --mode full --quick-test`
