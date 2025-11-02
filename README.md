# Food Weight Prediction System

A comprehensive deep learning pipeline for predicting the weight of leftover Indonesian food using computer vision and CNN-based models.

## Overview

This project implements a 5-step pipeline to predict food weight from images:

1. **Data Reading** - Loads and preprocesses food image-weight pairs
2. **Segmentation** - U-Net model for separating food from background
3. **Augmentation** - Advanced data augmentation techniques
4. **Classification** - EfficientNet for food category identification
5. **Regression** - CNN-based weight prediction

## Dataset

The dataset consists of **524 pairs** of Indonesian food images (before and after consumption) with the following properties:

- **ID**: Food category identifier
- **Name**: Food name
- **Image Before Eaten**: Filename of pre-consumption image
- **Weight Before Eaten (g)**: Measured weight before eating
- **Image After Eaten**: Filename of post-consumption image
- **Weight After Eaten (g)**: Measured weight after eating
- **Visual Estimation**: 7-level scale from observer assessment

### Food Category Naming Convention

Images follow the pattern: `XXX_YYY_ZZZ_aft/bef.JPG`
- First 3 digits (`XXX`): Food category ID (e.g., `001` = "Nasi")

## Project Structure

```
cnn_food_weight/source/
├── data/
│   ├── data_original.xlsx           # Dataset metadata
│   └── leftover_dataset/
│       ├── data_before/             # Pre-consumption images
│       │   ├── 001/
│       │   ├── 002/
│       │   └── ...
│       └── data_after/              # Post-consumption images
│           ├── 001/
│           ├── 002/
│           └── ...
├── modules/
│   ├── config.py                    # Configuration settings
│   ├── step1_data_reader.py        # Data loading and preprocessing
│   ├── step2_segmentation.py       # U-Net segmentation
│   ├── step3_augmentation.py       # Data augmentation
│   ├── step4_classification.py     # EfficientNet classification
│   └── step5_regression.py         # CNN regression
├── tests/
│   ├── test_unit.py                # Unit tests
│   ├── test_pipeline.py            # Pipeline integration tests
│   └── test_e2e.py                 # End-to-end tests
├── models/                          # Saved models (generated)
├── outputs/                         # Results and logs (generated)
├── main.py                          # Main pipeline orchestrator
└── requirements.txt                 # Python dependencies
```

## Installation

### 1. Create Conda Environment

```bash
# Activate the required conda environment
conda activate requqira
```

### 2. Install Dependencies

```bash
cd cnn_food_weight/source
pip install -r requirements.txt
```

## Usage

### Quick Start - Full Pipeline

Run the complete pipeline with default settings:

```bash
python main.py --mode full
```

### Quick Test Mode

Test the pipeline with reduced epochs (for validation):

```bash
python main.py --mode full --quick-test
```

### Training Only

Train models without evaluation:

```bash
python main.py --mode train
```

### Evaluation Only

Evaluate pre-trained models:

```bash
python main.py --mode evaluate
```

### Custom Configuration

```bash
# Custom epochs for each model
python main.py --mode full \
    --seg-epochs 30 \
    --cls-epochs 50 \
    --reg-epochs 50

# Force retraining even if models exist
python main.py --mode full --no-skip-existing
```

## Pipeline Steps Details

### Step 1: Data Reading

- Reads Excel metadata
- Validates image paths
- Computes weight differences
- Creates train/val/test split (70/15/15)
- Stratified by food category

**Run individually:**
```bash
cd modules
python step1_data_reader.py
```

### Step 2: Segmentation (U-Net)

- Architecture: U-Net with 5 encoder-decoder levels
- Purpose: Segment food from background
- Uses pseudo ground truth masks (Otsu thresholding)
- Input: 256×256 RGB images
- Output: Binary masks

**Run individually:**
```bash
cd modules
python step2_segmentation.py
```

### Step 3: Augmentation

- **Geometric**: Rotation, flipping, translation, scaling
- **Color**: Brightness, contrast, hue, saturation
- **Quality**: Blur, noise, compression
- **Lighting**: Shadows, fog, sun flare

Supports both Keras ImageDataGenerator and Albumentations.

**Run individually:**
```bash
cd modules
python step3_augmentation.py
```

### Step 4: Classification (EfficientNet)

- Model: EfficientNet-B0 (transfer learning from ImageNet)
- Purpose: Classify food categories
- Two-phase training:
  1. Frozen base model (transfer learning)
  2. Fine-tuning (last 30 layers unfrozen)
- Metrics: Accuracy, Top-5 Accuracy

**Run individually:**
```bash
cd modules
python step4_classification.py
```

### Step 5: Regression (CNN)

- Architecture: Dual-input CNN (before + after images)
- Purpose: Predict food weight
- Feature extractor: EfficientNet-B0
- Output: Single continuous value (weight in grams)
- Metrics: MAE, RMSE, R², MAPE

**Run individually:**
```bash
cd modules
python step5_regression.py
```

## Testing

### Run All Tests

```bash
# From source directory
python -m pytest tests/ -v
```

### Unit Tests

Test individual modules:

```bash
python tests/test_unit.py
```

### Pipeline Tests

Test integration between modules:

```bash
python tests/test_pipeline.py
```

### End-to-End Tests

Test complete workflow:

```bash
python tests/test_e2e.py
```

## Configuration

Edit `modules/config.py` to customize:

- Image sizes
- Batch sizes
- Learning rates
- Number of epochs
- Model architectures
- Data split ratios
- Augmentation parameters

## Model Architecture

### U-Net Segmentation
```
Input (256×256×3)
↓ Conv2D + Pool (5 levels)
Bottleneck (1024 filters)
↑ Conv2DTranspose + Concat (5 levels)
Output (256×256×1)
```

### EfficientNet Classification
```
Input (224×224×3)
↓ EfficientNet-B0 (ImageNet pretrained)
Global Average Pooling
↓ Dense(512) + BN + Dropout
↓ Dense(256) + BN + Dropout
Output (num_classes, softmax)
```

### CNN Regression (Dual Input)
```
Before Image (224×224×3) ──→ Feature Extractor
                              (EfficientNet-B0)
                                    ↓
After Image (224×224×3) ──→ Feature Extractor ──→ Concatenate
                              (EfficientNet-B0)      ↓
                                              Dense Layers
                                                    ↓
                                            Output (1, linear)
```

## Results

Results are saved in the `outputs/` directory:

- `pipeline_results_YYYYMMDD_HHMMSS.json` - Training and evaluation metrics
- `data_split.csv` - Train/val/test split information
- `logs/` - TensorBoard logs
- `segmentation_results/` - Sample segmentation outputs

### Viewing TensorBoard

```bash
tensorboard --logdir=outputs/logs
```

## Performance Metrics

### Classification
- **Accuracy**: Overall classification accuracy
- **Top-5 Accuracy**: Whether correct class is in top 5 predictions

### Regression
- **MAE** (Mean Absolute Error): Average prediction error in grams
- **RMSE** (Root Mean Squared Error): Square root of average squared errors
- **R²** (R-squared): Proportion of variance explained (0-1, higher is better)
- **MAPE** (Mean Absolute Percentage Error): Percentage error

## Troubleshooting

### Issue: CUDA/GPU not detected
```bash
# Check TensorFlow GPU support
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Install GPU version if needed
pip install tensorflow-gpu
```

### Issue: Out of memory
- Reduce batch size in `config.py`
- Use smaller image sizes
- Enable mixed precision training

### Issue: Missing images
- Verify image paths in Excel file
- Check folder structure matches expected format
- Run data validation: `python modules/step1_data_reader.py`

## Future Improvements

- [ ] Add more segmentation methods (Mask R-CNN, DeepLab)
- [ ] Implement attention mechanisms
- [ ] Add model ensemble for better predictions
- [ ] Web interface for inference
- [ ] Mobile deployment
- [ ] Real-time video processing

## Citation

If you use this code, please cite:

```
@software{food_weight_prediction,
  title={Food Weight Prediction System},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/food-weight-prediction}
}
```

## License

[Your License Here]

## Contact

For questions or issues, please contact [your email] or open an issue on GitHub.

---

**Note**: This project requires the Indonesian food dataset with before/after images and weight measurements. Ensure your data follows the expected format before running the pipeline.
