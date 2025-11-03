"""
Main Pipeline for Food Weight Prediction
Orchestrates all 5 steps: data reading, segmentation, augmentation, classification, and regression.
"""
import sys
from pathlib import Path
import argparse
import json
from datetime import datetime
from loguru import logger

# Add modules directory to path
sys.path.insert(0, str(Path(__file__).parent / 'modules'))

from modules.config import *
from modules.step1_data_reader import FoodDataReader
from modules.step2_segmentation import UNetSegmentation
from modules.step3_augmentation import DataAugmentation
from modules.step4_classification import FoodClassification
from modules.step5_regression import WeightRegression

# Setup loguru logging
logger.add(
    OUTPUTS_DIR / 'pipeline.log',
    rotation="10 MB",
    retention="7 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
)


class FoodWeightPredictionPipeline:
    """Main pipeline for food weight prediction."""

    def __init__(self, config_overrides: dict = None):
        """
        Initialize the pipeline.

        Args:
            config_overrides: Dictionary of configuration overrides
        """
        self.config = config_overrides or {}
        self.data = None
        self.segmentation_model = None
        self.classification_model = None
        self.regression_model = None
        self.results = {}

        logger.info("=" * 80)
        logger.info("Food Weight Prediction Pipeline Initialized")
        logger.info("=" * 80)

    def step1_load_data(self):
        """Step 1: Load and preprocess data."""
        logger.info("\n" + "=" * 80)
        logger.info("STEP 1: DATA READING AND PREPROCESSING")
        logger.info("=" * 80)

        reader = FoodDataReader()
        self.data = reader.process()

        logger.info(f"\nData loaded successfully:")
        logger.info(f"  - Training samples: {len(self.data['train'])}")
        logger.info(f"  - Validation samples: {len(self.data['val'])}")
        logger.info(f"  - Test samples: {len(self.data['test'])}")
        logger.info(f"  - Unique food categories: {len(self.data['food_mapping'])}")

        self.results['step1'] = {
            'train_size': len(self.data['train']),
            'val_size': len(self.data['val']),
            'test_size': len(self.data['test']),
            'num_categories': len(self.data['food_mapping'])
        }

    def step2_train_segmentation(self, epochs: int = 30, batch_size: int = 16,
                                 skip_if_exists: bool = True):
        """Step 2: Train U-Net segmentation model."""
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: IMAGE SEGMENTATION (U-NET)")
        logger.info("=" * 80)

        if skip_if_exists and SEGMENTATION_MODEL_PATH.exists():
            logger.info(f"Loading existing segmentation model from {SEGMENTATION_MODEL_PATH}")
            self.segmentation_model = UNetSegmentation()
            self.segmentation_model.load_model()
            self.results['step2'] = {'status': 'loaded_existing'}
            return

        self.segmentation_model = UNetSegmentation()

        history = self.segmentation_model.train(
            self.data['train'],
            self.data['val'],
            epochs=epochs,
            batch_size=batch_size
        )

        self.segmentation_model.save_model()

        self.results['step2'] = {
            'final_train_loss': float(history.history['loss'][-1]),
            'final_val_loss': float(history.history['val_loss'][-1]),
            'final_train_accuracy': float(history.history['accuracy'][-1]),
            'final_val_accuracy': float(history.history['val_accuracy'][-1])
        }

        logger.info(f"\nSegmentation training completed!")
        logger.info(f"  - Final training loss: {self.results['step2']['final_train_loss']:.4f}")
        logger.info(f"  - Final validation loss: {self.results['step2']['final_val_loss']:.4f}")

    def step3_setup_augmentation(self):
        """Step 3: Setup data augmentation."""
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: DATA AUGMENTATION SETUP")
        logger.info("=" * 80)

        augmenter = DataAugmentation()

        logger.info("Data augmentation configured with parameters:")
        for key, value in AUGMENTATION_PARAMS.items():
            logger.info(f"  - {key}: {value}")

        self.results['step3'] = {
            'augmentation_params': AUGMENTATION_PARAMS
        }

    def step4_train_classification(self, epochs: int = None, batch_size: int = None,
                                   fine_tune: bool = True, skip_if_exists: bool = True):
        """Step 4: Train EfficientNet classification model."""
        logger.info("\n" + "=" * 80)
        logger.info("STEP 4: FOOD CLASSIFICATION (EFFICIENTNET)")
        logger.info("=" * 80)

        epochs = epochs or CLASSIFICATION_EPOCHS
        batch_size = batch_size or CLASSIFICATION_BATCH_SIZE

        if skip_if_exists and CLASSIFICATION_MODEL_PATH.exists():
            logger.info(f"Loading existing classification model from {CLASSIFICATION_MODEL_PATH}")

            # Get number of classes
            num_classes = self.data['train']['food_category_id'].nunique()
            self.classification_model = FoodClassification(num_classes=num_classes)
            self.classification_model.load_model()
            self.results['step4'] = {'status': 'loaded_existing'}
            return

        # Get number of unique food categories
        num_classes = self.data['train']['food_category_id'].nunique()

        self.classification_model = FoodClassification(num_classes=num_classes)

        history = self.classification_model.train(
            self.data['train'],
            self.data['val'],
            epochs=epochs,
            batch_size=batch_size,
            use_augmentation=True,
            fine_tune=fine_tune,
            fine_tune_epochs=20
        )

        self.classification_model.save_model()

        self.results['step4'] = {
            'num_classes': num_classes,
            'final_train_loss': float(history.history['loss'][-1]),
            'final_val_loss': float(history.history['val_loss'][-1]),
            'final_train_accuracy': float(history.history['accuracy'][-1]),
            'final_val_accuracy': float(history.history['val_accuracy'][-1]),
            'final_top5_accuracy': float(history.history['val_top5_accuracy'][-1])
        }

        logger.info(f"\nClassification training completed!")
        logger.info(f"  - Final training accuracy: {self.results['step4']['final_train_accuracy']:.4f}")
        logger.info(f"  - Final validation accuracy: {self.results['step4']['final_val_accuracy']:.4f}")
        logger.info(f"  - Final top-5 accuracy: {self.results['step4']['final_top5_accuracy']:.4f}")

    def step5_train_regression(self, epochs: int = None, batch_size: int = None,
                              use_dual_input: bool = True, predict_difference: bool = False,
                              skip_if_exists: bool = True):
        """Step 5: Train CNN regression model for weight prediction."""
        logger.info("\n" + "=" * 80)
        logger.info("STEP 5: WEIGHT PREDICTION (CNN REGRESSION)")
        logger.info("=" * 80)

        epochs = epochs or REGRESSION_EPOCHS
        batch_size = batch_size or REGRESSION_BATCH_SIZE

        if skip_if_exists and REGRESSION_MODEL_PATH.exists():
            logger.info(f"Loading existing regression model from {REGRESSION_MODEL_PATH}")
            self.regression_model = WeightRegression(use_dual_input=use_dual_input)
            self.regression_model.load_model()
            self.results['step5'] = {'status': 'loaded_existing'}
            return

        self.regression_model = WeightRegression(use_dual_input=use_dual_input)

        history = self.regression_model.train(
            self.data['train'],
            self.data['val'],
            epochs=epochs,
            batch_size=batch_size,
            use_augmentation=True,
            predict_difference=predict_difference
        )

        self.regression_model.save_model()

        self.results['step5'] = {
            'dual_input': use_dual_input,
            'predict_difference': predict_difference,
            'final_train_mae': float(history.history['mae'][-1]),
            'final_val_mae': float(history.history['val_mae'][-1]),
            'final_train_rmse': float(history.history['rmse'][-1]),
            'final_val_rmse': float(history.history['val_rmse'][-1])
        }

        # Denormalize metrics
        if hasattr(self.regression_model, 'weight_stats'):
            std = self.regression_model.weight_stats.get('std', 1.0)
            self.results['step5']['final_train_mae_g'] = self.results['step5']['final_train_mae'] * std
            self.results['step5']['final_val_mae_g'] = self.results['step5']['final_val_mae'] * std

        logger.info(f"\nRegression training completed!")
        logger.info(f"  - Final training MAE: {self.results['step5'].get('final_train_mae_g', 'N/A')} g")
        logger.info(f"  - Final validation MAE: {self.results['step5'].get('final_val_mae_g', 'N/A')} g")

    def evaluate_all_models(self):
        """Evaluate all models on test set."""
        logger.info("\n" + "=" * 80)
        logger.info("FINAL EVALUATION ON TEST SET")
        logger.info("=" * 80)

        test_results = {}

        # Evaluate classification
        if self.classification_model is not None:
            logger.info("\nEvaluating Classification Model...")
            classification_metrics = self.classification_model.evaluate(self.data['test'])
            test_results['classification'] = classification_metrics

        # Evaluate regression
        if self.regression_model is not None:
            logger.info("\nEvaluating Regression Model...")
            regression_metrics = self.regression_model.evaluate(self.data['test'], predict_difference=False)
            test_results['regression'] = regression_metrics

        self.results['test_evaluation'] = test_results

        logger.info("\n" + "=" * 80)
        logger.info("TEST SET RESULTS SUMMARY")
        logger.info("=" * 80)

        if 'classification' in test_results:
            logger.info("\nClassification:")
            logger.info(f"  - Accuracy: {test_results['classification']['accuracy']:.4f}")
            logger.info(f"  - Top-5 Accuracy: {test_results['classification']['top5_accuracy']:.4f}")

        if 'regression' in test_results:
            logger.info("\nRegression:")
            logger.info(f"  - MAE: {test_results['regression']['mae']:.2f} g")
            logger.info(f"  - RMSE: {test_results['regression']['rmse']:.2f} g")
            logger.info(f"  - R²: {test_results['regression']['r2']:.4f}")
            logger.info(f"  - MAPE: {test_results['regression']['mape']:.2f}%")

    def save_results(self):
        """Save pipeline results to JSON."""
        results_path = OUTPUTS_DIR / f"pipeline_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=4)

        logger.info(f"\nPipeline results saved to: {results_path}")

    def run_full_pipeline(self, skip_if_exists: bool = True,
                         segmentation_epochs: int = 30,
                         classification_epochs: int = None,
                         regression_epochs: int = None):
        """
        Run the complete pipeline.

        Args:
            skip_if_exists: Skip training if models already exist
            segmentation_epochs: Number of epochs for segmentation
            classification_epochs: Number of epochs for classification
            regression_epochs: Number of epochs for regression
        """
        start_time = datetime.now()

        try:
            # Step 1: Load data
            self.step1_load_data()

            # Step 2: Train segmentation
            self.step2_train_segmentation(
                epochs=segmentation_epochs,
                skip_if_exists=skip_if_exists
            )

            # Step 3: Setup augmentation
            self.step3_setup_augmentation()

            # Step 4: Train classification
            self.step4_train_classification(
                epochs=classification_epochs,
                skip_if_exists=skip_if_exists
            )

            # Step 5: Train regression
            self.step5_train_regression(
                epochs=regression_epochs,
                skip_if_exists=skip_if_exists
            )

            # Evaluate all models
            self.evaluate_all_models()

            # Save results
            self.save_results()

            end_time = datetime.now()
            duration = end_time - start_time

            logger.info("\n" + "=" * 80)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("=" * 80)
            logger.info(f"Total duration: {duration}")

        except Exception as e:
            logger.error(f"Pipeline failed with error: {e}", exc_info=True)
            raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Food Weight Prediction Pipeline')

    parser.add_argument('--mode', type=str, default='full',
                       choices=['full', 'train', 'evaluate'],
                       help='Pipeline mode: full, train, or evaluate')

    parser.add_argument('--skip-existing', action='store_true', default=True,
                       help='Skip training if models already exist')

    parser.add_argument('--no-skip-existing', action='store_false', dest='skip_existing',
                       help='Retrain even if models exist')

    parser.add_argument('--seg-epochs', type=int, default=30,
                       help='Number of epochs for segmentation training')

    parser.add_argument('--cls-epochs', type=int, default=None,
                       help='Number of epochs for classification training')

    parser.add_argument('--reg-epochs', type=int, default=None,
                       help='Number of epochs for regression training')

    parser.add_argument('--quick-test', action='store_true',
                       help='Run quick test with reduced epochs (for testing)')

    args = parser.parse_args()

    # Quick test mode
    if args.quick_test:
        args.seg_epochs = 2
        args.cls_epochs = 2
        args.reg_epochs = 2
        logger.info("Running in QUICK TEST mode with reduced epochs")

    # Initialize pipeline
    pipeline = FoodWeightPredictionPipeline()

    if args.mode == 'full':
        # Run full pipeline
        pipeline.run_full_pipeline(
            skip_if_exists=args.skip_existing,
            segmentation_epochs=args.seg_epochs,
            classification_epochs=args.cls_epochs,
            regression_epochs=args.reg_epochs
        )

    elif args.mode == 'train':
        # Only training
        pipeline.step1_load_data()
        pipeline.step2_train_segmentation(epochs=args.seg_epochs, skip_if_exists=args.skip_existing)
        pipeline.step3_setup_augmentation()
        pipeline.step4_train_classification(epochs=args.cls_epochs, skip_if_exists=args.skip_existing)
        pipeline.step5_train_regression(epochs=args.reg_epochs, skip_if_exists=args.skip_existing)
        pipeline.save_results()

    elif args.mode == 'evaluate':
        # Only evaluation
        pipeline.step1_load_data()

        # Load models
        num_classes = pipeline.data['train']['food_category_id'].nunique()
        pipeline.classification_model = FoodClassification(num_classes=num_classes)
        pipeline.classification_model.load_model()

        pipeline.regression_model = WeightRegression(use_dual_input=True)
        pipeline.regression_model.load_model()

        # Evaluate
        pipeline.evaluate_all_models()
        pipeline.save_results()


if __name__ == "__main__":
    main()
