"""
Main Pipeline for Food Weight Prediction
UPDATED: Now uses classification output to guide regression predictions.
"""
import sys
from pathlib import Path
import argparse
import json
from datetime import datetime
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent / 'modules'))

from modules.config import *
from modules.step1_data_reader import FoodDataReader
from modules.step2_segmentation import UNetSegmentation
from modules.step3_augmentation import DataAugmentation
from modules.step4_classification import FoodClassification
from modules.step5_regression import WeightRegression  # Updated version

logger.add(
    OUTPUTS_DIR / 'pipeline.log',
    rotation="10 MB",
    retention="7 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
)


class FoodWeightPredictionPipeline:
    """Main pipeline with class-conditioned regression."""

    def __init__(self, config_overrides: dict = None):
        """Initialize the pipeline."""
        self.config = config_overrides or {}
        self.data = None
        self.segmentation_model = None
        self.classification_model = None
        self.regression_model = None
        self.results = {}

        logger.info("=" * 80)
        logger.info("Food Weight Prediction Pipeline (Class-Conditioned)")
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
            'final_train_dice': float(history.history['dice'][-1]),
            'final_val_dice': float(history.history['val_dice'][-1])
        }

        logger.info(f"\nSegmentation training completed!")
        logger.info(f"  - Final validation Dice: {self.results['step2']['final_val_dice']:.4f}")

    def step3_setup_augmentation(self):
        """Step 3: Setup data augmentation."""
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: DATA AUGMENTATION SETUP")
        logger.info("=" * 80)

        augmenter = DataAugmentation()

        logger.info("Data augmentation configured")
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

        num_classes = self.data['train']['food_category_id'].nunique()

        if skip_if_exists and CLASSIFICATION_MODEL_PATH.exists():
            logger.info(f"Loading existing classification model from {CLASSIFICATION_MODEL_PATH}")
            self.classification_model = FoodClassification(num_classes=num_classes)
            self.classification_model.load_model()
            self.results['step4'] = {'status': 'loaded_existing', 'num_classes': num_classes}
            return

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
        logger.info(f"  - Final validation accuracy: {self.results['step4']['final_val_accuracy']:.4f}")
        logger.info(f"  - Final top-5 accuracy: {self.results['step4']['final_top5_accuracy']:.4f}")

    def step5_train_regression(self, epochs: int = None, batch_size: int = None,
                              use_dual_input: bool = True, predict_difference: bool = False,
                              skip_if_exists: bool = True):
        """Step 5: Train class-conditioned CNN regression model."""
        logger.info("\n" + "=" * 80)
        logger.info("STEP 5: WEIGHT PREDICTION (CLASS-CONDITIONED REGRESSION)")
        logger.info("=" * 80)

        epochs = epochs or REGRESSION_EPOCHS
        batch_size = batch_size or REGRESSION_BATCH_SIZE

        # Ensure classification model is loaded
        if self.classification_model is None:
            logger.info("Loading classification model for regression...")
            num_classes = self.data['train']['food_category_id'].nunique()
            self.classification_model = FoodClassification(num_classes=num_classes)
            self.classification_model.load_model()

        num_classes = self.data['train']['food_category_id'].nunique()

        if skip_if_exists and REGRESSION_MODEL_PATH.exists():
            logger.info(f"Loading existing regression model from {REGRESSION_MODEL_PATH}")
            self.regression_model = WeightRegression(
                num_classes=num_classes,
                use_dual_input=use_dual_input,
                classification_model=self.classification_model
            )
            self.regression_model.load_model()
            self.results['step5'] = {'status': 'loaded_existing'}
            return

        # Initialize class-conditioned regression model
        logger.info(f"Initializing class-conditioned regression with {num_classes} food categories")
        self.regression_model = WeightRegression(
            num_classes=num_classes,
            use_dual_input=use_dual_input,
            classification_model=self.classification_model
        )

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
            'class_conditioned': True,
            'num_classes': num_classes,
            'final_train_mae': float(history.history['mae'][-1]),
            'final_val_mae': float(history.history['val_mae'][-1]),
            'final_train_rmse': float(history.history['rmse'][-1]),
            'final_val_rmse': float(history.history['val_rmse'][-1])
        }

        logger.info(f"\nClass-conditioned regression training completed!")
        logger.info(f"  - Final training MAE: {self.results['step5']['final_train_mae']:.2f}g")
        logger.info(f"  - Final validation MAE: {self.results['step5']['final_val_mae']:.2f}g")
        logger.info(f"  - Uses classification guidance: YES")

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
            
            # Log per-class performance
            logger.info(f"  - Test Accuracy: {classification_metrics['accuracy']:.4f}")
            logger.info(f"  - Test Top-5 Accuracy: {classification_metrics['top5_accuracy']:.4f}")

        # Evaluate regression
        if self.regression_model is not None:
            logger.info("\nEvaluating Class-Conditioned Regression Model...")
            regression_metrics = self.regression_model.evaluate(
                self.data['test'], 
                predict_difference=False
            )
            test_results['regression'] = regression_metrics
            
            logger.info(f"  - Test MAE: {regression_metrics['mae']:.2f}g")
            logger.info(f"  - Test RMSE: {regression_metrics['rmse']:.2f}g")
            logger.info(f"  - Test R²: {regression_metrics['r2']:.4f}")

        self.results['test_evaluation'] = test_results

        logger.info("\n" + "=" * 80)
        logger.info("TEST SET RESULTS SUMMARY")
        logger.info("=" * 80)

        if 'classification' in test_results:
            logger.info("\nClassification Performance:")
            logger.info(f"  - Accuracy: {test_results['classification']['accuracy']:.4f}")
            logger.info(f"  - Top-5 Accuracy: {test_results['classification']['top5_accuracy']:.4f}")

        if 'regression' in test_results:
            logger.info("\nClass-Conditioned Regression Performance:")
            logger.info(f"  - MAE: {test_results['regression']['mae']:.2f}g")
            logger.info(f"  - RMSE: {test_results['regression']['rmse']:.2f}g")
            logger.info(f"  - R²: {test_results['regression']['r2']:.4f}")
            logger.info(f"  - MAPE: {test_results['regression']['mape']:.2f}%")

    def analyze_per_class_performance(self):
        """Analyze regression performance per food class."""
        logger.info("\n" + "=" * 80)
        logger.info("PER-CLASS PERFORMANCE ANALYSIS")
        logger.info("=" * 80)

        if self.regression_model is None or self.classification_model is None:
            logger.warning("Models not loaded, skipping per-class analysis")
            return

        test_df = self.data['test']
        
        # Get predictions
        if self.regression_model.use_dual_input:
            predictions = self.regression_model.predict(
                test_df['image_before_path'].tolist(),
                test_df['image_after_path'].tolist()
            )
        else:
            predictions = self.regression_model.predict(
                image_after_paths=test_df['image_after_path'].tolist()
            )

        ground_truth = test_df['Weight After Eaten (g)'].values
        food_categories = test_df['food_category_id'].values

        # Calculate per-class metrics
        per_class_metrics = {}
        for category in np.unique(food_categories):
            mask = food_categories == category
            if mask.sum() > 0:
                cat_gt = ground_truth[mask]
                cat_pred = predictions[mask]
                
                mae = mean_absolute_error(cat_gt, cat_pred)
                rmse = np.sqrt(mean_squared_error(cat_gt, cat_pred))
                
                per_class_metrics[category] = {
                    'count': mask.sum(),
                    'mae': mae,
                    'rmse': rmse
                }

        # Log top 5 best and worst performing classes
        sorted_classes = sorted(per_class_metrics.items(), 
                               key=lambda x: x[1]['mae'])

        logger.info("\nTop 5 Best Performing Classes (Lowest MAE):")
        for i, (cat, metrics) in enumerate(sorted_classes[:5], 1):
            food_name = self.data['food_mapping'].get(cat, cat)
            logger.info(f"  {i}. {food_name} ({cat}): MAE={metrics['mae']:.2f}g, "
                       f"RMSE={metrics['rmse']:.2f}g, n={metrics['count']}")

        logger.info("\nTop 5 Worst Performing Classes (Highest MAE):")
        for i, (cat, metrics) in enumerate(sorted_classes[-5:][::-1], 1):
            food_name = self.data['food_mapping'].get(cat, cat)
            logger.info(f"  {i}. {food_name} ({cat}): MAE={metrics['mae']:.2f}g, "
                       f"RMSE={metrics['rmse']:.2f}g, n={metrics['count']}")

        self.results['per_class_analysis'] = per_class_metrics

    def save_results(self):
        """Save pipeline results to JSON."""
        results_path = OUTPUTS_DIR / f"pipeline_results_class_conditioned_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=4)

        logger.info(f"\nPipeline results saved to: {results_path}")

    def run_full_pipeline(self, skip_if_exists: bool = True,
                         segmentation_epochs: int = 30,
                         classification_epochs: int = None,
                         regression_epochs: int = None):
        """Run the complete class-conditioned pipeline."""
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

            # Step 4: Train classification (CRITICAL for step 5)
            self.step4_train_classification(
                epochs=classification_epochs,
                skip_if_exists=skip_if_exists
            )

            # Step 5: Train class-conditioned regression
            logger.info("\n*** NOTE: Regression will use classification predictions ***")
            self.step5_train_regression(
                epochs=regression_epochs,
                skip_if_exists=skip_if_exists
            )

            # Evaluate all models
            self.evaluate_all_models()

            # Analyze per-class performance
            self.analyze_per_class_performance()

            # Save results
            self.save_results()

            end_time = datetime.now()
            duration = end_time - start_time

            logger.info("\n" + "=" * 80)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("=" * 80)
            logger.info(f"Total duration: {duration}")
            logger.info(f"Architecture: Classification → Regression (Class-Conditioned)")

        except Exception as e:
            logger.error(f"Pipeline failed with error: {e}", exc_info=True)
            raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Food Weight Prediction Pipeline (Class-Conditioned)'
    )

    parser.add_argument('--mode', type=str, default='full',
                       choices=['full', 'train', 'evaluate'],
                       help='Pipeline mode')

    parser.add_argument('--skip-existing', action='store_true', default=True,
                       help='Skip training if models already exist')

    parser.add_argument('--no-skip-existing', action='store_false', 
                       dest='skip_existing',
                       help='Retrain even if models exist')

    parser.add_argument('--seg-epochs', type=int, default=30,
                       help='Number of epochs for segmentation')

    parser.add_argument('--cls-epochs', type=int, default=None,
                       help='Number of epochs for classification')

    parser.add_argument('--reg-epochs', type=int, default=None,
                       help='Number of epochs for regression')

    parser.add_argument('--quick-test', action='store_true',
                       help='Quick test with reduced epochs')

    args = parser.parse_args()

    if args.quick_test:
        args.seg_epochs = 2
        args.cls_epochs = 2
        args.reg_epochs = 2
        logger.info("Running in QUICK TEST mode with reduced epochs")

    # Initialize pipeline
    pipeline = FoodWeightPredictionPipeline()

    if args.mode == 'full':
        pipeline.run_full_pipeline(
            skip_if_exists=args.skip_existing,
            segmentation_epochs=args.seg_epochs,
            classification_epochs=args.cls_epochs,
            regression_epochs=args.reg_epochs
        )

    elif args.mode == 'train':
        pipeline.step1_load_data()
        pipeline.step2_train_segmentation(
            epochs=args.seg_epochs, 
            skip_if_exists=args.skip_existing
        )
        pipeline.step3_setup_augmentation()
        pipeline.step4_train_classification(
            epochs=args.cls_epochs, 
            skip_if_exists=args.skip_existing
        )
        pipeline.step5_train_regression(
            epochs=args.reg_epochs, 
            skip_if_exists=args.skip_existing
        )
        pipeline.save_results()

    elif args.mode == 'evaluate':
        pipeline.step1_load_data()

        # Load models
        num_classes = pipeline.data['train']['food_category_id'].nunique()
        
        pipeline.classification_model = FoodClassification(num_classes=num_classes)
        pipeline.classification_model.load_model()

        pipeline.regression_model = WeightRegression(
            num_classes=num_classes,
            use_dual_input=True,
            classification_model=pipeline.classification_model
        )
        pipeline.regression_model.load_model()

        # Evaluate
        pipeline.evaluate_all_models()
        pipeline.analyze_per_class_performance()
        pipeline.save_results()


if __name__ == "__main__":
    main()