"""
Main Pipeline for Food Weight Prediction and Classification.
"""
import sys
from pathlib import Path
import argparse
import json
import pandas as pd
from datetime import datetime
from loguru import logger

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from source.config import config
from source.modules.step1_data_reader import FoodDataReader
from source.modules.step2_segmentation import UNetSegmentation
from source.modules.step3_augmentation import DataAugmentation
from source.modules.step4_classification import FoodClassification
from source.models.svm_model import SVMClassifier
from source.models.rf_model import RFClassifier
from source.models.dt_model import DTClassifier
from source.models.knn_model import KNNClassifier
from source.models.cnn_model import CNNClassifier
from source.models.cnn.cnn_non_pretrained import CNNScratchClassifier
from source.models.cnn.cnn_pretrained import CNNPretrainedScratchClassifier
from source.modules.step5_regression import WeightRegression

logger.add(
    config.OUTPUTS_DIR / 'pipeline.log',
    rotation="10 MB",
    retention="7 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
)

def get_model(model_name: str, num_classes: int, use_visualization: bool = False, pretrained: bool = True):
    """Factory to get model instance."""
    if model_name.upper() == 'SVM':
        return SVMClassifier(num_classes)
    elif model_name.upper() == 'RF':
        return RFClassifier(num_classes)
    elif model_name.upper() == 'DT':
        return DTClassifier(num_classes)
    elif model_name.upper() == 'KNN':
        return KNNClassifier(num_classes)
    elif model_name.upper() == 'CNN':
        if use_visualization:
            if pretrained:
                logger.info("Using CNNPretrainedScratchClassifier with Mathematical Visualization (Transfer Learning & Tracer)")
                return CNNPretrainedScratchClassifier(num_classes, use_visualization=True)
            else:
                logger.info("Using CNNScratchClassifier (Scratch parameters) with Mathematical Visualization")
                return CNNScratchClassifier(num_classes, use_visualization=True)
        return CNNClassifier(num_classes, pretrained=pretrained)
    elif model_name.upper() == 'COMPLEXCNN':
        return 'COMPLEXCNN'
    else:
        raise ValueError(f"Unknown model: {model_name}")

class FoodPipeline:
    def __init__(self):
        self.data = None
        self.results = {}
        self.segmentation_model = None
        self.classification_model = None
        self.regression_model = None

    def load_data(self):
        logger.info("Loading data...")
        reader = FoodDataReader()
        self.data = reader.process()
        self.num_classes = self.data['train']['food_category_id'].nunique()
        logger.info(f"Data loaded. {self.num_classes} classes found.")

    def run_segmentation(self, epochs: int = 30, batch_size: int = 16):
        """Run U-Net segmentation training."""
        logger.info("Running U-Net segmentation...")
        
        self.segmentation_model = UNetSegmentation(
            img_size=config.SEGMENTATION_IMG_SIZE,
            filters=config.UNET_FILTERS
        )
        
        history = self.segmentation_model.train(
            self.data['train'],
            self.data['val'],
            epochs=epochs,
            batch_size=batch_size
        )
        
        logger.info("Segmentation training completed!")
        return history

    def run_classification(self, model_name: str, optimize: bool = False, use_visualization: bool = False, pretrained: bool = True):
        logger.info(f"Running classification with {model_name}...")
        
        # Special handling for ComplexCNN - run only classification component
        if model_name.upper() == 'COMPLEXCNN':
            logger.info("Running ComplexCNN classification component for comparison...")
            
            # Create and train classification model
            classification_model = FoodClassification(
                num_classes=self.num_classes,
                img_size=config.IMG_HEIGHT,
                pretrained=pretrained
            )
            
            # Train with default or optimized parameters
            if optimize:
                logger.warning("Optimization not yet implemented for ComplexCNN in comparison mode")
            
            classification_model.train(
                self.data['train'],
                self.data['val'],
                epochs=config.CLASSIFICATION_EPOCHS,
                batch_size=config.CLASSIFICATION_BATCH_SIZE,
                use_augmentation=True,
                fine_tune=True,
                fine_tune_epochs=20,
                num_workers=0
            )
            
            # Evaluate on test set
            metrics = classification_model.evaluate(
                self.data['test'],
                batch_size=config.CLASSIFICATION_BATCH_SIZE
            )
            
            # Save model
            classification_model.save_model(config.MODELS_DIR / "complexcnn_classification.pth")
            
            self.results[model_name] = metrics
            return metrics
        
        # Standard classification models
        model = get_model(model_name, self.num_classes, use_visualization, pretrained=pretrained)
        
        # Optimization
        if optimize:
            logger.info(f"Optimizing {model_name}...")
            best_params = model.optimize(
                self.data['train'], 
                self.data['val'], 
                n_trials=config.OPTUNA_N_TRIALS
            )
            # Model is already retrained with best params inside optimize() for most non-DL models,
            # but for CNN we might need to be careful. The interface says optimize returns params.
            # Our CNN implementation does not auto-retrain fully.
            if model_name.upper() == 'CNN':
                logger.info("Retraining CNN with best params...")
                model.train(
                    self.data['train'], 
                    self.data['val'], 
                    epochs=config.CLASSIFICATION_EPOCHS,
                    **best_params
                )
        else:
            model.train(self.data['train'], self.data['val'])
            
        # Evaluation
        metrics = model.evaluate(self.data['test'])
        model.save(config.MODELS_DIR / f"{model_name.lower()}_model.pkl") # or .pth for CNN
        
        self.results[model_name] = metrics
        return metrics

    def optimize_complex_cnn(self, n_trials: int = 20):
        """
        Optimize ComplexCNN hyperparameters using Optuna.
        
        Args:
            n_trials: Number of Optuna trials
            
        Returns:
            Best hyperparameters dictionary
        """
        import optuna
        from optuna.samplers import TPESampler
        
        logger.info("="*80)
        logger.info(f"Starting ComplexCNN Hyperparameter Optimization ({n_trials} trials)")
        logger.info("="*80)
        
        def objective(trial):
            """Optuna objective function."""
            
            # Suggest hyperparameters
            params = {
                'classification_lr': trial.suggest_float('classification_lr', 1e-5, 1e-3, log=True),
                'classification_batch_size': trial.suggest_categorical('classification_batch_size', [16, 32, 64]),
                'classification_dropout': trial.suggest_float('classification_dropout', 0.3, 0.6),
                'regression_lr': trial.suggest_float('regression_lr', 1e-5, 1e-3, log=True),
                'regression_batch_size': trial.suggest_categorical('regression_batch_size', [16, 32, 64]),
                'use_augmentation': trial.suggest_categorical('use_augmentation', [True, False]),
                'dual_input_regression': trial.suggest_categorical('dual_input_regression', [True, False]),
            }
            
            logger.info(f"\nTrial {trial.number + 1}/{n_trials}")
            logger.info(f"Parameters: {params}")
            
            try:
                # Train classification model
                classification_model = FoodClassification(
                    num_classes=self.num_classes,
                    img_size=config.IMG_HEIGHT,
                    pretrained=True # Enforced True for optimization defaults unless added to optuna space
                )
                
                # Quick training (reduced epochs for optimization)
                classification_model.train(
                    self.data['train'],
                    self.data['val'],
                    epochs=10,  # Reduced for faster optimization
                    batch_size=params['classification_batch_size'],
                    learning_rate=params['classification_lr'],
                    use_augmentation=params['use_augmentation'],
                    fine_tune=False,  # Skip fine-tuning during optimization
                    num_workers=0
                )
                
                # Evaluate classification
                class_metrics = classification_model.evaluate(
                    self.data['val'],
                    batch_size=params['classification_batch_size']
                )
                
                # Train regression model
                regression_model = WeightRegression(
                    num_classes=self.num_classes,
                    img_size=config.IMG_HEIGHT,
                    use_dual_input=params['dual_input_regression'],
                    classification_model=classification_model
                )
                
                reg_history = regression_model.train(
                    self.data['train'],
                    self.data['val'],
                    epochs=20,  # Reduced for faster optimization
                    batch_size=params['regression_batch_size'],
                    learning_rate=params['regression_lr'],
                    use_augmentation=params['use_augmentation'],
                    predict_difference=False,
                    num_workers=0
                )
                
                # Combined metric: classification accuracy + regression performance
                # Normalize MAE to [0, 1] range (assuming max error ~500g)
                normalized_mae = reg_history.history['val_mae'][-1] / 500.0
                
                # Combined score (higher is better)
                # 70% weight on classification, 30% on regression
                score = (0.7 * class_metrics['accuracy']) + (0.3 * (1 - normalized_mae))
                
                logger.info(f"Trial {trial.number + 1} Results:")
                logger.info(f"  Classification Accuracy: {class_metrics['accuracy']:.4f}")
                logger.info(f"  Regression MAE: {reg_history.history['val_mae'][-1]:.2f}g")
                logger.info(f"  Combined Score: {score:.4f}")
                
                return score
                
            except Exception as e:
                logger.error(f"Trial {trial.number + 1} failed: {e}")
                return 0.0  # Return worst score on failure
        
        # Create Optuna study
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=config.RANDOM_SEED)
        )
        
        # Run optimization
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        # Get best parameters
        best_params = study.best_params
        best_score = study.best_value
        
        logger.info("\n" + "="*80)
        logger.info("Optimization Completed!")
        logger.info("="*80)
        logger.info(f"Best Combined Score: {best_score:.4f}")
        logger.info(f"Best Parameters: {json.dumps(best_params, indent=2)}")
        
        # Save optimization results
        results_path = config.OUTPUTS_DIR / f"complex_cnn_optuna_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_path, 'w') as f:
            json.dump({
                'best_params': best_params,
                'best_score': best_score,
                'n_trials': n_trials
            }, f, indent=4)
        logger.info(f"Optimization results saved to {results_path}")
        
        return best_params

    def run_complex_cnn(self, 
                       segmentation_epochs: int = 30,
                       classification_epochs: int = 50,
                       regression_epochs: int = 100,
                       batch_size: int = 32,
                       classification_lr: float = None,
                       regression_lr: float = None,
                       use_segmentation: bool = True,
                       use_augmentation: bool = True,
                       fine_tune_classification: bool = True,
                       dual_input_regression: bool = True,
                       predict_difference: bool = False,
                       optimized_params: dict = None):
        """
        Run the complete ComplexCNN pipeline with all modules.

        Args:
            segmentation_epochs: Epochs for U-Net training
            classification_epochs: Epochs for classification training
            regression_epochs: Epochs for regression training
            batch_size: Batch size for training
            classification_lr: Learning rate for classification
            regression_lr: Learning rate for regression
            use_segmentation: Whether to train and use segmentation
            use_augmentation: Whether to use data augmentation
            fine_tune_classification: Whether to fine-tune classification model
            dual_input_regression: Whether to use both before/after images for regression
            predict_difference: Whether to predict weight difference or absolute weight
            optimized_params: Dictionary of optimized parameters from Optuna
        """
        if optimized_params:
            logger.info("Applying optimized parameters...")
            classification_lr = optimized_params.get('classification_lr', classification_lr)
            batch_size = optimized_params.get('classification_batch_size', batch_size)
            regression_lr = optimized_params.get('regression_lr', regression_lr)
            # Note: We use classification_batch_size for both for simplicity or separate if needed
            # In optimize_complex_cnn we have both, but here we only have one batch_size arg.
            # Let's favor the classification one or handle them separately if we update signature.
            use_augmentation = optimized_params.get('use_augmentation', use_augmentation)
            dual_input_regression = optimized_params.get('dual_input_regression', dual_input_regression)
            logger.info(f"Updated params: LR_Class={classification_lr}, LR_Reg={regression_lr}, Batch={batch_size}, Aug={use_augmentation}")

        logger.info("="*80)
        logger.info("Starting ComplexCNN Pipeline")
        logger.info("="*80)
        
        results = {
            'segmentation': None,
            'classification': None,
            'regression': None
        }
        
        # Step 1: Data is already loaded
        logger.info(f"Training data: {len(self.data['train'])} samples")
        logger.info(f"Validation data: {len(self.data['val'])} samples")
        logger.info(f"Test data: {len(self.data['test'])} samples")
        
        # Step 2: Segmentation (Optional)
        if use_segmentation:
            logger.info("\n" + "="*80)
            logger.info("STEP 2: U-Net Segmentation")
            logger.info("="*80)
            
            try:
                self.segmentation_model = UNetSegmentation(
                    img_size=config.SEGMENTATION_IMG_SIZE,
                    filters=config.UNET_FILTERS
                )
                
                seg_history = self.segmentation_model.train(
                    self.data['train'],
                    self.data['val'],
                    epochs=segmentation_epochs,
                    batch_size=batch_size,
                    learning_rate=classification_lr if classification_lr else 0.001
                )
                
                results['segmentation'] = {
                    'final_loss': seg_history.history['loss'][-1],
                    'final_val_loss': seg_history.history['val_loss'][-1],
                    'final_dice': seg_history.history['dice'][-1],
                    'final_val_dice': seg_history.history['val_dice'][-1]
                }
                
                logger.info(f"Segmentation - Final Dice: {results['segmentation']['final_val_dice']:.4f}")
                
            except Exception as e:
                logger.error(f"Segmentation failed: {e}")
                logger.warning("Continuing without segmentation...")
                use_segmentation = False
        
        # Step 3: Augmentation is handled internally by classification and regression
        if use_augmentation:
            logger.info("\n" + "="*80)
            logger.info("STEP 3: Data Augmentation will be applied during training")
            logger.info("="*80)
        
        # Step 4: Classification
        logger.info("\n" + "="*80)
        logger.info("STEP 4: Food Classification (EfficientNet)")
        logger.info("="*80)
        
        try:
            self.classification_model = FoodClassification(
                num_classes=self.num_classes,
                img_size=config.IMG_HEIGHT,
                pretrained=True
            )
            
            class_history = self.classification_model.train(
                self.data['train'],
                self.data['val'],
                epochs=classification_epochs,
                batch_size=batch_size,
                learning_rate=classification_lr if classification_lr else config.CLASSIFICATION_LEARNING_RATE,
                use_augmentation=use_augmentation,
                fine_tune=fine_tune_classification,
                fine_tune_epochs=20,
                num_workers=0
            )
            
            # Evaluate classification on test set
            class_metrics = self.classification_model.evaluate(
                self.data['test'],
                batch_size=batch_size
            )
            
            results['classification'] = {
                'test_accuracy': class_metrics['accuracy'],
                'test_loss': class_metrics['loss'],
                'test_top5_accuracy': class_metrics['top5_accuracy']
            }
            
            logger.info(f"Classification - Test Accuracy: {results['classification']['test_accuracy']:.4f}")
            logger.info(f"Classification - Test Top-5 Accuracy: {results['classification']['test_top5_accuracy']:.4f}")
            
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            raise
        
        # Step 5: Weight Regression with Classification Guidance
        logger.info("\n" + "="*80)
        logger.info("STEP 5: Weight Regression (Class-Conditioned)")
        logger.info("="*80)
        
        try:
            self.regression_model = WeightRegression(
                num_classes=self.num_classes,
                img_size=config.IMG_HEIGHT,
                use_dual_input=dual_input_regression,
                classification_model=self.classification_model
            )
            
            reg_history = self.regression_model.train(
                self.data['train'],
                self.data['val'],
                epochs=regression_epochs,
                batch_size=batch_size,
                learning_rate=regression_lr if regression_lr else config.REGRESSION_LEARNING_RATE,
                use_augmentation=use_augmentation,
                predict_difference=predict_difference,
                num_workers=0
            )
            
            # Evaluate regression on test set
            reg_metrics = self.regression_model.evaluate(
                self.data['test'],
                predict_difference=predict_difference
            )
            
            results['regression'] = {
                'test_mae': reg_metrics['mae'],
                'test_rmse': reg_metrics['rmse'],
                'test_r2': reg_metrics['r2'],
                'test_mape': reg_metrics['mape']
            }
            
            logger.info(f"Regression - Test MAE: {results['regression']['test_mae']:.2f}g")
            logger.info(f"Regression - Test RMSE: {results['regression']['test_rmse']:.2f}g")
            logger.info(f"Regression - Test R²: {results['regression']['test_r2']:.4f}")
            
        except Exception as e:
            logger.error(f"Regression failed: {e}")
            raise
        
        # Save results
        logger.info("\n" + "="*80)
        logger.info("ComplexCNN Pipeline Completed Successfully!")
        logger.info("="*80)
        
        # Save comprehensive results
        results_df = pd.DataFrame({
            'Model': ['ComplexCNN'],
            'Segmentation_Dice': [results['segmentation']['final_val_dice'] if results['segmentation'] else 'N/A'],
            'Classification_Accuracy': [results['classification']['test_accuracy']],
            'Classification_Top5_Accuracy': [results['classification']['test_top5_accuracy']],
            'Regression_MAE': [results['regression']['test_mae']],
            'Regression_RMSE': [results['regression']['test_rmse']],
            'Regression_R2': [results['regression']['test_r2']],
            'Regression_MAPE': [results['regression']['test_mape']]
        })
        
        save_path = config.OUTPUTS_DIR / f"complex_cnn_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results_df.to_csv(save_path, index=False)
        logger.info(f"Results saved to {save_path}")
        
        # Save detailed results as JSON
        json_path = config.OUTPUTS_DIR / f"complex_cnn_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=4)
        logger.info(f"Detailed results saved to {json_path}")
        
        self.results['ComplexCNN'] = results
        
        return results

    def compare_models(self, models_to_run: list = None, optimize: bool = False, pretrained: bool = True):
        if models_to_run is None:
            models_to_run = ['SVM', 'RF', 'DT', 'KNN', 'CNN', 'ComplexCNN']
            
        logger.info(f"Comparing models: {models_to_run}")
        logger.info("Note: ComplexCNN comparison uses only its classification component (EfficientNet)")
        comparison = []
        
        for name in models_to_run:
            try:
                metrics = self.run_classification(name, optimize, pretrained=pretrained)
                metrics['Model'] = name
                comparison.append(metrics)
            except Exception as e:
                logger.error(f"Failed to run {name}: {e}")
                
        # Create comparison table
        df_comp = pd.DataFrame(comparison)
        logger.info("\nModel Comparison Results:")
        logger.info(f"\n{df_comp}")
        
        # Save results
        save_path = config.OUTPUTS_DIR / f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df_comp.to_csv(save_path, index=False)
        logger.info(f"Comparison saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Food Classification & Weight Prediction Pipeline")
    parser.add_argument('--model', type=str, 
                        choices=['SVM', 'RF', 'DT', 'KNN', 'CNN', 'ComplexCNN', 'ALL'], 
                        default='ComplexCNN', 
                        help='Model to train/evaluate')
    parser.add_argument('--optimize', action='store_true', 
                        help='Perform hyperparameter optimization with Optuna')
    parser.add_argument('--compare', action='store_true', 
                        help='Run comparison of all models')
    
    # Visualization argument
    parser.add_argument('--use-visualization', action='store_true',
                        help='Enable mathematical visualization for CNN models')
    
    # Pretrained toggle for CNN
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pre-trained weights for CNN and ComplexCNN (default)')
    parser.add_argument('--non-pretrained', action='store_false', dest='pretrained',
                        help='Do not use pre-trained weights for CNN and ComplexCNN')

    # ComplexCNN specific arguments
    parser.add_argument('--seg-epochs', type=int, default=30,
                        help='Epochs for segmentation training')
    parser.add_argument('--class-epochs', type=int, default=50,
                        help='Epochs for classification training')
    parser.add_argument('--reg-epochs', type=int, default=100,
                        help='Epochs for regression training')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--no-segmentation', action='store_true',
                        help='Skip segmentation step')
    parser.add_argument('--no-augmentation', action='store_true',
                        help='Disable data augmentation')
    parser.add_argument('--no-fine-tune', action='store_true',
                        help='Disable fine-tuning for classification')
    parser.add_argument('--single-input', action='store_true',
                        help='Use single input (after image only) for regression')
    parser.add_argument('--predict-difference', action='store_true',
                        help='Predict weight difference instead of absolute weight')
    
    args = parser.parse_args()
    
    pipeline = FoodPipeline()
    pipeline.load_data()
    
    if args.model == 'ComplexCNN':
        best_params = None
        if args.optimize:
            best_params = pipeline.optimize_complex_cnn(n_trials=config.OPTUNA_N_TRIALS)
            
        logger.info("Running ComplexCNN pipeline with all modules...")
        pipeline.run_complex_cnn(
            segmentation_epochs=args.seg_epochs,
            classification_epochs=args.class_epochs,
            regression_epochs=args.reg_epochs,
            batch_size=args.batch_size,
            use_segmentation=not args.no_segmentation,
            use_augmentation=not args.no_augmentation,
            fine_tune_classification=not args.no_fine_tune,
            dual_input_regression=not args.single_input,
            predict_difference=args.predict_difference,
            optimized_params=best_params
        )
    elif args.compare or args.model == 'ALL':
        pipeline.compare_models(optimize=args.optimize, pretrained=args.pretrained)
    else:
        pipeline.run_classification(args.model, optimize=args.optimize, use_visualization=args.use_visualization, pretrained=args.pretrained)

if __name__ == "__main__":
    main()