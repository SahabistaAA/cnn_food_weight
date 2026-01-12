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
# from source.modules.step3_augmentation import DataAugmentation # Optional, can be imported if needed
from source.models.svm_model import SVMClassifier
from source.models.rf_model import RFClassifier
from source.models.dt_model import DTClassifier
from source.models.knn_model import KNNClassifier
from source.models.cnn_model import CNNClassifier
from source.modules.step5_regression import WeightRegression

logger.add(
    config.OUTPUTS_DIR / 'pipeline.log',
    rotation="10 MB",
    retention="7 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
)

def get_model(model_name: str, num_classes: int):
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
        return CNNClassifier(num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")

class FoodPipeline:
    def __init__(self):
        self.data = None
        self.results = {}

    def load_data(self):
        logger.info("Loading data...")
        reader = FoodDataReader()
        self.data = reader.process()
        self.num_classes = self.data['train']['food_category_id'].nunique()
        logger.info(f"Data loaded. {self.num_classes} classes found.")

    def run_classification(self, model_name: str, optimize: bool = False):
        logger.info(f"Running classification with {model_name}...")
        model = get_model(model_name, self.num_classes)
        
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

    def compare_models(self, models_to_run: list = None, optimize: bool = False):
        if models_to_run is None:
            models_to_run = ['SVM', 'RF', 'DT', 'KNN', 'CNN']
            
        logger.info(f"Comparing models: {models_to_run}")
        comparison = []
        
        for name in models_to_run:
            try:
                metrics = self.run_classification(name, optimize)
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
    parser.add_argument('--model', type=str, choices=['SVM', 'RF', 'DT', 'KNN', 'CNN', 'ALL'], 
                        default='CNN', help='Model to train/evaluate')
    parser.add_argument('--optimize', action='store_true', help='Perform hyperparameter optimization with Optuna')
    parser.add_argument('--compare', action='store_true', help='Run comparison of all models')
    
    args = parser.parse_args()
    
    pipeline = FoodPipeline()
    pipeline.load_data()
    
    if args.compare or args.model == 'ALL':
        pipeline.compare_models(optimize=args.optimize)
    else:
        pipeline.run_classification(args.model, optimize=args.optimize)

if __name__ == "__main__":
    main()
