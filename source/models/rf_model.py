"""
Random Forest implementation for food classification.
"""
from typing import Dict, Any, List, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from loguru import logger
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from source.models.base_model import BaseFoodClassifier
from source.models.data_utils import prepare_sklearn_data, load_and_preprocess_images

class RFClassifier(BaseFoodClassifier):
    """Random Forest Classifier with Optuna optimization."""
    
    def __init__(self, num_classes: int, img_size: int = 64, **kwargs):
        super().__init__(num_classes, **kwargs)
        self.img_size = img_size
        self.model = RandomForestClassifier(random_state=42, n_jobs=-1)
        
    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Train the Random Forest model."""
        logger.info("Preparing data for Random Forest...")
        X_train, y_train = prepare_sklearn_data(train_df, self.img_size)
        X_val, y_val = prepare_sklearn_data(val_df, self.img_size)
        
        if 'params' in kwargs:
            self.model.set_params(**kwargs['params'])
            
        logger.info(f"Training Random Forest with {len(X_train)} samples...")
        self.model.fit(X_train, y_train)
        
        val_acc = self.model.score(X_val, y_val)
        logger.info(f"RF Validation Accuracy: {val_acc:.4f}")
        
        return {'accuracy': val_acc}
        
    def predict(self, image_paths: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Predict using Random Forest."""
        if self.model is None:
            raise ValueError("Model not trained")
            
        X = load_and_preprocess_images(image_paths, self.img_size)
        probs = self.model.predict_proba(X)
        preds = self.model.classes_[np.argmax(probs, axis=1)]
        
        return preds, probs
        
    def evaluate(self, test_df: pd.DataFrame) -> Dict[str, float]:
        """Evaluate Random Forest."""
        X_test, y_test = prepare_sklearn_data(test_df, self.img_size)
        y_pred = self.model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        logger.info(f"RF Test Accuracy: {acc:.4f}")
        logger.info(f"RF Test F1 Score: {f1:.4f}")
        
        return {'accuracy': acc, 'f1_score': f1}
        
    def optimize(self, train_df: pd.DataFrame, val_df: pd.DataFrame, n_trials: int = 20) -> Dict[str, Any]:
        """Optimize Random Forest hyperparameters."""
        logger.info("Starting Random Forest optimization...")
        
        X_train, y_train = prepare_sklearn_data(train_df, self.img_size)
        X_val, y_val = prepare_sklearn_data(val_df, self.img_size)
        
        def objective(trial):
            n_estimators = trial.suggest_int('n_estimators', 50, 300)
            max_depth = trial.suggest_int('max_depth', 5, 50)
            min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
            
            clf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                n_jobs=-1,
                random_state=42
            )
            clf.fit(X_train, y_train)
            return clf.score(X_val, y_val)
            
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        self.best_params = study.best_params
        logger.info(f"Best RF params: {self.best_params}")
        
        # Retrain
        self.model.set_params(**self.best_params)
        self.model.fit(X_train, y_train)
        
        return self.best_params

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        logger.info(f"RF model saved to {path}")
        
    def load(self, path: Path):
        self.model = joblib.load(path)
        logger.info(f"RF model loaded from {path}")
