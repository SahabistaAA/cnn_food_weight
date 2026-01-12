"""
k-Nearest Neighbors (kNN) implementation for food classification.
"""
from typing import Dict, Any, List, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import optuna
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from loguru import logger
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from source.models.base_model import BaseFoodClassifier
from source.models.data_utils import prepare_sklearn_data, load_and_preprocess_images

class KNNClassifier(BaseFoodClassifier):
    """kNN Classifier with Optuna optimization."""
    
    def __init__(self, num_classes: int, img_size: int = 64, **kwargs):
        super().__init__(num_classes, **kwargs)
        self.img_size = img_size
        self.model = KNeighborsClassifier(n_jobs=-1)
        
    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Train the kNN model."""
        logger.info("Preparing data for kNN...")
        X_train, y_train = prepare_sklearn_data(train_df, self.img_size)
        X_val, y_val = prepare_sklearn_data(val_df, self.img_size)
        
        if 'params' in kwargs:
            self.model.set_params(**kwargs['params'])
            
        logger.info(f"Training kNN with {len(X_train)} samples...")
        self.model.fit(X_train, y_train)
        
        val_acc = self.model.score(X_val, y_val)
        logger.info(f"kNN Validation Accuracy: {val_acc:.4f}")
        
        return {'accuracy': val_acc}
        
    def predict(self, image_paths: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Predict using kNN."""
        if self.model is None:
            raise ValueError("Model not trained")
            
        X = load_and_preprocess_images(image_paths, self.img_size)
        probs = self.model.predict_proba(X)
        preds = self.model.classes_[np.argmax(probs, axis=1)]
        
        return preds, probs
        
    def evaluate(self, test_df: pd.DataFrame) -> Dict[str, float]:
        """Evaluate kNN."""
        X_test, y_test = prepare_sklearn_data(test_df, self.img_size)
        y_pred = self.model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        logger.info(f"kNN Test Accuracy: {acc:.4f}")
        logger.info(f"kNN Test F1 Score: {f1:.4f}")
        
        return {'accuracy': acc, 'f1_score': f1}
        
    def optimize(self, train_df: pd.DataFrame, val_df: pd.DataFrame, n_trials: int = 20) -> Dict[str, Any]:
        """Optimize kNN hyperparameters."""
        logger.info("Starting kNN optimization...")
        
        X_train, y_train = prepare_sklearn_data(train_df, self.img_size)
        X_val, y_val = prepare_sklearn_data(val_df, self.img_size)
        
        def objective(trial):
            n_neighbors = trial.suggest_int('n_neighbors', 3, 20)
            weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
            metric = trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'minkowski'])
            
            clf = KNeighborsClassifier(
                n_neighbors=n_neighbors,
                weights=weights,
                metric=metric,
                n_jobs=-1
            )
            clf.fit(X_train, y_train)
            return clf.score(X_val, y_val)
            
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        self.best_params = study.best_params
        logger.info(f"Best kNN params: {self.best_params}")
        
        # Retrain
        self.model.set_params(**self.best_params)
        self.model.fit(X_train, y_train)
        
        return self.best_params

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        logger.info(f"kNN model saved to {path}")
        
    def load(self, path: Path):
        self.model = joblib.load(path)
        logger.info(f"kNN model loaded from {path}")
