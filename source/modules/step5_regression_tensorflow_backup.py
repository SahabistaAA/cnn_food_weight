"""
Step 5: Weight Prediction using CNN Regression
Predicts the weight of leftover food using regression.
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
import cv2
from pathlib import Path
import pandas as pd
from typing import Tuple, List, Dict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WeightRegression:
    """CNN-based regression model for predicting food weight."""

    def __init__(self, img_size: int = config.IMG_HEIGHT, use_dual_input: bool = True):
        """
        Initialize weight regression model.

        Args:
            img_size: Input image size
            use_dual_input: Whether to use both before and after images
        """
        self.img_size = img_size
        self.use_dual_input = use_dual_input
        self.model = None
        self.weight_stats = {}  # For normalization

    def build_cnn_regression(self) -> keras.Model:
        """
        Build CNN regression model.
        Can use either single image (after) or dual images (before + after).

        Returns:
            Keras model
        """
        if self.use_dual_input:
            # Dual input model (before and after images)
            input_before = layers.Input(shape=(self.img_size, self.img_size, 3), name='image_before')
            input_after = layers.Input(shape=(self.img_size, self.img_size, 3), name='image_after')

            # Shared feature extractor
            feature_extractor = self.create_feature_extractor()

            # Extract features from both images
            features_before = feature_extractor(input_before)
            features_after = feature_extractor(input_after)

            # Combine features
            combined = layers.Concatenate()([features_before, features_after])

            # Regression head
            x = combined

        else:
            # Single input model (only after image)
            input_after = layers.Input(shape=(self.img_size, self.img_size, 3), name='image_after')

            # Feature extractor
            feature_extractor = self.create_feature_extractor()
            x = feature_extractor(input_after)

        # Dense layers for regression
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)

        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)

        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)

        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.2)(x)

        # Output layer (single value - weight)
        output = layers.Dense(1, activation='linear', name='weight_output')(x)

        # Create model
        if self.use_dual_input:
            model = keras.Model(
                inputs=[input_before, input_after],
                outputs=output,
                name='DualInput_Weight_Regression'
            )
        else:
            model = keras.Model(
                inputs=input_after,
                outputs=output,
                name='SingleInput_Weight_Regression'
            )

        logger.info(f"Regression model built (dual_input={self.use_dual_input})")
        logger.info(f"Total parameters: {model.count_params():,}")

        return model

    def create_feature_extractor(self) -> keras.Model:
        """
        Create a feature extractor using EfficientNet.

        Returns:
            Feature extractor model
        """
        # Use EfficientNetB0 as backbone
        base_model = EfficientNetB0(
            include_top=False,
            weights='imagenet',
            pooling='avg'
        )

        # Freeze most layers
        base_model.trainable = True
        for layer in base_model.layers[:-20]:
            layer.trainable = False

        # Add additional layers
        inputs = layers.Input(shape=(self.img_size, self.img_size, 3))
        x = base_model(inputs, training=False)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)

        feature_extractor = keras.Model(inputs=inputs, outputs=x, name='feature_extractor')

        return feature_extractor

    def compile_model(self, learning_rate: float = config.REGRESSION_LEARNING_RATE):
        """Compile the regression model."""
        if self.model is None:
            self.model = self.build_cnn_regression()

        # Custom metrics
        def mean_absolute_percentage_error(y_true, y_pred):
            """MAPE metric."""
            epsilon = 1e-7  # Avoid division by zero
            diff = tf.abs((y_true - y_pred) / (y_true + epsilon))
            return 100. * tf.reduce_mean(diff)

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mean_squared_error',
            metrics=[
                'mae',
                'mse',
                keras.metrics.RootMeanSquaredError(name='rmse'),
                mean_absolute_percentage_error
            ]
        )

        logger.info("Regression model compiled successfully")

    def load_and_preprocess_images(self, image_paths: List[str]) -> np.ndarray:
        """
        Load and preprocess images for regression.

        Args:
            image_paths: List of image file paths

        Returns:
            Preprocessed images array [N, H, W, C]
        """
        images = []

        for img_path in image_paths:
            # Read image
            img = cv2.imread(str(img_path))
            if img is None:
                logger.warning(f"Failed to load image: {img_path}")
                # Create black image as placeholder
                img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            else:
                # Convert BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Resize
            img_resized = cv2.resize(img, (self.img_size, self.img_size))

            # Normalize to [0, 1]
            img_normalized = img_resized.astype(np.float32) / 255.0

            # Apply EfficientNet preprocessing
            img_preprocessed = tf.keras.applications.efficientnet.preprocess_input(
                img_normalized * 255.0
            )

            images.append(img_preprocessed)

        return np.array(images)

    def normalize_weights(self, weights: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        Normalize weight values for better training.

        Args:
            weights: Array of weights
            fit: Whether to fit normalization parameters

        Returns:
            Normalized weights
        """
        if fit:
            self.weight_stats['mean'] = np.mean(weights)
            self.weight_stats['std'] = np.std(weights)
            logger.info(f"Weight statistics - Mean: {self.weight_stats['mean']:.2f}, "
                       f"Std: {self.weight_stats['std']:.2f}")

        normalized = (weights - self.weight_stats['mean']) / (self.weight_stats['std'] + 1e-7)
        return normalized

    def denormalize_weights(self, normalized_weights: np.ndarray) -> np.ndarray:
        """
        Denormalize weight predictions.

        Args:
            normalized_weights: Normalized weight values

        Returns:
            Original scale weights
        """
        return normalized_weights * self.weight_stats['std'] + self.weight_stats['mean']

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
              epochs: int = config.REGRESSION_EPOCHS,
              batch_size: int = config.REGRESSION_BATCH_SIZE,
              use_augmentation: bool = True,
              predict_difference: bool = False):
        """
        Train the regression model.

        Args:
            train_df: Training dataframe
            val_df: Validation dataframe
            epochs: Number of training epochs
            batch_size: Batch size
            use_augmentation: Whether to use data augmentation
            predict_difference: If True, predict weight difference instead of absolute weight
        """
        logger.info("Preparing training data for regression...")

        # Load images
        if self.use_dual_input:
            train_before = self.load_and_preprocess_images(train_df['image_before_path'].tolist())
            train_after = self.load_and_preprocess_images(train_df['image_after_path'].tolist())
            val_before = self.load_and_preprocess_images(val_df['image_before_path'].tolist())
            val_after = self.load_and_preprocess_images(val_df['image_after_path'].tolist())

            train_images = [train_before, train_after]
            val_images = [val_before, val_after]
        else:
            train_after = self.load_and_preprocess_images(train_df['image_after_path'].tolist())
            val_after = self.load_and_preprocess_images(val_df['image_after_path'].tolist())

            train_images = train_after
            val_images = val_after

        # Prepare target weights
        if predict_difference:
            train_weights = train_df['weight_difference'].values
            val_weights = val_df['weight_difference'].values
            logger.info("Training to predict weight difference (leftover)")
        else:
            train_weights = train_df['Weight After Eaten (g)'].values
            val_weights = val_df['Weight After Eaten (g)'].values
            logger.info("Training to predict absolute weight after eating")

        # Normalize weights
        train_weights_norm = self.normalize_weights(train_weights, fit=True)
        val_weights_norm = self.normalize_weights(val_weights, fit=False)

        logger.info(f"Training samples: {len(train_weights)}")
        logger.info(f"Validation samples: {len(val_weights)}")

        # Compile model if not already compiled
        if self.model is None:
            self.compile_model()

        # Data augmentation for dual input
        if use_augmentation and self.use_dual_input:
            from step3_augmentation import DataAugmentation
            augmenter = DataAugmentation()

            # Create datasets with augmentation
            def create_dual_dataset(before_imgs, after_imgs, weights, batch_size, augment):
                # Create individual datasets
                dataset = tf.data.Dataset.from_tensor_slices((
                    {'image_before': before_imgs, 'image_after': after_imgs},
                    weights
                ))

                if augment:
                    def augment_dual(inputs, weight):
                        # Apply same augmentation to both images
                        seed = tf.random.uniform(shape=[2], minval=0, maxval=1000, dtype=tf.int32)

                        # Augment before image
                        before_aug = tf.image.stateless_random_flip_left_right(
                            inputs['image_before'], seed
                        )
                        before_aug = tf.image.stateless_random_brightness(
                            before_aug, 0.2, seed
                        )

                        # Augment after image with same transformations
                        after_aug = tf.image.stateless_random_flip_left_right(
                            inputs['image_after'], seed
                        )
                        after_aug = tf.image.stateless_random_brightness(
                            after_aug, 0.2, seed
                        )

                        # Clip values
                        before_aug = tf.clip_by_value(before_aug, 0.0, 1.0)
                        after_aug = tf.clip_by_value(after_aug, 0.0, 1.0)

                        return {'image_before': before_aug, 'image_after': after_aug}, weight

                    dataset = dataset.map(augment_dual, num_parallel_calls=tf.data.AUTOTUNE)
                    dataset = dataset.shuffle(len(weights))

                dataset = dataset.batch(batch_size)
                dataset = dataset.prefetch(tf.data.AUTOTUNE)

                return dataset

            train_dataset = create_dual_dataset(
                train_before, train_after, train_weights_norm, batch_size, augment=True
            )
            val_dataset = create_dual_dataset(
                val_before, val_after, val_weights_norm, batch_size, augment=False
            )

        else:
            # Simple dataset without augmentation or single input
            if self.use_dual_input:
                train_dataset = tf.data.Dataset.from_tensor_slices((
                    {'image_before': train_before, 'image_after': train_after},
                    train_weights_norm
                ))
                val_dataset = tf.data.Dataset.from_tensor_slices((
                    {'image_before': val_before, 'image_after': val_after},
                    val_weights_norm
                ))
            else:
                train_dataset = tf.data.Dataset.from_tensor_slices((
                    train_after, train_weights_norm
                ))
                val_dataset = tf.data.Dataset.from_tensor_slices((
                    val_after, val_weights_norm
                ))

            train_dataset = train_dataset.shuffle(len(train_weights_norm)).batch(batch_size)
            val_dataset = val_dataset.batch(batch_size)

        # Callbacks
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                config.REGRESSION_MODEL_PATH,
                save_best_only=True,
                monitor='val_loss',
                mode='min',
                verbose=1
            ),
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=config.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=config.REDUCE_LR_PATIENCE,
                verbose=1,
                min_lr=1e-7
            ),
            keras.callbacks.TensorBoard(
                log_dir=str(config.OUTPUTS_DIR / 'logs' / 'regression'),
                histogram_freq=1
            )
        ]

        # Train model
        logger.info("Starting regression training...")

        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )

        logger.info("Regression training completed!")

        return history

    def predict(self, image_before_paths: List[str] = None,
               image_after_paths: List[str] = None) -> np.ndarray:
        """
        Predict weights for given images.

        Args:
            image_before_paths: List of before-image paths (if dual input)
            image_after_paths: List of after-image paths

        Returns:
            Predicted weights
        """
        if self.use_dual_input:
            if image_before_paths is None or image_after_paths is None:
                raise ValueError("Both before and after images required for dual input model")

            images_before = self.load_and_preprocess_images(image_before_paths)
            images_after = self.load_and_preprocess_images(image_after_paths)
            images = [images_before, images_after]
        else:
            if image_after_paths is None:
                raise ValueError("After images required")

            images = self.load_and_preprocess_images(image_after_paths)

        # Predict
        predictions_norm = self.model.predict(images, verbose=0)

        # Denormalize
        predictions = self.denormalize_weights(predictions_norm.flatten())

        return predictions

    def evaluate(self, test_df: pd.DataFrame, predict_difference: bool = False) -> Dict:
        """
        Evaluate model on test set.

        Args:
            test_df: Test dataframe
            predict_difference: Whether model predicts difference or absolute weight

        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Evaluating regression model...")

        # Predict
        if self.use_dual_input:
            predictions = self.predict(
                test_df['image_before_path'].tolist(),
                test_df['image_after_path'].tolist()
            )
        else:
            predictions = self.predict(
                image_after_paths=test_df['image_after_path'].tolist()
            )

        # Get ground truth
        if predict_difference:
            ground_truth = test_df['weight_difference'].values
        else:
            ground_truth = test_df['Weight After Eaten (g)'].values

        # Calculate metrics
        mae = mean_absolute_error(ground_truth, predictions)
        mse = mean_squared_error(ground_truth, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(ground_truth, predictions)

        # MAPE
        mape = np.mean(np.abs((ground_truth - predictions) / (ground_truth + 1e-7))) * 100

        metrics = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'mape': mape
        }

        logger.info(f"Test MAE: {mae:.2f}g")
        logger.info(f"Test RMSE: {rmse:.2f}g")
        logger.info(f"Test R²: {r2:.4f}")
        logger.info(f"Test MAPE: {mape:.2f}%")

        return metrics

    def save_model(self, path: Path = config.REGRESSION_MODEL_PATH):
        """Save the trained model."""
        if self.model is not None:
            self.model.save(path)
            logger.info(f"Model saved to: {path}")

            # Save weight statistics
            import json
            stats_path = path.parent / "weight_stats.json"
            with open(stats_path, 'w') as f:
                json.dump(self.weight_stats, f)
            logger.info(f"Weight statistics saved to: {stats_path}")

    def load_model(self, path: Path = config.REGRESSION_MODEL_PATH):
        """Load a trained model."""
        self.model = keras.models.load_model(path)
        logger.info(f"Model loaded from: {path}")

        # Load weight statistics
        import json
        stats_path = path.parent / "weight_stats.json"
        if stats_path.exists():
            with open(stats_path, 'r') as f:
                self.weight_stats = json.load(f)
            logger.info(f"Weight statistics loaded from: {stats_path}")


def main():
    """Main function for testing."""
    from step1_data_reader import FoodDataReader

    # Load data
    reader = FoodDataReader()
    data = reader.process()

    # Initialize regression model
    regressor = WeightRegression(use_dual_input=True)

    # Train model
    history = regressor.train(
        data['train'].head(100),  # Use subset for testing
        data['val'].head(20),
        epochs=5,
        batch_size=16,
        use_augmentation=True,
        predict_difference=False
    )

    print(f"\nTraining completed!")
    print(f"Final training MAE: {history.history['mae'][-1]:.2f}g")
    print(f"Final validation MAE: {history.history['val_mae'][-1]:.2f}g")


if __name__ == "__main__":
    main()
