"""
Step 4: Food Classification using EfficientNet
Classifies food images into different food categories.
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
from sklearn.preprocessing import LabelEncoder
import logging
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FoodClassification:
    """EfficientNet-based food classification model."""

    def __init__(self, num_classes: int, img_size: int = config.IMG_HEIGHT,
                 pretrained: bool = True):
        """
        Initialize food classification model.

        Args:
            num_classes: Number of food categories
            img_size: Input image size
            pretrained: Whether to use ImageNet pretrained weights
        """
        self.num_classes = num_classes
        self.img_size = img_size
        self.pretrained = pretrained
        self.model = None
        self.label_encoder = LabelEncoder()

    def build_efficientnet(self) -> keras.Model:
        """
        Build EfficientNet-B0 model for classification.

        Returns:
            Keras model
        """
        # Input layer
        inputs = layers.Input(shape=(self.img_size, self.img_size, 3))

        # Load EfficientNetB0 as base model
        base_model = EfficientNetB0(
            include_top=False,
            weights='imagenet' if self.pretrained else None,
            input_tensor=inputs,
            pooling='avg'
        )

        # Freeze base model initially for transfer learning
        base_model.trainable = False

        # Build classification head
        x = base_model.output
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)

        # Dense layers
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)

        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)

        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax', name='classification_output')(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name='EfficientNet_Classification')

        logger.info(f"EfficientNet model built with {self.num_classes} classes")
        logger.info(f"Total parameters: {model.count_params():,}")

        return model

    def compile_model(self, learning_rate: float = config.CLASSIFICATION_LEARNING_RATE):
        """Compile the classification model."""
        if self.model is None:
            self.model = self.build_efficientnet()

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=5, name='top5_accuracy')]
        )

        logger.info("Classification model compiled successfully")

    def unfreeze_base_model(self, num_layers_to_unfreeze: int = 20):
        """
        Unfreeze layers of the base model for fine-tuning.

        Args:
            num_layers_to_unfreeze: Number of layers to unfreeze from the end
        """
        if self.model is None:
            raise ValueError("Model not built yet!")

        # Get the base model (EfficientNet)
        base_model = self.model.layers[1]  # Assuming EfficientNet is second layer

        # Unfreeze the last N layers
        for layer in base_model.layers[-num_layers_to_unfreeze:]:
            layer.trainable = True

        # Recompile with lower learning rate for fine-tuning
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-5),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=5, name='top5_accuracy')]
        )

        logger.info(f"Unfroze last {num_layers_to_unfreeze} layers for fine-tuning")

    def load_and_preprocess_images(self, image_paths: List[str]) -> np.ndarray:
        """
        Load and preprocess images for classification.

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

    def prepare_labels(self, food_category_ids: List[str]) -> np.ndarray:
        """
        Encode food category IDs to numerical labels.

        Args:
            food_category_ids: List of food category IDs (e.g., ['001', '002', ...])

        Returns:
            Encoded labels array
        """
        labels = self.label_encoder.fit_transform(food_category_ids)
        logger.info(f"Encoded {len(np.unique(labels))} unique food categories")
        return labels

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
              epochs: int = config.CLASSIFICATION_EPOCHS,
              batch_size: int = config.CLASSIFICATION_BATCH_SIZE,
              use_augmentation: bool = True,
              fine_tune: bool = True,
              fine_tune_epochs: int = 20):
        """
        Train the classification model.

        Args:
            train_df: Training dataframe
            val_df: Validation dataframe
            epochs: Number of training epochs for initial training
            batch_size: Batch size
            use_augmentation: Whether to use data augmentation
            fine_tune: Whether to perform fine-tuning after initial training
            fine_tune_epochs: Number of epochs for fine-tuning
        """
        logger.info("Preparing training data...")

        # Use 'before' images for classification (food type doesn't change)
        train_images = self.load_and_preprocess_images(train_df['image_before_path'].tolist())
        val_images = self.load_and_preprocess_images(val_df['image_before_path'].tolist())

        # Prepare labels
        train_labels = self.prepare_labels(train_df['food_category_id'].tolist())
        val_labels = self.label_encoder.transform(val_df['food_category_id'].tolist())

        logger.info(f"Training images: {train_images.shape}")
        logger.info(f"Training labels: {train_labels.shape}")
        logger.info(f"Number of classes: {self.num_classes}")

        # Compile model if not already compiled
        if self.model is None:
            self.compile_model()

        # Data augmentation
        if use_augmentation:
            from step3_augmentation import DataAugmentation
            augmenter = DataAugmentation()

            train_dataset = augmenter.create_tf_dataset(
                train_images, train_labels,
                batch_size=batch_size,
                shuffle=True,
                augment=True
            )

            val_dataset = augmenter.create_tf_dataset(
                val_images, val_labels,
                batch_size=batch_size,
                shuffle=False,
                augment=False
            )
        else:
            train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
            train_dataset = train_dataset.shuffle(len(train_images)).batch(batch_size)

            val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
            val_dataset = val_dataset.batch(batch_size)

        # Callbacks
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                config.CLASSIFICATION_MODEL_PATH,
                save_best_only=True,
                monitor='val_accuracy',
                mode='max',
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
                log_dir=str(config.OUTPUTS_DIR / 'logs' / 'classification'),
                histogram_freq=1
            )
        ]

        # Phase 1: Train with frozen base model
        logger.info("Phase 1: Training with frozen base model...")

        history1 = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )

        # Phase 2: Fine-tuning
        if fine_tune:
            logger.info("\nPhase 2: Fine-tuning with unfrozen layers...")

            self.unfreeze_base_model(num_layers_to_unfreeze=30)

            history2 = self.model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=fine_tune_epochs,
                callbacks=callbacks,
                verbose=1
            )

            # Combine histories
            for key in history1.history.keys():
                history1.history[key].extend(history2.history[key])

        logger.info("Classification training completed!")

        return history1

    def predict(self, image_paths: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict food categories for given images.

        Args:
            image_paths: List of image paths

        Returns:
            Tuple of (predicted_labels, prediction_probabilities)
        """
        images = self.load_and_preprocess_images(image_paths)
        predictions = self.model.predict(images, verbose=0)

        predicted_labels = np.argmax(predictions, axis=1)
        predicted_categories = self.label_encoder.inverse_transform(predicted_labels)

        return predicted_categories, predictions

    def evaluate(self, test_df: pd.DataFrame, batch_size: int = 32) -> Dict:
        """
        Evaluate model on test set.

        Args:
            test_df: Test dataframe
            batch_size: Batch size for evaluation

        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Evaluating classification model...")

        test_images = self.load_and_preprocess_images(test_df['image_before_path'].tolist())
        test_labels = self.label_encoder.transform(test_df['food_category_id'].tolist())

        # Evaluate
        results = self.model.evaluate(test_images, test_labels, batch_size=batch_size, verbose=1)

        metrics = {
            'loss': results[0],
            'accuracy': results[1],
            'top5_accuracy': results[2]
        }

        logger.info(f"Test Loss: {metrics['loss']:.4f}")
        logger.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Test Top-5 Accuracy: {metrics['top5_accuracy']:.4f}")

        return metrics

    def get_class_mapping(self) -> Dict[int, str]:
        """Get mapping from encoded labels to food category IDs."""
        return {i: label for i, label in enumerate(self.label_encoder.classes_)}

    def save_model(self, path: Path = config.CLASSIFICATION_MODEL_PATH):
        """Save the trained model."""
        if self.model is not None:
            self.model.save(path)
            logger.info(f"Model saved to: {path}")

            # Save label encoder
            import joblib
            encoder_path = path.parent / "label_encoder.pkl"
            joblib.dump(self.label_encoder, encoder_path)
            logger.info(f"Label encoder saved to: {encoder_path}")

    def load_model(self, path: Path = config.CLASSIFICATION_MODEL_PATH):
        """Load a trained model."""
        self.model = keras.models.load_model(path)
        logger.info(f"Model loaded from: {path}")

        # Load label encoder
        import joblib
        encoder_path = path.parent / "label_encoder.pkl"
        if encoder_path.exists():
            self.label_encoder = joblib.load(encoder_path)
            logger.info(f"Label encoder loaded from: {encoder_path}")


def main():
    """Main function for testing."""
    from step1_data_reader import FoodDataReader

    # Load data
    reader = FoodDataReader()
    data = reader.process()

    # Get number of unique food categories
    num_classes = data['train']['food_category_id'].nunique()
    print(f"Number of food categories: {num_classes}")

    # Initialize classification model
    classifier = FoodClassification(num_classes=num_classes)

    # Train model
    history = classifier.train(
        data['train'].head(100),  # Use subset for testing
        data['val'].head(20),
        epochs=5,
        batch_size=16,
        use_augmentation=True,
        fine_tune=False
    )

    print(f"\nTraining completed!")
    print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")


if __name__ == "__main__":
    main()
