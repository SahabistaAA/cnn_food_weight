"""
Step 1: Data Reading and Preprocessing
Reads the Excel file and organizes image-weight pairs with train/val/test split.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict
from loguru import logger
import config


class FoodDataReader:
    """Handles reading and preprocessing of food weight dataset."""

    def __init__(self, excel_path: Path = config.EXCEL_PATH,
                 images_before_dir: Path = config.IMAGES_BEFORE_DIR,
                 images_after_dir: Path = config.IMAGES_AFTER_DIR):
        """
        Initialize the data reader.

        Args:
            excel_path: Path to the Excel file with metadata
            images_before_dir: Directory containing before-eating images
            images_after_dir: Directory containing after-eating images
        """
        self.excel_path = excel_path
        self.images_before_dir = images_before_dir
        self.images_after_dir = images_after_dir
        self.df = None

    def read_excel(self) -> pd.DataFrame:
        """Read and validate the Excel file."""
        logger.info(f"Reading Excel file from: {self.excel_path}")

        try:
            self.df = pd.read_excel(self.excel_path)
            logger.info(f"Successfully read {len(self.df)} records")
            logger.info(f"Columns: {self.df.columns.tolist()}")

            # Validate required columns
            required_columns = [
                'ID', 'Name of the food', 'Image Before Eaten',
                'Weight Before Eaten (g)', 'Image After Eaten',
                'Weight After Eaten (g)', 'Visual Estimation by Observer (1-7)'
            ]

            missing_cols = set(required_columns) - set(self.df.columns)
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            return self.df

        except Exception as e:
            logger.error(f"Error reading Excel file: {e}")
            raise

    def extract_food_category_id(self, image_filename: str) -> str:
        """
        Extract the food category ID from image filename.
        Example: '001_002_DSC_0066_bef.JPG' -> '001'

        Args:
            image_filename: The image filename

        Returns:
            Three-digit food category ID
        """
        return image_filename[:3]

    def validate_image_paths(self) -> pd.DataFrame:
        """Validate that image files exist and add full paths to dataframe."""
        logger.info("Validating image paths...")

        valid_rows = []
        missing_before = []
        missing_after = []

        for idx, row in self.df.iterrows():
            # Extract food category ID from image filename
            food_category_id = self.extract_food_category_id(row['Image Before Eaten'])

            # Construct full paths
            before_path = self.images_before_dir / food_category_id / row['Image Before Eaten']
            after_path = self.images_after_dir / food_category_id / row['Image After Eaten']

            # Check if files exist
            before_exists = before_path.exists()
            after_exists = after_path.exists()

            if before_exists and after_exists:
                row['image_before_path'] = str(before_path)
                row['image_after_path'] = str(after_path)
                row['food_category_id'] = food_category_id
                valid_rows.append(row)
            else:
                if not before_exists:
                    missing_before.append(str(before_path))
                if not after_exists:
                    missing_after.append(str(after_path))

        if missing_before or missing_after:
            logger.warning(f"Missing {len(missing_before)} before images")
            logger.warning(f"Missing {len(missing_after)} after images")

        validated_df = pd.DataFrame(valid_rows)
        logger.info(f"Valid image pairs: {len(validated_df)}/{len(self.df)}")

        return validated_df

    def compute_weight_difference(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute the weight difference (leftover weight)."""
        df['weight_difference'] = df['Weight After Eaten (g)']
        df['weight_consumed'] = df['Weight Before Eaten (g)'] - df['Weight After Eaten (g)']
        df['consumption_ratio'] = df['weight_consumed'] / df['Weight Before Eaten (g)']

        return df

    def create_train_val_test_split(self, df: pd.DataFrame,
                                     train_ratio: float = config.TRAIN_RATIO,
                                     val_ratio: float = config.VAL_RATIO,
                                     test_ratio: float = config.TEST_RATIO,
                                     random_state: int = config.RANDOM_SEED) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.

        Args:
            df: Input dataframe
            train_ratio: Proportion for training set (default: 0.70)
            val_ratio: Proportion for validation set (default: 0.15)
            test_ratio: Proportion for test set (default: 0.15)
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, \
            "Ratios must sum to 1.0"

        # Check class distribution
        class_counts = df['food_category_id'].value_counts()
        min_samples = class_counts.min()

        logger.info(f"Class distribution: min={min_samples}, max={class_counts.max()}, "
                   f"unique_classes={len(class_counts)}")

        # Filter out classes with too few samples for stratification
        # Need at least 2 samples per class for stratification
        valid_classes = class_counts[class_counts >= 2].index
        classes_to_remove = class_counts[class_counts < 2].index

        if len(classes_to_remove) > 0:
            logger.warning(f"Removing {len(classes_to_remove)} classes with only 1 sample: "
                          f"{classes_to_remove.tolist()}")
            df_filtered = df[df['food_category_id'].isin(valid_classes)].copy()
            logger.info(f"Filtered dataset: {len(df_filtered)}/{len(df)} samples remaining")
        else:
            df_filtered = df.copy()

        # Determine if we can use stratification
        # Need at least 2 samples per class for stratified split
        can_stratify = df_filtered['food_category_id'].value_counts().min() >= 2

        if can_stratify:
            logger.info("Using stratified split by food category")
            stratify_col = df_filtered['food_category_id']
        else:
            logger.warning("Cannot use stratified split - some classes have too few samples")
            logger.warning("Using random split instead")
            stratify_col = None

        # First split: separate out test set
        train_val_df, test_df = train_test_split(
            df_filtered,
            test_size=test_ratio,
            random_state=random_state,
            stratify=stratify_col
        )

        # Second split: separate train and validation
        val_size_adjusted = val_ratio / (train_ratio + val_ratio)

        # Re-check if we can stratify for second split
        if can_stratify and train_val_df['food_category_id'].value_counts().min() >= 2:
            stratify_col_2 = train_val_df['food_category_id']
        else:
            stratify_col_2 = None

        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=stratify_col_2
        )

        logger.info(f"Data split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        logger.info(f"Percentages - Train: {len(train_df)/len(df_filtered)*100:.1f}%, "
                   f"Val: {len(val_df)/len(df_filtered)*100:.1f}%, "
                   f"Test: {len(test_df)/len(df_filtered)*100:.1f}%")

        # Add split column
        train_df['split'] = 'train'
        val_df['split'] = 'val'
        test_df['split'] = 'test'

        return train_df, val_df, test_df

    def get_food_name_mapping(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Create a mapping from food category ID to food name.

        Args:
            df: Input dataframe

        Returns:
            Dictionary mapping category ID to food name
        """
        mapping = df.groupby('food_category_id')['Name of the food'].first().to_dict()
        logger.info(f"Found {len(mapping)} unique food categories")
        return mapping

    def save_split_data(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
                       test_df: pd.DataFrame, output_path: Path = config.TRAIN_VAL_TEST_SPLIT_PATH):
        """Save the split data to CSV for reproducibility."""
        combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        combined_df.to_csv(output_path, index=False)
        logger.info(f"Saved data split to: {output_path}")

    def process(self) -> Dict[str, pd.DataFrame]:
        """
        Execute the complete data reading pipeline.

        Returns:
            Dictionary with keys 'train', 'val', 'test' containing respective DataFrames
        """
        # Read Excel
        self.read_excel()

        # Validate image paths
        validated_df = self.validate_image_paths()

        # Compute weight differences
        validated_df = self.compute_weight_difference(validated_df)

        # Create train/val/test split
        train_df, val_df, test_df = self.create_train_val_test_split(validated_df)

        # Save split information
        self.save_split_data(train_df, val_df, test_df)

        # Get food name mapping
        food_mapping = self.get_food_name_mapping(validated_df)

        logger.info("Data reading and preprocessing completed successfully!")

        return {
            'train': train_df,
            'val': val_df,
            'test': test_df,
            'food_mapping': food_mapping,
            'all_data': pd.concat([train_df, val_df, test_df], ignore_index=True)
        }


def main():
    """Main function for testing."""
    reader = FoodDataReader()
    data = reader.process()

    print(f"\nTrain samples: {len(data['train'])}")
    print(f"Val samples: {len(data['val'])}")
    print(f"Test samples: {len(data['test'])}")
    print(f"\nFood categories: {len(data['food_mapping'])}")
    print(f"\nSample data:\n{data['train'].head()}")


if __name__ == "__main__":
    main()
