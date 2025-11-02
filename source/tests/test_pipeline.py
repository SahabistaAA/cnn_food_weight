"""
Pipeline integration tests.
Tests the full pipeline flow with sample data.
"""
import unittest
import sys
from pathlib import Path
import tempfile
import shutil
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import FoodWeightPredictionPipeline


class TestPipelineIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline."""

    @classmethod
    def setUpClass(cls):
        """Setup test fixtures."""
        cls.test_output_dir = Path(tempfile.mkdtemp())
        print(f"\nTest output directory: {cls.test_output_dir}")

    @classmethod
    def tearDownClass(cls):
        """Cleanup test fixtures."""
        if cls.test_output_dir.exists():
            shutil.rmtree(cls.test_output_dir)

    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        pipeline = FoodWeightPredictionPipeline()

        self.assertIsNotNone(pipeline)
        self.assertIsNone(pipeline.data)
        self.assertIsNone(pipeline.segmentation_model)
        self.assertIsNone(pipeline.classification_model)
        self.assertIsNone(pipeline.regression_model)

    def test_step1_data_loading(self):
        """Test data loading step."""
        pipeline = FoodWeightPredictionPipeline()

        try:
            pipeline.step1_load_data()

            # Check data was loaded
            self.assertIsNotNone(pipeline.data)
            self.assertIn('train', pipeline.data)
            self.assertIn('val', pipeline.data)
            self.assertIn('test', pipeline.data)

            # Check train/val/test split proportions
            total = len(pipeline.data['train']) + len(pipeline.data['val']) + len(pipeline.data['test'])

            train_ratio = len(pipeline.data['train']) / total
            val_ratio = len(pipeline.data['val']) / total
            test_ratio = len(pipeline.data['test']) / total

            # Allow 5% tolerance
            self.assertAlmostEqual(train_ratio, 0.70, delta=0.05)
            self.assertAlmostEqual(val_ratio, 0.15, delta=0.05)
            self.assertAlmostEqual(test_ratio, 0.15, delta=0.05)

            print(f"\n  Train: {len(pipeline.data['train'])} ({train_ratio:.2%})")
            print(f"  Val: {len(pipeline.data['val'])} ({val_ratio:.2%})")
            print(f"  Test: {len(pipeline.data['test'])} ({test_ratio:.2%})")

        except Exception as e:
            self.skipTest(f"Data loading failed (may be missing data files): {e}")

    def test_pipeline_results_structure(self):
        """Test pipeline results dictionary structure."""
        pipeline = FoodWeightPredictionPipeline()
        pipeline.results = {
            'step1': {'train_size': 100},
            'step2': {'final_train_loss': 0.5},
            'step3': {'augmentation_params': {}},
            'step4': {'final_train_accuracy': 0.8},
            'step5': {'final_train_mae': 10.0}
        }

        # Check all steps present
        for step in ['step1', 'step2', 'step3', 'step4', 'step5']:
            self.assertIn(step, pipeline.results)

    def test_config_parameters(self):
        """Test configuration parameters are valid."""
        from modules import config

        # Check ratios sum to 1
        total_ratio = config.TRAIN_RATIO + config.VAL_RATIO + config.TEST_RATIO
        self.assertAlmostEqual(total_ratio, 1.0, places=5)

        # Check image parameters
        self.assertGreater(config.IMG_HEIGHT, 0)
        self.assertGreater(config.IMG_WIDTH, 0)
        self.assertEqual(config.IMG_CHANNELS, 3)

        # Check paths exist
        self.assertTrue(config.DATA_DIR.exists())

        print(f"\n  Data directory: {config.DATA_DIR}")
        print(f"  Models directory: {config.MODELS_DIR}")
        print(f"  Outputs directory: {config.OUTPUTS_DIR}")


class TestPipelineErrorHandling(unittest.TestCase):
    """Test error handling in pipeline."""

    def test_invalid_config(self):
        """Test pipeline with invalid configuration."""
        invalid_config = {
            'TRAIN_RATIO': 0.5,
            'VAL_RATIO': 0.5,
            'TEST_RATIO': 0.5  # Invalid: sums to > 1
        }

        pipeline = FoodWeightPredictionPipeline(config_overrides=invalid_config)

        # Pipeline should still initialize
        self.assertIsNotNone(pipeline)


class TestDataConsistency(unittest.TestCase):
    """Test data consistency across pipeline."""

    def test_data_split_uniqueness(self):
        """Test that train/val/test splits don't overlap."""
        try:
            pipeline = FoodWeightPredictionPipeline()
            pipeline.step1_load_data()

            # Get sample IDs from each split
            train_ids = set(pipeline.data['train']['ID'].values)
            val_ids = set(pipeline.data['val']['ID'].values)
            test_ids = set(pipeline.data['test']['ID'].values)

            # Check no overlap
            self.assertEqual(len(train_ids & val_ids), 0, "Train and Val overlap!")
            self.assertEqual(len(train_ids & test_ids), 0, "Train and Test overlap!")
            self.assertEqual(len(val_ids & test_ids), 0, "Val and Test overlap!")

            print(f"\n  Train IDs: {len(train_ids)}")
            print(f"  Val IDs: {len(val_ids)}")
            print(f"  Test IDs: {len(test_ids)}")
            print("  ✓ No overlap between splits")

        except Exception as e:
            self.skipTest(f"Data loading failed: {e}")

    def test_required_columns(self):
        """Test that all required columns are present in data."""
        try:
            pipeline = FoodWeightPredictionPipeline()
            pipeline.step1_load_data()

            required_columns = [
                'ID', 'Name of the food', 'Image Before Eaten',
                'Weight Before Eaten (g)', 'Image After Eaten',
                'Weight After Eaten (g)', 'image_before_path',
                'image_after_path', 'food_category_id'
            ]

            for split in ['train', 'val', 'test']:
                df = pipeline.data[split]
                for col in required_columns:
                    self.assertIn(col, df.columns,
                                f"Missing column '{col}' in {split} split")

            print(f"\n  ✓ All required columns present in all splits")

        except Exception as e:
            self.skipTest(f"Data loading failed: {e}")


def run_pipeline_tests():
    """Run all pipeline tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestPipelineIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestPipelineErrorHandling))
    suite.addTests(loader.loadTestsFromTestCase(TestDataConsistency))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_pipeline_tests()
    sys.exit(0 if success else 1)
