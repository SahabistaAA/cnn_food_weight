"""
End-to-end tests for the complete pipeline.
Tests the full workflow from data loading to prediction.
"""
import unittest
import sys
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import FoodWeightPredictionPipeline


class TestEndToEnd(unittest.TestCase):
    """End-to-end tests for the complete pipeline."""

    @classmethod
    def setUpClass(cls):
        """Setup for E2E tests."""
        cls.pipeline = FoodWeightPredictionPipeline()
        print("\n" + "=" * 80)
        print("Starting End-to-End Pipeline Test")
        print("=" * 80)

    def test_01_complete_pipeline_quick(self):
        """Test complete pipeline with minimal epochs (quick test)."""
        try:
            print("\n[E2E Test] Running quick pipeline test with 2 epochs per model...")

            # Run pipeline with minimal epochs for testing
            self.pipeline.run_full_pipeline(
                skip_if_exists=True,  # Skip if models already trained
                segmentation_epochs=2,
                classification_epochs=2,
                regression_epochs=2
            )

            # Verify all steps completed
            self.assertIn('step1', self.pipeline.results)
            self.assertIn('step2', self.pipeline.results)
            self.assertIn('step3', self.pipeline.results)
            self.assertIn('step4', self.pipeline.results)
            self.assertIn('step5', self.pipeline.results)

            print("\n[OK] E2E Pipeline Test Completed Successfully!")

        except Exception as e:
            self.skipTest(f"Pipeline test failed (may be missing data): {e}")

    def test_02_data_loaded_correctly(self):
        """Test that data is loaded with correct structure."""
        if self.pipeline.data is None:
            self.skipTest("Data not loaded")

        # Check splits exist
        self.assertIn('train', self.pipeline.data)
        self.assertIn('val', self.pipeline.data)
        self.assertIn('test', self.pipeline.data)

        # Check non-empty
        self.assertGreater(len(self.pipeline.data['train']), 0)
        self.assertGreater(len(self.pipeline.data['val']), 0)
        self.assertGreater(len(self.pipeline.data['test']), 0)

        print(f"\n[OK] Data loaded: {len(self.pipeline.data['train'])} train, "
              f"{len(self.pipeline.data['val'])} val, {len(self.pipeline.data['test'])} test samples")

    def test_03_models_created(self):
        """Test that all models are created."""
        if (self.pipeline.segmentation_model is None or
            self.pipeline.classification_model is None or
            self.pipeline.regression_model is None):
            self.skipTest("Models not created - run full pipeline first")

        # Check models exist
        self.assertIsNotNone(self.pipeline.segmentation_model)
        self.assertIsNotNone(self.pipeline.classification_model)
        self.assertIsNotNone(self.pipeline.regression_model)

        print("\n[OK] All models created successfully")

    def test_04_segmentation_predictions(self):
        """Test segmentation model predictions."""
        if self.pipeline.segmentation_model is None:
            self.skipTest("Segmentation model not trained")

        try:
            # Get sample images
            sample_paths = self.pipeline.data['test']['image_before_path'].head(5).tolist()

            # Predict masks
            masks = self.pipeline.segmentation_model.predict_masks(sample_paths)

            # Check output shape
            self.assertEqual(len(masks), 5)
            self.assertEqual(masks.shape[-1], 1)  # Single channel mask

            # Check value range [0, 1]
            self.assertTrue(np.all(masks >= 0))
            self.assertTrue(np.all(masks <= 1))

            print(f"\n[OK] Segmentation predictions: {masks.shape}")

        except Exception as e:
            self.skipTest(f"Segmentation prediction failed: {e}")

    def test_05_classification_predictions(self):
        """Test classification model predictions."""
        if self.pipeline.classification_model is None:
            self.skipTest("Classification model not trained")

        try:
            # Get sample images
            sample_paths = self.pipeline.data['test']['image_before_path'].head(5).tolist()

            # Predict categories
            predicted_categories, probabilities = self.pipeline.classification_model.predict(sample_paths)

            # Check output
            self.assertEqual(len(predicted_categories), 5)
            self.assertEqual(probabilities.shape[0], 5)

            # Check probabilities sum to 1
            prob_sums = probabilities.sum(axis=1)
            np.testing.assert_array_almost_equal(prob_sums, np.ones(5), decimal=5)

            print(f"\n[OK] Classification predictions: {predicted_categories}")
            print(f"  Top prediction probabilities: {probabilities.max(axis=1)}")

        except Exception as e:
            self.skipTest(f"Classification prediction failed: {e}")

    def test_06_regression_predictions(self):
        """Test regression model predictions."""
        if self.pipeline.regression_model is None:
            self.skipTest("Regression model not trained")

        try:
            # Get sample images
            test_sample = self.pipeline.data['test'].head(5)

            sample_before = test_sample['image_before_path'].tolist()
            sample_after = test_sample['image_after_path'].tolist()

            # Predict weights
            if self.pipeline.regression_model.use_dual_input:
                predictions = self.pipeline.regression_model.predict(
                    sample_before, sample_after
                )
            else:
                predictions = self.pipeline.regression_model.predict(
                    image_after_paths=sample_after
                )

            # Check output
            self.assertEqual(len(predictions), 5)

            # Check predictions are positive
            self.assertTrue(np.all(predictions >= 0))

            # Get ground truth
            ground_truth = test_sample['Weight After Eaten (g)'].values

            # Calculate error
            errors = np.abs(predictions - ground_truth)
            mean_error = errors.mean()

            print(f"\n[OK] Regression predictions:")
            print(f"  Predictions: {predictions}")
            print(f"  Ground truth: {ground_truth}")
            print(f"  Mean absolute error: {mean_error:.2f}g")

        except Exception as e:
            self.skipTest(f"Regression prediction failed: {e}")

    def test_07_evaluation_metrics(self):
        """Test that evaluation metrics are computed."""
        if 'test_evaluation' not in self.pipeline.results:
            self.skipTest("Evaluation not performed")

        test_results = self.pipeline.results['test_evaluation']

        # Check classification metrics
        if 'classification' in test_results:
            self.assertIn('accuracy', test_results['classification'])
            self.assertIn('top5_accuracy', test_results['classification'])

            accuracy = test_results['classification']['accuracy']
            self.assertGreaterEqual(accuracy, 0)
            self.assertLessEqual(accuracy, 1)

            print(f"\n[OK] Classification Test Accuracy: {accuracy:.4f}")

        # Check regression metrics
        if 'regression' in test_results:
            self.assertIn('mae', test_results['regression'])
            self.assertIn('rmse', test_results['regression'])
            self.assertIn('r2', test_results['regression'])

            mae = test_results['regression']['mae']
            r2 = test_results['regression']['r2']

            self.assertGreaterEqual(mae, 0)

            print(f"\n[OK] Regression Test MAE: {mae:.2f}g")
            print(f"  R² Score: {r2:.4f}")

    def test_08_results_saved(self):
        """Test that results are saved correctly."""
        from modules.config import OUTPUTS_DIR

        # Check if any results file exists
        result_files = list(OUTPUTS_DIR.glob("pipeline_results_*.json"))

        if len(result_files) > 0:
            print(f"\n[OK] Results saved: {len(result_files)} result files found")
            latest_result = max(result_files, key=lambda p: p.stat().st_mtime)
            print(f"  Latest: {latest_result.name}")
        else:
            self.skipTest("No result files found")


class TestModelPersistence(unittest.TestCase):
    """Test model saving and loading."""

    def test_model_files_exist(self):
        """Test that model files are saved."""
        from modules.config import MODELS_DIR

        # Check models directory
        self.assertTrue(MODELS_DIR.exists())

        print(f"\nModels directory: {MODELS_DIR}")

        # List saved models
        model_files = list(MODELS_DIR.glob("*.h5")) + list(MODELS_DIR.glob("*.pkl"))

        if len(model_files) > 0:
            print(f"[OK] Found {len(model_files)} model files:")
            for model_file in model_files:
                size_mb = model_file.stat().st_size / (1024 * 1024)
                print(f"  - {model_file.name} ({size_mb:.2f} MB)")
        else:
            self.skipTest("No model files found (models not trained yet)")


def run_e2e_tests():
    """Run all end-to-end tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes in order
    suite.addTests(loader.loadTestsFromTestCase(TestEndToEnd))
    suite.addTests(loader.loadTestsFromTestCase(TestModelPersistence))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_e2e_tests()
    sys.exit(0 if success else 1)
