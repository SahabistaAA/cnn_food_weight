"""
Unit tests for individual modules.
"""
import unittest
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import tempfile
import shutil

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'modules'))

from modules import config
from modules.step1_data_reader import FoodDataReader
from modules.step2_segmentation import UNetSegmentation
from modules.step3_augmentation import DataAugmentation
from modules.step4_classification import FoodClassification
from modules.step5_regression import WeightRegression


class TestDataReader(unittest.TestCase):
    """Unit tests for data reading module."""

    def test_extract_food_category_id(self):
        """Test food category ID extraction from filename."""
        reader = FoodDataReader()

        test_cases = [
            ('001_002_DSC_0066_bef.JPG', '001'),
            ('099_001_IMG_1234_aft.jpg', '099'),
            ('042_003_photo.png', '042'),
        ]

        for filename, expected_id in test_cases:
            result = reader.extract_food_category_id(filename)
            self.assertEqual(result, expected_id)

    def test_compute_weight_difference(self):
        """Test weight difference computation."""
        reader = FoodDataReader()

        df = pd.DataFrame({
            'Weight Before Eaten (g)': [100, 200, 150],
            'Weight After Eaten (g)': [20, 50, 0]
        })

        result_df = reader.compute_weight_difference(df)

        self.assertIn('weight_difference', result_df.columns)
        self.assertIn('weight_consumed', result_df.columns)
        self.assertIn('consumption_ratio', result_df.columns)

        np.testing.assert_array_almost_equal(
            result_df['weight_consumed'].values,
            [80, 150, 150]
        )

        np.testing.assert_array_almost_equal(
            result_df['consumption_ratio'].values,
            [0.8, 0.75, 1.0]
        )


class TestSegmentation(unittest.TestCase):
    """Unit tests for segmentation module."""

    def test_unet_build(self):
        """Test U-Net model building (PyTorch)."""
        import torch
        segmenter = UNetSegmentation(img_size=128, filters=[32, 64, 128])

        # Build model
        unet_model = segmenter.build_unet()

        # Check model exists
        self.assertIsNotNone(unet_model)

        # Test forward pass
        unet_model.eval()
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 128, 128)
            output = unet_model(dummy_input)

        # Check output shape [B, C, H, W]
        self.assertEqual(output.shape, (1, 1, 128, 128))

    def test_create_pseudo_masks(self):
        """Test pseudo mask creation."""
        model = UNetSegmentation(img_size=128)

        # Create dummy images
        images = np.random.rand(5, 128, 128, 3).astype(np.float32)

        masks = model.create_pseudo_masks(images, method='otsu')

        # Check shape
        self.assertEqual(masks.shape, (5, 128, 128, 1))

        # Check value range
        self.assertTrue(np.all(masks >= 0))
        self.assertTrue(np.all(masks <= 1))

    def test_apply_mask_to_image(self):
        """Test mask application."""
        model = UNetSegmentation()

        image = np.random.rand(128, 128, 3).astype(np.float32)
        mask = np.random.rand(128, 128, 1).astype(np.float32)

        masked = model.apply_mask_to_image(image, mask, threshold=0.5)

        # Check shape
        self.assertEqual(masked.shape, image.shape)

        # Check that pixels are either 0 or original value
        self.assertTrue(np.all(masked <= image))


class TestAugmentation(unittest.TestCase):
    """Unit tests for augmentation module."""

    def test_augment_image_albumentations(self):
        """Test image augmentation with albumentations."""
        augmenter = DataAugmentation()

        image = np.random.rand(224, 224, 3).astype(np.float32)

        augmented = augmenter.augment_image(image, method='albumentations')

        # Check shape is preserved
        self.assertEqual(augmented.shape, image.shape)

        # Check value range
        self.assertTrue(np.all(augmented >= 0))
        self.assertTrue(np.all(augmented <= 1))

    def test_augment_batch(self):
        """Test batch augmentation."""
        augmenter = DataAugmentation()

        images = np.random.rand(10, 224, 224, 3).astype(np.float32)
        labels = np.random.randint(0, 5, size=(10,))

        aug_images, aug_labels = augmenter.augment_batch(
            images, labels, augment_factor=3
        )

        # Check size increase
        self.assertEqual(len(aug_images), len(images) * 3)
        self.assertEqual(len(aug_labels), len(labels) * 3)

    def test_visualize_augmentations(self):
        """Test augmentation visualization."""
        augmenter = DataAugmentation()

        image = np.random.rand(224, 224, 3).astype(np.float32)

        grid = augmenter.visualize_augmentations(image, num_augmentations=9)

        # Check grid shape
        expected_size = int(np.ceil(np.sqrt(9)))
        expected_shape = (expected_size * 224, expected_size * 224, 3)
        self.assertEqual(grid.shape, expected_shape)


class TestClassification(unittest.TestCase):
    """Unit tests for classification module."""

    def test_efficientnet_build(self):
        """Test EfficientNet model building (PyTorch)."""
        import torch
        model = FoodClassification(num_classes=10, img_size=224, pretrained=False)

        # Build model
        model.build_model()

        # Check model exists
        self.assertIsNotNone(model.model)

        # Test forward pass
        dummy_input = torch.randn(2, 3, 224, 224)
        output = model.model(dummy_input)

        # Check output shape [B, num_classes]
        self.assertEqual(output.shape, (2, 10))

    def test_prepare_labels(self):
        """Test label encoding."""
        model = FoodClassification(num_classes=5)

        food_ids = ['001', '002', '001', '003', '002', '001']
        labels = model.prepare_labels(food_ids)

        # Check encoded
        self.assertEqual(len(labels), len(food_ids))
        self.assertTrue(np.all(labels >= 0))
        self.assertTrue(np.all(labels < 3))  # 3 unique categories

        # Check consistency
        self.assertEqual(labels[0], labels[2])
        self.assertEqual(labels[0], labels[5])
        self.assertEqual(labels[1], labels[4])

    def test_get_class_mapping(self):
        """Test class mapping retrieval."""
        model = FoodClassification(num_classes=5)

        food_ids = ['001', '002', '003']
        model.prepare_labels(food_ids)

        mapping = model.get_class_mapping()

        # Check mapping exists
        self.assertEqual(len(mapping), 3)
        self.assertIn(0, mapping)
        self.assertIn(1, mapping)
        self.assertIn(2, mapping)


class TestRegression(unittest.TestCase):
    """Unit tests for regression module."""

    def test_cnn_regression_single_input_build(self):
        """Test CNN regression model building (single input, PyTorch)."""
        import torch
        model = WeightRegression(img_size=224, use_dual_input=False)

        # Build model
        model.build_model(pretrained=False)

        # Check model exists
        self.assertIsNotNone(model.model)

        # Test forward pass
        dummy_input = torch.randn(2, 3, 224, 224)
        output = model.model(dummy_input)

        # Check output shape [B, 1]
        self.assertEqual(output.shape, (2, 1))

    def test_cnn_regression_dual_input_build(self):
        """Test CNN regression model building (dual input, PyTorch)."""
        import torch
        model = WeightRegression(img_size=224, use_dual_input=True)

        # Build model
        model.build_model(pretrained=False)

        # Check model exists
        self.assertIsNotNone(model.model)

        # Test forward pass
        dummy_before = torch.randn(2, 3, 224, 224)
        dummy_after = torch.randn(2, 3, 224, 224)
        output = model.model(dummy_before, dummy_after)

        # Check output shape [B, 1]
        self.assertEqual(output.shape, (2, 1))

    def test_normalize_denormalize_weights(self):
        """Test weight normalization and denormalization."""
        model = WeightRegression()

        weights = np.array([100, 200, 150, 50, 300])

        # Normalize
        normalized = model.normalize_weights(weights, fit=True)

        # Check statistics were saved
        self.assertIn('mean', model.weight_stats)
        self.assertIn('std', model.weight_stats)

        # Denormalize
        denormalized = model.denormalize_weights(normalized)

        # Check roundtrip
        np.testing.assert_array_almost_equal(weights, denormalized, decimal=5)


class TestIntegration(unittest.TestCase):
    """Integration tests across modules."""

    def test_data_flow(self):
        """Test data flow through multiple modules."""
        # Create dummy dataframe
        df = pd.DataFrame({
            'food_category_id': ['001', '002', '001'],
            'image_before_path': ['dummy1.jpg', 'dummy2.jpg', 'dummy3.jpg'],
            'image_after_path': ['dummy1.jpg', 'dummy2.jpg', 'dummy3.jpg'],
            'Weight Before Eaten (g)': [100, 200, 150],
            'Weight After Eaten (g)': [20, 50, 30]
        })

        # Test classification label encoding
        classifier = FoodClassification(num_classes=2)
        labels = classifier.prepare_labels(df['food_category_id'].tolist())

        self.assertEqual(len(labels), len(df))

        # Test regression weight normalization
        regressor = WeightRegression()
        normalized_weights = regressor.normalize_weights(
            df['Weight After Eaten (g)'].values,
            fit=True
        )

        self.assertEqual(len(normalized_weights), len(df))


def run_unit_tests():
    """Run all unit tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestDataReader))
    suite.addTests(loader.loadTestsFromTestCase(TestSegmentation))
    suite.addTests(loader.loadTestsFromTestCase(TestAugmentation))
    suite.addTests(loader.loadTestsFromTestCase(TestClassification))
    suite.addTests(loader.loadTestsFromTestCase(TestRegression))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_unit_tests()
    sys.exit(0 if success else 1)
