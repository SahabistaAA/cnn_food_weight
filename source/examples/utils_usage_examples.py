# -*- coding: utf-8 -*-
"""
Examples of using the reorganized utils package.

This file demonstrates how to use all the utility functions
from the modular utils package structure.
"""
import sys
from pathlib import Path
import numpy as np

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# EXAMPLE 1: Using Visualization Utilities
# ============================================================================

def example_visualization():
    """Examples of using visualization utilities."""
    print("=" * 80)
    print("EXAMPLE 1: Visualization Utilities")
    print("=" * 80)

    # Method 1: Import specific functions
    from utils.visualization import visualize_image_pair, plot_training_history

    # Method 2: Import from utils package directly
    from utils import plot_regression_results, plot_confusion_matrix

    # Example: Visualize image pair (requires actual image files)
    # visualize_image_pair(
    #     before_path='data/leftover_dataset/data_before/001/001.jpg',
    #     after_path='data/leftover_dataset/data_after/001/001.jpg',
    #     before_weight=500.0,
    #     after_weight=350.0,
    #     predicted_weight=345.0,
    #     save_path='outputs/prediction_example.png'
    # )

    # Example: Plot training history
    # dummy_history = {
    #     'loss': [0.5, 0.4, 0.3, 0.2],
    #     'val_loss': [0.6, 0.5, 0.4, 0.35],
    #     'accuracy': [0.7, 0.75, 0.8, 0.85],
    #     'val_accuracy': [0.65, 0.7, 0.75, 0.78]
    # }
    # plot_training_history(dummy_history, save_path='outputs/training_history.png')

    # Example: Plot regression results
    # y_true = np.array([100, 150, 200, 250, 300])
    # y_pred = np.array([105, 145, 210, 245, 295])
    # plot_regression_results(y_true, y_pred, save_path='outputs/regression.png')

    print("\nVisualization functions available:")
    print("  - visualize_image_pair()")
    print("  - visualize_segmentation()")
    print("  - visualize_augmentations()")
    print("  - plot_training_history()")
    print("  - plot_regression_results()")
    print("  - plot_confusion_matrix()")
    print("  - create_inference_report()")
    print("\n[OK] All visualization utilities imported successfully!")


# ============================================================================
# EXAMPLE 2: Using Results Utilities
# ============================================================================

def example_results():
    """Examples of using result handling utilities."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Results Utilities")
    print("=" * 80)

    from utils.results import load_results, print_results_summary

    # Example: Load results from JSON file
    # results_file = Path('outputs/pipeline_results_20251102_123456.json')
    # if results_file.exists():
    #     results = load_results(results_file)
    #     print_results_summary(results)

    print("\nResults functions available:")
    print("  - load_results(path)")
    print("  - print_results_summary(results)")
    print("\n[OK] All results utilities imported successfully!")


# ============================================================================
# EXAMPLE 3: Using Testing Utilities
# ============================================================================

def example_testing():
    """Examples of using testing utilities."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Testing Utilities")
    print("=" * 80)

    from utils.testing import (
        run_all_tests,
        run_unit_tests_only,
        run_pipeline_tests_only,
        run_e2e_tests_only
    )

    # Example: Run specific test suite
    # success = run_unit_tests_only()
    # if success:
    #     print("Unit tests passed!")

    # Example: Run all tests
    # all_success = run_all_tests(verbose=True)

    print("\nTesting functions available:")
    print("  - run_all_tests(verbose=True)")
    print("  - run_unit_tests_only()")
    print("  - run_pipeline_tests_only()")
    print("  - run_e2e_tests_only()")
    print("\n[OK] All testing utilities imported successfully!")


# ============================================================================
# EXAMPLE 4: Using Validation Utilities
# ============================================================================

def example_validation():
    """Examples of using validation utilities."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Validation Utilities")
    print("=" * 80)

    from utils.validation import (
        check_python_version,
        check_dependencies,
        check_directory_structure,
        check_gpu,
        run_all_checks
    )

    # Example: Check individual components
    print("\n1. Checking Python version...")
    python_ok = check_python_version()

    print("\n2. Checking dependencies...")
    deps_ok = check_dependencies()

    print("\n3. Checking GPU...")
    check_gpu()  # Always returns True (optional check)

    # Example: Run all validation checks
    # exit_code = run_all_checks()
    # if exit_code == 0:
    #     print("All validation checks passed!")

    print("\n[OK] All validation utilities imported successfully!")


# ============================================================================
# EXAMPLE 5: Combined Usage in a Script
# ============================================================================

def example_complete_workflow():
    """Example of using multiple utility modules together."""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Complete Workflow")
    print("=" * 80)

    # Import everything you need
    from utils import (
        # Visualization
        plot_training_history,
        plot_regression_results,
        visualize_image_pair,
        # Results
        load_results,
        print_results_summary,
        # Validation
        check_dependencies,
        check_gpu
    )

    print("\n1. Verify environment...")
    if check_dependencies():
        print("   [OK] Dependencies OK")

    check_gpu()

    print("\n2. Train model and get history...")
    # model.fit(...) would return history
    # plot_training_history(history.history)

    print("\n3. Evaluate and visualize results...")
    # y_true, y_pred = model.predict(test_data)
    # plot_regression_results(y_true, y_pred)

    print("\n4. Load and display results...")
    # results = load_results('outputs/pipeline_results.json')
    # print_results_summary(results)

    print("\n[OK] Complete workflow example finished!")


# ============================================================================
# EXAMPLE 6: Alternative Import Styles
# ============================================================================

def example_import_styles():
    """Different ways to import from utils package."""
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Import Styles")
    print("=" * 80)

    # Style 1: Import from specific module
    from utils.visualization import plot_training_history
    from utils.results import load_results
    print("[OK] Style 1: Direct module imports")

    # Style 2: Import from utils package
    from utils import plot_regression_results, check_dependencies
    print("[OK] Style 2: Package-level imports")

    # Style 3: Import entire module
    from utils import visualization
    from utils import validation
    # Then use: visualization.plot_training_history(...)
    print("[OK] Style 3: Module imports")

    # Style 4: Import with aliases
    from utils.visualization import plot_confusion_matrix as plot_cm
    from utils.testing import run_unit_tests_only as run_unit_tests
    print("[OK] Style 4: Aliased imports")

    print("\nAll import styles work correctly!")


# ============================================================================
# Main Example Runner
# ============================================================================

def main():
    """Run all examples."""
    print("\n")
    print("*" * 80)
    print("UTILS PACKAGE - USAGE EXAMPLES")
    print("*" * 80)
    print("\nThis script demonstrates how to use the reorganized utils package.")
    print("The utils package is now modular with the following structure:")
    print("\n  utils/")
    print("  |-- __init__.py          # Package exports")
    print("  |-- visualization.py     # Plotting functions")
    print("  |-- results.py          # Result handling")
    print("  |-- testing.py          # Test runners")
    print("  +-- validation.py       # Installation checks")
    print("\n" + "*" * 80)

    try:
        # Run all examples
        example_visualization()
        example_results()
        example_testing()
        example_validation()
        example_complete_workflow()
        example_import_styles()

        print("\n" + "=" * 80)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\nKey Takeaways:")
        print("  1. Import from specific modules: from utils.visualization import ...")
        print("  2. Or import from package: from utils import ...")
        print("  3. All functions documented with docstrings")
        print("  4. Use run_tests.py and verify_installation.py as wrappers")
        print("\nFor more info, see: UTILS_REORGANIZATION.md")

    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("Make sure you're running from the source/ directory")
        print("or adjust the sys.path configuration")
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
