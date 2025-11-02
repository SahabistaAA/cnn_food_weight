"""
Utils package for the food weight prediction pipeline.

This package provides utilities for:
- Visualization (plotting and image display)
- Results handling (loading and printing results)
- Testing (test runners)
- Validation (installation verification)
"""

# Visualization utilities
from .visualization import (
    visualize_image_pair,
    visualize_segmentation,
    visualize_augmentations,
    plot_training_history,
    plot_regression_results,
    plot_confusion_matrix,
    create_inference_report,
)

# Results utilities
from .results import (
    load_results,
    print_results_summary,
)

# Testing utilities
from .testing import (
    run_all_tests,
    run_unit_tests_only,
    run_pipeline_tests_only,
    run_e2e_tests_only,
)

# Validation utilities
from .validation import (
    check_python_version,
    check_dependencies,
    check_directory_structure,
    check_data_files,
    test_module_imports,
    check_gpu,
    run_all_checks,
)

__all__ = [
    # Visualization
    'visualize_image_pair',
    'visualize_segmentation',
    'visualize_augmentations',
    'plot_training_history',
    'plot_regression_results',
    'plot_confusion_matrix',
    'create_inference_report',
    # Results
    'load_results',
    'print_results_summary',
    # Testing
    'run_all_tests',
    'run_unit_tests_only',
    'run_pipeline_tests_only',
    'run_e2e_tests_only',
    # Validation
    'check_python_version',
    'check_dependencies',
    'check_directory_structure',
    'check_data_files',
    'test_module_imports',
    'check_gpu',
    'run_all_checks',
]
