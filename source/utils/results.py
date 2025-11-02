"""
Result handling utilities for the food weight prediction pipeline.
"""
import json
from pathlib import Path
from typing import Dict


def load_results(results_path: Path) -> Dict:
    """
    Load pipeline results from JSON file.

    Args:
        results_path: Path to JSON results file

    Returns:
        Dictionary containing results
    """
    with open(results_path, 'r') as f:
        return json.load(f)


def print_results_summary(results: Dict):
    """
    Print a formatted summary of results.

    Args:
        results: Dictionary containing pipeline results
    """
    print("=" * 80)
    print("PIPELINE RESULTS SUMMARY")
    print("=" * 80)

    for step, step_results in results.items():
        if step.startswith('step'):
            print(f"\n{step.upper()}")
            print("-" * 40)
            for key, value in step_results.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
