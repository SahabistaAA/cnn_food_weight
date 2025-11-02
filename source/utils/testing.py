"""
Test runner utilities for running all tests with detailed reporting.
"""
import sys
import unittest
from pathlib import Path
from datetime import datetime
import json


def run_all_tests(verbose=True):
    """
    Run all test suites and generate a report.

    Args:
        verbose: Whether to show detailed output

    Returns:
        bool: True if all tests passed, False otherwise
    """
    # Add tests directory to path
    sys.path.insert(0, str(Path(__file__).parent.parent / 'tests'))

    from tests.test_unit import run_unit_tests
    from tests.test_pipeline import run_pipeline_tests
    from tests.test_e2e import run_e2e_tests

    print("=" * 80)
    print("FOOD WEIGHT PREDICTION - TEST SUITE")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    results = {}

    # Run unit tests
    print("\n" + "=" * 80)
    print("1. UNIT TESTS")
    print("=" * 80)
    try:
        unit_success = run_unit_tests()
        results['unit_tests'] = 'PASS' if unit_success else 'FAIL'
        print(f"\nUnit Tests: {'[PASS]' if unit_success else '[FAIL]'}")
    except Exception as e:
        results['unit_tests'] = 'ERROR'
        print(f"\nUnit Tests: [ERROR] - {e}")

    # Run pipeline tests
    print("\n" + "=" * 80)
    print("2. PIPELINE INTEGRATION TESTS")
    print("=" * 80)
    try:
        pipeline_success = run_pipeline_tests()
        results['pipeline_tests'] = 'PASS' if pipeline_success else 'FAIL'
        print(f"\nPipeline Tests: {'[PASS]' if pipeline_success else '[FAIL]'}")
    except Exception as e:
        results['pipeline_tests'] = 'ERROR'
        print(f"\nPipeline Tests: [ERROR] - {e}")

    # Run E2E tests
    print("\n" + "=" * 80)
    print("3. END-TO-END TESTS")
    print("=" * 80)
    try:
        e2e_success = run_e2e_tests()
        results['e2e_tests'] = 'PASS' if e2e_success else 'FAIL'
        print(f"\nE2E Tests: {'[PASS]' if e2e_success else '[FAIL]'}")
    except Exception as e:
        results['e2e_tests'] = 'ERROR'
        print(f"\nE2E Tests: [ERROR] - {e}")

    # Final report
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    for test_name, status in results.items():
        status_symbol = "[+]" if status == "PASS" else "[-]"
        print(f"{status_symbol} {test_name}: {status}")

    all_passed = all(status == 'PASS' for status in results.values())

    print("\n" + "=" * 80)
    if all_passed:
        print("ALL TESTS PASSED [SUCCESS]")
    else:
        print("SOME TESTS FAILED [WARNING]")
    print("=" * 80)

    print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Save results
    output_dir = Path(__file__).parent.parent / 'outputs'
    output_dir.mkdir(exist_ok=True)

    results_file = output_dir / f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': results,
            'all_passed': all_passed
        }, f, indent=4)

    print(f"\nTest results saved to: {results_file}")

    return all_passed


def run_unit_tests_only():
    """
    Run only unit tests.

    Returns:
        bool: True if all tests passed, False otherwise
    """
    sys.path.insert(0, str(Path(__file__).parent.parent / 'tests'))
    from tests.test_unit import run_unit_tests
    return run_unit_tests()


def run_pipeline_tests_only():
    """
    Run only pipeline tests.

    Returns:
        bool: True if all tests passed, False otherwise
    """
    sys.path.insert(0, str(Path(__file__).parent.parent / 'tests'))
    from tests.test_pipeline import run_pipeline_tests
    return run_pipeline_tests()


def run_e2e_tests_only():
    """
    Run only E2E tests.

    Returns:
        bool: True if all tests passed, False otherwise
    """
    sys.path.insert(0, str(Path(__file__).parent.parent / 'tests'))
    from tests.test_e2e import run_e2e_tests
    return run_e2e_tests()
