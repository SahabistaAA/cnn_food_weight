"""
Test runner script for running all tests with detailed reporting.
"""
import sys
import unittest
from pathlib import Path
from datetime import datetime
import json

# Add tests directory to path
sys.path.insert(0, str(Path(__file__).parent / 'tests'))

from tests.test_unit import run_unit_tests
from tests.test_pipeline import run_pipeline_tests
from tests.test_e2e import run_e2e_tests


def run_all_tests(verbose=True):
    """
    Run all test suites and generate a report.

    Args:
        verbose: Whether to show detailed output
    """
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
        print(f"\nUnit Tests: {'✓ PASSED' if unit_success else '✗ FAILED'}")
    except Exception as e:
        results['unit_tests'] = 'ERROR'
        print(f"\nUnit Tests: ✗ ERROR - {e}")

    # Run pipeline tests
    print("\n" + "=" * 80)
    print("2. PIPELINE INTEGRATION TESTS")
    print("=" * 80)
    try:
        pipeline_success = run_pipeline_tests()
        results['pipeline_tests'] = 'PASS' if pipeline_success else 'FAIL'
        print(f"\nPipeline Tests: {'✓ PASSED' if pipeline_success else '✗ FAILED'}")
    except Exception as e:
        results['pipeline_tests'] = 'ERROR'
        print(f"\nPipeline Tests: ✗ ERROR - {e}")

    # Run E2E tests
    print("\n" + "=" * 80)
    print("3. END-TO-END TESTS")
    print("=" * 80)
    try:
        e2e_success = run_e2e_tests()
        results['e2e_tests'] = 'PASS' if e2e_success else 'FAIL'
        print(f"\nE2E Tests: {'✓ PASSED' if e2e_success else '✗ FAILED'}")
    except Exception as e:
        results['e2e_tests'] = 'ERROR'
        print(f"\nE2E Tests: ✗ ERROR - {e}")

    # Final report
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    for test_name, status in results.items():
        status_symbol = "✓" if status == "PASS" else "✗"
        print(f"{status_symbol} {test_name}: {status}")

    all_passed = all(status == 'PASS' for status in results.values())

    print("\n" + "=" * 80)
    if all_passed:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("=" * 80)

    print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Save results
    output_dir = Path(__file__).parent / 'outputs'
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


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run test suites')
    parser.add_argument('--suite', type=str, choices=['all', 'unit', 'pipeline', 'e2e'],
                       default='all', help='Which test suite to run')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Verbose output')

    args = parser.parse_args()

    if args.suite == 'all':
        success = run_all_tests(verbose=args.verbose)
    elif args.suite == 'unit':
        success = run_unit_tests()
    elif args.suite == 'pipeline':
        success = run_pipeline_tests()
    elif args.suite == 'e2e':
        success = run_e2e_tests()

    sys.exit(0 if success else 1)
