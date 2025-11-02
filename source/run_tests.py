"""
Test runner script for running all tests with detailed reporting.

This is a wrapper script that uses the utils.testing module.
"""
import sys
from utils.testing import (
    run_all_tests,
    run_unit_tests_only,
    run_pipeline_tests_only,
    run_e2e_tests_only
)


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
        success = run_unit_tests_only()
    elif args.suite == 'pipeline':
        success = run_pipeline_tests_only()
    elif args.suite == 'e2e':
        success = run_e2e_tests_only()

    sys.exit(0 if success else 1)
