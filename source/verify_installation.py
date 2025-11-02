"""
Installation and setup verification script.

This is a wrapper script that uses the utils.validation module.
Run this to check if everything is properly configured.
"""
import sys
from utils.validation import run_all_checks


def main():
    """Run all verification checks."""
    return run_all_checks()

if __name__ == '__main__':
    sys.exit(main())
