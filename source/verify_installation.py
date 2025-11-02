"""
Installation and setup verification script.
Run this to check if everything is properly configured.
"""
import sys
from pathlib import Path
import importlib.util

def check_python_version():
    """Check Python version."""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")

    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("  ❌ Python 3.8+ required")
        return False
    else:
        print("  ✓ Python version OK")
        return True

def check_dependencies():
    """Check if required packages are installed."""
    required_packages = [
        'numpy',
        'pandas',
        'tensorflow',
        'keras',
        'cv2',  # opencv-python
        'sklearn',  # scikit-learn
        'albumentations',
        'openpyxl',
        'matplotlib',
        'PIL',  # Pillow
    ]

    missing = []
    installed = []

    print("\nChecking dependencies:")
    for package in required_packages:
        # Handle special cases
        import_name = package
        if package == 'cv2':
            package_name = 'opencv-python'
        elif package == 'sklearn':
            package_name = 'scikit-learn'
        elif package == 'PIL':
            package_name = 'Pillow'
        else:
            package_name = package

        spec = importlib.util.find_spec(import_name)
        if spec is None:
            print(f"  ❌ {package_name}")
            missing.append(package_name)
        else:
            print(f"  ✓ {package_name}")
            installed.append(package_name)

    if missing:
        print(f"\n⚠ Missing packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False
    else:
        print("\n✓ All dependencies installed")
        return True

def check_directory_structure():
    """Check if directory structure is correct."""
    print("\nChecking directory structure:")

    base_dir = Path(__file__).parent

    required_dirs = [
        'modules',
        'tests',
        'data',
        'models',
        'outputs',
    ]

    required_files = [
        'main.py',
        'requirements.txt',
        'README.md',
        'modules/config.py',
        'modules/step1_data_reader.py',
        'modules/step2_segmentation.py',
        'modules/step3_augmentation.py',
        'modules/step4_classification.py',
        'modules/step5_regression.py',
        'tests/test_unit.py',
        'tests/test_pipeline.py',
        'tests/test_e2e.py',
    ]

    all_good = True

    # Check directories
    for dir_name in required_dirs:
        dir_path = base_dir / dir_name
        if dir_path.exists():
            print(f"  ✓ {dir_name}/")
        else:
            print(f"  ❌ {dir_name}/ (missing)")
            all_good = False

    # Check files
    for file_name in required_files:
        file_path = base_dir / file_name
        if file_path.exists():
            print(f"  ✓ {file_name}")
        else:
            print(f"  ❌ {file_name} (missing)")
            all_good = False

    return all_good

def check_data_files():
    """Check if data files exist."""
    print("\nChecking data files:")

    base_dir = Path(__file__).parent
    data_dir = base_dir / 'data'

    excel_file = data_dir / 'data_original.xlsx'
    before_dir = data_dir / 'leftover_dataset' / 'data_before'
    after_dir = data_dir / 'leftover_dataset' / 'data_after'

    all_good = True

    if excel_file.exists():
        print(f"  ✓ data_original.xlsx")
    else:
        print(f"  ❌ data_original.xlsx (missing)")
        all_good = False

    if before_dir.exists():
        num_folders = len(list(before_dir.iterdir()))
        print(f"  ✓ leftover_dataset/data_before/ ({num_folders} folders)")
    else:
        print(f"  ❌ leftover_dataset/data_before/ (missing)")
        all_good = False

    if after_dir.exists():
        num_folders = len(list(after_dir.iterdir()))
        print(f"  ✓ leftover_dataset/data_after/ ({num_folders} folders)")
    else:
        print(f"  ❌ leftover_dataset/data_after/ (missing)")
        all_good = False

    if not all_good:
        print("\n⚠ Data files are missing. Please ensure dataset is in place.")

    return all_good

def test_module_imports():
    """Test if modules can be imported."""
    print("\nTesting module imports:")

    sys.path.insert(0, str(Path(__file__).parent / 'modules'))

    modules_to_test = [
        'config',
        'step1_data_reader',
        'step2_segmentation',
        'step3_augmentation',
        'step4_classification',
        'step5_regression',
    ]

    all_good = True

    for module_name in modules_to_test:
        try:
            __import__(f'modules.{module_name}')
            print(f"  ✓ {module_name}")
        except ImportError as e:
            print(f"  ❌ {module_name} ({e})")
            all_good = False

    return all_good

def check_gpu():
    """Check if GPU is available."""
    print("\nChecking GPU availability:")

    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices('GPU')

        if gpus:
            print(f"  ✓ {len(gpus)} GPU(s) detected:")
            for gpu in gpus:
                print(f"    - {gpu.name}")
        else:
            print("  ⚠ No GPU detected (will use CPU)")
            print("    Training will be slower but still functional")
    except Exception as e:
        print(f"  ⚠ Could not check GPU: {e}")

    return True  # Not critical

def main():
    """Run all verification checks."""
    print("=" * 70)
    print("FOOD WEIGHT PREDICTION - INSTALLATION VERIFICATION")
    print("=" * 70)

    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Directory Structure", check_directory_structure),
        ("Data Files", check_data_files),
        ("Module Imports", test_module_imports),
        ("GPU", check_gpu),
    ]

    results = {}

    for check_name, check_func in checks:
        try:
            results[check_name] = check_func()
        except Exception as e:
            print(f"\n❌ Error in {check_name}: {e}")
            results[check_name] = False

    # Summary
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)

    for check_name, result in results.items():
        if check_name == "GPU":
            status = "⚠ Optional"
        elif result:
            status = "✓ PASS"
        else:
            status = "❌ FAIL"

        print(f"{status:15} {check_name}")

    # Overall status
    critical_checks = [k for k in results.keys() if k != "GPU"]
    all_critical_passed = all(results[k] for k in critical_checks)

    print("\n" + "=" * 70)

    if all_critical_passed:
        print("✓ ALL CHECKS PASSED - Ready to run!")
        print("\nNext steps:")
        print("  1. Quick test: python main.py --quick-test")
        print("  2. Full pipeline: python main.py --mode full")
        print("  3. Run tests: python run_tests.py")
        return 0
    else:
        print("❌ SOME CHECKS FAILED")
        print("\nPlease fix the issues above before running the pipeline.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
