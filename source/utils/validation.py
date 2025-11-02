"""
Installation and setup verification utilities.
"""
import sys
from pathlib import Path
import importlib.util


def check_python_version():
    """
    Check Python version.

    Returns:
        bool: True if version is compatible, False otherwise
    """
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")

    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("  [X] Python 3.8+ required")
        return False
    else:
        print("  [OK] Python version OK")
        return True


def check_dependencies():
    """
    Check if required packages are installed.

    Returns:
        bool: True if all dependencies are installed, False otherwise
    """
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
            print(f"  [X] {package_name}")
            missing.append(package_name)
        else:
            print(f"  [OK] {package_name}")
            installed.append(package_name)

    if missing:
        print(f"\n[WARNING] Missing packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False
    else:
        print("\n[OK] All dependencies installed")
        return True


def check_directory_structure():
    """
    Check if directory structure is correct.

    Returns:
        bool: True if all required directories and files exist, False otherwise
    """
    print("\nChecking directory structure:")

    base_dir = Path(__file__).parent.parent

    required_dirs = [
        'modules',
        'tests',
        'data',
        'models',
        'outputs',
        'utils',
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
        'utils/visualization.py',
        'utils/results.py',
        'utils/testing.py',
        'utils/validation.py',
    ]

    all_good = True

    # Check directories
    for dir_name in required_dirs:
        dir_path = base_dir / dir_name
        if dir_path.exists():
            print(f"  [OK] {dir_name}/")
        else:
            print(f"  [X] {dir_name}/ (missing)")
            all_good = False

    # Check files
    for file_name in required_files:
        file_path = base_dir / file_name
        if file_path.exists():
            print(f"  [OK] {file_name}")
        else:
            print(f"  [X] {file_name} (missing)")
            all_good = False

    return all_good


def check_data_files():
    """
    Check if data files exist.

    Returns:
        bool: True if data files exist, False otherwise
    """
    print("\nChecking data files:")

    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data'

    excel_file = data_dir / 'data_original.xlsx'
    before_dir = data_dir / 'leftover_dataset' / 'data_before'
    after_dir = data_dir / 'leftover_dataset' / 'data_after'

    all_good = True

    if excel_file.exists():
        print(f"  [OK] data_original.xlsx")
    else:
        print(f"  [X] data_original.xlsx (missing)")
        all_good = False

    if before_dir.exists():
        num_folders = len(list(before_dir.iterdir()))
        print(f"  [OK] leftover_dataset/data_before/ ({num_folders} folders)")
    else:
        print(f"  [X] leftover_dataset/data_before/ (missing)")
        all_good = False

    if after_dir.exists():
        num_folders = len(list(after_dir.iterdir()))
        print(f"  [OK] leftover_dataset/data_after/ ({num_folders} folders)")
    else:
        print(f"  [X] leftover_dataset/data_after/ (missing)")
        all_good = False

    if not all_good:
        print("\n[WARNING] Data files are missing. Please ensure dataset is in place.")

    return all_good


def test_module_imports():
    """
    Test if modules can be imported.

    Returns:
        bool: True if all modules can be imported, False otherwise
    """
    print("\nTesting module imports:")

    sys.path.insert(0, str(Path(__file__).parent.parent / 'modules'))

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
            print(f"  [OK] {module_name}")
        except ImportError as e:
            print(f"  [X] {module_name} ({e})")
            all_good = False

    return all_good


def check_gpu():
    """
    Check if GPU is available.

    Returns:
        bool: Always returns True (GPU is optional)
    """
    print("\nChecking GPU availability:")

    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices('GPU')

        if gpus:
            print(f"  [OK] {len(gpus)} GPU(s) detected:")
            for gpu in gpus:
                print(f"    - {gpu.name}")
        else:
            print("  [WARNING] No GPU detected (will use CPU)")
            print("    Training will be slower but still functional")
    except Exception as e:
        print(f"  [WARNING] Could not check GPU: {e}")

    return True  # Not critical


def run_all_checks():
    """
    Run all verification checks.

    Returns:
        int: 0 if all checks passed, 1 otherwise
    """
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
            print(f"\n[X] Error in {check_name}: {e}")
            results[check_name] = False

    # Summary
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)

    for check_name, result in results.items():
        if check_name == "GPU":
            status = "[WARNING] Optional"
        elif result:
            status = "[OK] PASS"
        else:
            status = "[X] FAIL"

        print(f"{status:20} {check_name}")

    # Overall status
    critical_checks = [k for k in results.keys() if k != "GPU"]
    all_critical_passed = all(results[k] for k in critical_checks)

    print("\n" + "=" * 70)

    if all_critical_passed:
        print("[OK] ALL CHECKS PASSED - Ready to run!")
        print("\nNext steps:")
        print("  1. Quick test: python main.py --quick-test")
        print("  2. Full pipeline: python main.py --mode full")
        print("  3. Run tests: python run_tests.py")
        return 0
    else:
        print("[X] SOME CHECKS FAILED")
        print("\nPlease fix the issues above before running the pipeline.")
        return 1
