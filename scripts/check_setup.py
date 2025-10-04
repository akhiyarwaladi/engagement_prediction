#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quick setup validation script."""

import sys
import os
from pathlib import Path

# Fix Windows emoji encoding
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'

def check_setup():
    """Validate project setup."""

    print("\n" + "=" * 70)
    print(" " * 15 + "SETUP VALIDATION CHECK")
    print("=" * 70 + "\n")

    checks = []
    errors = []

    # Check 1: Data file exists
    data_file = Path('fst_unja_from_gallery_dl.csv')
    if data_file.exists():
        try:
            import pandas as pd
            df = pd.read_csv(data_file)
            print(f"[OK] Data file found: {len(df)} posts")
            checks.append(True)
        except ImportError:
            print(f"[WARN]  Data file found but pandas not installed")
            print(f"   Run: pip install -r requirements.txt")
            checks.append(False)
    else:
        print(f"[FAIL] Data file NOT found: {data_file}")
        errors.append("Run gallery-dl first")
        checks.append(False)

    # Check 2: Config file exists
    config_file = Path('config.yaml')
    if config_file.exists():
        print(f"[OK] Config file found")
        checks.append(True)
    else:
        print(f"[FAIL] Config file NOT found")
        errors.append("Config missing")
        checks.append(False)

    # Check 3: Source modules exist
    src_modules = [
        'src/utils/__init__.py',
        'src/features/__init__.py',
        'src/models/__init__.py'
    ]

    modules_ok = True
    for module in src_modules:
        if not Path(module).exists():
            print(f"[FAIL] Module missing: {module}")
            modules_ok = False
            errors.append(f"Missing {module}")

    if modules_ok:
        print(f"[OK] All source modules found")
        checks.append(True)
    else:
        checks.append(False)

    # Check 4: Dependencies
    try:
        import pandas
        import numpy
        import sklearn
        import yaml
        print(f"[OK] Core dependencies installed")
        checks.append(True)
    except ImportError as e:
        print(f"[FAIL] Missing dependency: {e}")
        errors.append("Run: pip install -r requirements.txt")
        checks.append(False)

    # Check 5: Directories exist
    dirs = ['data/processed', 'data/features', 'models', 'logs', 'docs/figures']
    dirs_ok = all(Path(d).exists() for d in dirs)

    if dirs_ok:
        print(f"[OK] All directories exist")
        checks.append(True)
    else:
        print(f"[WARN]  Some directories missing (will be created automatically)")
        checks.append(True)  # Not critical

    # Summary
    print("\n" + "=" * 70)

    if all(checks):
        print("[OK] ALL CHECKS PASSED - Ready to run!")
        print("\nNext step: python run_pipeline.py")
    else:
        print("[FAIL] SOME CHECKS FAILED - Please fix:")
        for error in errors:
            print(f"  - {error}")

    print("=" * 70 + "\n")

    return all(checks)


if __name__ == '__main__':
    success = check_setup()
    sys.exit(0 if success else 1)
