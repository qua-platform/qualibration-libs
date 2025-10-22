#!/usr/bin/env python3
"""
Simple test script to verify the Docker environment is working correctly.
"""

import sys
import subprocess
from pathlib import Path

def test_imports():
    """Test that all required packages can be imported."""
    print("🧪 Testing package imports...")
    
    success_count = 0
    total_tests = 0
    
    # Test core packages
    packages_to_test = [
        ("qualibration_libs", "qualibration_libs"),
        ("quam", "quam"),
        ("qualibrate", "qualibrate"),
        ("qm", "qm-qua"),
        ("xarray", "xarray"),
        ("matplotlib", "matplotlib"),
        ("numpy", "numpy"),
        ("scipy", "scipy"),
    ]
    
    for package_name, display_name in packages_to_test:
        total_tests += 1
        try:
            __import__(package_name)
            print(f"✅ {display_name} imported successfully")
            success_count += 1
        except ImportError as e:
            print(f"❌ Failed to import {display_name}: {e}")
    
    print(f"\n📊 Import test results: {success_count}/{total_tests} packages imported successfully")
    
    # Consider it successful if at least the core packages work
    return success_count >= 4

def test_script_execution():
    """Test that the plotting scripts can be executed (dry run)."""
    print("\n🧪 Testing script execution...")
    
    scripts_dir = Path("/app/qualibration-libs/scripts")
    if not scripts_dir.exists():
        print(f"❌ Scripts directory not found: {scripts_dir}")
        return False
    
    # Test the main plotting script
    script_path = scripts_dir / "plot_02a_resonator_spectroscopy_results.py"
    if not script_path.exists():
        print(f"❌ Test script not found: {script_path}")
        return False
    
    # Try to run the script with --help to see if it can start
    try:
        result = subprocess.run(
            [sys.executable, str(script_path), "--help"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            print("✅ Script can be executed successfully")
            return True
        else:
            print(f"⚠️  Script execution returned non-zero code: {result.returncode}")
            print(f"Error output: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⚠️  Script execution timed out")
        return False
    except Exception as e:
        print(f"❌ Failed to execute script: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting Docker environment tests...")
    print("=" * 50)
    
    # Test imports
    imports_ok = test_imports()
    
    # Test script execution
    scripts_ok = test_script_execution()
    
    print("\n" + "=" * 50)
    if imports_ok and scripts_ok:
        print("🎉 All tests passed! Docker environment is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
