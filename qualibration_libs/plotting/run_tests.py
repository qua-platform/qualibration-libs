#!/usr/bin/env python3
"""
Test runner for qualibration plotting module.

This script runs all unit tests and provides a summary of results.
"""
import sys
import subprocess
from pathlib import Path


def run_tests():
    """Run all unit tests for the plotting module."""
    print("Running Qualibration Plotting Module Tests")
    print("=" * 50)
    
    # Get the test directory
    test_dir = Path(__file__).parent / "tests"
    
    if not test_dir.exists():
        print(f"Test directory not found: {test_dir}")
        return False
    
    # Run pytest on the test directory
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            str(test_dir),
            "-v",
            "--tb=short",
            "--color=yes"
        ], capture_output=True, text=True)
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        print(f"\nTest exit code: {result.returncode}")
        
        if result.returncode == 0:
            print("[OK] All tests passed!")
            return True
        else:
            print("[ERROR] Some tests failed!")
            return False
            
    except Exception as e:
        print(f"Error running tests: {e}")
        return False


def run_demos():
    """Run demo scripts."""
    print("\nRunning Demo Scripts")
    print("=" * 50)
    
    demo_dir = Path(__file__).parent / "demos"
    
    if not demo_dir.exists():
        print(f"Demo directory not found: {demo_dir}")
        return False
    
    demos = [
        "basic_plots.py",
        "advanced_plots.py",
        "qubit_grid_demo.py"
    ]
    
    success_count = 0
    
    for demo in demos:
        demo_path = demo_dir / demo
        if demo_path.exists():
            print(f"\nRunning {demo}...")
            try:
                result = subprocess.run([
                    sys.executable, str(demo_path)
                ], capture_output=True, text=True, timeout=300)  # 5 minute timeout
                
                if result.returncode == 0:
                    print(f"[OK] {demo} completed successfully")
                    success_count += 1
                else:
                    print(f"[ERROR] {demo} failed with exit code {result.returncode}")
                    if result.stderr:
                        print("Error output:")
                        print(result.stderr)
            except subprocess.TimeoutExpired:
                print(f"[ERROR] {demo} timed out")
            except Exception as e:
                print(f"[ERROR] {demo} failed with error: {e}")
        else:
            print(f"Demo file not found: {demo_path}")
    
    print(f"\nCompleted {success_count}/{len(demos)} demos successfully")
    return success_count == len(demos)


def main():
    """Main test runner."""
    print("Qualibration Plotting Module - Test Runner")
    print("=" * 60)
    
    # Run unit tests
    test_success = run_tests()
    
    # Run demos
    demo_success = run_demos()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Unit Tests: {'[OK] PASSED' if test_success else '[ERROR] FAILED'}")
    print(f"Demos: {'[OK] PASSED' if demo_success else '[ERROR] FAILED'}")
    
    if test_success and demo_success:
        print("\n[SUCCESS] All tests and demos completed successfully!")
        return 0
    else:
        print("\n[ERROR] Some tests or demos failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
