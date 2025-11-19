#!/usr/bin/env python3
"""
Script to run all plotting tests with --save option.
This script automatically discovers and runs all plotting scripts in the scripts directory.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import time

def find_plotting_scripts(scripts_dir):
    """Find all Python plotting scripts in the scripts directory."""
    scripts = []
    for file in scripts_dir.glob("plot_*.py"):
        # Skip this script itself and any non-plotting scripts
        if file.name == "run_all_tests.py":
            continue
        scripts.append(file)
    return sorted(scripts)

def run_script(script_path, save=True, verbose=False):
    """Run a single script with optional --save flag."""
    cmd = [sys.executable, str(script_path)]
    if save:
        cmd.append("--save")
    
    if verbose:
        print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Script timed out after 5 minutes"
    except Exception as e:
        return False, "", str(e)

def main():
    parser = argparse.ArgumentParser(description="Run all plotting tests with --save option")
    parser.add_argument("--no-save", action="store_true", help="Run without --save flag")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue running other scripts if one fails")
    args = parser.parse_args()
    
    # Get the scripts directory
    scripts_dir = Path(__file__).resolve().parent
    
    # Find all plotting scripts
    scripts = find_plotting_scripts(scripts_dir)
    
    if not scripts:
        print("❌ No plotting scripts found in the scripts directory")
        return 1
    
    print(f"🔍 Found {len(scripts)} plotting scripts:")
    for script in scripts:
        print(f"  - {script.name}")
    print()
    
    # Run each script
    save_flag = not args.no_save
    successful = 0
    failed = 0
    
    start_time = time.time()
    
    for i, script in enumerate(scripts, 1):
        print(f"📊 [{i}/{len(scripts)}] Running {script.name}...")
        
        success, stdout, stderr = run_script(script, save=save_flag, verbose=args.verbose)
        
        if success:
            print(f"✅ {script.name} completed successfully")
            successful += 1
            if args.verbose and stdout:
                print(f"   Output: {stdout.strip()}")
        else:
            print(f"❌ {script.name} failed")
            failed += 1
            if stderr:
                print(f"   Error: {stderr.strip()}")
            if not args.continue_on_error:
                print("🛑 Stopping due to error (use --continue-on-error to continue)")
                break
        
        print()  # Add spacing between scripts
    
    # Summary
    elapsed = time.time() - start_time
    print("=" * 50)
    print(f"📈 Test Summary:")
    print(f"   ✅ Successful: {successful}")
    print(f"   ❌ Failed: {failed}")
    print(f"   ⏱️  Total time: {elapsed:.1f} seconds")
    
    if failed > 0:
        print(f"\n💡 Tip: Use --continue-on-error to run all scripts even if some fail")
        return 1
    
    print(f"\n🎉 All tests completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
