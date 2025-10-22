#!/usr/bin/env python3
"""
Manual setup script for qualibrate-config.
This can be run inside the container if the automated setup fails.
"""

import subprocess
import sys
import os

def main():
    """Run setup-qualibrate-config manually with user interaction."""
    print("🔧 Manual setup-qualibrate-config")
    print("=" * 40)
    print("This script will run setup-qualibrate-config interactively.")
    print("You will need to respond to the prompts manually.")
    print()
    print("Expected prompts:")
    print("1. 'Use all default values? (y/n)' - Answer: y")
    print("2. 'Do you confirm config? [Y/n]' - Answer: y")
    print()
    
    # Change to the superconducting directory
    superconducting_dir = "/app/qualibration_graphs/superconducting"
    if not os.path.exists(superconducting_dir):
        print(f"❌ Superconducting directory not found: {superconducting_dir}")
        sys.exit(1)
    
    print(f"Running setup-qualibrate-config in: {superconducting_dir}")
    print()
    
    try:
        # Run the command interactively
        result = subprocess.run(
            ["setup-qualibrate-config"],
            cwd=superconducting_dir
        )
        
        if result.returncode == 0:
            print("✅ setup-qualibrate-config completed successfully!")
            return 0
        else:
            print(f"❌ setup-qualibrate-config failed with exit code {result.returncode}")
            return 1
            
    except KeyboardInterrupt:
        print("\n⚠️  Setup interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error running setup-qualibrate-config: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())



