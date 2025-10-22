#!/usr/bin/env python3
"""
Test the setup-qualibrate-config part locally.
"""

import subprocess
import sys
from pathlib import Path

def test_setup_qualibrate_config():
    """Test setup-qualibrate-config with automated responses."""
    print("Testing setup-qualibrate-config with default settings...")
    
    try:
        # Use the simple approach that works locally
        result = subprocess.run(
            ["setup-qualibrate-config"],
            input="y\ny\n",
            text=True,
            capture_output=True,
            timeout=30,
            cwd="."
        )
        
        print("Setup output:")
        print(result.stdout)
        if result.stderr:
            print("Error output:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("SUCCESS: setup-qualibrate-config completed successfully!")
            return True
        else:
            print(f"ERROR: setup-qualibrate-config failed with exit code {result.returncode}")
            return False
                
    except subprocess.TimeoutExpired:
        print("ERROR: setup-qualibrate-config timed out.")
        return False
    except Exception as e:
        print(f"ERROR: setup-qualibrate-config encountered an error: {e}")
        return False

if __name__ == "__main__":
    success = test_setup_qualibrate_config()
    if success:
        print("\nSUCCESS: Local test passed!")
    else:
        print("\nERROR: Local test failed!")
    sys.exit(0 if success else 1)



