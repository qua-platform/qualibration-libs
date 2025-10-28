#!/usr/bin/env python3
"""
Test script to debug setup-qualibrate-config interaction.
"""

import subprocess
import sys
import time
import threading

def test_setup_qualibrate_config():
    """Test setup-qualibrate-config with different approaches."""
    print("Testing setup-qualibrate-config...")
    
    # First, let's see what happens when we run it without any input
    print("\n1. Testing without input (should show prompts):")
    try:
        result = subprocess.run(
            ["setup-qualibrate-config"],
            timeout=10,
            cwd="."
        )
        print(f"Exit code: {result.returncode}")
    except subprocess.TimeoutExpired:
        print("Timed out (expected)")
    except Exception as e:
        print(f"Error: {e}")
    
    # Now test with input
    print("\n2. Testing with input:")
    try:
        result = subprocess.run(
            ["setup-qualibrate-config"],
            input="y\ny\n",
            text=True,
            timeout=30,
            cwd="."
        )
        print(f"Exit code: {result.returncode}")
        print(f"Stdout: {result.stdout}")
        print(f"Stderr: {result.stderr}")
    except subprocess.TimeoutExpired:
        print("Timed out")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_setup_qualibrate_config()



