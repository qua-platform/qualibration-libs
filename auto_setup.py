#!/usr/bin/env python3
"""
Automated setup script for qualibration-libs test environment.
This script handles the interactive setup-qualibrate-config command.
"""

import subprocess
import sys
from pathlib import Path


def run_setup_qualibrate_config():
    """Run setup-qualibrate-config with automated responses using pexpect."""
    print("Running setup-qualibrate-config with default settings...")
    
    try:
        import pexpect
        import time
        
        print("[DEBUG] Starting setup-qualibrate-config process...")
        
        # Start the process with pexpect
        child = pexpect.spawn(
            "setup-qualibrate-config",
            cwd="/app/qualibration_graphs/superconducting",
            timeout=30,  # 30 second timeout for each expect
            encoding='utf-8'
        )
        
        # Don't use logfile_read to avoid encoding issues
        # We'll capture output manually
        
        print("[DEBUG] Process environment:")
        print(f"[DEBUG] Working directory: /app/qualibration_graphs/superconducting")
        print(f"[DEBUG] Timeout settings: 30s per expect, 60s for completion")
        
        print("[DEBUG] Process started, waiting for prompts...")
        
        # Wait for first prompt: "Use all default values? (y/n)"
        try:
            print("[DEBUG] Waiting for first prompt...")
            # Look for the prompt text directly, ignoring ANSI codes
            print("[DEBUG] Looking for first prompt pattern...")
            try:
                child.expect("Use all default values.*\\(y/n\\)", timeout=30)
                print("[DEBUG] Found first prompt with first pattern, sending 'y'...")
            except pexpect.TIMEOUT:
                print("[DEBUG] First pattern failed, trying alternative...")
                # Try a more general pattern
                child.expect(".*default.*values.*\\(y/n\\)", timeout=30)
                print("[DEBUG] Found first prompt with alternative pattern, sending 'y'...")
            
            child.sendline("y")
            print("[DEBUG] Sent first 'y' input")
        except pexpect.TIMEOUT:
            print("[DEBUG] TIMEOUT waiting for first prompt")
            print(f"[DEBUG] Current output: {repr(child.before)}")
            print(f"[DEBUG] Buffer content: {repr(child.buffer)}")
            child.close()
            return False
        except Exception as e:
            print(f"[DEBUG] Error waiting for first prompt: {e}")
            print(f"[DEBUG] Current output: {repr(child.before)}")
            print(f"[DEBUG] Buffer content: {repr(child.buffer)}")
            child.close()
            return False
        
        # Wait for second prompt: "Do you confirm config? [Y/n]:"
        try:
            print("[DEBUG] Waiting for second prompt...")
            print("[DEBUG] Looking for second prompt pattern...")
            # Look for the prompt text, ignoring ANSI codes
            child.expect("Do you confirm config.*\\[Y/n\\]:", timeout=30)
            print("[DEBUG] Found second prompt, sending 'y'...")
            child.sendline("y")
            print("[DEBUG] Sent second 'y' input")
        except pexpect.TIMEOUT:
            print("[DEBUG] TIMEOUT waiting for second prompt")
            print(f"[DEBUG] Current output: {repr(child.before)}")
            print(f"[DEBUG] Buffer content: {repr(child.buffer)}")
            child.close()
            return False
        except Exception as e:
            print(f"[DEBUG] Error waiting for second prompt: {e}")
            print(f"[DEBUG] Current output: {repr(child.before)}")
            print(f"[DEBUG] Buffer content: {repr(child.buffer)}")
            child.close()
            return False
        
        # Wait for process to complete
        print("[DEBUG] Waiting for process to complete...")
        try:
            child.expect(pexpect.EOF, timeout=60)  # 60 second timeout for completion
            print("[DEBUG] Process completed")
        except pexpect.TIMEOUT:
            print("[DEBUG] TIMEOUT waiting for process completion")
            print(f"[DEBUG] Current output: {child.before}")
            child.close()
            return False
        
        # Get the exit status
        child.close()
        exit_status = child.exitstatus
        
        print(f"[DEBUG] Process completed with exit status: {exit_status}")
        
        if exit_status == 0:
            print("SUCCESS: setup-qualibrate-config completed successfully!")
            return True
        else:
            print(f"ERROR: setup-qualibrate-config failed with exit code {exit_status}")
            print("This is critical for the setup. Please check the configuration manually.")
            return False
                
    except ImportError:
        print("[DEBUG] pexpect not available, falling back to subprocess method...")
        return run_setup_qualibrate_config_fallback()
    except Exception as e:
        print(f"ERROR: setup-qualibrate-config encountered an error: {e}")
        print("This is critical for the setup.")
        return False


def run_setup_qualibrate_config_fallback():
    """Fallback method using subprocess if pexpect is not available."""
    print("[DEBUG] Using fallback subprocess method...")
    
    try:
        import subprocess
        import time
        
        # Start the process
        process = subprocess.Popen(
            ["setup-qualibrate-config"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd="/app/qualibration_graphs/superconducting"
        )
        
        print("[DEBUG] Process started, sending inputs...")
        
        # Send the inputs
        process.stdin.write("y\n")  # First prompt
        process.stdin.write("y\n")  # Second prompt
        process.stdin.close()
        
        print("[DEBUG] Inputs sent, waiting for completion...")
        
        # Wait for completion with timeout
        try:
            stdout, stderr = process.communicate(timeout=120)  # 2 minute timeout
            print(f"[DEBUG] Process completed with return code: {process.returncode}")
            print(f"[DEBUG] Output: {stdout}")
            
            if process.returncode == 0:
                print("SUCCESS: setup-qualibrate-config completed successfully!")
                return True
            else:
                print(f"ERROR: setup-qualibrate-config failed with exit code {process.returncode}")
                return False
                
        except subprocess.TimeoutExpired:
            print("[DEBUG] Process timed out, terminating...")
            process.kill()
            process.wait()
            return False
            
    except Exception as e:
        print(f"ERROR: Fallback method failed: {e}")
        return False


def copy_quam_state():
    """Copy quam_state folder from data directory to superconducting directory."""
    data_dir = Path("/app/qualibration_graphs/superconducting/data/QPU_project/2025-10-01")
    superconducting_dir = Path("/app/qualibration_graphs/superconducting")
    
    # Find any data folder that contains quam_state
    quam_state_found = False
    for date_folder in data_dir.iterdir():
        if date_folder.is_dir():
            quam_state_src = date_folder / "quam_state"
            if quam_state_src.exists():
                quam_state_dst = superconducting_dir / "quam_state"
                print(f"Copying quam_state from {quam_state_src} to {quam_state_dst}")
                
                # Remove existing quam_state if it exists
                if quam_state_dst.exists():
                    import shutil
                    shutil.rmtree(quam_state_dst)
                
                # Copy the quam_state folder
                import shutil
                shutil.copytree(quam_state_src, quam_state_dst)
                quam_state_found = True
                print("SUCCESS: quam_state copied successfully!")
                break
    
    if not quam_state_found:
        print("WARNING:  No quam_state folder found in data directories")
    
    return quam_state_found


def setup_preliminary_data():
    """Setup preliminary datasets if zip file is available."""
    print("Checking for preliminary datasets...")
    
    zip_file = Path("/app/preliminary_datasets.zip")
    if not zip_file.exists():
        print("No preliminary_datasets.zip found, skipping data setup")
        return True
    
    print("Found preliminary_datasets.zip, setting up data...")
    
    try:
        # Import the data setup functionality
        import zipfile
        import shutil
        
        # Extract zip file
        extract_dir = Path("/tmp/preliminary_data")
        extract_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Extracting {zip_file} to {extract_dir}")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        # Find result folders (those starting with #)
        result_folders = []
        for item in extract_dir.rglob("*"):
            if item.is_dir() and item.name.startswith("#"):
                result_folders.append(item)
        
        print(f"Found {len(result_folders)} result folders")
        
        # Copy result folders to target date directory
        target_date_dir = Path("/app/qualibration_graphs/superconducting/data/QPU_project/2025-10-01")
        target_date_dir.mkdir(parents=True, exist_ok=True)
        
        for folder in result_folders:
            target_folder = target_date_dir / folder.name
            if target_folder.exists():
                shutil.rmtree(target_folder)
            shutil.copytree(folder, target_folder)
            print(f"Copied {folder.name} to {target_folder}")
        
        # Find and copy quam_state files
        quam_state_files = []
        for root, dirs, files in extract_dir.rglob("*"):
            if root.name == "quam_state" and root.is_dir():
                quam_state_files.append(root)
        
        if quam_state_files:
            superconducting_dir = Path("/app/qualibration_graphs/superconducting")
            target_quam_state = superconducting_dir / "quam_state"
            
            if target_quam_state.exists():
                shutil.rmtree(target_quam_state)
            
            # Use the first quam_state found
            shutil.copytree(quam_state_files[0], target_quam_state)
            print(f"Copied quam_state to {target_quam_state}")
        else:
            print("No quam_state directories found")
        
        print("SUCCESS: Preliminary data setup completed!")
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to setup preliminary data: {e}")
        return False


def main():
    """Main setup function."""
    print("Starting automated qualibration-libs setup...")
    
    # Install qualibration-libs first (mounted at runtime)
    qualibration_libs_dir = Path("/app/qualibration-libs")
    if qualibration_libs_dir.exists():
        print("Installing qualibration-libs package...")
        
        # First try to install quam-builder from git if it's not available
        print("Ensuring quam-builder is available...")
        quam_builder_result = subprocess.run(
            ["pip", "install", "git+https://github.com/qua-platform/quam-builder.git"],
            capture_output=True,
            text=True
        )
        
        if quam_builder_result.returncode != 0:
            print(f"WARNING:  Warning: Failed to install quam-builder: {quam_builder_result.stderr}")
        
        # Now try to install qualibration-libs
        result = subprocess.run(
            ["pip", "install", "-e", "."],
            cwd=qualibration_libs_dir,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"WARNING:  Editable install failed, trying regular install: {result.stderr}")
            result = subprocess.run(
                ["pip", "install", "."],
                cwd=qualibration_libs_dir,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                print(f"ERROR: Failed to install qualibration-libs: {result.stderr}")
                print("Continuing anyway...")
            else:
                print("SUCCESS: Qualibration-libs installed successfully!")
        else:
            print("SUCCESS: Qualibration-libs installed successfully!")
    else:
        print("WARNING:  Qualibration-libs directory not found, skipping installation")
    
    # Change to the superconducting directory
    superconducting_dir = Path("/app/qualibration_graphs/superconducting")
    if not superconducting_dir.exists():
        print(f"ERROR: Superconducting directory not found: {superconducting_dir}")
        sys.exit(1)
    
    # Install the superconducting calibrations package
    print("Installing superconducting calibrations package...")
    
    # First, ensure README.md exists to avoid build issues
    readme_path = superconducting_dir / "README.md"
    if not readme_path.exists():
        print("Creating README.md for superconducting calibrations...")
        readme_content = """# Superconducting Calibrations

QM Superconducting Calibration Graphs

This package provides calibration utilities for superconducting quantum systems.
"""
        readme_path.write_text(readme_content)
    
    result = subprocess.run(
        ["pip", "install", "-e", "."],
        cwd=superconducting_dir,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"WARNING:  Editable install failed, trying regular install: {result.stderr}")
        result = subprocess.run(
            ["pip", "install", "."],
            cwd=superconducting_dir,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"ERROR: Failed to install superconducting calibrations: {result.stderr}")
            print("Continuing anyway...")
        else:
            print("SUCCESS: Superconducting calibrations installed successfully!")
    else:
        print("SUCCESS: Superconducting calibrations installed successfully!")
    
    # Run setup-qualibrate-config (critical step)
    if not run_setup_qualibrate_config():
        print("ERROR: Critical setup step failed. The environment may not work properly.")
        print("You may need to run 'setup-qualibrate-config' manually inside the container.")
        print("Continuing with the rest of the setup...")
    
    # Copy quam_state folder
    copy_quam_state()
    
    # Setup preliminary data if available
    setup_preliminary_data()
    
    print("SUCCESS: Setup completed successfully!")
    print("You can now run the test scripts from /app/qualibration-libs/scripts/")


if __name__ == "__main__":
    main()
