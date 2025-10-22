#!/usr/bin/env python3
"""
Automated setup script for qualibration-libs test environment.
This script handles the interactive setup-qualibrate-config command.
"""

import subprocess
import sys
from pathlib import Path


def run_setup_qualibrate_config():
    """Run setup-qualibrate-config with automated responses."""
    print("Running setup-qualibrate-config with default settings...")
    
    try:
        import time
        import threading
        import select
        import sys
        
        # Start the process with unbuffered output
        process = subprocess.Popen(
            ["setup-qualibrate-config"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=0,  # Unbuffered
            cwd="/app/qualibration_graphs/superconducting"
        )
        
        output_lines = []
        first_prompt_sent = False
        second_prompt_sent = False
        
        def read_output():
            """Read output and respond to prompts."""
            nonlocal first_prompt_sent, second_prompt_sent
            try:
                while True:
                    # Use select to check if data is available
                    if select.select([process.stdout], [], [], 1.0)[0]:
                        char = process.stdout.read(1)
                        if not char:
                            break
                        
                        # Print character immediately
                        sys.stdout.write(char)
                        sys.stdout.flush()
                        
                        # Accumulate line for pattern matching
                        if char == '\n':
                            line = ''.join(output_lines[-100:])  # Get last 100 chars for context
                            output_lines.append('')
                            
                            # Check for first prompt: "Use all default values? (y/n)"
                            if "Use all default values? (y/n)" in line and not first_prompt_sent:
                                print("\n[DEBUG] Found first prompt, sending 'y'...")
                                process.stdin.write("y\n")
                                process.stdin.flush()
                                first_prompt_sent = True
                                print("[DEBUG] Sent first 'y' input")
                            
                            # Check for second prompt: "Do you confirm config? [Y/n]:"
                            elif "Do you confirm config? [Y/n]:" in line and not second_prompt_sent:
                                print("\n[DEBUG] Found second prompt, sending 'y'...")
                                process.stdin.write("y\n")
                                process.stdin.flush()
                                second_prompt_sent = True
                                print("[DEBUG] Sent second 'y' input")
                        else:
                            output_lines.append(char)
                    else:
                        # Timeout - check if process is still running
                        if process.poll() is not None:
                            break
                        
            except Exception as e:
                print(f"\n[DEBUG] Error reading output: {e}")
        
        # Start output reading thread
        output_thread = threading.Thread(target=read_output)
        output_thread.start()
        
        # Wait for process to complete
        process.wait()
        
        # Wait for output thread to finish
        output_thread.join(timeout=5)
        
        # Close stdin if still open
        if not process.stdin.closed:
            process.stdin.close()
        
        print(f"\n[DEBUG] Process completed with return code: {process.returncode}")
        print(f"[DEBUG] First prompt sent: {first_prompt_sent}")
        print(f"[DEBUG] Second prompt sent: {second_prompt_sent}")
        
        if process.returncode == 0:
            print("SUCCESS: setup-qualibrate-config completed successfully!")
            return True
        else:
            print(f"ERROR: setup-qualibrate-config failed with exit code {process.returncode}")
            print("This is critical for the setup. Please check the configuration manually.")
            return False  # This is critical, so we should fail
                
    except Exception as e:
        print(f"ERROR: setup-qualibrate-config encountered an error: {e}")
        print("This is critical for the setup.")
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
    
    print("SUCCESS: Setup completed successfully!")
    print("You can now run the test scripts from /app/qualibration-libs/scripts/")


if __name__ == "__main__":
    main()
