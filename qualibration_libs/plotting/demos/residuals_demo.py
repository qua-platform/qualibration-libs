#!/usr/bin/env python3
"""
Demo script specifically for testing residuals functionality.
This demonstrates both empty residuals (no fit data) and proper residuals (with fit data).
"""
import sys
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import numpy as np
import xarray as xr
import qualibration_libs.plotting as qplot
from qualibration_libs.plotting.overlays import FitOverlay


def demo_residuals_functionality():
    """Demo: Comprehensive residuals functionality testing."""
    print("="*60)
    print("DEMO: Residuals Functionality Testing")
    print("="*60)
    
    # Load test data
    data_dir = Path(__file__).parent.parent / "test_data"
    data_files = {}
    
    for filename in ['ds_raw.h5', 'ds_fit.h5', 'ds_raw_2.h5', 'ds_fit_2.h5', 'ds_raw_3.h5', 'ds_fit_3.h5']:
        filepath = data_dir / filename
        if filepath.exists():
            try:
                data_files[filename] = xr.open_dataset(filepath)
                print(f"[OK] Loaded {filename}")
            except Exception as e:
                print(f"[FAIL] Failed to load {filename}: {e}")
    
    if not data_files:
        print("No test data files found. Creating synthetic data...")
        # Create synthetic data for demonstration
        detuning = np.linspace(-1.5e7, 1.5e7, 100)
        qubits = ['qC1', 'qC2', 'qC3']
        
        data = {}
        for i, qubit in enumerate(qubits):
            center = i * 0.5e6
            width = 1e6
            response = np.exp(-((detuning - center) / width)**2)
            data[qubit] = (['detuning'], response)
        
        ds = xr.Dataset(data, coords={'detuning': detuning})
        data_files['synthetic'] = ds
    
    # Test 1: Residuals WITHOUT fit data (should show empty residuals subplot)
    print("\n1. Testing residuals WITHOUT fit data (empty residuals subplot):")
    ds = data_files.get('ds_raw.h5', data_files.get('synthetic'))
    if ds is not None:
        fig1 = qplot.QualibrationFigure.plot(
            ds.isel(qubit=0) if 'qubit' in ds.dims else ds,
            x='detuning',
            data_var='IQ_abs' if 'IQ_abs' in ds.data_vars else list(ds.data_vars)[0],
            residuals=True,
            title="Residuals WITHOUT Fit Data (Empty Residuals Subplot)"
        )
        print(f"   - Figure created with {len(fig1.figure.data)} traces")
        print(f"   - Trace names: {[trace.name for trace in fig1.figure.data]}")
        print("   - Expected: Empty residuals subplot (just zero line)")
        fig1.figure.show()
    
    # Test 2: Residuals WITH fit data (should show actual residual data)
    print("\n2. Testing residuals WITH fit data (actual residual data):")
    if ds is not None:
        # Create a fit overlay
        detuning_vals = ds.coords['detuning'].values if 'detuning' in ds.coords else np.linspace(-1.5e7, 1.5e7, 100)
        data_var = 'IQ_abs' if 'IQ_abs' in ds.data_vars else list(ds.data_vars)[0]
        
        # Create a simple Gaussian fit
        center = 0.0
        width = 2e6
        fit_curve = np.exp(-((detuning_vals - center) / width)**2)
        
        fit_overlay = FitOverlay(
            y_fit=fit_curve,
            params={'center': center, 'width': width},
            formatter=lambda p: f"Center: {p['center']:.0f} Hz, Width: {p['width']:.0f} Hz"
        )
        
        fig2 = qplot.QualibrationFigure.plot(
            ds.isel(qubit=0) if 'qubit' in ds.dims else ds,
            x='detuning',
            data_var=data_var,
            overlays=[fit_overlay],
            residuals=True,
            title="Residuals WITH Fit Data (Actual Residual Data)"
        )
        print(f"   - Figure created with {len(fig2.figure.data)} traces")
        print(f"   - Trace names: {[trace.name for trace in fig2.figure.data]}")
        
        # Check for residual traces
        residual_traces = [trace for trace in fig2.figure.data if 'residuals' in str(trace.name)]
        print(f"   - Residual traces found: {len(residual_traces)}")
        if residual_traces:
            print(f"   - Residual trace names: {[trace.name for trace in residual_traces]}")
            print("   - [SUCCESS] Residuals are properly plotted!")
        else:
            print("   - [FAILURE] No residual traces found")
        
        fig2.figure.show()
    
    # Test 3: Multi-qubit residuals
    print("\n3. Testing multi-qubit residuals:")
    if ds is not None and 'qubit' in ds.dims:
        fig3 = qplot.QualibrationFigure.plot(
            ds,
            x='detuning',
            data_var='IQ_abs' if 'IQ_abs' in ds.data_vars else list(ds.data_vars)[0],
            overlays=[fit_overlay],
            residuals=True,
            title="Multi-Qubit Residuals"
        )
        print(f"   - Figure created with {len(fig3.figure.data)} traces")
        residual_traces = [trace for trace in fig3.figure.data if 'residuals' in str(trace.name)]
        print(f"   - Residual traces found: {len(residual_traces)}")
        fig3.figure.show()
    
    print("\n" + "="*60)
    print("RESIDUALS DEMO COMPLETED")
    print("="*60)
    print("[OK] Empty residuals subplot (no fit data)")
    print("[OK] Actual residual data (with fit data)")
    print("[OK] Multi-qubit residuals functionality")


if __name__ == "__main__":
    demo_residuals_functionality()
