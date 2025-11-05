#!/usr/bin/env python3
"""
Correct Raw vs Fit Demo - Shows raw measurement data with fitted curves overlaid.

This script demonstrates the proper usage:
- ds_raw.h5 = Raw measurement data (I, Q, IQ_abs, phase)
- ds_fit.h5 = Fitted curves and parameters (base_line, amplitude, width, position)
"""
import sys
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import qualibration_libs.plotting as qplot
from qualibration_libs.plotting.overlays import FitOverlay, RefLine
import xarray as xr
import numpy as np


def demo_raw_data_with_fits():
    """Demo: Raw measurement data with fitted curves overlaid."""
    print("\n" + "="*60)
    print("CORRECT DEMO: Raw Measurement Data with Fitted Curves")
    print("="*60)
    
    # Load the data correctly
    data_dir = Path(__file__).parent.parent / "test_data"
    ds_raw = xr.open_dataset(data_dir / "ds_raw.h5")  # Raw measurement data
    ds_fit = xr.open_dataset(data_dir / "ds_fit.h5")  # Fitted curves and parameters
    
    print(f"Raw data shape: {dict(ds_raw.dims)}")
    print(f"Raw data variables: {list(ds_raw.data_vars)}")
    print(f"Fit data shape: {dict(ds_fit.dims)}")
    print(f"Fit data variables: {list(ds_fit.data_vars)}")
    print(f"Successful fits: {ds_fit.coords['success'].sum().values}/{len(ds_fit.coords['qubit'])}")
    
    # Find successful fits
    successful_qubits = []
    for i, success in enumerate(ds_fit.coords['success'].values):
        if success:
            successful_qubits.append(i)
    
    print(f"Successful qubits: {[ds_raw.coords['qubit'].values[i] for i in successful_qubits]}")
    
    if successful_qubits:
        # Test with first successful qubit
        qubit_idx = successful_qubits[0]
        qubit_name = ds_raw.coords['qubit'].values[qubit_idx]
        
        print(f"\nAnalyzing qubit {qubit_name} (index {qubit_idx})...")
        
        # Get raw data
        raw_data = ds_raw['IQ_abs'].isel(qubit=qubit_idx).values
        detuning = ds_raw.coords['detuning'].values
        
        print(f"Raw data range: {raw_data.min():.6f} to {raw_data.max():.6f}")
        print(f"Raw data sample (center): {raw_data[140:160]}")
        
        # Get fitted curve
        fit_curve = ds_fit['base_line'].isel(qubit=qubit_idx).values
        
        print(f"Fit curve range: {fit_curve.min():.6f} to {fit_curve.max():.6f}")
        print(f"Fit curve sample (center): {fit_curve[140:160]}")
        
        # Get fit parameters
        amplitude = ds_fit['amplitude'].isel(qubit=qubit_idx).values
        width = ds_fit['width'].isel(qubit=qubit_idx).values
        position = ds_fit['position'].isel(qubit=qubit_idx).values
        
        print(f"Fit parameters:")
        print(f"  Amplitude: {amplitude:.6f}")
        print(f"  Width: {width:.0f} Hz")
        print(f"  Position: {position:.0f} Hz")
        
        # Create fit overlay using the fitted curve
        fit_overlay = FitOverlay(
            y_fit=fit_curve,  # The fitted curve from ds_fit
            params={'amplitude': amplitude, 'width': width, 'position': position},
            formatter=lambda p: f"Fit: A={p['amplitude']:.6f}, W={p['width']:.0f}Hz, P={p['position']:.0f}Hz",
            name="Fitted Curve"
        )
        
        # Add reference line at fit position
        ref_line = RefLine(x=position, name=f"Fit Center ({position:.0f} Hz)")
        
        # Create plot: Raw data with fitted curve overlaid
        fig = qplot.QualibrationFigure.plot(
            ds_raw.isel(qubit=qubit_idx),  # Plot raw data
            x='detuning',
            data_var='IQ_abs',  # Raw measurement data
            overlays=[fit_overlay, ref_line],  # Overlay fitted curve
            residuals=True,
            title=f"Raw Data with Fitted Curve - {qubit_name}"
        )
        
        print(f"[OK] Created plot with {len(fig.figure.data)} traces")
        print(f"[OK] Trace names: {[trace.name for trace in fig.figure.data]}")
        fig.figure.show()
        
        # Calculate fit quality
        residuals = raw_data - fit_curve
        
        print(f"\nFit Quality Analysis:")
        print(f"  Raw data range: {raw_data.min():.6f} to {raw_data.max():.6f}")
        print(f"  Fitted curve range: {fit_curve.min():.6f} to {fit_curve.max():.6f}")
        print(f"  Residuals range: {residuals.min():.6f} to {residuals.max():.6f}")
        print(f"  RMS residual: {np.sqrt(np.mean(residuals**2)):.6f}")
        print(f"  R²: {1 - np.sum(residuals**2) / np.sum((raw_data - np.mean(raw_data))**2):.6f}")
    else:
        print("No successful fits found in the data")


def demo_multi_qubit_raw_with_fits():
    """Demo: Multi-qubit raw data with fitted curves."""
    print("\n" + "="*60)
    print("CORRECT DEMO: Multi-Qubit Raw Data with Fitted Curves")
    print("="*60)
    
    # Load the data correctly
    data_dir = Path(__file__).parent.parent / "test_data"
    ds_raw = xr.open_dataset(data_dir / "ds_raw.h5")  # Raw measurement data
    ds_fit = xr.open_dataset(data_dir / "ds_fit.h5")  # Fitted curves and parameters
    
    # Find successful fits
    successful_qubits = []
    for i, success in enumerate(ds_fit.coords['success'].values):
        if success:
            successful_qubits.append(i)
    
    print(f"Successful qubits: {[ds_raw.coords['qubit'].values[i] for i in successful_qubits]}")
    
    if len(successful_qubits) >= 2:
        # Create per-qubit overlays using fitted curves
        def create_qubit_fit_overlay(qubit_name, qubit_data):
            # Find qubit index
            qubit_idx = None
            for i, name in enumerate(ds_raw.coords['qubit'].values):
                if name == qubit_name:
                    qubit_idx = i
                    break
            
            if qubit_idx is not None and ds_fit.coords['success'].values[qubit_idx]:
                # Get the fitted curve from ds_fit
                fit_curve = ds_fit['base_line'].isel(qubit=qubit_idx).values
                
                # Get fit parameters
                amplitude = ds_fit['amplitude'].isel(qubit=qubit_idx).values
                width = ds_fit['width'].isel(qubit=qubit_idx).values
                position = ds_fit['position'].isel(qubit=qubit_idx).values
                
                # Create fit overlay with fitted curve
                fit_overlay = FitOverlay(
                    y_fit=fit_curve,  # Fitted curve from ds_fit
                    params={'amplitude': amplitude, 'width': width, 'position': position},
                    formatter=lambda p: f"{qubit_name}: A={p['amplitude']:.4f}, W={p['width']:.0f}Hz",
                    name=f"{qubit_name} Fit"
                )
                
                # Add reference line at fit position
                ref_line = RefLine(x=position, name=f"{qubit_name} Center")
                
                return [fit_overlay, ref_line]
            
            return []
        
        # Create multi-qubit plot: Raw data with fitted curves
        fig = qplot.QualibrationFigure.plot(
            ds_raw.isel(qubit=successful_qubits),  # Plot raw data for successful qubits
            x='detuning',
            data_var='IQ_abs',  # Raw measurement data
            overlays=create_qubit_fit_overlay,  # Overlay fitted curves
            title="Multi-Qubit Raw Data with Fitted Curves"
        )
        
        print(f"[OK] Created multi-qubit plot with {len(fig.figure.data)} traces")
        print(f"[OK] Using raw data from ds_raw.h5 with fitted curves from ds_fit.h5")
        fig.figure.show()
    else:
        print("Need at least 2 successful fits for multi-qubit comparison")


def demo_fit_parameter_analysis():
    """Demo: Analyze fit parameters from ds_fit.h5."""
    print("\n" + "="*60)
    print("CORRECT DEMO: Fit Parameter Analysis from ds_fit.h5")
    print("="*60)
    
    # Load fit data
    data_dir = Path(__file__).parent.parent / "test_data"
    ds_fit = xr.open_dataset(data_dir / "ds_fit.h5")  # Fitted curves and parameters
    
    print(f"Fit data shape: {dict(ds_fit.dims)}")
    print(f"Successful fits: {ds_fit.coords['success'].sum().values}/{len(ds_fit.coords['qubit'])}")
    
    # Create parameter analysis plots
    successful_qubits = []
    for i, success in enumerate(ds_fit.coords['success'].values):
        if success:
            successful_qubits.append(i)
    
    if successful_qubits:
        print(f"Analyzing fit parameters for {len(successful_qubits)} successful fits...")
        
        # Plot amplitude vs qubit
        fig = qplot.QualibrationFigure.plot(
            ds_fit,
            x='qubit',
            data_var='amplitude',
            title="Fit Amplitude vs Qubit (from ds_fit.h5)"
        )
        print("[OK] Amplitude analysis plot created")
        fig.figure.show()
        
        # Plot width vs qubit
        fig = qplot.QualibrationFigure.plot(
            ds_fit,
            x='qubit',
            data_var='width',
            title="Fit Width vs Qubit (from ds_fit.h5)"
        )
        print("[OK] Width analysis plot created")
        fig.figure.show()
        
        # Plot position vs qubit
        fig = qplot.QualibrationFigure.plot(
            ds_fit,
            x='qubit',
            data_var='position',
            title="Fit Position vs Qubit (from ds_fit.h5)"
        )
        print("[OK] Position analysis plot created")
        fig.figure.show()
        
        # Show parameter statistics
        print(f"\nFit Parameter Statistics (from ds_fit.h5):")
        for param in ['amplitude', 'width', 'position']:
            values = ds_fit[param].values
            valid_values = values[~np.isnan(values)]
            if len(valid_values) > 0:
                print(f"  {param}:")
                print(f"    Mean: {np.mean(valid_values):.6f}")
                print(f"    Std: {np.std(valid_values):.6f}")
                print(f"    Range: {np.min(valid_values):.6f} to {np.max(valid_values):.6f}")
    else:
        print("No successful fits found for parameter analysis")


def demo_raw_data_only():
    """Demo: Just raw measurement data without fits."""
    print("\n" + "="*60)
    print("CORRECT DEMO: Raw Measurement Data Only")
    print("="*60)
    
    # Load raw data
    data_dir = Path(__file__).parent.parent / "test_data"
    ds_raw = xr.open_dataset(data_dir / "ds_raw.h5")  # Raw measurement data
    
    print(f"Raw data shape: {dict(ds_raw.dims)}")
    print(f"Raw data variables: {list(ds_raw.data_vars)}")
    print(f"Available qubits: {list(ds_raw.coords['qubit'].values)}")
    
    # Plot raw data only
    fig = qplot.QualibrationFigure.plot(
        ds_raw,
        x='detuning',
        data_var='IQ_abs',  # Raw measurement data
        title="Raw Measurement Data (ds_raw.h5)"
    )
    
    print(f"[OK] Created plot with {len(fig.figure.data)} traces")
    print(f"[OK] Showing raw measurement data only")
    fig.figure.show()
    
    # Plot individual components
    print("\nPlotting individual components...")
    
    # I component
    fig = qplot.QualibrationFigure.plot(
        ds_raw.isel(qubit=0),
        x='detuning',
        data_var='I',
        title="Raw I Component (ds_raw.h5)"
    )
    print("[OK] I component plot created")
    fig.figure.show()
    
    # Q component
    fig = qplot.QualibrationFigure.plot(
        ds_raw.isel(qubit=0),
        x='detuning',
        data_var='Q',
        title="Raw Q Component (ds_raw.h5)"
    )
    print("[OK] Q component plot created")
    fig.figure.show()
    
    # Phase
    fig = qplot.QualibrationFigure.plot(
        ds_raw.isel(qubit=0),
        x='detuning',
        data_var='phase',
        title="Raw Phase (ds_raw.h5)"
    )
    print("[OK] Phase plot created")
    fig.figure.show()


def main():
    """Run all correct demos."""
    print("Qualibration Plotting Module - Correct Raw vs Fit Demos")
    print("=" * 60)
    print("ds_raw.h5 = Raw measurement data (I, Q, IQ_abs, phase)")
    print("ds_fit.h5 = Fitted curves and parameters (base_line, amplitude, width, position)")
    print("=" * 60)
    
    try:
        demo_raw_data_only()
        demo_raw_data_with_fits()
        demo_multi_qubit_raw_with_fits()
        demo_fit_parameter_analysis()
        
        print("\n" + "="*60)
        print("🎉 ALL CORRECT DEMOS COMPLETED!")
        print("="*60)
        print("[OK] Raw measurement data (ds_raw.h5) is working")
        print("[OK] Fitted curves (ds_fit.h5) are working")
        print("[OK] Raw data with fitted curves overlaid is working")
        print("[OK] Multi-qubit raw data with fits is working")
        print("[OK] Fit parameter analysis is working")
        print("[OK] Using correct data sources!")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
