#!/usr/bin/env python3
"""
Fit Overlay Demo - Demonstrates the fixed fit overlay functionality.

This script shows how to use fit overlays with real quantum device data.
"""
import sys
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import qualibration_libs.plotting as qplot
from qualibration_libs.plotting.overlays import FitOverlay, RefLine
import xarray as xr
import numpy as np


def demo_simple_fit_overlay():
    """Demo: Simple fit overlay with synthetic data."""
    print("\n" + "="*60)
    print("FIT OVERLAY DEMO 1: Simple Synthetic Fit")
    print("="*60)
    
    # Load data
    data_dir = Path(__file__).parent.parent / "test_data"
    ds = xr.open_dataset(data_dir / "ds_raw.h5")
    
    # Create a simple Gaussian fit
    detuning = ds.coords['detuning'].values
    center = 0.0  # Center at zero detuning
    width = 2e6   # 2 MHz width
    amplitude = 0.05  # Visible amplitude
    fit_curve = amplitude * np.exp(-((detuning - center) / width)**2)
    
    print(f"Created Gaussian fit:")
    print(f"  Center: {center:.0f} Hz")
    print(f"  Width: {width:.0f} Hz")
    print(f"  Amplitude: {amplitude:.3f}")
    print(f"  Fit curve range: {fit_curve.min():.6f} to {fit_curve.max():.6f}")
    
    # Create fit overlay
    fit_overlay = FitOverlay(
        y_fit=fit_curve,
        params={'center': center, 'width': width, 'amplitude': amplitude},
        formatter=lambda p: f"Center: {p['center']:.0f} Hz\nWidth: {p['width']:.0f} Hz\nAmplitude: {p['amplitude']:.3f}",
        name="Gaussian Fit"
    )
    
    # Create plot
    fig = qplot.QualibrationFigure.plot(
        ds.isel(qubit=0),
        x='detuning',
        data_var='IQ_abs',
        overlays=[fit_overlay],
        title="Simple Gaussian Fit Overlay"
    )
    
    print(f"[OK] Created plot with {len(fig.figure.data)} traces")
    print(f"[OK] Trace names: {[trace.name for trace in fig.figure.data]}")
    fig.figure.show()


def demo_real_fit_data():
    """Demo: Real fit data from quantum device measurements."""
    print("\n" + "="*60)
    print("FIT OVERLAY DEMO 2: Real Quantum Device Fit Data")
    print("="*60)
    
    # Load both raw and fit data
    data_dir = Path(__file__).parent.parent / "test_data"
    ds_raw = xr.open_dataset(data_dir / "ds_raw.h5")
    ds_fit = xr.open_dataset(data_dir / "ds_fit.h5")
    
    print(f"Raw data shape: {dict(ds_raw.dims)}")
    print(f"Fit data shape: {dict(ds_fit.dims)}")
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
        
        # Get fit parameters
        amplitude = ds_fit['amplitude'].isel(qubit=qubit_idx).values
        width = ds_fit['width'].isel(qubit=qubit_idx).values
        position = ds_fit['position'].isel(qubit=qubit_idx).values
        
        print(f"Fit parameters:")
        print(f"  Amplitude: {amplitude:.6f}")
        print(f"  Width: {width:.0f} Hz")
        print(f"  Position: {position:.0f} Hz")
        
        # Create fit curve using real parameters
        detuning = ds_raw.coords['detuning'].values
        fit_curve = amplitude * np.exp(-((detuning - position) / width)**2)
        
        print(f"Fit curve range: {fit_curve.min():.6f} to {fit_curve.max():.6f}")
        
        # Create fit overlay with real parameters
        fit_overlay = FitOverlay(
            y_fit=fit_curve,
            params={'amplitude': amplitude, 'width': width, 'position': position},
            formatter=lambda p: f"Amplitude: {p['amplitude']:.6f}\nWidth: {p['width']:.0f} Hz\nPosition: {p['position']:.0f} Hz",
            name="Real Fit"
        )
        
        # Add reference line at fit position
        ref_line = RefLine(x=position, name=f"Fit Center ({position:.0f} Hz)")
        
        # Create plot with real data and fit
        fig = qplot.QualibrationFigure.plot(
            ds_raw.isel(qubit=qubit_idx),
            x='detuning',
            data_var='IQ_abs',
            overlays=[fit_overlay, ref_line],
            residuals=True,
            title=f"Real Fit Analysis - {qubit_name}"
        )
        
        print(f"[OK] Created plot with {len(fig.figure.data)} traces")
        print(f"[OK] Trace names: {[trace.name for trace in fig.figure.data]}")
        fig.figure.show()
        
        # Test with multiple successful qubits
        if len(successful_qubits) > 1:
            print(f"\nCreating multi-qubit fit comparison...")
            
            # Create per-qubit overlays
            def create_qubit_fit_overlay(qubit_name, qubit_data):
                # Find qubit index
                qubit_idx = None
                for i, name in enumerate(ds_raw.coords['qubit'].values):
                    if name == qubit_name:
                        qubit_idx = i
                        break
                
                if qubit_idx is not None and ds_fit.coords['success'].values[qubit_idx]:
                    # Get fit parameters for this qubit
                    amplitude = ds_fit['amplitude'].isel(qubit=qubit_idx).values
                    width = ds_fit['width'].isel(qubit=qubit_idx).values
                    position = ds_fit['position'].isel(qubit=qubit_idx).values
                    
                    # Create fit curve
                    detuning = ds_raw.coords['detuning'].values
                    fit_curve = amplitude * np.exp(-((detuning - position) / width)**2)
                    
                    # Create fit overlay
                    fit_overlay = FitOverlay(
                        y_fit=fit_curve,
                        params={'amplitude': amplitude, 'width': width, 'position': position},
                        formatter=lambda p: f"{qubit_name}: A={p['amplitude']:.4f}, W={p['width']:.0f}Hz",
                        name=f"{qubit_name} Fit"
                    )
                    
                    return [fit_overlay]
                
                return []
            
            # Create multi-qubit plot
            fig = qplot.QualibrationFigure.plot(
                ds_raw.isel(qubit=successful_qubits[:4]),  # Limit to 4 qubits
                x='detuning',
                data_var='IQ_abs',
                overlays=create_qubit_fit_overlay,
                title="Multi-Qubit Fit Comparison"
            )
            
            print(f"[OK] Created multi-qubit plot with {len(fig.figure.data)} traces")
            fig.figure.show()
    else:
        print("No successful fits found in the data")


def demo_fit_quality_analysis():
    """Demo: Fit quality analysis with residuals."""
    print("\n" + "="*60)
    print("FIT OVERLAY DEMO 3: Fit Quality Analysis")
    print("="*60)
    
    # Load data
    data_dir = Path(__file__).parent.parent / "test_data"
    ds_raw = xr.open_dataset(data_dir / "ds_raw.h5")
    ds_fit = xr.open_dataset(data_dir / "ds_fit.h5")
    
    # Find successful fits
    successful_qubits = []
    for i, success in enumerate(ds_fit.coords['success'].values):
        if success:
            successful_qubits.append(i)
    
    if successful_qubits:
        qubit_idx = successful_qubits[0]
        qubit_name = ds_raw.coords['qubit'].values[qubit_idx]
        
        print(f"Analyzing fit quality for {qubit_name}...")
        
        # Get fit parameters
        amplitude = ds_fit['amplitude'].isel(qubit=qubit_idx).values
        width = ds_fit['width'].isel(qubit=qubit_idx).values
        position = ds_fit['position'].isel(qubit=qubit_idx).values
        
        # Create fit curve
        detuning = ds_raw.coords['detuning'].values
        fit_curve = amplitude * np.exp(-((detuning - position) / width)**2)
        
        # Create comprehensive overlays
        overlays = [
            # Fit curve
            FitOverlay(
                y_fit=fit_curve,
                params={'amplitude': amplitude, 'width': width, 'position': position},
                formatter=lambda p: f"Fit: A={p['amplitude']:.6f}, W={p['width']:.0f}Hz, P={p['position']:.0f}Hz",
                name="Gaussian Fit"
            ),
            # Reference lines
            RefLine(x=position, name=f"Fit Center"),
            RefLine(x=position - width, name=f"Fit -1σ"),
            RefLine(x=position + width, name=f"Fit +1σ")
        ]
        
        # Create plot with residuals
        fig = qplot.QualibrationFigure.plot(
            ds_raw.isel(qubit=qubit_idx),
            x='detuning',
            data_var='IQ_abs',
            overlays=overlays,
            residuals=True,
            title=f"Fit Quality Analysis - {qubit_name}"
        )
        
        print(f"[OK] Created fit quality plot with {len(fig.figure.data)} traces")
        print(f"[OK] Includes residuals subplot")
        fig.figure.show()
        
        # Calculate and display fit statistics
        raw_data = ds_raw['IQ_abs'].isel(qubit=qubit_idx).values
        residuals = raw_data - fit_curve
        
        print(f"\nFit Statistics:")
        print(f"  Raw data range: {raw_data.min():.6f} to {raw_data.max():.6f}")
        print(f"  Fit curve range: {fit_curve.min():.6f} to {fit_curve.max():.6f}")
        print(f"  Residuals range: {residuals.min():.6f} to {residuals.max():.6f}")
        print(f"  RMS residual: {np.sqrt(np.mean(residuals**2)):.6f}")
        print(f"  R²: {1 - np.sum(residuals**2) / np.sum((raw_data - np.mean(raw_data))**2):.6f}")
    else:
        print("No successful fits found for quality analysis")


def demo_custom_fit_functions():
    """Demo: Custom fit functions and parameter display."""
    print("\n" + "="*60)
    print("FIT OVERLAY DEMO 4: Custom Fit Functions")
    print("="*60)
    
    # Load data
    data_dir = Path(__file__).parent.parent / "test_data"
    ds = xr.open_dataset(data_dir / "ds_raw.h5")
    
    # Create different types of fits
    detuning = ds.coords['detuning'].values
    
    # 1. Lorentzian fit
    center_lorentz = 0.0
    width_lorentz = 1e6
    amplitude_lorentz = 0.03
    lorentz_fit = amplitude_lorentz / (1 + ((detuning - center_lorentz) / width_lorentz)**2)
    
    # 2. Double Gaussian fit
    center1, center2 = -0.5e6, 0.5e6
    width1, width2 = 0.8e6, 0.8e6
    amp1, amp2 = 0.02, 0.02
    gauss1 = amp1 * np.exp(-((detuning - center1) / width1)**2)
    gauss2 = amp2 * np.exp(-((detuning - center2) / width2)**2)
    double_gauss_fit = gauss1 + gauss2
    
    # Create overlays
    overlays = [
        # Lorentzian fit
        FitOverlay(
            y_fit=lorentz_fit,
            params={'center': center_lorentz, 'width': width_lorentz, 'amplitude': amplitude_lorentz},
            formatter=lambda p: f"Lorentzian: C={p['center']:.0f}Hz, W={p['width']:.0f}Hz, A={p['amplitude']:.4f}",
            name="Lorentzian Fit"
        ),
        # Double Gaussian fit
        FitOverlay(
            y_fit=double_gauss_fit,
            params={'center1': center1, 'center2': center2, 'width1': width1, 'width2': width2, 'amp1': amp1, 'amp2': amp2},
            formatter=lambda p: f"Double Gauss: C1={p['center1']:.0f}Hz, C2={p['center2']:.0f}Hz",
            name="Double Gaussian Fit"
        )
    ]
    
    # Create plot
    fig = qplot.QualibrationFigure.plot(
        ds.isel(qubit=0),
        x='detuning',
        data_var='IQ_abs',
        overlays=overlays,
        title="Custom Fit Functions Comparison"
    )
    
    print(f"[OK] Created plot with {len(fig.figure.data)} traces")
    print(f"[OK] Includes Lorentzian and Double Gaussian fits")
    fig.figure.show()


def main():
    """Run all fit overlay demos."""
    print("Qualibration Plotting Module - Fit Overlay Demos")
    print("=" * 60)
    
    try:
        demo_simple_fit_overlay()
        demo_real_fit_data()
        demo_fit_quality_analysis()
        demo_custom_fit_functions()
        
        print("\n" + "="*60)
        print("🎉 ALL FIT OVERLAY DEMOS COMPLETED!")
        print("="*60)
        print("[OK] Simple synthetic fits are working")
        print("[OK] Real quantum device fit data is working")
        print("[OK] Fit quality analysis with residuals is working")
        print("[OK] Custom fit functions are working")
        print("[OK] Parameter display and formatting is working")
        
    except Exception as e:
        print(f"\n❌ Fit overlay demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
