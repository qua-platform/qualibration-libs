#!/usr/bin/env python3
"""
Real Fit Data Demo - Using actual fit data from quantum device measurements.

This script demonstrates how to use the real fit data files that contain
pre-computed fit curves and parameters.
"""
import sys
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import qualibration_libs.plotting as qplot
from qualibration_libs.plotting.overlays import FitOverlay, RefLine
import xarray as xr
import numpy as np


def demo_raw_vs_fit_data():
    """Demo: Compare raw data with actual fit data."""
    print("\n" + "="*60)
    print("REAL FIT DATA DEMO 1: Raw vs Fit Data Comparison")
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
        
        # Get the actual fit curve from the fit data
        fit_curve = ds_fit['base_line'].isel(qubit=qubit_idx).values
        detuning = ds_raw.coords['detuning'].values
        
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
        
        # Create fit overlay using the actual fit curve
        fit_overlay = FitOverlay(
            y_fit=fit_curve,
            params={'amplitude': amplitude, 'width': width, 'position': position},
            formatter=lambda p: f"Real Fit: A={p['amplitude']:.6f}, W={p['width']:.0f}Hz, P={p['position']:.0f}Hz",
            name="Real Fit Curve"
        )
        
        # Add reference line at fit position
        ref_line = RefLine(x=position, name=f"Fit Center ({position:.0f} Hz)")
        
        # Create plot comparing raw data with real fit
        fig = qplot.QualibrationFigure.plot(
            ds_raw.isel(qubit=qubit_idx),
            x='detuning',
            data_var='IQ_abs',
            overlays=[fit_overlay, ref_line],
            residuals=True,
            title=f"Raw Data vs Real Fit - {qubit_name}"
        )
        
        print(f"✓ Created plot with {len(fig.figure.data)} traces")
        print(f"✓ Trace names: {[trace.name for trace in fig.figure.data]}")
        fig.figure.show()
        
        # Calculate fit quality
        raw_data = ds_raw['IQ_abs'].isel(qubit=qubit_idx).values
        residuals = raw_data - fit_curve
        
        print(f"\nFit Quality Analysis:")
        print(f"  Raw data range: {raw_data.min():.6f} to {raw_data.max():.6f}")
        print(f"  Fit curve range: {fit_curve.min():.6f} to {fit_curve.max():.6f}")
        print(f"  Residuals range: {residuals.min():.6f} to {residuals.max():.6f}")
        print(f"  RMS residual: {np.sqrt(np.mean(residuals**2)):.6f}")
        print(f"  R²: {1 - np.sum(residuals**2) / np.sum((raw_data - np.mean(raw_data))**2):.6f}")
    else:
        print("No successful fits found in the data")


def demo_multi_qubit_real_fits():
    """Demo: Multi-qubit comparison with real fit data."""
    print("\n" + "="*60)
    print("REAL FIT DATA DEMO 2: Multi-Qubit Real Fit Comparison")
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
    
    print(f"Successful qubits: {[ds_raw.coords['qubit'].values[i] for i in successful_qubits]}")
    
    if len(successful_qubits) >= 2:
        # Create per-qubit overlays using real fit data
        def create_real_qubit_fit_overlay(qubit_name, qubit_data):
            # Find qubit index
            qubit_idx = None
            for i, name in enumerate(ds_raw.coords['qubit'].values):
                if name == qubit_name:
                    qubit_idx = i
                    break
            
            if qubit_idx is not None and ds_fit.coords['success'].values[qubit_idx]:
                # Get the actual fit curve
                fit_curve = ds_fit['base_line'].isel(qubit=qubit_idx).values
                
                # Get fit parameters
                amplitude = ds_fit['amplitude'].isel(qubit=qubit_idx).values
                width = ds_fit['width'].isel(qubit=qubit_idx).values
                position = ds_fit['position'].isel(qubit=qubit_idx).values
                
                # Create fit overlay with real data
                fit_overlay = FitOverlay(
                    y_fit=fit_curve,
                    params={'amplitude': amplitude, 'width': width, 'position': position},
                    formatter=lambda p: f"{qubit_name}: A={p['amplitude']:.4f}, W={p['width']:.0f}Hz",
                    name=f"{qubit_name} Real Fit"
                )
                
                # Add reference line at fit position
                ref_line = RefLine(x=position, name=f"{qubit_name} Center")
                
                return [fit_overlay, ref_line]
            
            return []
        
        # Create multi-qubit plot with real fits
        fig = qplot.QualibrationFigure.plot(
            ds_raw.isel(qubit=successful_qubits),
            x='detuning',
            data_var='IQ_abs',
            overlays=create_real_qubit_fit_overlay,
            title="Multi-Qubit Real Fit Comparison"
        )
        
        print(f"✓ Created multi-qubit plot with {len(fig.figure.data)} traces")
        print(f"✓ Using real fit curves from ds_fit.h5")
        fig.figure.show()
    else:
        print("Need at least 2 successful fits for multi-qubit comparison")


def demo_fit_parameter_analysis():
    """Demo: Analyze fit parameters across qubits."""
    print("\n" + "="*60)
    print("REAL FIT DATA DEMO 3: Fit Parameter Analysis")
    print("="*60)
    
    # Load fit data
    data_dir = Path(__file__).parent.parent / "test_data"
    ds_fit = xr.open_dataset(data_dir / "ds_fit.h5")
    
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
            title="Fit Amplitude vs Qubit"
        )
        print("✓ Amplitude analysis plot created")
        fig.figure.show()
        
        # Plot width vs qubit
        fig = qplot.QualibrationFigure.plot(
            ds_fit,
            x='qubit',
            data_var='width',
            title="Fit Width vs Qubit"
        )
        print("✓ Width analysis plot created")
        fig.figure.show()
        
        # Plot position vs qubit
        fig = qplot.QualibrationFigure.plot(
            ds_fit,
            x='qubit',
            data_var='position',
            title="Fit Position vs Qubit"
        )
        print("✓ Position analysis plot created")
        fig.figure.show()
        
        # Show parameter statistics
        print(f"\nFit Parameter Statistics:")
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


def demo_flux_tuning_fit_data():
    """Demo: Flux tuning fit data (ds_fit_2.h5)."""
    print("\n" + "="*60)
    print("REAL FIT DATA DEMO 4: Flux Tuning Fit Data")
    print("="*60)
    
    # Load flux tuning fit data
    data_dir = Path(__file__).parent.parent / "test_data"
    ds_fit_2 = xr.open_dataset(data_dir / "ds_fit_2.h5")
    
    print(f"Flux tuning fit data shape: {dict(ds_fit_2.dims)}")
    print(f"Successful fits: {ds_fit_2.coords['success'].sum().values}/{len(ds_fit_2.coords['qubit'])}")
    print(f"Available qubits: {list(ds_fit_2.coords['qubit'].values)}")
    
    # Plot peak frequency vs flux bias for each qubit
    for i, qubit in enumerate(ds_fit_2.coords['qubit'].values):
        if ds_fit_2.coords['success'].values[i]:
            print(f"\nAnalyzing flux tuning for {qubit}...")
            
            fig = qplot.QualibrationFigure.plot(
                ds_fit_2.isel(qubit=i),
                x='flux_bias',
                data_var='peak_freq',
                title=f"Peak Frequency vs Flux Bias - {qubit}"
            )
            print(f"✓ Created flux tuning plot for {qubit}")
            fig.figure.show()
    
    # Plot fit results
    print(f"\nAnalyzing fit results...")
    fit_results_ds = xr.Dataset({
        'a': (['qubit'], ds_fit_2['fit_results'].isel(fit_vals=0).values),
        'f': (['qubit'], ds_fit_2['fit_results'].isel(fit_vals=1).values),
        'phi': (['qubit'], ds_fit_2['fit_results'].isel(fit_vals=2).values),
        'offset': (['qubit'], ds_fit_2['fit_results'].isel(fit_vals=3).values)
    }, coords={'qubit': ds_fit_2.coords['qubit']})
    
    for param in ['a', 'f', 'phi', 'offset']:
        fig = qplot.QualibrationFigure.plot(
            fit_results_ds,
            x='qubit',
            data_var=param,
            title=f"Flux Tuning Fit Parameter {param}"
        )
        print(f"✓ Created {param} parameter plot")
        fig.figure.show()


def demo_power_sweep_fit_data():
    """Demo: Power sweep fit data (ds_fit_3.h5)."""
    print("\n" + "="*60)
    print("REAL FIT DATA DEMO 5: Power Sweep Fit Data")
    print("="*60)
    
    # Load power sweep fit data
    data_dir = Path(__file__).parent.parent / "test_data"
    ds_fit_3 = xr.open_dataset(data_dir / "ds_fit_3.h5")
    
    print(f"Power sweep fit data shape: {dict(ds_fit_3.dims)}")
    print(f"Successful fits: {ds_fit_3.coords['success'].sum().values}/{len(ds_fit_3.coords['qubit'])}")
    print(f"Available qubits: {list(ds_fit_3.coords['qubit'].values)}")
    
    # Find successful fits
    successful_qubits = []
    for i, success in enumerate(ds_fit_3.coords['success'].values):
        if success:
            successful_qubits.append(i)
    
    if successful_qubits:
        print(f"Successful qubits: {[ds_fit_3.coords['qubit'].values[i] for i in successful_qubits]}")
        
        # Plot optimal power for each qubit
        for qubit_idx in successful_qubits:
            qubit_name = ds_fit_3.coords['qubit'].values[qubit_idx]
            optimal_power = ds_fit_3.coords['optimal_power'].values[qubit_idx]
            
            print(f"\nAnalyzing {qubit_name} with optimal power {optimal_power:.1f} dBm...")
            
            # Create 2D heatmap
            fig = qplot.QualibrationFigure.plot(
                ds_fit_3.isel(qubit=qubit_idx),
                x='detuning',
                y='power',
                data_var='IQ_abs',
                title=f"Power Sweep Analysis - {qubit_name} (Optimal: {optimal_power:.1f} dBm)"
            )
            print(f"✓ Created power sweep heatmap for {qubit_name}")
            fig.figure.show()
    else:
        print("No successful fits found in power sweep data")


def main():
    """Run all real fit data demos."""
    print("Qualibration Plotting Module - Real Fit Data Demos")
    print("=" * 60)
    
    try:
        demo_raw_vs_fit_data()
        demo_multi_qubit_real_fits()
        demo_fit_parameter_analysis()
        demo_flux_tuning_fit_data()
        demo_power_sweep_fit_data()
        
        print("\n" + "="*60)
        print("🎉 ALL REAL FIT DATA DEMOS COMPLETED!")
        print("="*60)
        print("✓ Raw vs fit data comparison is working")
        print("✓ Multi-qubit real fit comparison is working")
        print("✓ Fit parameter analysis is working")
        print("✓ Flux tuning fit data is working")
        print("✓ Power sweep fit data is working")
        print("✓ Using actual fit curves from HDF5 files")
        
    except Exception as e:
        print(f"\n❌ Real fit data demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
