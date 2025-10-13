"""
Advanced plotting demos using real test data.

This script demonstrates advanced plotting capabilities including
multi-dimensional data, complex overlays, and publication-ready figures.
"""
import os
import sys
import numpy as np
import xarray as xr
from pathlib import Path

# Add the parent directory to the path to import qualibration_libs
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import qualibration_libs.plotting as qplot
from qualibration_libs.plotting.overlays import RefLine, LineOverlay, FitOverlay, ScatterOverlay, TextBoxOverlay
from qualibration_libs.plotting import QubitGrid


def load_test_data():
    """Load all available test data files."""
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
    
    return data_files


def demo_flux_tuning_analysis(data_files):
    """Demo: Flux tuning analysis with 3D data."""
    print("\n" + "="*60)
    print("ADVANCED DEMO 1: Flux Tuning Analysis")
    print("="*60)
    
    if 'ds_raw_2.h5' not in data_files:
        print("No flux tuning data available for this demo")
        return
    
    ds = data_files['ds_raw_2.h5']
    print(f"Data shape: {dict(ds.dims)}")
    print(f"Flux bias range: {ds.coords['flux_bias'].min().values:.2f} to {ds.coords['flux_bias'].max().values:.2f}")
    
    # Create a comprehensive flux tuning plot
    print("\nCreating comprehensive flux tuning analysis...")
    
    # Select a single qubit for detailed analysis
    qubit_idx = 0
    ds_qubit = ds.isel(qubit=qubit_idx)
    qubit_name = ds.coords['qubit'].values[qubit_idx]
    
    # 2D heatmap of IQ magnitude
    fig = qplot.QualibrationFigure.plot(
        ds_qubit,
        x='detuning',
        y='flux_bias',
        data_var='IQ_abs',
        title=f"Flux Tuning Map: IQ Magnitude - {qubit_name}"
    )
    fig.figure.show()
    
    # 2D heatmap of phase
    fig = qplot.QualibrationFigure.plot(
        ds_qubit,
        x='detuning',
        y='flux_bias',
        data_var='phase',
        title=f"Flux Tuning Map: Phase - {qubit_name}"
    )
    fig.figure.show()
    
    # 1D slices at different flux bias points
    flux_points = [0, 0.1, 0.2, 0.3, 0.4]
    print(f"\nCreating 1D slices at flux bias points: {flux_points}")
    
    for flux_val in flux_points:
        # Find closest flux bias index
        flux_idx = np.argmin(np.abs(ds.coords['flux_bias'].values - flux_val))
        actual_flux = ds.coords['flux_bias'].values[flux_idx]
        
        fig = qplot.QualibrationFigure.plot(
            ds_qubit.isel(flux_bias=flux_idx),
            x='detuning',
            data_var='IQ_abs',
            title=f"IQ Magnitude at Flux Bias = {actual_flux:.2f} - {qubit_name}"
        )
        fig.figure.show()


def demo_power_optimization(data_files):
    """Demo: Power optimization analysis."""
    print("\n" + "="*60)
    print("ADVANCED DEMO 2: Power Optimization Analysis")
    print("="*60)
    
    if 'ds_raw_3.h5' not in data_files:
        print("No power sweep data available for this demo")
        return
    
    ds = data_files['ds_raw_3.h5']
    print(f"Data shape: {dict(ds.dims)}")
    print(f"Power range: {ds.coords['power'].min().values:.1f} to {ds.coords['power'].max().values:.1f} dBm")
    
    # 2D power optimization heatmap
    print("\nCreating power optimization heatmap...")
    fig = qplot.QualibrationFigure.plot(
        ds.isel(qubit=0),
        x='detuning',
        y='power',
        data_var='IQ_abs',
        title="Power Optimization: IQ Magnitude vs Detuning and Power"
    )
    fig.figure.show()
    
    # Analyze optimal power for each qubit
    if 'optimal_power' in ds.coords:
        print("\nAnalyzing optimal power for each qubit...")
        optimal_powers = ds.coords['optimal_power'].values
        
        for i, (qubit, optimal_power) in enumerate(zip(ds.coords['qubit'].values, optimal_powers)):
            if not np.isnan(optimal_power):
                print(f"{qubit}: Optimal power = {optimal_power:.1f} dBm")
                
                # Find closest power index
                power_idx = np.argmin(np.abs(ds.coords['power'].values - optimal_power))
                
                # Plot at optimal power
                fig = qplot.QualibrationFigure.plot(
                    ds.isel(qubit=i, power=power_idx),
                    x='detuning',
                    data_var='IQ_abs',
                    title=f"Optimal Power Response - {qubit} ({optimal_power:.1f} dBm)"
                )
                fig.figure.show()
    
    # Power sweep analysis - plot response vs power at zero detuning
    zero_detuning_idx = len(ds.coords['detuning']) // 2  # Middle detuning point
    
    print("\nCreating power sweep analysis...")
    for i, qubit in enumerate(ds.coords['qubit'].values):
        # Extract power sweep data
        power_sweep_data = ds.isel(qubit=i, detuning=zero_detuning_idx)
        
        fig = qplot.QualibrationFigure.plot(
            power_sweep_data,
            x='power',
            data_var='IQ_abs',
            title=f"Power Sweep at Zero Detuning - {qubit}"
        )
        fig.figure.show()


def demo_fit_quality_analysis(data_files):
    """Demo: Fit quality analysis and comparison."""
    print("\n" + "="*60)
    print("ADVANCED DEMO 3: Fit Quality Analysis")
    print("="*60)
    
    if 'ds_fit.h5' not in data_files or 'ds_raw.h5' not in data_files:
        print("No fit data available for this demo")
        return
    
    ds_raw = data_files['ds_raw.h5']
    ds_fit = data_files['ds_fit.h5']
    
    print("Creating fit quality analysis...")
    
    # Compare raw data with fit results
    for i, qubit in enumerate(ds_raw.coords['qubit'].values):
        if i >= len(ds_fit.coords['qubit']):
            break
            
        # Check if fit was successful
        if ds_fit.coords['success'].values[i]:
            print(f"\nAnalyzing fit quality for {qubit}...")
            
            # Raw data
            raw_data = ds_raw.isel(qubit=i)
            
            # Create fit overlay
            detuning = ds_raw.coords['detuning'].values
            baseline = ds_fit['base_line'].isel(qubit=i).values
            
            # Create overlays for fit analysis
            overlays = [
                LineOverlay(
                    x=detuning,
                    y=baseline,
                    name="Baseline Fit",
                    dash="dash"
                ),
                RefLine(x=0, name="Zero Detuning")
            ]
            
            # Add fit parameters as text
            if 'amplitude' in ds_fit.data_vars:
                amplitude = ds_fit['amplitude'].isel(qubit=i).values
                width = ds_fit['width'].isel(qubit=i).values
                position = ds_fit['position'].isel(qubit=i).values
                
                param_text = f"Amplitude: {amplitude:.3f}\nWidth: {width:.0f} Hz\nPosition: {position:.0f} Hz"
                text_overlay = TextBoxOverlay(
                    text=param_text,
                    anchor="top left"
                )
                overlays.append(text_overlay)
            
            fig = qplot.QualibrationFigure.plot(
                raw_data,
                x='detuning',
                data_var='IQ_abs',
                overlays=overlays,
                residuals=True,
                title=f"Fit Quality Analysis - {qubit}"
            )
            fig.figure.show()
    
    # Fit parameter comparison across qubits
    print("\nCreating fit parameter comparison...")
    
    # Amplitude comparison
    fig = qplot.QualibrationFigure.plot(
        ds_fit,
        x='qubit',
        data_var='amplitude',
        title="Fit Amplitude Comparison Across Qubits"
    )
    fig.figure.show()
    
    # Width comparison
    fig = qplot.QualibrationFigure.plot(
        ds_fit,
        x='qubit',
        data_var='width',
        title="Fit Width Comparison Across Qubits"
    )
    fig.figure.show()


def demo_multi_qubit_comparison(data_files):
    """Demo: Multi-qubit comparison and analysis."""
    print("\n" + "="*60)
    print("ADVANCED DEMO 4: Multi-Qubit Comparison")
    print("="*60)
    
    if 'ds_raw.h5' not in data_files:
        print("No multi-qubit data available for this demo")
        return
    
    ds = data_files['ds_raw.h5']
    print(f"Available qubits: {list(ds.coords['qubit'].values)}")
    
    # Create custom grid layouts for different qubit arrangements
    print("\nCreating custom grid layouts...")
    
    # 2x4 grid for 8 qubits
    qubit_coords_2x4 = {}
    for i, qubit in enumerate(ds.coords['qubit'].values):
        row = i // 4
        col = i % 4
        qubit_coords_2x4[qubit] = (row, col)
    
    grid_2x4 = QubitGrid(coords=qubit_coords_2x4, shape=(2, 4))
    
    fig = qplot.QualibrationFigure.plot(
        ds,
        x='detuning',
        data_var='IQ_abs',
        grid=grid_2x4,
        title="Multi-Qubit IQ Magnitude - 2x4 Grid"
    )
    fig.figure.show()
    
    # 4x2 grid
    qubit_coords_4x2 = {}
    for i, qubit in enumerate(ds.coords['qubit'].values):
        row = i // 2
        col = i % 2
        qubit_coords_4x2[qubit] = (row, col)
    
    grid_4x2 = QubitGrid(coords=qubit_coords_4x2, shape=(4, 2))
    
    fig = qplot.QualibrationFigure.plot(
        ds,
        x='detuning',
        data_var='IQ_abs',
        grid=grid_4x2,
        title="Multi-Qubit IQ Magnitude - 4x2 Grid"
    )
    fig.figure.show()
    
    # Per-qubit overlays with different reference lines
    print("\nCreating per-qubit overlays...")
    
    def create_qubit_overlays(qubit_name, qubit_data):
        """Create qubit-specific overlays."""
        overlays = []
        
        # Add qubit-specific reference line
        if 'qC' in qubit_name:
            overlays.append(RefLine(x=0, name="C-Qubit Zero"))
        elif 'qD' in qubit_name:
            overlays.append(RefLine(x=1e6, name="D-Qubit Offset"))
        
        # Add qubit-specific text annotation
        overlays.append(TextBoxOverlay(
            text=f"Qubit: {qubit_name}",
            anchor="top right"
        ))
        
        return overlays
    
    fig = qplot.QualibrationFigure.plot(
        ds,
        x='detuning',
        data_var='IQ_abs',
        overlays=create_qubit_overlays,
        title="Multi-Qubit with Per-Qubit Overlays"
    )
    fig.figure.show()


def demo_publication_ready_plots(data_files):
    """Demo: Publication-ready plots with advanced styling."""
    print("\n" + "="*60)
    print("ADVANCED DEMO 5: Publication-Ready Plots")
    print("="*60)
    
    if 'ds_raw.h5' not in data_files:
        print("No data available for this demo")
        return
    
    ds = data_files['ds_raw.h5']
    
    # Create publication-ready theme
    print("\nCreating publication-ready plots...")
    
    pub_theme = qplot.PlotTheme(
        font_size=16,
        title_size=20,
        tick_label_size=14,
        marker_size=8,
        line_width=3,
        show_grid=True,
        grid_opacity=0.3,
        figure_bg="white",
        paper_bg="white"
    )
    
    # Set publication theme
    qplot.set_theme(theme=pub_theme, palette="deep")
    
    # Create high-quality multi-qubit plot
    fig = qplot.QualibrationFigure.plot(
        ds,
        x='detuning',
        data_var='IQ_abs',
        title="Quantum Device Calibration: IQ Magnitude vs Detuning",
        residuals=True
    )
    fig.figure.show()
    
    # Create detailed single-qubit analysis
    qubit_idx = 0
    qubit_name = ds.coords['qubit'].values[qubit_idx]
    
    # Add comprehensive overlays
    detuning = ds.coords['detuning'].values
    
    # Create Gaussian fit overlay
    center = 0.5e6
    width = 1e6
    gaussian_fit = np.exp(-((detuning - center) / width)**2) * 0.1
    
    overlays = [
        LineOverlay(
            x=detuning,
            y=gaussian_fit,
            name="Gaussian Fit",
            dash="dash"
        ),
        RefLine(x=0, name="Zero Detuning"),
        RefLine(x=center, name="Fit Center"),
        TextBoxOverlay(
            text=f"Center: {center:.0f} Hz\nWidth: {width:.0f} Hz",
            anchor="top left"
        )
    ]
    
    fig = qplot.QualibrationFigure.plot(
        ds.isel(qubit=qubit_idx),
        x='detuning',
        data_var='IQ_abs',
        overlays=overlays,
        residuals=True,
        title=f"Detailed Analysis: {qubit_name}"
    )
    fig.figure.show()
    
    # Create comparison plot with multiple data variables
    print("\nCreating multi-variable comparison...")
    
    # Plot both I and Q components
    fig = qplot.QualibrationFigure.plot(
        ds.isel(qubit=qubit_idx),
        x='detuning',
        data_var='I',
        title=f"I Component - {qubit_name}"
    )
    fig.figure.show()
    
    fig = qplot.QualibrationFigure.plot(
        ds.isel(qubit=qubit_idx),
        x='detuning',
        data_var='Q',
        title=f"Q Component - {qubit_name}"
    )
    fig.figure.show()
    
    # Reset theme
    qplot.set_theme(theme=qplot.PlotTheme())


def demo_flux_tuning_fit_analysis(data_files):
    """Demo: Flux tuning fit analysis."""
    print("\n" + "="*60)
    print("ADVANCED DEMO 6: Flux Tuning Fit Analysis")
    print("="*60)
    
    if 'ds_fit_2.h5' not in data_files:
        print("No flux tuning fit data available for this demo")
        return
    
    ds = data_files['ds_fit_2.h5']
    print(f"Flux tuning fit data shape: {dict(ds.dims)}")
    print(f"Available qubits: {list(ds.coords['qubit'].values)}")
    
    # Plot peak frequency vs flux bias
    print("\nCreating peak frequency vs flux bias plots...")
    
    for i, qubit in enumerate(ds.coords['qubit'].values):
        fig = qplot.QualibrationFigure.plot(
            ds.isel(qubit=i),
            x='flux_bias',
            data_var='peak_freq',
            title=f"Peak Frequency vs Flux Bias - {qubit}"
        )
        fig.figure.show()
    
    # Plot fit results
    print("\nCreating fit results analysis...")
    
    # Create a dataset with fit results as data variables
    fit_results_ds = xr.Dataset({
        'a': (['qubit'], ds['fit_results'].isel(fit_vals=0).values),
        'f': (['qubit'], ds['fit_results'].isel(fit_vals=1).values),
        'phi': (['qubit'], ds['fit_results'].isel(fit_vals=2).values),
        'offset': (['qubit'], ds['fit_results'].isel(fit_vals=3).values)
    }, coords={'qubit': ds.coords['qubit']})
    
    for param in ['a', 'f', 'phi', 'offset']:
        fig = qplot.QualibrationFigure.plot(
            fit_results_ds,
            x='qubit',
            data_var=param,
            title=f"Fit Parameter {param} vs Qubit"
        )
        fig.figure.show()


def main():
    """Run all advanced demos."""
    print("Qualibration Plotting Module - Advanced Demos")
    print("=" * 60)
    
    # Load test data
    print("Loading test data...")
    data_files = load_test_data()
    
    if not data_files:
        print("No test data files found!")
        return
    
    print(f"Loaded {len(data_files)} data files")
    
    # Run advanced demos
    try:
        demo_flux_tuning_analysis(data_files)
        demo_power_optimization(data_files)
        demo_fit_quality_analysis(data_files)
        demo_multi_qubit_comparison(data_files)
        demo_publication_ready_plots(data_files)
        demo_flux_tuning_fit_analysis(data_files)
        
        print("\n" + "="*60)
        print("All advanced demos completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
