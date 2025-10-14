"""
Basic plotting demos using real test data.

This script demonstrates fundamental plotting capabilities with the provided
test data files.
"""
import os
import sys
import numpy as np
import xarray as xr
from pathlib import Path

# Add the parent directory to the path to import qualibration_libs
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import qualibration_libs.plotting as qplot
from qualibration_libs.plotting.overlays import RefLine, LineOverlay, FitOverlay


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
                print(f"[ERROR] Failed to load {filename}: {e}")
    
    return data_files


def demo_1d_frequency_sweep(data_files):
    """Demo: 1D frequency sweep plots."""
    print("\n" + "="*60)
    print("DEMO 1: 1D Frequency Sweep Plots")
    print("="*60)
    
    if 'ds_raw.h5' not in data_files:
        print("No raw data available for this demo")
        return
    
    ds = data_files['ds_raw.h5']
    print(f"Data shape: {dict(ds.dims)}")
    print(f"Qubits: {list(ds.coords['qubit'].values)}")
    print(f"Detuning range: {ds.coords['detuning'].min().values:.2e} to {ds.coords['detuning'].max().values:.2e} Hz")
    
    # Plot IQ magnitude for all qubits
    print("\nCreating IQ magnitude plot for all qubits...")
    fig = qplot.QualibrationFigure.plot(
        ds,
        x='detuning',
        data_var='IQ_abs',
        title="IQ Magnitude vs Detuning - All Qubits"
    )
    fig.figure.show()
    
    # Plot phase for a single qubit
    print("\nCreating phase plot for single qubit...")
    fig = qplot.QualibrationFigure.plot(
        ds.isel(qubit=0),  # Select first qubit
        x='detuning',
        data_var='phase',
        title="Phase vs Detuning - qC1"
    )
    fig.figure.show()
    
    # Plot with residuals (empty residuals subplot - no fit data)
    print("\nCreating plot with residuals (no fit data)...")
    fig = qplot.QualibrationFigure.plot(
        ds.isel(qubit=0),
        x='detuning',
        data_var='IQ_abs',
        residuals=True,
        title="IQ Magnitude with Residuals (No Fit Data) - qC1"
    )
    fig.figure.show()
    
    # Plot with residuals AND fit data (proper residuals demo)
    print("\nCreating plot with residuals AND fit data...")
    from qualibration_libs.plotting.overlays import FitOverlay
    import numpy as np
    
    # Create a simple fit overlay for demonstration
    detuning = ds.coords['detuning'].values
    # Create a simple Gaussian fit
    center = 0.0  # Center at 0 Hz
    width = 2e6   # 2 MHz width
    fit_curve = np.exp(-((detuning - center) / width)**2)
    
    fit_overlay = FitOverlay(
        y_fit=fit_curve,
        params={'center': center, 'width': width},
        formatter=lambda p: f"Center: {p['center']:.0f} Hz, Width: {p['width']:.0f} Hz"
    )
    
    fig = qplot.QualibrationFigure.plot(
        ds.isel(qubit=0),
        x='detuning',
        data_var='IQ_abs',
        overlays=[fit_overlay],
        residuals=True,
        title="IQ Magnitude with Residuals AND Fit Data - qC1"
    )
    fig.figure.show()


def demo_2d_heatmaps(data_files):
    """Demo: 2D heatmap plots."""
    print("\n" + "="*60)
    print("DEMO 2: 2D Heatmap Plots")
    print("="*60)
    
    if 'ds_raw_2.h5' not in data_files:
        print("No 2D raw data available for this demo")
        return
    
    ds = data_files['ds_raw_2.h5']
    print(f"Data shape: {dict(ds.dims)}")
    print(f"Flux bias range: {ds.coords['flux_bias'].min().values:.2f} to {ds.coords['flux_bias'].max().values:.2f}")
    print(f"Detuning range: {ds.coords['detuning'].min().values:.2e} to {ds.coords['detuning'].max().values:.2e} Hz")
    
    # 2D heatmap: IQ magnitude vs detuning and flux bias
    print("\nCreating 2D heatmap: IQ magnitude vs detuning and flux bias...")
    fig = qplot.QualibrationFigure.plot(
        ds.isel(qubit=0),  # Select first qubit
        x='detuning',
        y='flux_bias',
        data_var='IQ_abs',
        title="IQ Magnitude: Detuning vs Flux Bias - qC1"
    )
    fig.figure.show()
    
    # 2D heatmap: Phase vs detuning and flux bias
    print("\nCreating 2D heatmap: Phase vs detuning and flux bias...")
    fig = qplot.QualibrationFigure.plot(
        ds.isel(qubit=0),
        x='detuning',
        y='flux_bias',
        data_var='phase',
        title="Phase: Detuning vs Flux Bias - qC1"
    )
    fig.figure.show()


def demo_power_sweep(data_files):
    """Demo: Power sweep plots."""
    print("\n" + "="*60)
    print("DEMO 3: Power Sweep Plots")
    print("="*60)
    
    if 'ds_raw_3.h5' not in data_files:
        print("No power sweep data available for this demo")
        return
    
    ds = data_files['ds_raw_3.h5']
    print(f"Data shape: {dict(ds.dims)}")
    print(f"Power range: {ds.coords['power'].min().values:.1f} to {ds.coords['power'].max().values:.1f} dBm")
    
    # 2D heatmap: IQ magnitude vs detuning and power
    print("\nCreating 2D heatmap: IQ magnitude vs detuning and power...")
    fig = qplot.QualibrationFigure.plot(
        ds.isel(qubit=0),  # Select first qubit
        x='detuning',
        y='power',
        data_var='IQ_abs',
        title="IQ Magnitude: Detuning vs Power - qC1"
    )
    fig.figure.show()
    
    # 1D plot at optimal power
    if 'optimal_power' in ds.coords:
        optimal_power = ds.coords['optimal_power'].values[0]
        print(f"\nOptimal power for qC1: {optimal_power:.1f} dBm")
        
        # Find closest power index
        power_idx = np.argmin(np.abs(ds.coords['power'].values - optimal_power))
        
        print(f"Creating 1D plot at optimal power ({optimal_power:.1f} dBm)...")
        fig = qplot.QualibrationFigure.plot(
            ds.isel(qubit=0, power=power_idx),
            x='detuning',
            data_var='IQ_abs',
            title=f"IQ Magnitude at Optimal Power ({optimal_power:.1f} dBm) - qC1"
        )
        fig.figure.show()


def demo_fit_results(data_files):
    """Demo: Fit results visualization."""
    print("\n" + "="*60)
    print("DEMO 4: Fit Results Visualization")
    print("="*60)
    
    if 'ds_fit.h5' not in data_files:
        print("No fit data available for this demo")
        return
    
    ds = data_files['ds_fit.h5']
    print(f"Fit data shape: {dict(ds.dims)}")
    print(f"Successful fits: {ds.coords['success'].sum().values}/{len(ds.coords['qubit'])}")
    
    # Plot baseline fits
    print("\nCreating baseline fit plots...")
    fig = qplot.QualibrationFigure.plot(
        ds,
        x='detuning',
        data_var='base_line',
        title="Baseline Fits - All Qubits"
    )
    fig.figure.show()
    
    # Plot fit parameters
    print("\nCreating fit parameter plots...")
    
    # Amplitude vs qubit
    fig = qplot.QualibrationFigure.plot(
        ds,
        x='qubit',
        data_var='amplitude',
        title="Fit Amplitude vs Qubit"
    )
    fig.figure.show()
    
    # Width vs qubit
    fig = qplot.QualibrationFigure.plot(
        ds,
        x='qubit',
        data_var='width',
        title="Fit Width vs Qubit"
    )
    fig.figure.show()


def demo_overlays(data_files):
    """Demo: Overlay functionality."""
    print("\n" + "="*60)
    print("DEMO 5: Overlay Functionality")
    print("="*60)
    
    if 'ds_raw.h5' not in data_files:
        print("No raw data available for this demo")
        return
    
    ds = data_files['ds_raw.h5']
    
    # Reference lines
    print("\nCreating plot with reference lines...")
    overlays = [
        RefLine(x=0, name="Zero Detuning"),
        RefLine(x=1e6, name="1 MHz Offset"),
        RefLine(x=-1e6, name="-1 MHz Offset")
    ]
    
    fig = qplot.QualibrationFigure.plot(
        ds.isel(qubit=0),
        x='detuning',
        data_var='IQ_abs',
        overlays=overlays,
        title="IQ Magnitude with Reference Lines - qC1"
    )
    fig.figure.show()
    
    # Custom line overlay
    print("\nCreating plot with custom line overlay...")
    detuning = ds.coords['detuning'].values
    # Create a Gaussian fit line
    center = 0.5e6  # 0.5 MHz offset
    width = 1e6    # 1 MHz width
    gaussian_line = np.exp(-((detuning - center) / width)**2)
    
    line_overlay = LineOverlay(
        x=detuning,
        y=gaussian_line,
        name="Gaussian Fit",
        dash="dash"
    )
    
    fig = qplot.QualibrationFigure.plot(
        ds.isel(qubit=0),
        x='detuning',
        data_var='IQ_abs',
        overlays=[line_overlay],
        title="IQ Magnitude with Gaussian Fit - qC1"
    )
    fig.figure.show()
    
    # Fit overlay with parameters
    print("\nCreating plot with fit overlay and parameters...")
    fit_params = {
        'center': center,
        'width': width,
        'amplitude': 0.1
    }
    
    def param_formatter(params):
        return f"Center: {params['center']:.0f} Hz\nWidth: {params['width']:.0f} Hz\nAmplitude: {params['amplitude']:.3f}"
    
    fit_overlay = FitOverlay(
        y_fit=gaussian_line,
        params=fit_params,
        formatter=param_formatter,
        name="Fit"
    )
    
    fig = qplot.QualibrationFigure.plot(
        ds.isel(qubit=0),
        x='detuning',
        data_var='IQ_abs',
        overlays=[fit_overlay],
        title="IQ Magnitude with Fit Parameters - qC1"
    )
    fig.figure.show()


def demo_custom_grids(data_files):
    """Demo: Custom qubit grid layouts."""
    print("\n" + "="*60)
    print("DEMO 6: Custom Qubit Grid Layouts")
    print("="*60)
    
    if 'ds_raw.h5' not in data_files:
        print("No raw data available for this demo")
        return
    
    ds = data_files['ds_raw.h5']
    
    # Custom 2x4 grid layout
    print("\nCreating custom 2x4 grid layout...")
    from qualibration_libs.plotting import QubitGrid
    
    # Create a 2x4 grid for 8 qubits
    qubit_coords = {}
    for i, qubit in enumerate(ds.coords['qubit'].values):
        row = i // 4
        col = i % 4
        qubit_coords[qubit] = (row, col)
    
    custom_grid = QubitGrid(
        qubit_coords,
        shape=(2, 4)
    )
    
    fig = qplot.QualibrationFigure.plot(
        ds,
        x='detuning',
        data_var='IQ_abs',
        grid=custom_grid,
        title="Custom 2x4 Qubit Grid Layout"
    )
    fig.figure.show()
    
    # Custom 4x2 grid layout
    print("\nCreating custom 4x2 grid layout...")
    qubit_coords_4x2 = {}
    for i, qubit in enumerate(ds.coords['qubit'].values):
        row = i // 2
        col = i % 2
        qubit_coords_4x2[qubit] = (row, col)
    
    custom_grid_4x2 = QubitGrid(
        qubit_coords_4x2,
        shape=(4, 2)
    )
    
    fig = qplot.QualibrationFigure.plot(
        ds,
        x='detuning',
        data_var='IQ_abs',
        grid=custom_grid_4x2,
        title="Custom 4x2 Qubit Grid Layout"
    )
    fig.figure.show()


def demo_styling(data_files):
    """Demo: Styling and theming."""
    print("\n" + "="*60)
    print("DEMO 7: Styling and Theming")
    print("="*60)
    
    if 'ds_raw.h5' not in data_files:
        print("No raw data available for this demo")
        return
    
    ds = data_files['ds_raw.h5']
    
    # Default theme
    print("\nCreating plot with default theme...")
    fig = qplot.QualibrationFigure.plot(
        ds.isel(qubit=0),
        x='detuning',
        data_var='IQ_abs',
        title="Default Theme"
    )
    fig.figure.show()
    
    # Custom theme
    print("\nCreating plot with custom theme...")
    from qualibration_libs.plotting.config import PlotTheme
    custom_theme = PlotTheme(
        font_size=18,
        marker_size=10,
        line_width=3,
        show_grid=True,
        grid_opacity=0.5,
        figure_bg="lightgray"
    )
    
    qplot.set_theme(theme=custom_theme)
    
    fig = qplot.QualibrationFigure.plot(
        ds.isel(qubit=0),
        x='detuning',
        data_var='IQ_abs',
        title="Custom Theme"
    )
    fig.figure.show()
    
    # Custom color palette
    print("\nCreating plot with custom color palette...")
    custom_palette = ["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF"]
    qplot.set_palette(custom_palette)
    
    fig = qplot.QualibrationFigure.plot(
        ds.isel(qubit=0),
        x='detuning',
        data_var='IQ_abs',
        title="Custom Color Palette"
    )
    fig.figure.show()
    
    # Theme context manager
    print("\nCreating plot with theme context manager...")
    with qplot.theme_context(
        theme=PlotTheme(font_size=20, marker_size=12),
        palette="deep"
    ):
        fig = qplot.QualibrationFigure.plot(
            ds.isel(qubit=0),
            x='detuning',
            data_var='IQ_abs',
            title="Theme Context Manager"
        )
        fig.figure.show()
    
    # Reset to default
    qplot.set_theme(theme=PlotTheme())
    qplot.set_palette("qualibrate")


def main():
    """Run all demos."""
    print("Qualibration Plotting Module - Basic Demos")
    print("=" * 60)
    
    # Load test data
    print("Loading test data...")
    data_files = load_test_data()
    
    if not data_files:
        print("No test data files found!")
        return
    
    print(f"Loaded {len(data_files)} data files")
    
    # Run demos
    try:
        demo_1d_frequency_sweep(data_files)
        demo_2d_heatmaps(data_files)
        demo_power_sweep(data_files)
        demo_fit_results(data_files)
        demo_overlays(data_files)
        demo_custom_grids(data_files)
        demo_styling(data_files)
        
        print("\n" + "="*60)
        print("All demos completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
