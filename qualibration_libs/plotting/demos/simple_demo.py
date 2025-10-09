#!/usr/bin/env python3
"""
Simple demo script to test plotting functionality with real data.
"""
import sys
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import qualibration_libs.plotting as qplot
from qualibration_libs.plotting.config import PlotTheme
from qualibration_libs.plotting.overlays import RefLine
import xarray as xr


def main():
    """Run a simple demo with real data."""
    print("Simple Qualibration Plotting Demo")
    print("=" * 40)
    
    # Load test data
    data_dir = Path(__file__).parent.parent / "test_data"
    data_file = data_dir / "ds_raw.h5"
    
    if not data_file.exists():
        print(f"Data file not found: {data_file}")
        return
    
    print(f"Loading data from {data_file.name}...")
    ds = xr.open_dataset(data_file)
    print(f"✓ Loaded data with shape: {dict(ds.dims)}")
    print(f"  Qubits: {list(ds.coords['qubit'].values)}")
    
    # Create a simple plot
    print("\nCreating simple IQ magnitude plot...")
    fig = qplot.QualibrationFigure.plot(
        ds.isel(qubit=0),  # Select first qubit
        x='detuning',
        data_var='IQ_abs',
        title="IQ Magnitude vs Detuning - qC1"
    )
    print("✓ Plot created successfully")
    
    # Show the plot
    print("Displaying plot...")
    fig.figure.show()
    
    # Create plot with overlays
    print("\nCreating plot with reference lines...")
    overlays = [
        RefLine(x=0, name="Zero Detuning"),
        RefLine(x=1e6, name="1 MHz Offset")
    ]
    
    fig = qplot.QualibrationFigure.plot(
        ds.isel(qubit=0),
        x='detuning',
        data_var='IQ_abs',
        overlays=overlays,
        title="IQ Magnitude with Reference Lines - qC1"
    )
    print("✓ Plot with overlays created successfully")
    fig.figure.show()
    
    # Create multi-qubit plot
    print("\nCreating multi-qubit plot...")
    fig = qplot.QualibrationFigure.plot(
        ds,
        x='detuning',
        data_var='IQ_abs',
        title="Multi-Qubit IQ Magnitude"
    )
    print("✓ Multi-qubit plot created successfully")
    fig.figure.show()
    
    # Test theme functionality
    print("\nTesting theme functionality...")
    custom_theme = PlotTheme(
        font_size=18,
        marker_size=10,
        line_width=3
    )
    
    qplot.set_theme(theme=custom_theme)
    
    fig = qplot.QualibrationFigure.plot(
        ds.isel(qubit=0),
        x='detuning',
        data_var='IQ_abs',
        title="Custom Theme Plot"
    )
    print("✓ Custom theme applied successfully")
    fig.figure.show()
    
    print("\n🎉 Demo completed successfully!")


if __name__ == "__main__":
    main()
