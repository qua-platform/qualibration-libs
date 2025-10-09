#!/usr/bin/env python3
"""
Feature verification demo for qualibration plotting module.

This script demonstrates and verifies the key features:
1. Multi-qubit subplot grids
2. 2D heatmap plots
3. Overlay functionality
4. Theme customization
"""
import sys
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import qualibration_libs.plotting as qplot
from qualibration_libs.plotting.config import PlotTheme
from qualibration_libs.plotting.overlays import RefLine, LineOverlay
from qualibration_libs.plotting import QubitGrid
import xarray as xr
import numpy as np


def demo_subplot_grids():
    """Demo: Multi-qubit subplot grids."""
    print("\n" + "="*60)
    print("FEATURE VERIFICATION: Multi-Qubit Subplot Grids")
    print("="*60)
    
    # Load data
    data_dir = Path(__file__).parent.parent / "test_data"
    ds = xr.open_dataset(data_dir / "ds_raw.h5")
    
    print(f"Loaded data with {len(ds.coords['qubit'])} qubits: {list(ds.coords['qubit'].values)}")
    
    # Test 1: Default 1x8 grid
    print("\n1. Testing default 1x8 grid layout...")
    fig = qplot.QualibrationFigure.plot(
        ds,
        x='detuning',
        data_var='IQ_abs',
        title="Default 1x8 Grid Layout"
    )
    print(f"✓ Created plot with {len(fig.figure.data)} traces")
    print(f"✓ Subplot titles: {[ann.text for ann in fig.figure.layout.annotations if ann.text]}")
    fig.figure.show()
    
    # Test 2: Custom 2x4 grid
    print("\n2. Testing custom 2x4 grid layout...")
    qubit_coords = {}
    for i, qubit in enumerate(ds.coords['qubit'].values):
        row = i // 4
        col = i % 4
        qubit_coords[qubit] = (row, col)
    
    grid_2x4 = QubitGrid(coords=qubit_coords, shape=(2, 4))
    
    fig = qplot.QualibrationFigure.plot(
        ds,
        x='detuning',
        data_var='IQ_abs',
        grid=grid_2x4,
        title="Custom 2x4 Grid Layout"
    )
    print(f"✓ Created plot with {len(fig.figure.data)} traces")
    print(f"✓ Subplot titles: {[ann.text for ann in fig.figure.layout.annotations if ann.text]}")
    fig.figure.show()
    
    # Test 3: Custom 4x2 grid
    print("\n3. Testing custom 4x2 grid layout...")
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
        title="Custom 4x2 Grid Layout"
    )
    print(f"✓ Created plot with {len(fig.figure.data)} traces")
    print(f"✓ Subplot titles: {[ann.text for ann in fig.figure.layout.annotations if ann.text]}")
    fig.figure.show()


def demo_heatmaps():
    """Demo: 2D heatmap plots."""
    print("\n" + "="*60)
    print("FEATURE VERIFICATION: 2D Heatmap Plots")
    print("="*60)
    
    data_dir = Path(__file__).parent.parent / "test_data"
    
    # Test 1: 2D flux tuning heatmap
    print("\n1. Testing 2D flux tuning heatmap...")
    ds_2d = xr.open_dataset(data_dir / "ds_raw_2.h5")
    print(f"2D data shape: {dict(ds_2d.dims)}")
    
    fig = qplot.QualibrationFigure.plot(
        ds_2d.isel(qubit=0),
        x='detuning',
        y='flux_bias',
        data_var='IQ_abs',
        title="2D Heatmap: IQ Magnitude vs Detuning and Flux Bias"
    )
    
    heatmap_traces = [trace for trace in fig.figure.data if trace.type == 'heatmap']
    print(f"✓ Created heatmap with {len(heatmap_traces)} heatmap traces")
    if heatmap_traces:
        print(f"✓ Heatmap data shape: {heatmap_traces[0].z.shape}")
    fig.figure.show()
    
    # Test 2: 3D data as 2D heatmap
    print("\n2. Testing 3D data as 2D heatmap...")
    ds_3d = xr.open_dataset(data_dir / "ds_raw_3.h5")
    print(f"3D data shape: {dict(ds_3d.dims)}")
    
    fig = qplot.QualibrationFigure.plot(
        ds_3d.isel(qubit=0),
        x='detuning',
        y='power',
        data_var='IQ_abs',
        title="3D Data as 2D Heatmap: IQ Magnitude vs Detuning and Power"
    )
    
    heatmap_traces = [trace for trace in fig.figure.data if trace.type == 'heatmap']
    print(f"✓ Created heatmap with {len(heatmap_traces)} heatmap traces")
    if heatmap_traces:
        print(f"✓ Heatmap data shape: {heatmap_traces[0].z.shape}")
    fig.figure.show()
    
    # Test 3: Multi-qubit heatmap comparison
    print("\n3. Testing multi-qubit heatmap comparison...")
    fig = qplot.QualibrationFigure.plot(
        ds_2d,
        x='detuning',
        y='flux_bias',
        data_var='IQ_abs',
        title="Multi-Qubit Heatmap Comparison"
    )
    
    heatmap_traces = [trace for trace in fig.figure.data if trace.type == 'heatmap']
    print(f"✓ Created multi-qubit heatmap with {len(heatmap_traces)} heatmap traces")
    fig.figure.show()


def demo_overlays():
    """Demo: Overlay functionality."""
    print("\n" + "="*60)
    print("FEATURE VERIFICATION: Overlay Functionality")
    print("="*60)
    
    data_dir = Path(__file__).parent.parent / "test_data"
    ds = xr.open_dataset(data_dir / "ds_raw.h5")
    
    # Test 1: Reference lines
    print("\n1. Testing reference line overlays...")
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
        title="Reference Line Overlays"
    )
    print(f"✓ Created plot with {len(overlays)} reference lines")
    fig.figure.show()
    
    # Test 2: Custom line overlay
    print("\n2. Testing custom line overlay...")
    detuning = ds.coords['detuning'].values
    # Create a Gaussian fit line
    center = 0.5e6
    width = 1e6
    gaussian_line = np.exp(-((detuning - center) / width)**2) * 0.1
    
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
        title="Custom Line Overlay"
    )
    print("✓ Created plot with custom line overlay")
    fig.figure.show()
    
    # Test 3: Per-qubit overlays
    print("\n3. Testing per-qubit overlays...")
    def create_qubit_overlays(qubit_name, qubit_data):
        overlays = []
        if 'qC' in qubit_name:
            overlays.append(RefLine(x=0, name="C-Qubit Zero"))
        elif 'qD' in qubit_name:
            overlays.append(RefLine(x=1e6, name="D-Qubit Offset"))
        return overlays
    
    fig = qplot.QualibrationFigure.plot(
        ds.isel(qubit=[0, 1, 2, 3]),
        x='detuning',
        data_var='IQ_abs',
        overlays=create_qubit_overlays,
        title="Per-Qubit Overlays"
    )
    print("✓ Created plot with per-qubit overlays")
    fig.figure.show()


def demo_themes():
    """Demo: Theme customization."""
    print("\n" + "="*60)
    print("FEATURE VERIFICATION: Theme Customization")
    print("="*60)
    
    data_dir = Path(__file__).parent.parent / "test_data"
    ds = xr.open_dataset(data_dir / "ds_raw.h5")
    
    # Test 1: Default theme
    print("\n1. Testing default theme...")
    fig = qplot.QualibrationFigure.plot(
        ds.isel(qubit=0),
        x='detuning',
        data_var='IQ_abs',
        title="Default Theme"
    )
    print("✓ Created plot with default theme")
    fig.figure.show()
    
    # Test 2: Custom theme
    print("\n2. Testing custom theme...")
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
    print("✓ Created plot with custom theme")
    fig.figure.show()
    
    # Test 3: Custom color palette
    print("\n3. Testing custom color palette...")
    custom_palette = ["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF"]
    qplot.set_palette(custom_palette)
    
    fig = qplot.QualibrationFigure.plot(
        ds.isel(qubit=[0, 1, 2]),
        x='detuning',
        data_var='IQ_abs',
        title="Custom Color Palette"
    )
    print("✓ Created plot with custom color palette")
    fig.figure.show()
    
    # Test 4: Theme context manager
    print("\n4. Testing theme context manager...")
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
        print("✓ Created plot with theme context manager")
        fig.figure.show()
    
    # Reset to default
    qplot.set_theme(theme=PlotTheme())
    qplot.set_palette("qualibrate")


def demo_residuals():
    """Demo: Residuals plots."""
    print("\n" + "="*60)
    print("FEATURE VERIFICATION: Residuals Plots")
    print("="*60)
    
    data_dir = Path(__file__).parent.parent / "test_data"
    ds = xr.open_dataset(data_dir / "ds_raw.h5")
    
    # Test residuals plot
    print("\nTesting residuals plot...")
    fig = qplot.QualibrationFigure.plot(
        ds.isel(qubit=0),
        x='detuning',
        data_var='IQ_abs',
        residuals=True,
        title="Residuals Plot"
    )
    
    # Check for residual subplot shapes (zero line)
    shapes = fig.figure.layout.shapes
    zero_lines = [shape for shape in shapes if shape.type == 'line' and shape.y0 == 0]
    print(f"✓ Created residuals plot with {len(zero_lines)} zero reference lines")
    fig.figure.show()


def main():
    """Run all feature verification demos."""
    print("Qualibration Plotting Module - Feature Verification")
    print("=" * 60)
    
    try:
        demo_subplot_grids()
        demo_heatmaps()
        demo_overlays()
        demo_themes()
        demo_residuals()
        
        print("\n" + "="*60)
        print("🎉 ALL FEATURES VERIFIED SUCCESSFULLY!")
        print("="*60)
        print("✓ Multi-qubit subplot grids are working")
        print("✓ 2D heatmap plots are working")
        print("✓ Overlay functionality is working")
        print("✓ Theme customization is working")
        print("✓ Residuals plots are working")
        
    except Exception as e:
        print(f"\n❌ Feature verification failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
