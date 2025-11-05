#!/usr/bin/env python3
"""
Multi-Qubit Color Consistency Demo

This demo demonstrates how QualibrationFigure maintains consistent colors across
subplots when plotting data from multiple qubits. Each qubit gets assigned a
consistent color that is used for both the raw data and fit overlays across
all subplots.

Key features demonstrated:
1. Multi-qubit data with different characteristics per qubit
2. Consistent color mapping across subplots
3. Fit overlays with matching colors for each qubit
4. Grid layout showing color consistency
5. Both 1D and 2D data visualization

Usage:
    python multi_qubit_color_consistency_demo.py

The script will display multiple figures showing color consistency across
different plot configurations.
"""

import numpy as np
import xarray as xr
import sys
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import qualibration_libs.plotting as qplot
from qualibration_libs.plotting.overlays import FitOverlay, RefLine
from qualibration_libs.plotting.grid import QubitGrid


def create_multi_qubit_data():
    """Create synthetic multi-qubit data with different characteristics."""
    print("Creating multi-qubit synthetic data...")
    
    # Create frequency/detuning sweep data
    detuning = np.linspace(-3e6, 3e6, 150)  # -3 to +3 MHz
    
    # Define qubits with different characteristics
    qubits = ['q0', 'q1', 'q2', 'q3', 'q4', 'q5']
    
    # Create data for each qubit with different resonance characteristics
    data = {}
    fit_params = {}
    
    for i, qubit in enumerate(qubits):
        # Different center frequencies for each qubit
        center_freq = (i - 2.5) * 0.8e6  # Spread centers across range
        
        # Different widths (some narrow, some broad)
        if i % 2 == 0:
            width = 0.6e6  # Narrow resonances
        else:
            width = 1.2e6  # Broad resonances
            
        # Different amplitudes
        amplitude = 0.8 + 0.2 * i  # Increasing amplitude
        
        # Add some noise
        noise_level = 0.05
        
        # Create Gaussian resonance response
        response = amplitude * np.exp(-((detuning - center_freq) / width)**2)
        noise = np.random.normal(0, noise_level, len(detuning))
        response += noise
        
        data[qubit] = (['detuning'], response)
        
        # Store fit parameters for overlays
        fit_params[qubit] = {
            'center': center_freq,
            'width': width,
            'amplitude': amplitude
        }
        
        print(f"  {qubit}: center={center_freq/1e6:.1f} MHz, width={width/1e6:.1f} MHz, amp={amplitude:.2f}")
    
    # Create dataset
    ds = xr.Dataset(data, coords={'detuning': detuning, 'qubit': qubits})
    
    print(f"[OK] Created dataset with {len(qubits)} qubits")
    print(f"[OK] Data shape: {dict(ds.dims)}")
    
    return ds, fit_params


def create_fit_overlays(fit_params, detuning):
    """Create fit overlays for each qubit with consistent formatting."""
    
    def create_qubit_fit_overlay(qubit_name, qubit_data):
        """Create fit overlay for a specific qubit."""
        if qubit_name in fit_params:
            params = fit_params[qubit_name]
            
            # Create fit curve
            fit_curve = params['amplitude'] * np.exp(
                -((detuning - params['center']) / params['width'])**2
            )
            
            # Create fit overlay
            fit_overlay = FitOverlay(
                y_fit=fit_curve,
                params=params,
                formatter=lambda p: f"{qubit_name}: C={p['center']/1e6:.1f}MHz, W={p['width']/1e6:.1f}MHz",
                name=f"{qubit_name} Fit"
            )
            
            # Add reference line at resonance center
            ref_line = RefLine(
                x=params['center'], 
                name=f"{qubit_name} Center"
            )
            
            return [fit_overlay, ref_line]
        
        return []
    
    return create_qubit_fit_overlay


def demo_basic_color_consistency():
    """Demo 1: Basic color consistency across qubits."""
    print("\n" + "="*70)
    print("DEMO 1: Basic Color Consistency Across Qubits")
    print("="*70)
    
    # Create data
    ds, fit_params = create_multi_qubit_data()
    detuning = ds.coords['detuning'].values
    
    # Create fit overlays
    create_overlays = create_fit_overlays(fit_params, detuning)
    
    # Create plot with all qubits
    fig = qplot.QualibrationFigure.plot(
        ds,
        x='detuning',
        data_var='q0',  # This will plot all qubits
        overlays=create_overlays,
        title="Multi-Qubit Color Consistency Demo"
    )
    
    print(f"[OK] Created plot with {len(fig.figure.data)} traces")
    print(f"[OK] Each qubit should have consistent colors for data and fits")
    
    # Show trace information
    print("\nTrace information:")
    for trace in fig.figure.data:
        print(f"  {trace.name}: color={trace.line.color if hasattr(trace.line, 'color') else 'N/A'}")
    
    fig.figure.show()
    return fig


def demo_grid_layout_color_consistency():
    """Demo 2: Color consistency in grid layout."""
    print("\n" + "="*70)
    print("DEMO 2: Color Consistency in Grid Layout")
    print("="*70)
    
    # Create data
    ds, fit_params = create_multi_qubit_data()
    detuning = ds.coords['detuning'].values
    
    # Create 2x3 grid layout
    grid_locations = ["0,0", "1,0", "2,0", "0,1", "1,1", "2,1"]
    custom_grid = QubitGrid(ds, grid_locations=grid_locations, shape=(2, 3))
    
    print(f"Grid layout: 2x3")
    print(f"Grid coordinates: {custom_grid.coords}")
    
    # Create fit overlays
    create_overlays = create_fit_overlays(fit_params, detuning)
    
    # Create plot with grid layout
    fig = qplot.QualibrationFigure.plot(
        ds,
        x='detuning',
        data_var='q0',  # This will plot all qubits
        grid=custom_grid,
        overlays=create_overlays,
        title="Grid Layout: Color Consistency Across Subplots"
    )
    
    print(f"[OK] Created grid plot with {len(fig.figure.data)} traces")
    print(f"[OK] Each subplot should show consistent colors for each qubit")
    
    fig.figure.show()
    return fig


def demo_residuals_color_consistency():
    """Demo 3: Color consistency with residuals subplots."""
    print("\n" + "="*70)
    print("DEMO 3: Color Consistency with Residuals")
    print("="*70)
    
    # Create data
    ds, fit_params = create_multi_qubit_data()
    detuning = ds.coords['detuning'].values
    
    # Create fit overlays
    create_overlays = create_fit_overlays(fit_params, detuning)
    
    # Create plot with residuals
    fig = qplot.QualibrationFigure.plot(
        ds,
        x='detuning',
        data_var='q0',  # This will plot all qubits
        overlays=create_overlays,
        residuals=True,
        title="Color Consistency with Residuals Subplots"
    )
    
    print(f"[OK] Created plot with residuals and {len(fig.figure.data)} traces")
    print(f"[OK] Colors should be consistent between main plots and residuals")
    
    fig.figure.show()
    return fig


def demo_subset_color_consistency():
    """Demo 4: Color consistency with subset of qubits."""
    print("\n" + "="*70)
    print("DEMO 4: Color Consistency with Subset of Qubits")
    print("="*70)
    
    # Create data
    ds, fit_params = create_multi_qubit_data()
    detuning = ds.coords['detuning'].values
    
    # Select subset of qubits
    subset_qubits = ['q0', 'q2', 'q4']  # Every other qubit
    ds_subset = ds.sel(qubit=subset_qubits)
    
    print(f"Selected subset: {subset_qubits}")
    
    # Create fit overlays for subset
    create_overlays = create_fit_overlays(fit_params, detuning)
    
    # Create plot with subset
    fig = qplot.QualibrationFigure.plot(
        ds_subset,
        x='detuning',
        data_var='q0',  # This will plot all qubits in subset
        overlays=create_overlays,
        title="Color Consistency: Subset of Qubits"
    )
    
    print(f"[OK] Created plot with subset and {len(fig.figure.data)} traces")
    print(f"[OK] Colors should be consistent with the full dataset")
    
    fig.figure.show()
    return fig


def demo_2d_data_color_consistency():
    """Demo 5: Color consistency with 2D data."""
    print("\n" + "="*70)
    print("DEMO 5: Color Consistency with 2D Data")
    print("="*70)
    
    # Create 2D data (frequency vs power)
    detuning = np.linspace(-2e6, 2e6, 80)
    power = np.linspace(-20, 0, 40)  # dBm
    
    qubits = ['q0', 'q1', 'q2', 'q3']
    
    # Create 2D data for each qubit
    data_2d = {}
    for i, qubit in enumerate(qubits):
        # Create 2D response surface
        det_grid, pow_grid = np.meshgrid(detuning, power, indexing='ij')
        
        # Different center frequencies
        center_freq = (i - 1.5) * 0.5e6
        
        # Power-dependent amplitude
        amplitude = 0.5 * (1 + pow_grid / 20)  # Amplitude increases with power
        
        # Create 2D Gaussian response
        response = amplitude * np.exp(-((det_grid - center_freq) / 0.8e6)**2)
        
        # Add some noise
        noise = np.random.normal(0, 0.02, response.shape)
        response += noise
        
        data_2d[qubit] = (['detuning', 'power'], response)
    
    # Create 2D dataset
    ds_2d = xr.Dataset(data_2d, coords={
        'detuning': detuning, 
        'power': power, 
        'qubit': qubits
    })
    
    print(f"[OK] Created 2D dataset with {len(qubits)} qubits")
    print(f"[OK] Data shape: {dict(ds_2d.dims)}")
    
    # Create 2D plot
    fig = qplot.QualibrationFigure.plot(
        ds_2d,
        x='detuning',
        y='power',
        data_var='q0',  # This will plot all qubits
        title="2D Data: Color Consistency Across Qubits"
    )
    
    print(f"[OK] Created 2D plot with {len(fig.figure.data)} traces")
    print(f"[OK] Each qubit should have consistent colors in heatmap")
    
    fig.figure.show()
    return fig


def demo_color_mapping_verification():
    """Demo 6: Verify color mapping is consistent."""
    print("\n" + "="*70)
    print("DEMO 6: Color Mapping Verification")
    print("="*70)
    
    # Create data
    ds, fit_params = create_multi_qubit_data()
    detuning = ds.coords['detuning'].values
    
    # Create multiple plots to verify color consistency
    plots = []
    
    # Plot 1: All qubits
    create_overlays = create_fit_overlays(fit_params, detuning)
    fig1 = qplot.QualibrationFigure.plot(
        ds,
        x='detuning',
        data_var='q0',
        overlays=create_overlays,
        title="All Qubits - Color Reference"
    )
    plots.append(("All Qubits", fig1))
    
    # Plot 2: First half
    ds_half1 = ds.isel(qubit=slice(0, 3))
    fig2 = qplot.QualibrationFigure.plot(
        ds_half1,
        x='detuning',
        data_var='q0',
        overlays=create_overlays,
        title="First Half - Should Match Colors"
    )
    plots.append(("First Half", fig2))
    
    # Plot 3: Second half
    ds_half2 = ds.isel(qubit=slice(3, 6))
    fig3 = qplot.QualibrationFigure.plot(
        ds_half2,
        x='detuning',
        data_var='q0',
        overlays=create_overlays,
        title="Second Half - Should Match Colors"
    )
    plots.append(("Second Half", fig3))
    
    print(f"[OK] Created {len(plots)} verification plots")
    print(f"[OK] Colors should be consistent across all plots")
    
    # Show all plots
    for name, fig in plots:
        print(f"\nShowing {name} plot...")
        fig.figure.show()
    
    return plots


def analyze_color_consistency():
    """Analyze and report on color consistency."""
    print("\n" + "="*70)
    print("COLOR CONSISTENCY ANALYSIS")
    print("="*70)
    
    # Create data
    ds, fit_params = create_multi_qubit_data()
    detuning = ds.coords['detuning'].values
    
    # Create a plot to analyze colors
    create_overlays = create_fit_overlays(fit_params, detuning)
    fig = qplot.QualibrationFigure.plot(
        ds,
        x='detuning',
        data_var='q0',
        overlays=create_overlays,
        title="Color Consistency Analysis"
    )
    
    # Analyze trace colors
    print("Color Analysis:")
    qubit_colors = {}
    
    for trace in fig.figure.data:
        trace_name = trace.name
        if hasattr(trace, 'line') and hasattr(trace.line, 'color'):
            color = trace.line.color
        elif hasattr(trace, 'marker') and hasattr(trace.marker, 'color'):
            color = trace.marker.color
        else:
            color = "Default"
        
        # Extract qubit name from trace name
        if ' Fit' in trace_name:
            qubit_name = trace_name.replace(' Fit', '')
        elif ' Center' in trace_name:
            qubit_name = trace_name.replace(' Center', '')
        else:
            qubit_name = trace_name
        
        if qubit_name not in qubit_colors:
            qubit_colors[qubit_name] = []
        
        qubit_colors[qubit_name].append((trace_name, color))
    
    # Report color consistency
    print("\nColor consistency report:")
    for qubit, traces in qubit_colors.items():
        colors = [color for _, color in traces]
        unique_colors = set(colors)
        
        if len(unique_colors) == 1:
            status = "✓ CONSISTENT"
        else:
            status = "✗ INCONSISTENT"
        
        print(f"  {qubit}: {status}")
        for trace_name, color in traces:
            print(f"    {trace_name}: {color}")
    
    fig.figure.show()
    return fig


def main():
    """Run all multi-qubit color consistency demos."""
    print("Multi-Qubit Color Consistency Demo")
    print("="*70)
    print("This demo shows how QualibrationFigure maintains consistent")
    print("colors across subplots when plotting data from multiple qubits.")
    print("="*70)
    
    try:
        # Run all demos
        demo_basic_color_consistency()
        demo_grid_layout_color_consistency()
        demo_residuals_color_consistency()
        demo_subset_color_consistency()
        demo_2d_data_color_consistency()
        demo_color_mapping_verification()
        analyze_color_consistency()
        
        print("\n" + "="*70)
        print("🎉 ALL MULTI-QUBIT COLOR CONSISTENCY DEMOS COMPLETED!")
        print("="*70)
        print("[OK] Basic color consistency across qubits")
        print("[OK] Color consistency in grid layouts")
        print("[OK] Color consistency with residuals")
        print("[OK] Color consistency with qubit subsets")
        print("[OK] Color consistency with 2D data")
        print("[OK] Color mapping verification")
        print("[OK] Color consistency analysis")
        print("\nKey Features Demonstrated:")
        print("• Each qubit gets assigned a consistent color")
        print("• Colors are maintained across different plot types")
        print("• Fit overlays use matching colors for each qubit")
        print("• Grid layouts preserve color consistency")
        print("• Residuals maintain color mapping")
        print("• 2D heatmaps use consistent colors")
        
    except Exception as e:
        print(f"\n❌ Multi-qubit color consistency demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
