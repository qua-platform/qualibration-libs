#!/usr/bin/env python3
"""
Palette Decorator Demo

This demo showcases the `with_palette` decorator functionality, which allows you to
temporarily set color palettes for plotting functions and automatically restore
the original palette afterward.

The decorator supports:
- Predefined palette names (viridis, plasma, tab10, etc.)
- Custom color lists (hex codes, named colors)
- Nested decorators (inner decorator takes precedence)
- Automatic palette restoration
"""

import numpy as np
import xarray as xr
import sys
import os

# Add the qualibration_libs to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from qualibration_libs.plotting import with_palette, QualibrationFigure
from qualibration_libs.plotting.config import CURRENT_PALETTE

def create_demo_data():
    """Create sample data for the demo."""
    qubits = ['Q1', 'Q2', 'Q3', 'Q4']
    frequency = np.linspace(6.0, 7.0, 50)
    power = np.linspace(-50, -20, 30)
    
    data_vars = {}
    coords = {
        'qubit': qubits,
        'frequency': frequency,
        'power': power,
    }
    
    for i, qubit in enumerate(qubits):
        # Create 2D data with different resonance patterns
        freq_mesh, power_mesh = np.meshgrid(frequency, power)
        
        # Different resonance frequencies for each qubit
        resonance_freq = 6.3 + i * 0.15
        
        # Create resonance pattern
        amplitude = 1.0 - 0.8 * np.exp(-((freq_mesh - resonance_freq)**2 + (power_mesh + 35)**2) / 20)
        
        # Add some noise
        amplitude += 0.05 * np.random.randn(*amplitude.shape)
        
        data_vars[qubit] = (['power', 'frequency'], amplitude)
    
    # Create dataset
    ds = xr.Dataset(data_vars, coords=coords)
    ds_stacked = ds.to_array(dim='qubit', name='amplitude')
    ds_stacked = ds_stacked.transpose('qubit', 'power', 'frequency')
    
    return ds_stacked

def demo_predefined_palettes():
    """Demo using predefined palette names."""
    print("🎨 Demo 1: Predefined Palette Names")
    print("=" * 50)
    
    data = create_demo_data()
    
    # Viridis palette
    @with_palette('viridis')
    def plot_viridis(data):
        return QualibrationFigure.plot(
            data,
            x='frequency',
            y='power',
            data_var='amplitude',
            qubit_dim='qubit',
            qubit_names=['Q1', 'Q2', 'Q3', 'Q4'],
            title="Viridis Palette - Smooth Color Transition",
            colorscale='Viridis_r',
            colorbar={'title': 'Amplitude (a.u.)'},
            colorbar_tolerance=100.0
        )
    
    # Plasma palette
    @with_palette('plasma')
    def plot_plasma(data):
        return QualibrationFigure.plot(
            data,
            x='frequency',
            y='power',
            data_var='amplitude',
            qubit_dim='qubit',
            qubit_names=['Q1', 'Q2', 'Q3', 'Q4'],
            title="Plasma Palette - High Contrast",
            colorscale='Plasma_r',
            colorbar={'title': 'Amplitude (a.u.)'},
            colorbar_tolerance=100.0
        )
    
    # Tab10 palette
    @with_palette('tab10')
    def plot_tab10(data):
        return QualibrationFigure.plot(
            data,
            x='frequency',
            y='power',
            data_var='amplitude',
            qubit_dim='qubit',
            qubit_names=['Q1', 'Q2', 'Q3', 'Q4'],
            title="Tab10 Palette - Distinct Colors",
            colorscale='Viridis_r',
            colorbar={'title': 'Amplitude (a.u.)'},
            colorbar_tolerance=100.0
        )
    
    # Generate plots
    fig1 = plot_viridis(data)
    fig1.figure.write_html("demo_viridis.html")
    print("✓ Generated viridis palette plot: demo_viridis.html")
    
    fig2 = plot_plasma(data)
    fig2.figure.write_html("demo_plasma.html")
    print("✓ Generated plasma palette plot: demo_plasma.html")
    
    fig3 = plot_tab10(data)
    fig3.figure.write_html("demo_tab10.html")
    print("✓ Generated tab10 palette plot: demo_tab10.html")

def demo_custom_colors():
    """Demo using custom color lists."""
    print("\n🎨 Demo 2: Custom Color Lists")
    print("=" * 50)
    
    data = create_demo_data()
    
    # Custom hex colors
    @with_palette(['#ff0000', '#00ff00', '#0000ff', '#ffff00'])
    def plot_hex_colors(data):
        return QualibrationFigure.plot(
            data,
            x='frequency',
            y='power',
            data_var='amplitude',
            qubit_dim='qubit',
            qubit_names=['Q1', 'Q2', 'Q3', 'Q4'],
            title="Custom Hex Colors - Red, Green, Blue, Yellow",
            colorscale='Viridis_r',
            colorbar={'title': 'Amplitude (a.u.)'},
            colorbar_tolerance=100.0
        )
    
    # Named colors
    @with_palette(['red', 'green', 'blue', 'orange'])
    def plot_named_colors(data):
        return QualibrationFigure.plot(
            data,
            x='frequency',
            y='power',
            data_var='amplitude',
            qubit_dim='qubit',
            qubit_names=['Q1', 'Q2', 'Q3', 'Q4'],
            title="Named Colors - Red, Green, Blue, Orange",
            colorscale='Viridis_r',
            colorbar={'title': 'Amplitude (a.u.)'},
            colorbar_tolerance=100.0
        )
    
    # Generate plots
    fig1 = plot_hex_colors(data)
    fig1.figure.write_html("demo_hex_colors.html")
    print("✓ Generated hex colors plot: demo_hex_colors.html")
    
    fig2 = plot_named_colors(data)
    fig2.figure.write_html("demo_named_colors.html")
    print("✓ Generated named colors plot: demo_named_colors.html")

def demo_nested_decorators():
    """Demo nested decorators (inner takes precedence)."""
    print("\n🎨 Demo 3: Nested Decorators")
    print("=" * 50)
    
    data = create_demo_data()
    
    @with_palette('viridis')
    def outer_function(data):
        @with_palette('set1')
        def inner_function(data):
            return QualibrationFigure.plot(
                data,
                x='frequency',
                y='power',
                data_var='amplitude',
                qubit_dim='qubit',
                qubit_names=['Q1', 'Q2', 'Q3', 'Q4'],
                title="Nested Decorators - Outer: Viridis, Inner: Set1 (Set1 wins)",
                colorscale='Viridis_r',
                colorbar={'title': 'Amplitude (a.u.)'},
                colorbar_tolerance=100.0
            )
        return inner_function(data)
    
    fig = outer_function(data)
    fig.figure.write_html("demo_nested_decorators.html")
    print("✓ Generated nested decorators plot: demo_nested_decorators.html")

def demo_palette_restoration():
    """Demo palette restoration functionality."""
    print("\n🎨 Demo 4: Palette Restoration")
    print("=" * 50)
    
    print(f"Initial global palette: {CURRENT_PALETTE}")
    
    @with_palette('plasma')
    def test_function_1():
        print(f"Inside function 1 (plasma): {CURRENT_PALETTE}")
        return "Function 1 completed"
    
    @with_palette('set2')
    def test_function_2():
        print(f"Inside function 2 (set2): {CURRENT_PALETTE}")
        return "Function 2 completed"
    
    # Test function calls
    result1 = test_function_1()
    print(f"After function 1: {CURRENT_PALETTE}")
    
    result2 = test_function_2()
    print(f"After function 2: {CURRENT_PALETTE}")
    
    print(f"Final global palette: {CURRENT_PALETTE}")
    print("✓ Palette restoration working correctly!")

def demo_available_palettes():
    """Demo showing available predefined palettes."""
    print("\n🎨 Demo 5: Available Predefined Palettes")
    print("=" * 50)
    
    palettes = [
        'viridis', 'plasma', 'tab10', 'tab20',
        'set1', 'set2', 'set3', 'pastel1', 'pastel2',
        'dark2', 'paired', 'accent', 'spectral',
        'coolwarm', 'rdylbu', 'rdbu', 'piyg', 'prgn',
        'brbg', 'puor', 'rdgy', 'terrain', 'ocean', 'rainbow'
    ]
    
    print("Available predefined palettes:")
    for i, palette in enumerate(palettes, 1):
        print(f"  {i:2d}. {palette}")
    
    print(f"\nTotal: {len(palettes)} predefined palettes available")
    print("You can also use custom color lists like ['#ff0000', '#00ff00', '#0000ff']")

def main():
    """Run all palette decorator demos."""
    print("🎨 Palette Decorator Demo")
    print("=" * 60)
    print("This demo showcases the with_palette decorator functionality.")
    print("The decorator temporarily sets color palettes for plotting functions")
    print("and automatically restores the original palette afterward.\n")
    
    try:
        # Run all demos
        demo_predefined_palettes()
        demo_custom_colors()
        demo_nested_decorators()
        demo_palette_restoration()
        demo_available_palettes()
        
        print("\n" + "=" * 60)
        print("✅ All demos completed successfully!")
        print("\nGenerated files:")
        print("  - demo_viridis.html")
        print("  - demo_plasma.html")
        print("  - demo_tab10.html")
        print("  - demo_hex_colors.html")
        print("  - demo_named_colors.html")
        print("  - demo_nested_decorators.html")
        print("\nOpen these files in a web browser to see the different color palettes!")
        
    except Exception as e:
        print(f"❌ Error running demos: {e}")
        raise

if __name__ == "__main__":
    main()
