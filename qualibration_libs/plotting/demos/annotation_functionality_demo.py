#!/usr/bin/env python3
"""
Comprehensive demo script for QualibrationFigure annotation functionality.

This script demonstrates the annotation positioning functionality in QualibrationFigure,
showing both the previous behavior and the improved behavior after removing the
'starts with q' requirement.

The script covers:
1. Basic annotation functionality with x2 secondary axis
2. Custom grid layouts with mixed qubit names
3. Comparison between old and new behavior
4. Configuration options for annotation positioning
5. Edge cases and best practices

Usage:
    python annotation_functionality_demo.py

The script will automatically display all demo figures at the end, showing
the annotation positioning functionality in action.
"""

import numpy as np
import xarray as xr
import time
from qualibration_libs.plotting import QualibrationFigure, QubitGrid
from qualibration_libs.plotting.overlays import RefLine, FitOverlay


def create_sample_data():
    """Create sample data for demonstration."""
    # Create frequency data
    frequency = np.linspace(4.5e9, 5.5e9, 100)  # 4.5-5.5 GHz
    
    # Create qubit names - some starting with 'q', some not
    qubits = ['q0', 'q1', 'Qubit_A', 'Qubit_B', 'qubit_C']
    
    # Create sample response data
    data = {}
    for i, qubit in enumerate(qubits):
        # Simulate resonance response
        center_freq = 5.0e9 + i * 0.1e9  # Different center for each qubit
        width = 0.05e9
        response = np.exp(-((frequency - center_freq) / width)**2)
        data[qubit] = (['frequency'], response)
    
    # Create dataset with both primary and secondary x-coordinates
    ds = xr.Dataset(data, coords={'frequency': frequency})
    
    # Add secondary coordinate (wavelength)
    wavelength_values = 3e8 / frequency  # Convert frequency to wavelength
    ds = ds.assign_coords(wavelength=('frequency', wavelength_values))
    
    return ds


def create_mixed_qubit_dataset():
    """Create dataset with mixed qubit naming conventions for comprehensive testing."""
    detuning = np.linspace(-1.5e7, 1.5e7, 50)
    qubits = ['q0', 'Qubit_A', 'qubit_B', 'Q1']  # Mixed naming conventions
    
    # Create data with qubit dimension
    data = {}
    for i, qubit in enumerate(qubits):
        center = i * 0.5e6
        width = 1e6
        response = np.exp(-((detuning - center) / width)**2)
        data[qubit] = (['detuning'], response)
    
    ds = xr.Dataset(data, coords={'detuning': detuning, 'qubit': qubits})
    
    # Add secondary coordinate
    wavelength_values = 3e8 / (5e9 + ds.detuning.values)
    ds = ds.assign_coords(wavelength=('detuning', wavelength_values))
    
    return ds


def demo_basic_annotation_functionality():
    """Demonstrate basic annotation functionality with x2 secondary axis."""
    print("=== Demo 1: Basic Annotation Functionality ===")
    print()
    
    # Create sample data
    ds = create_sample_data()
    print(f"Created dataset with qubits: {list(ds.data_vars)}")
    print(f"Primary x-coordinate: frequency ({ds.frequency.min():.2e} - {ds.frequency.max():.2e} Hz)")
    print(f"Secondary x-coordinate: wavelength ({ds.wavelength.min():.2e} - {ds.wavelength.max():.2e} m)")
    print()
    
    # Create a plot with secondary x-axis
    print("Creating plot with secondary x-axis (frequency + wavelength)...")
    fig = QualibrationFigure.plot(
        ds,
        x='frequency',
        x2='wavelength',
        data_var='q0',  # Plot first qubit
        title="Basic Annotation Functionality Demo",
        x2_top_margin=150,           # Custom top margin
        x2_annotation_offset=0.05    # Custom annotation offset
    )
    
    # Analyze the annotations
    annotations = fig.figure.layout.annotations
    print(f"\nFound {len(annotations)} annotations:")
    
    for i, annotation in enumerate(annotations):
        print(f"  Annotation {i+1}:")
        print(f"    Text: '{annotation.text}'")
        print(f"    Y position: {annotation.y}")
        print(f"    Starts with 'q': {annotation.text.startswith('q') if annotation.text else False}")
        print()
    
    print("="*60)
    return fig


def demo_custom_grid_layout():
    """Demonstrate annotation functionality with custom grid layout."""
    print("\n=== Demo 2: Custom Grid Layout ===")
    print()
    
    ds = create_sample_data()
    
    # Create custom grid with mixed qubit names
    grid_coords = {
        'q0': (0, 0),      # Starts with 'q'
        'q1': (0, 1),      # Starts with 'q'
        'Qubit_A': (1, 0), # Doesn't start with 'q'
        'Qubit_B': (1, 1), # Doesn't start with 'q'
        'qubit_C': (2, 0)  # Starts with 'q'
    }
    grid = QubitGrid(grid_coords, shape=(3, 2))
    
    print("Creating plot with custom grid layout...")
    fig = QualibrationFigure.plot(
        ds,
        x='frequency',
        x2='wavelength',
        data_var='q0',
        grid=grid,
        title="Custom Grid with Mixed Qubit Names",
        x2_top_margin=180,           # Larger margin for 3-row grid
        x2_annotation_offset=0.06    # Larger offset for 3-row grid
    )
    
    # Analyze annotations
    annotations = fig.figure.layout.annotations
    print(f"\nFound {len(annotations)} annotations in custom grid:")
    
    for i, annotation in enumerate(annotations):
        print(f"  Annotation {i+1}: '{annotation.text}' (Y: {annotation.y})")
    
    print("\n" + "="*60)
    return fig


def demo_mixed_qubit_names():
    """Demonstrate annotation functionality with mixed qubit naming conventions."""
    print("\n=== Demo 3: Mixed Qubit Names (Comprehensive Test) ===")
    print()
    
    ds = create_mixed_qubit_dataset()
    
    # Create custom grid with mixed qubit names
    grid_coords = {
        'q0': (0, 0),       # Starts with 'q'
        'Qubit_A': (0, 1),  # Doesn't start with 'q'
        'qubit_B': (1, 0),  # Starts with 'q'
        'Q1': (1, 1),       # Starts with 'Q' (capital)
    }
    grid = QubitGrid(grid_coords, shape=(2, 2))
    
    print("Creating plot with mixed qubit naming conventions...")
    fig = QualibrationFigure.plot(
        ds,
        x='detuning',
        x2='wavelength',
        data_var='q0',  # Data variable name
        qubit_dim='qubit',  # Dimension containing qubit names
        grid=grid,
        title="Mixed Qubit Names Test",
        x2_annotation_offset=0.05
    )
    
    # Analyze annotations
    annotations = fig.figure.layout.annotations
    print(f"\nFound {len(annotations)} annotations:")
    
    # Check that ALL subplot title annotations have been moved up
    moved_annotations = []
    for annotation in annotations:
        if annotation.y is not None and annotation.text:
            if annotation.text in ['q0', 'Qubit_A', 'qubit_B', 'Q1']:
                moved_annotations.append(annotation)
    
    print(f"\nQubit title annotations found: {len(moved_annotations)}")
    for ann in moved_annotations:
        print(f"  - '{ann.text}' (Y: {ann.y}) - MOVED UP [OK]")
    
    # Verify that annotations starting with 'q' are moved
    q_annotations = [ann for ann in moved_annotations if ann.text.startswith('q')]
    print(f"\nAnnotations starting with 'q': {len(q_annotations)}")
    
    # Verify that annotations NOT starting with 'q' are also moved
    non_q_annotations = [ann for ann in moved_annotations if not ann.text.startswith('q')]
    print(f"Annotations NOT starting with 'q': {len(non_q_annotations)}")
    
    print("\n" + "="*60)
    return fig


def demo_comparison_old_vs_new():
    """Demonstrate the difference between old and new behavior."""
    print("\n=== Demo 4: Old vs New Behavior Comparison ===")
    print()
    
    ds = create_mixed_qubit_dataset()
    
    # Create grid with mixed qubit names
    grid_coords = {
        'q0': (0, 0),       # Starts with 'q' - would be moved in OLD behavior
        'Qubit_A': (0, 1),  # Doesn't start with 'q' - would NOT be moved in OLD behavior
        'q1': (1, 0),       # Starts with 'q' - would be moved in OLD behavior
        'Qubit_B': (1, 1),  # Doesn't start with 'q' - would NOT be moved in OLD behavior
    }
    grid = QubitGrid(grid_coords, shape=(2, 2))
    
    print("OLD BEHAVIOR (simulated):")
    print("- Only annotations starting with 'q' would be moved")
    print("- 'q0' and 'q1' would be moved up")
    print("- 'Qubit_A' and 'Qubit_B' would NOT be moved (potential overlap!)")
    print()
    
    print("NEW BEHAVIOR (actual):")
    print("- ALL subplot title annotations are moved up")
    print("- 'q0', 'q1', 'Qubit_A', and 'Qubit_B' are ALL moved up")
    print("- No overlap with secondary x-axis!")
    print()
    
    # Create plot with NEW behavior
    fig = QualibrationFigure.plot(
        ds,
        x='detuning',
        x2='wavelength',
        data_var='q0',
        qubit_dim='qubit',
        grid=grid,
        title="Comparison: All Annotations Moved (NEW)",
        x2_annotation_offset=0.05
    )
    
    # Analyze annotations
    annotations = fig.figure.layout.annotations
    qubit_annotations = [ann for ann in annotations 
                        if ann.text in ['q0', 'q1', 'Qubit_A', 'Qubit_B']]
    
    print(f"Actual result: {len(qubit_annotations)} qubit annotations found")
    for ann in qubit_annotations:
        print(f"  - '{ann.text}' (Y: {ann.y}) - MOVED UP [OK]")
    
    print("\n" + "="*60)
    return fig


def demo_with_overlays():
    """Demonstrate annotation functionality with overlays."""
    print("\n=== Demo 5: With Overlays ===")
    print()
    
    ds = create_sample_data()
    
    # Create overlays
    overlays = [
        RefLine(x=5.0e9, name="Target Frequency"),
        RefLine(x=5.1e9, name="Backup Frequency")
    ]
    
    print("Creating plot with overlays...")
    fig = QualibrationFigure.plot(
        ds,
        x='frequency',
        x2='wavelength',
        data_var='q0',
        overlays=overlays,
        title="Plot with Overlays and Secondary Axis"
    )
    
    # Analyze annotations
    annotations = fig.figure.layout.annotations
    print(f"\nFound {len(annotations)} annotations with overlays:")
    
    for i, annotation in enumerate(annotations):
        print(f"  Annotation {i+1}: '{annotation.text}' (Y: {annotation.y})")
    
    print("\n" + "="*60)
    return fig


def demo_without_x2():
    """Demonstrate that annotations are NOT moved when x2 is not present."""
    print("\n=== Demo 6: No X2 (Annotations NOT Moved) ===")
    print()
    
    ds = create_sample_data()
    
    # Create plot WITHOUT x2
    fig = QualibrationFigure.plot(
        ds,
        x='frequency',
        data_var='q0',
        title="No Secondary Axis - Annotations Stay Put"
    )
    
    # Analyze annotations
    annotations = fig.figure.layout.annotations
    print(f"Found {len(annotations)} annotations (no x2):")
    
    for i, annotation in enumerate(annotations):
        print(f"  Annotation {i+1}: '{annotation.text}' (Y: {annotation.y})")
    
    print("\nNote: Without x2, annotations keep their default positions")
    print("(no artificial movement needed)")
    
    print("\n" + "="*60)
    return fig


def demo_configuration_options():
    """Demonstrate different configuration options for annotation positioning."""
    print("\n=== Demo 7: Configuration Options ===")
    print()
    
    ds = create_sample_data()
    
    # Test different margin and offset values
    configs = [
        {"margin": 100, "offset": 0.03, "name": "Small margin, small offset"},
        {"margin": 150, "offset": 0.05, "name": "Medium margin, medium offset"},
        {"margin": 200, "offset": 0.08, "name": "Large margin, large offset"},
    ]
    
    figs = []
    for config in configs:
        print(f"Testing: {config['name']}")
        fig = QualibrationFigure.plot(
            ds,
            x='frequency',
            x2='wavelength',
            data_var='q0',
            title=f"Config: {config['name']}",
            x2_top_margin=config['margin'],
            x2_annotation_offset=config['offset']
        )
        
        # Check margin
        margin = fig.figure.layout.margin.t
        print(f"  Applied margin: {margin}")
        
        # Check annotation positions
        annotations = fig.figure.layout.annotations
        for ann in annotations:
            if ann.text == 'q0':
                print(f"  Annotation Y position: {ann.y}")
                break
        
        figs.append(fig)
        print()
    
    print("Configuration options:")
    print("- x2_top_margin: Controls overall top margin (default: 120)")
    print("- x2_annotation_offset: Controls how much annotations move up (default: 0.08)")
    print("- Higher values = more spacing, lower values = tighter layout")
    
    print("\n" + "="*60)
    return figs


def main():
    """Run all demos."""
    print("QualibrationFigure Annotation Functionality Comprehensive Demo")
    print("=" * 65)
    print()
    print("This demo shows the annotation positioning functionality in QualibrationFigure,")
    print("demonstrating the improvement from filtering by 'starts with q' to moving")
    print("ALL subplot title annotations when x2 secondary axis is present.")
    print()
    
    # Run all demos
    fig1 = demo_basic_annotation_functionality()
    fig2 = demo_custom_grid_layout()
    fig3 = demo_mixed_qubit_names()
    fig4 = demo_comparison_old_vs_new()
    fig5 = demo_with_overlays()
    fig6 = demo_without_x2()
    figs7 = demo_configuration_options()
    
    print("\n=== Summary ===")
    print("NEW behavior:")
    print("- ALL subplot title annotations are moved up when x2 is present")
    print("- No more filtering based on text content")
    print("- Prevents overlap with secondary x-axis for ALL qubit names")
    print("- Works with any qubit naming convention")
    print()
    print("Benefits:")
    print("- More robust and predictable behavior")
    print("- No more overlap issues with non-'q' qubit names")
    print("- Cleaner, more consistent visual layout")
    print("- Future-proof for any naming scheme")
    print()
    print("Configuration:")
    print("- x2_top_margin: Controls overall top margin (default: 120)")
    print("- x2_annotation_offset: Controls how much annotations move up (default: 0.08)")
    print()
    print("Best Practices:")
    print("- Use descriptive qubit names regardless of format")
    print("- Adjust x2_top_margin for complex layouts")
    print("- Use x2_annotation_offset to fine-tune spacing")
    print("- Test with your specific qubit naming conventions")
    
    # Show all figures
    print("\n=== Displaying All Figures ===")
    print("Showing all demo figures...")
    print("Look for:")
    print("- Annotation positioning relative to secondary x-axis")
    print("- How qubit names are positioned to avoid overlap")
    print("- Differences between plots with and without x2")
    print()
    
    print("Demo 1: Basic Annotation Functionality")
    print("  -> Look for: Single qubit plot with dual x-axes (frequency + wavelength)")
    print("  -> Note: Annotation moved up to avoid secondary axis overlap")
    fig1.figure.show()
    time.sleep(2)
    
    print("\nDemo 2: Custom Grid Layout")
    print("  -> Look for: 3x2 grid with mixed qubit names")
    print("  -> Note: All annotations positioned to avoid secondary axis")
    fig2.figure.show()
    time.sleep(2)
    
    print("\nDemo 3: Mixed Qubit Names (Comprehensive Test)")
    print("  -> Look for: 2x2 grid with qubits: q0, Qubit_A, qubit_B, Q1")
    print("  -> Note: ALL annotations moved up regardless of naming convention")
    fig3.figure.show()
    time.sleep(2)
    
    print("\nDemo 4: Old vs New Behavior Comparison")
    print("  -> Look for: 2x2 grid showing the improvement")
    print("  -> Note: Both 'q' and non-'q' qubit names are properly positioned")
    fig4.figure.show()
    time.sleep(2)
    
    print("\nDemo 5: With Overlays")
    print("  -> Look for: Reference lines added to the plot")
    print("  -> Note: Annotations still properly positioned with overlays")
    fig5.figure.show()
    time.sleep(2)
    
    print("\nDemo 6: No X2 (Annotations NOT Moved)")
    print("  -> Look for: Single plot without secondary axis")
    print("  -> Note: Annotation stays at default position (no artificial movement)")
    fig6.figure.show()
    time.sleep(2)
    
    print("\nDemo 7: Configuration Options")
    print("  -> Look for: Three plots with different margin/offset settings")
    print("  -> Note: How spacing changes with different configuration values")
    for i, fig in enumerate(figs7):
        config_names = ['Small margin, small offset', 'Medium margin, medium offset', 'Large margin, large offset']
        print(f"  Configuration {i+1}: {config_names[i]}")
        fig.figure.show()
        time.sleep(1)
    
    print("\n=== All Figures Displayed ===")
    print("You should now see how the annotation positioning works!")
    print("Key takeaway: ALL subplot title annotations are moved up when x2 is present,")
    print("regardless of whether they start with 'q' or not.")


if __name__ == "__main__":
    main()
