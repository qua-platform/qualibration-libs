#!/usr/bin/env python3
"""
QubitGrid Demo - Showcasing Enhanced Grid Functionality

This demo demonstrates the new QubitGrid features including:
- Dataset integration with automatic grid creation
- String-based grid location parsing
- Mixed coordinate formats
- Plotly figure integration with QualibrationFigure
- Custom grid layouts for different qubit arrangements
- Sparse grid layouts with empty spots
"""

import numpy as np
import xarray as xr
import sys
from pathlib import Path

# Import the plotting module
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import qualibration_libs.plotting as qplot
from qualibration_libs.plotting.grid import QubitGrid, grid_iter


def create_sample_data():
    """Create sample datasets for demonstration."""
    print("Creating sample datasets...")
    
    # Create 1D frequency sweep data
    detuning = np.linspace(-2e6, 2e6, 100)
    qubits_1d = ['qC1', 'qC2', 'qC3', 'qC4']
    
    data_1d = {}
    for i, qubit in enumerate(qubits_1d):
        center = i * 0.5e6
        width = 0.8e6
        response = np.exp(-((detuning - center) / width)**2)
        data_1d[qubit] = (['detuning'], response)
    
    ds_1d = xr.Dataset(data_1d, coords={'detuning': detuning, 'qubit': qubits_1d})
    
    # Create 2D flux tuning data
    detuning_2d = np.linspace(-1e6, 1e6, 50)
    flux_bias = np.linspace(-0.5, 0.5, 25)
    qubits_2d = ['qD1', 'qD2', 'qD3', 'qD4', 'qD5', 'qD6']
    
    data_2d = {}
    for i, qubit in enumerate(qubits_2d):
        det_grid, flux_grid = np.meshgrid(detuning_2d, flux_bias, indexing='ij')
        center_shift = i * 0.2e6 * flux_grid  # Flux-dependent frequency shift
        response = np.exp(-((det_grid - center_shift) / 0.5e6)**2)
        data_2d[qubit] = (['detuning', 'flux_bias'], response)
    
    ds_2d = xr.Dataset(data_2d, coords={
        'detuning': detuning_2d, 
        'flux_bias': flux_bias, 
        'qubit': qubits_2d
    })
    
    print(f"[OK] Created 1D dataset with {len(qubits_1d)} qubits")
    print(f"[OK] Created 2D dataset with {len(qubits_2d)} qubits")
    
    return ds_1d, ds_2d


def demo_basic_coordinate_grid():
    """Demo 1: Basic coordinate-based grid (backward compatibility)."""
    print("\n" + "="*60)
    print("DEMO 1: Basic Coordinate-Based Grid")
    print("="*60)
    
    # Create a simple coordinate-based grid
    coords = {
        'Q0': (0, 0),
        'Q1': (0, 1), 
        'Q2': (1, 0),
        'Q3': (1, 1)
    }
    
    grid = QubitGrid(coords, shape=(2, 2))
    print(f"Created grid with coordinates: {grid.coords}")
    print(f"Grid shape: {grid.shape}")
    
    # Test resolution
    n_rows, n_cols, positions = grid.resolve(['Q0', 'Q1', 'Q2', 'Q3'])
    print(f"Resolved to {n_rows}x{n_cols} grid")
    print(f"Positions: {positions}")
    
    return grid


def demo_string_based_grid(ds_1d):
    """Demo 2: String-based grid location parsing."""
    print("\n" + "="*60)
    print("DEMO 2: String-Based Grid Location Parsing")
    print("="*60)
    
    # Create grid using string locations
    grid_locations = ["0,0", "0,1", "1,0", "1,1"]
    print(f"Grid locations (col,row format): {grid_locations}")
    
    grid = QubitGrid(ds_1d, grid_locations=grid_locations, shape=(2, 2))
    print(f"Parsed coordinates: {grid.coords}")
    print(f"Matplotlib figure created: {hasattr(grid, 'fig')}")
    
    # Show the coordinate mapping
    print("\nCoordinate mapping:")
    for i, loc in enumerate(grid_locations):
        qubit = ds_1d.coords['qubit'].values[i]
        coords = grid.coords[qubit]
        print(f"  '{loc}' -> {qubit}: {coords}")
    
    return grid


def demo_mixed_coordinate_formats(ds_1d):
    """Demo 3: Mixed string and tuple coordinate formats."""
    print("\n" + "="*60)
    print("DEMO 3: Mixed String and Tuple Coordinate Formats")
    print("="*60)
    
    # Mix string and tuple formats
    grid_locations = ["0,0", (0, 1), "1,0", (1, 1)]
    print(f"Mixed grid locations: {grid_locations}")
    
    grid = QubitGrid(ds_1d, grid_locations=grid_locations, shape=(2, 2))
    print(f"Parsed coordinates: {grid.coords}")
    
    # Show how different formats map to the same result
    print("\nFormat comparison:")
    for i, loc in enumerate(grid_locations):
        qubit = ds_1d.coords['qubit'].values[i]
        coords = grid.coords[qubit]
        loc_type = "string" if isinstance(loc, str) else "tuple"
        print(f"  {loc_type:6} {loc} -> {qubit}: {coords}")
    
    return grid


def demo_automatic_shape_detection(ds_2d):
    """Demo 4: Automatic shape detection from coordinates."""
    print("\n" + "="*60)
    print("DEMO 4: Automatic Shape Detection")
    print("="*60)
    
    # Create a larger grid without specifying shape
    grid_locations = [
        "0,0", "0,1", "0,2",
        "1,0", "1,1", "1,2"
    ]
    print(f"Grid locations: {grid_locations}")
    
    grid = QubitGrid(ds_2d, grid_locations=grid_locations)  # No shape specified
    print(f"Automatic shape detection: {grid.shape}")
    
    # Test resolution
    n_rows, n_cols, positions = grid.resolve(ds_2d.coords['qubit'].values)
    print(f"Resolved shape: {n_rows}x{n_cols}")
    print(f"Number of positioned qubits: {len(positions)}")
    
    return grid


def demo_custom_qubit_arrangements(ds_1d):
    """Demo 5: Custom qubit arrangements for different layouts."""
    print("\n" + "="*60)
    print("DEMO 5: Custom Qubit Arrangements")
    print("="*60)
    
    # Linear arrangement
    print("\nLinear arrangement (1x4):")
    linear_locations = ["0,0", "0,1", "0,2", "0,3"]
    linear_grid = QubitGrid(ds_1d, grid_locations=linear_locations, shape=(1, 4))
    print(f"  Coordinates: {linear_grid.coords}")
    
    # Square arrangement
    print("\nSquare arrangement (2x2):")
    square_locations = ["0,0", "0,1", "1,0", "1,1"]
    square_grid = QubitGrid(ds_1d, grid_locations=square_locations, shape=(2, 2))
    print(f"  Coordinates: {square_grid.coords}")
    
    # L-shaped arrangement
    print("\nL-shaped arrangement:")
    l_locations = ["0,0", "0,1", "1,0"]  # Only 3 qubits
    l_grid = QubitGrid(ds_1d, grid_locations=l_locations, shape=(2, 2))
    print(f"  Coordinates: {l_grid.coords}")
    
    return linear_grid, square_grid, l_grid


def demo_plotly_integration(ds_1d):
    """Demo 6: Plotly figure integration with QubitGrid."""
    print("\n" + "="*60)
    print("DEMO 6: Plotly Integration with QubitGrid")
    print("="*60)
    
    # Create custom grid
    grid_locations = ["0,0", "0,1", "1,0", "1,1"]
    custom_grid = QubitGrid(ds_1d, grid_locations=grid_locations, shape=(2, 2))
    
    print(f"Grid coordinates: {custom_grid.coords}")
    print(f"Grid shape: {custom_grid.shape}")
    
    # Test grid resolution
    n_rows, n_cols, positions = custom_grid.resolve(ds_1d.coords['qubit'].values)
    print(f"Resolved to {n_rows}x{n_cols} grid")
    print(f"Positions: {positions}")
    
    # Demonstrate grid iteration
    print("\nGrid iteration:")
    for ax, qubit_info in grid_iter(custom_grid):
        print(f"  {qubit_info['qubit']}: row={qubit_info['row']}, col={qubit_info['col']}")
    
    # Use the grid with QualibrationFigure
    print("\nCreating Plotly figure with custom QubitGrid...")
    fig = qplot.QualibrationFigure.plot(
        ds_1d,
        x='detuning',
        data_var='qC1',  # This will plot all qubits
        grid=custom_grid,
        title="QubitGrid Plotly Integration Demo"
    )
    
    print(f"Plot created with {len(fig.figure.data)} traces")
    fig.figure.show()
    
    return fig


def demo_plotting_integration(ds_1d):
    """Demo 7: Integration with QualibrationFigure plotting."""
    print("\n" + "="*60)
    print("DEMO 7: QualibrationFigure Integration")
    print("="*60)
    
    # Create custom grid
    grid_locations = ["0,0", "0,1", "1,0", "1,1"]
    custom_grid = QubitGrid(ds_1d, grid_locations=grid_locations, shape=(2, 2))
    
    print("Creating plot with custom QubitGrid...")
    
    # Use the grid with QualibrationFigure
    fig = qplot.QualibrationFigure.plot(
        ds_1d,
        x='detuning',
        data_var='qC1',  # This will plot all qubits
        grid=custom_grid,
        title="Custom QubitGrid Layout"
    )
    
    print(f"Plot created with {len(fig.figure.data)} traces")
    fig.figure.show()
    
    return fig


def demo_sparse_3x3_grid():
    """Demo 8: Sparse 3x3 grid with empty spots."""
    print("\n" + "="*60)
    print("DEMO 8: Sparse 3x3 Grid with Empty Spots")
    print("="*60)
    
    # Create a dataset with 5 qubits for the sparse demo
    detuning = np.linspace(-2e6, 2e6, 100)
    qubits_sparse = ['qS1', 'qS2', 'qS3', 'qS4', 'qS5']
    
    data_sparse = {}
    for i, qubit in enumerate(qubits_sparse):
        center = i * 0.4e6
        width = 0.6e6
        response = np.exp(-((detuning - center) / width)**2)
        data_sparse[qubit] = (['detuning'], response)
    
    ds_sparse = xr.Dataset(data_sparse, coords={'detuning': detuning, 'qubit': qubits_sparse})
    print(f"Created sparse dataset with {len(qubits_sparse)} qubits")
    
    # Create a 3x3 grid with only 5 qubits (4 empty spots)
    # Position qubits in a cross pattern: center + 4 corners
    sparse_locations = [
        "0,0",  # Top-left
        "2,0",  # Top-right  
        "1,1",  # Center
        "0,2",  # Bottom-left
        "2,2"   # Bottom-right
    ]
    print(f"Sparse grid locations: {sparse_locations}")
    print("Grid layout (X = qubit, O = empty):")
    print("X O X")
    print("O X O") 
    print("X O X")
    
    # Create the sparse grid
    sparse_grid = QubitGrid(ds_sparse, grid_locations=sparse_locations, shape=(3, 3))
    print(f"\nParsed coordinates: {sparse_grid.coords}")
    
    # Show coordinate mapping
    print("\nCoordinate mapping:")
    for i, loc in enumerate(sparse_locations):
        qubit = ds_sparse.coords['qubit'].values[i]
        coords = sparse_grid.coords[qubit]
        print(f"  '{loc}' -> {qubit}: {coords}")
    
    # Test resolution
    n_rows, n_cols, positions = sparse_grid.resolve(ds_sparse.coords['qubit'].values)
    print(f"\nResolved to {n_rows}x{n_cols} grid")
    print(f"Number of positioned qubits: {len(positions)}")
    print(f"Empty spots: {n_rows * n_cols - len(positions)}")
    
    # Demonstrate grid iteration (only iterates over positioned qubits)
    print("\nGrid iteration (only positioned qubits):")
    for ax, qubit_info in grid_iter(sparse_grid):
        print(f"  {qubit_info['qubit']}: row={qubit_info['row']}, col={qubit_info['col']}")
    
    # Create Plotly figure with sparse grid
    print("\nCreating Plotly figure with sparse 3x3 QubitGrid...")
    fig = qplot.QualibrationFigure.plot(
        ds_sparse,
        x='detuning',
        data_var='qS1',  # This will plot all qubits
        grid=sparse_grid,
        title="Sparse 3x3 QubitGrid (5 qubits, 4 empty spots)"
    )
    
    print(f"Plot created with {len(fig.figure.data)} traces")
    fig.figure.show()
    
    return sparse_grid


def demo_real_data_grid():
    """Demo 9: Using real data with logical grid layout."""
    print("\n" + "="*60)
    print("DEMO 9: Real Data with Logical Grid Layout")
    print("="*60)
    
    # Load real data
    import xarray as xr
    data_path = Path(__file__).parent.parent / "test_data" / "ds_raw.h5"
    ds_real = xr.open_dataset(data_path)
    
    print(f"Loaded real dataset with {len(ds_real.coords['qubit'])} qubits")
    print(f"Qubit names: {list(ds_real.coords['qubit'].values)}")
    print(f"Data variables: {list(ds_real.data_vars)}")
    print(f"Dimensions: {dict(ds_real.dims)}")
    
    # Create logical grid layout based on qubit naming convention
    # qC1-qC4: Control qubits (top row)
    # qD1-qD4: Data qubits (bottom row)
    # This suggests a 2x4 grid layout
    real_grid_locations = [
        "0,0",  # qC1 - top-left
        "1,0",  # qC2 - top-center-left  
        "2,0",  # qC3 - top-center-right
        "3,0",  # qC4 - top-right
        "0,1",  # qD1 - bottom-left
        "1,1",  # qD2 - bottom-center-left
        "2,1",  # qD3 - bottom-center-right
        "3,1"   # qD4 - bottom-right
    ]
    
    print(f"\nLogical grid layout (2x4):")
    print("qC1 qC2 qC3 qC4  (Control qubits)")
    print("qD1 qD2 qD3 qD4  (Data qubits)")
    print(f"Grid locations: {real_grid_locations}")
    
    # Create the real data grid
    real_grid = QubitGrid(ds_real, grid_locations=real_grid_locations, shape=(2, 4))
    print(f"\nParsed coordinates: {real_grid.coords}")
    
    # Show coordinate mapping
    print("\nCoordinate mapping:")
    for i, loc in enumerate(real_grid_locations):
        qubit = ds_real.coords['qubit'].values[i]
        coords = real_grid.coords[qubit]
        print(f"  '{loc}' -> {qubit}: {coords}")
    
    # Test resolution
    n_rows, n_cols, positions = real_grid.resolve(ds_real.coords['qubit'].values)
    print(f"\nResolved to {n_rows}x{n_cols} grid")
    print(f"Number of positioned qubits: {len(positions)}")
    
    # Demonstrate grid iteration
    print("\nGrid iteration:")
    for ax, qubit_info in grid_iter(real_grid):
        print(f"  {qubit_info['qubit']}: row={qubit_info['row']}, col={qubit_info['col']}")
    
    # Create Plotly figure with real data
    print("\nCreating Plotly figure with real data and logical grid...")
    fig = qplot.QualibrationFigure.plot(
        ds_real,
        x='detuning',
        data_var='IQ_abs',  # Use the magnitude data
        grid=real_grid,
        title="Real Data: 8-Qubit System (2x4 Grid Layout)"
    )
    
    print(f"Plot created with {len(fig.figure.data)} traces")
    fig.figure.show()
    
    return real_grid


def demo_edge_cases():
    """Demo 10: Edge cases and error handling."""
    print("\n" + "="*60)
    print("DEMO 10: Edge Cases and Error Handling")
    print("="*60)
    
    # Empty grid
    print("\nEmpty grid:")
    empty_grid = QubitGrid()
    print(f"  Coordinates: {empty_grid.coords}")
    print(f"  Shape: {empty_grid.shape}")
    
    # Grid with missing qubits
    print("\nGrid with missing qubits:")
    coords = {'Q0': (0, 0), 'Q1': (0, 1)}
    partial_grid = QubitGrid(coords, shape=(2, 2))
    n_rows, n_cols, positions = partial_grid.resolve(['Q0', 'Q1', 'Q2'])  # Q2 missing
    print(f"  Requested: ['Q0', 'Q1', 'Q2']")
    print(f"  Found: {list(positions.keys())}")
    
    # String parsing edge cases
    print("\nString parsing edge cases:")
    detuning = np.linspace(-1e6, 1e6, 50)
    qubits = ['Q1', 'Q2']
    ds = xr.Dataset({
        'signal': (['detuning'], np.random.rand(50))
    }, coords={'detuning': detuning, 'qubit': qubits})
    
    edge_locations = ["0, 0", "0,1", "1, 0", "1,1"]  # With spaces
    edge_grid = QubitGrid(ds, grid_locations=edge_locations, shape=(2, 2))
    print(f"  Locations with spaces: {edge_locations}")
    print(f"  Parsed coordinates: {edge_grid.coords}")


def main():
    """Run all QubitGrid demos."""
    print("QubitGrid Enhanced Functionality Demo")
    print("="*60)
    
    # Create sample data
    ds_1d, ds_2d = create_sample_data()
    
    # Run all demos
    demo_basic_coordinate_grid()
    demo_string_based_grid(ds_1d)
    demo_mixed_coordinate_formats(ds_1d)
    demo_automatic_shape_detection(ds_2d)
    demo_custom_qubit_arrangements(ds_1d)
    demo_plotly_integration(ds_1d)
    demo_plotting_integration(ds_1d)
    demo_sparse_3x3_grid()
    demo_real_data_grid()
    demo_edge_cases()
    
    print("\n" + "="*60)
    print("All QubitGrid demos completed successfully!")
    print("="*60)
    print("\nKey Features Demonstrated:")
    print("[OK] Backward compatibility with coordinate-based grids")
    print("[OK] String-based grid location parsing ('col,row' format)")
    print("[OK] Mixed string and tuple coordinate formats")
    print("[OK] Automatic shape detection from coordinates")
    print("[OK] Custom qubit arrangements (linear, square, L-shaped)")
    print("[OK] Sparse grid layouts (3x3 with empty spots)")
    print("[OK] Real data integration with logical grid layouts")
    print("[OK] Plotly figure integration and grid iteration")
    print("[OK] Integration with QualibrationFigure plotting")
    print("[OK] Edge case handling and error resilience")


if __name__ == "__main__":
    main()
