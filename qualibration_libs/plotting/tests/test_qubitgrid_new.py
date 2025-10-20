"""
Unit tests for the new QubitGrid implementation.
Tests the enhanced features including dataset integration and string parsing.
"""
import pytest
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from qualibration_libs.plotting.grid import QubitGrid, grid_iter


class TestQubitGridNew:
    """Test cases for the new QubitGrid implementation."""
    
    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset for testing."""
        detuning = np.linspace(-1e6, 1e6, 50)
        qubits = ['qC1', 'qC2', 'qC3', 'qC4']
        
        data = {}
        for i, qubit in enumerate(qubits):
            # Create realistic IQ data
            center = i * 0.2e6
            width = 0.5e6
            response = np.exp(-((detuning - center) / width)**2)
            data[qubit] = (['detuning'], response)
        
        return xr.Dataset(data, coords={'detuning': detuning, 'qubit': qubits})
    
    def test_basic_coords_constructor(self):
        """Test basic coordinate-based constructor (backward compatibility)."""
        coords = {'Q0': (0, 0), 'Q1': (0, 1), 'Q2': (1, 0)}
        grid = QubitGrid(coords, shape=(2, 2))
        
        assert grid.coords == coords
        assert grid.shape == (2, 2)
        assert not hasattr(grid, 'fig')  # No matplotlib figure created
    
    def test_dataset_with_string_locations(self, sample_dataset):
        """Test creating QubitGrid with dataset and string grid locations."""
        grid_locations = ["0,0", "0,1", "1,0", "1,1"]
        grid = QubitGrid(sample_dataset, grid_locations=grid_locations, shape=(2, 2))
        
        # Check that coordinates were parsed correctly
        # String format is "col,row", stored as (row, col)
        expected_coords = {
            'qC1': (0, 0),  # "0,0" -> row=0, col=0
            'qC2': (1, 0),  # "0,1" -> row=1, col=0
            'qC3': (0, 1),  # "1,0" -> row=0, col=1
            'qC4': (1, 1)   # "1,1" -> row=1, col=1
        }
        assert grid.coords == expected_coords
        assert grid.shape == (2, 2)
        
        # Check that matplotlib figure is created lazily (not immediately)
        assert not hasattr(grid, 'fig')
        assert not hasattr(grid, '_axes')
        assert hasattr(grid, '_qubit_names')
        assert not hasattr(grid, '_positions')
    
    def test_dataset_with_tuple_locations(self, sample_dataset):
        """Test creating QubitGrid with dataset and tuple grid locations."""
        grid_locations = [(0, 0), (0, 1), (1, 0), (1, 1)]
        grid = QubitGrid(sample_dataset, grid_locations=grid_locations, shape=(2, 2))
        
        expected_coords = {
            'qC1': (0, 0),  # (0,0) -> (0,0)
            'qC2': (1, 0),  # (0,1) -> (1,0)
            'qC3': (0, 1),  # (1,0) -> (0,1)
            'qC4': (1, 1)   # (1,1) -> (1,1)
        }
        assert grid.coords == expected_coords
        assert not hasattr(grid, 'fig')  # Created lazily
    
    def test_mixed_string_tuple_locations(self, sample_dataset):
        """Test creating QubitGrid with mixed string and tuple locations."""
        grid_locations = ["0,0", (0, 1), "1,0", (1, 1)]
        grid = QubitGrid(sample_dataset, grid_locations=grid_locations, shape=(2, 2))
        
        expected_coords = {
            'qC1': (0, 0),  # "0,0" -> row=0, col=0
            'qC2': (1, 0),  # (0, 1) -> row=1, col=0
            'qC3': (0, 1),  # "1,0" -> row=0, col=1
            'qC4': (1, 1)   # (1, 1) -> row=1, col=1
        }
        assert grid.coords == expected_coords
    
    def test_automatic_shape_detection(self, sample_dataset):
        """Test automatic shape detection from coordinates."""
        grid_locations = ["0,0", "0,1", "1,0", "1,1"]
        grid = QubitGrid(sample_dataset, grid_locations=grid_locations)  # No shape specified
        
        n_rows, n_cols, positions = grid.resolve(['qC1', 'qC2', 'qC3', 'qC4'])
        assert n_rows == 2
        assert n_cols == 2
    
    def test_resolve_with_present_qubits(self, sample_dataset):
        """Test resolve method with subset of qubits."""
        grid_locations = ["0,0", "0,1", "1,0", "1,1"]
        grid = QubitGrid(sample_dataset, grid_locations=grid_locations, shape=(2, 2))
        
        # Test with subset of qubits
        n_rows, n_cols, positions = grid.resolve(['qC1', 'qC3'])
        
        assert n_rows == 2
        assert n_cols == 2
        assert 'qC1' in positions
        assert 'qC3' in positions
        assert 'qC2' not in positions
        assert 'qC4' not in positions
        assert positions['qC1'] == (2, 1)  # 1-indexed: (0,0) -> (2,1)
        assert positions['qC3'] == (2, 2)  # 1-indexed: (0,1) -> (2,2)
    
    def test_resolve_with_missing_qubits(self, sample_dataset):
        """Test resolve method with qubits not in grid."""
        grid_locations = ["0,0", "0,1"]
        grid = QubitGrid(sample_dataset, grid_locations=grid_locations, shape=(2, 2))
        
        n_rows, n_cols, positions = grid.resolve(['qC1', 'qC2', 'qC3'])  # qC3 not in grid
        
        assert n_rows == 2
        assert n_cols == 2
        assert 'qC1' in positions
        assert 'qC2' in positions
        assert 'qC3' not in positions
    
    def test_grid_iter_functionality(self, sample_dataset):
        """Test grid_iter function with new QubitGrid."""
        grid_locations = ["0,0", "0,1", "1,0", "1,1"]
        grid = QubitGrid(sample_dataset, grid_locations=grid_locations, shape=(2, 2))
        
        # Test grid_iter
        iter_results = list(grid_iter(grid))
        
        assert len(iter_results) == 4  # All 4 qubits
        
        # Check that we get axes and qubit info
        for ax, qubit_info in iter_results:
            assert hasattr(ax, 'plot')  # It's a matplotlib axes
            assert 'qubit' in qubit_info
            assert qubit_info['qubit'] in ['qC1', 'qC2', 'qC3', 'qC4']
    
    def test_grid_iter_with_old_style_grid(self):
        """Test that grid_iter raises error with old-style grid."""
        coords = {'Q0': (0, 0), 'Q1': (0, 1)}
        grid = QubitGrid(coords, shape=(1, 2))
        
        with pytest.raises(ValueError, match="grid_iter requires a QubitGrid created with the Dataset interface"):
            list(grid_iter(grid))
    
    def test_string_parsing_edge_cases(self, sample_dataset):
        """Test string parsing with various formats."""
        # Test with spaces
        grid_locations = ["0, 0", "0,1", "1, 0", "1,1"]
        grid = QubitGrid(sample_dataset, grid_locations=grid_locations, shape=(2, 2))
        
        expected_coords = {
            'qC1': (0, 0),  # "0, 0" -> row=0, col=0
            'qC2': (1, 0),  # "0,1" -> row=1, col=0
            'qC3': (0, 1),  # "1, 0" -> row=0, col=1
            'qC4': (1, 1)   # "1,1" -> row=1, col=1
        }
        assert grid.coords == expected_coords
    
    def test_dataset_without_qubit_coord(self):
        """Test QubitGrid with dataset that doesn't have 'qubit' coordinate."""
        # Create dataset without 'qubit' coordinate
        detuning = np.linspace(-1e6, 1e6, 50)
        data = {'signal': (['detuning'], np.random.rand(50))}
        ds = xr.Dataset(data, coords={'detuning': detuning})
        
        grid_locations = ["0,0", "0,1"]
        grid = QubitGrid(ds, grid_locations=grid_locations, shape=(1, 2))
        
        # Should handle gracefully with empty qubit names
        assert grid.coords == {}
        assert not hasattr(grid, 'fig')  # Created lazily
    
    def test_dataset_with_qubit_attribute(self):
        """Test QubitGrid with dataset that has qubit as attribute."""
        detuning = np.linspace(-1e6, 1e6, 50)
        qubits = ['Q1', 'Q2']
        
        # Create dataset with qubit as coordinate (not attribute)
        ds = xr.Dataset({
            'signal': (['detuning', 'qubit'], np.random.rand(50, 2))
        }, coords={'detuning': detuning, 'qubit': qubits})
        
        grid_locations = ["0,0", "0,1"]
        grid = QubitGrid(ds, grid_locations=grid_locations, shape=(1, 2))
        
        expected_coords = {
            'Q1': (0, 0),  # "0,0" -> row=0, col=0
            'Q2': (1, 0)   # "0,1" -> row=1, col=0
        }
        assert grid.coords == expected_coords
    
    def test_matplotlib_figure_properties(self, sample_dataset):
        """Test that matplotlib figure has correct properties."""
        grid_locations = ["0,0", "0,1", "1,0", "1,1"]
        grid = QubitGrid(sample_dataset, grid_locations=grid_locations, shape=(2, 2))
        
        # Create the figure by calling grid_iter
        list(grid_iter(grid))
        
        # Check figure properties
        assert grid.fig is not None
        assert hasattr(grid.fig, 'get_figwidth')
        assert hasattr(grid.fig, 'get_figheight')
        
        # Check axes array
        assert grid._axes.shape == (2, 2)
        assert all(hasattr(ax, 'plot') for ax in grid._axes.flat)
        
        # Check stored data
        assert grid._qubit_names == ['qC1', 'qC2', 'qC3', 'qC4']
        assert len(grid._positions) == 4
    
    def test_coordinate_indexing(self, sample_dataset):
        """Test that coordinates are properly indexed (0-based vs 1-based)."""
        grid_locations = ["0,0", "0,1", "1,0", "1,1"]
        grid = QubitGrid(sample_dataset, grid_locations=grid_locations, shape=(2, 2))
        
        # Test resolve returns 1-indexed positions
        n_rows, n_cols, positions = grid.resolve(['qC1', 'qC2', 'qC3', 'qC4'])
        
        assert positions['qC1'] == (2, 1)  # 1-indexed: (0,0) -> (2,1)
        assert positions['qC2'] == (1, 1)  # 1-indexed: (1,0) -> (1,1)
        assert positions['qC3'] == (2, 2)  # 1-indexed: (0,1) -> (2,2)
        assert positions['qC4'] == (1, 2)  # 1-indexed: (1,1) -> (1,2)
    
    def test_empty_constructor(self):
        """Test QubitGrid with no arguments."""
        grid = QubitGrid()
        
        assert grid.coords == {}
        assert grid.shape is None
        assert not hasattr(grid, 'fig')
    
    def test_none_constructor(self):
        """Test QubitGrid with None arguments."""
        grid = QubitGrid(None)
        
        assert grid.coords == {}
        assert grid.shape is None
        assert not hasattr(grid, 'fig')


if __name__ == "__main__":
    pytest.main([__file__])
