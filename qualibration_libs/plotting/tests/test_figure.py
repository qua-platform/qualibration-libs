"""
Unit tests for QualibrationFigure plotting functionality.
"""
import pytest
import numpy as np
import xarray as xr
import os
from pathlib import Path

from qualibration_libs.plotting import QualibrationFigure, QubitGrid
from qualibration_libs.plotting.overlays import RefLine, LineOverlay, FitOverlay


class TestQualibrationFigure:
    """Test cases for QualibrationFigure class."""
    
    @pytest.fixture
    def sample_data_1d(self):
        """Create sample 1D data for testing."""
        detuning = np.linspace(-1.5e7, 1.5e7, 100)
        qubits = ['qC1', 'qC2', 'qC3']
        
        # Create realistic IQ data
        data = {}
        for i, qubit in enumerate(qubits):
            # Simulate resonance response
            center = i * 0.5e6  # Different center for each qubit
            width = 1e6
            response = np.exp(-((detuning - center) / width)**2)
            data[qubit] = (['detuning'], response)
        
        return xr.Dataset(data, coords={'detuning': detuning})
    
    @pytest.fixture
    def sample_data_2d(self):
        """Create sample 2D data for testing."""
        detuning = np.linspace(-7.5e6, 7.5e6, 30)
        power = np.linspace(-55, -20, 12)
        qubits = ['qC1', 'qC2']
        
        data = {}
        for i, qubit in enumerate(qubits):
            # Create 2D response surface
            det_grid, pow_grid = np.meshgrid(detuning, power, indexing='ij')
            response = np.exp(-((det_grid - i * 1e6) / 2e6)**2) * (pow_grid + 55) / 35
            data[qubit] = (['detuning', 'power'], response)
        
        return xr.Dataset(data, coords={'detuning': detuning, 'power': power})
    
    @pytest.fixture
    def real_data_files(self):
        """Load real test data files."""
        data_dir = Path(__file__).parent.parent / "test_data"
        files = {}
        for filename in ['ds_raw.h5', 'ds_fit.h5', 'ds_raw_2.h5', 'ds_fit_2.h5']:
            filepath = data_dir / filename
            if filepath.exists():
                files[filename] = xr.open_dataset(filepath)
        return files
    
    def test_basic_1d_plot(self, sample_data_1d):
        """Test basic 1D plotting functionality."""
        fig = QualibrationFigure.plot(
            sample_data_1d,
            x='detuning',
            data_var='qC1',
            title="Test 1D Plot"
        )
        
        assert fig is not None
        assert fig.figure is not None
        assert len(fig.figure.data) > 0  # Should have traces
        
        # Check that subplot titles are set
        subplot_titles = fig.figure.layout.annotations
        assert len(subplot_titles) > 0
    
    def test_2d_heatmap_plot(self, sample_data_2d):
        """Test 2D heatmap plotting."""
        fig = QualibrationFigure.plot(
            sample_data_2d,
            x='detuning',
            y='power',
            data_var='qC1',
            title="Test 2D Heatmap"
        )
        
        assert fig is not None
        assert fig.figure is not None
        
        # Check for heatmap traces
        heatmap_traces = [trace for trace in fig.figure.data if trace.type == 'heatmap']
        assert len(heatmap_traces) > 0
    
    def test_multi_qubit_layout(self, sample_data_1d):
        """Test multi-qubit subplot layout."""
        fig = QualibrationFigure.plot(
            sample_data_1d,
            x='detuning',
            data_var='qC1',  # This will plot all qubits
            title="Multi-Qubit Layout"
        )
        
        assert fig is not None
        # Should have subplots for each qubit
        assert len(fig.figure.data) >= len(sample_data_1d.coords['qubit'])
    
    def test_custom_grid_layout(self, sample_data_1d):
        """Test custom qubit grid layout."""
        grid = QubitGrid(
            coords={'qC1': (0, 0), 'qC2': (0, 1), 'qC3': (1, 0)},
            shape=(2, 2)
        )
        
        fig = QualibrationFigure.plot(
            sample_data_1d,
            x='detuning',
            data_var='qC1',
            grid=grid,
            title="Custom Grid Layout"
        )
        
        assert fig is not None
        assert fig.figure is not None
    
    def test_residuals_plot(self, sample_data_1d):
        """Test residuals subplot functionality."""
        fig = QualibrationFigure.plot(
            sample_data_1d,
            x='detuning',
            data_var='qC1',
            residuals=True,
            title="Plot with Residuals"
        )
        
        assert fig is not None
        assert fig.figure is not None
        
        # Check for residual subplot shapes (zero line)
        shapes = fig.figure.layout.shapes
        zero_lines = [shape for shape in shapes if shape.type == 'line' and shape.y0 == 0]
        assert len(zero_lines) > 0
    
    def test_with_overlays(self, sample_data_1d):
        """Test plotting with overlays."""
        overlays = [
            RefLine(x=0, name="Zero Detuning"),
            RefLine(x=1e6, name="1 MHz Offset")
        ]
        
        fig = QualibrationFigure.plot(
            sample_data_1d,
            x='detuning',
            data_var='qC1',
            overlays=overlays,
            title="Plot with Overlays"
        )
        
        assert fig is not None
        assert fig.figure is not None
        
        # Check for overlay traces
        overlay_traces = [trace for trace in fig.figure.data if 'Zero' in str(trace.name) or '1 MHz' in str(trace.name)]
        assert len(overlay_traces) > 0
    
    def test_fit_overlay(self, sample_data_1d):
        """Test fit overlay functionality."""
        # Create mock fit data
        detuning = sample_data_1d.coords['detuning'].values
        fit_curve = np.exp(-((detuning - 0.5e6) / 1e6)**2)
        fit_params = {'center': 0.5e6, 'width': 1e6, 'amplitude': 1.0}
        
        fit_overlay = FitOverlay(
            y_fit=fit_curve,
            params=fit_params,
            formatter=lambda p: f"Center: {p['center']:.0f} Hz, Width: {p['width']:.0f} Hz"
        )
        
        fig = QualibrationFigure.plot(
            sample_data_1d,
            x='detuning',
            data_var='qC1',
            overlays=[fit_overlay],
            title="Plot with Fit Overlay"
        )
        
        assert fig is not None
        assert fig.figure is not None
    
    def test_hue_dimension(self, sample_data_1d):
        """Test color-coding by additional dimension."""
        # Add a hue dimension
        data_with_hue = sample_data_1d.expand_dims('power', axis=1)
        data_with_hue = data_with_hue.assign_coords(power=[-30, -20, -10])
        
        fig = QualibrationFigure.plot(
            data_with_hue,
            x='detuning',
            data_var='qC1',
            hue='power',
            title="Plot with Hue Dimension"
        )
        
        assert fig is not None
        assert fig.figure is not None
    
    def test_secondary_x_axis(self, sample_data_1d):
        """Test secondary x-axis functionality."""
        # Add secondary coordinate
        data_with_x2 = sample_data_1d.copy()
        data_with_x2 = data_with_x2.assign_coords(
            wavelength=('detuning', 3e8 / (5e9 + data_with_x2.detuning))
        )
        
        fig = QualibrationFigure.plot(
            data_with_x2,
            x='detuning',
            x2='wavelength',
            data_var='qC1',
            title="Plot with Secondary X-Axis"
        )
        
        assert fig is not None
        assert fig.figure is not None
    
    def test_real_data_loading(self, real_data_files):
        """Test loading and plotting real data files."""
        if 'ds_raw.h5' in real_data_files:
            ds = real_data_files['ds_raw.h5']
            
            fig = QualibrationFigure.plot(
                ds,
                x='detuning',
                data_var='IQ_abs',
                title="Real Data: IQ Magnitude"
            )
            
            assert fig is not None
            assert fig.figure is not None
            assert len(fig.figure.data) > 0
    
    def test_real_fit_data(self, real_data_files):
        """Test plotting real fit data."""
        if 'ds_fit.h5' in real_data_files:
            ds = real_data_files['ds_fit.h5']
            
            fig = QualibrationFigure.plot(
                ds,
                x='detuning',
                data_var='base_line',
                title="Real Fit Data: Baseline"
            )
            
            assert fig is not None
            assert fig.figure is not None
    
    def test_2d_real_data(self, real_data_files):
        """Test 2D plotting with real data."""
        if 'ds_raw_2.h5' in real_data_files:
            ds = real_data_files['ds_raw_2.h5']
            
            # Select a single flux bias point for 2D plot
            ds_slice = ds.isel(flux_bias=25)  # Middle flux bias point
            
            fig = QualibrationFigure.plot(
                ds_slice,
                x='detuning',
                data_var='IQ_abs',
                title="Real 2D Data: IQ Magnitude vs Detuning"
            )
            
            assert fig is not None
            assert fig.figure is not None
    
    def test_3d_data_heatmap(self, real_data_files):
        """Test 3D data as 2D heatmap."""
        if 'ds_raw_2.h5' in real_data_files:
            ds = real_data_files['ds_raw_2.h5']
            
            # Select a single qubit for 2D heatmap
            ds_single_qubit = ds.isel(qubit=0)
            
            fig = QualibrationFigure.plot(
                ds_single_qubit,
                x='detuning',
                y='flux_bias',
                data_var='IQ_abs',
                title="Real 3D Data as 2D Heatmap"
            )
            
            assert fig is not None
            assert fig.figure is not None
            
            # Check for heatmap traces
            heatmap_traces = [trace for trace in fig.figure.data if trace.type == 'heatmap']
            assert len(heatmap_traces) > 0
    
    def test_per_qubit_overlays(self, sample_data_1d):
        """Test per-qubit overlay functionality."""
        overlays = {
            'qC1': [RefLine(x=0, name="Q1 Zero")],
            'qC2': [RefLine(x=1e6, name="Q2 Offset")],
            'qC3': [RefLine(x=-1e6, name="Q3 Negative")]
        }
        
        fig = QualibrationFigure.plot(
            sample_data_1d,
            x='detuning',
            data_var='qC1',
            overlays=overlays,
            title="Per-Qubit Overlays"
        )
        
        assert fig is not None
        assert fig.figure is not None
    
    def test_dynamic_overlays(self, sample_data_1d):
        """Test dynamic overlay function."""
        def overlay_func(qubit_name, qubit_data):
            if qubit_name == 'qC1':
                return [RefLine(x=0, name="Q1 Special")]
            return []
        
        fig = QualibrationFigure.plot(
            sample_data_1d,
            x='detuning',
            data_var='qC1',
            overlays=overlay_func,
            title="Dynamic Overlays"
        )
        
        assert fig is not None
        assert fig.figure is not None
    
    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test with invalid data type
        with pytest.raises(TypeError):
            QualibrationFigure.plot("invalid_data", x='test')
        
        # Test with missing coordinate
        invalid_data = xr.Dataset({'test': (['x'], [1, 2, 3])}, coords={'x': [1, 2, 3]})
        with pytest.raises(KeyError):
            QualibrationFigure.plot(invalid_data, x='nonexistent_coord')
    
    def test_dict_data_input(self):
        """Test plotting with dictionary data input."""
        data_dict = {
            'x': np.linspace(0, 10, 100),
            'y': np.sin(np.linspace(0, 10, 100)),
            'qubit': ['Q0'] * 100
        }
        
        fig = QualibrationFigure.plot(
            data_dict,
            x='x',
            data_var='y',
            title="Dictionary Data Input"
        )
        
        assert fig is not None
        assert fig.figure is not None
    
    def test_pandas_dataframe_input(self):
        """Test plotting with pandas DataFrame input."""
        import pandas as pd
        
        df = pd.DataFrame({
            'frequency': np.linspace(4.5, 5.5, 100),
            'amplitude': np.random.rand(100),
            'qubit': ['Q0'] * 100
        })
        
        fig = QualibrationFigure.plot(
            df,
            x='frequency',
            data_var='amplitude',
            title="Pandas DataFrame Input"
        )
        
        assert fig is not None
        assert fig.figure is not None


class TestQubitGrid:
    """Test cases for QubitGrid class."""
    
    def test_basic_grid(self):
        """Test basic grid creation."""
        coords = {'Q0': (0, 0), 'Q1': (0, 1), 'Q2': (1, 0)}
        grid = QubitGrid(coords=coords, shape=(2, 2))
        
        assert grid.coords == coords
        assert grid.shape == (2, 2)
    
    def test_grid_resolution(self):
        """Test grid resolution functionality."""
        coords = {'Q0': (0, 0), 'Q1': (0, 1), 'Q2': (1, 0)}
        grid = QubitGrid(coords=coords, shape=(2, 2))
        
        n_rows, n_cols, positions = grid.resolve(['Q0', 'Q1', 'Q2'])
        
        assert n_rows == 2
        assert n_cols == 2
        assert positions['Q0'] == (1, 1)  # 1-indexed
        assert positions['Q1'] == (1, 2)
        assert positions['Q2'] == (2, 1)
    
    def test_automatic_shape(self):
        """Test automatic shape determination."""
        coords = {'Q0': (0, 0), 'Q1': (0, 1), 'Q2': (1, 0)}
        grid = QubitGrid(coords=coords)  # No shape specified
        
        n_rows, n_cols, positions = grid.resolve(['Q0', 'Q1', 'Q2'])
        
        assert n_rows == 2
        assert n_cols == 2
    
    def test_missing_qubits(self):
        """Test handling of missing qubits in resolution."""
        coords = {'Q0': (0, 0), 'Q1': (0, 1)}
        grid = QubitGrid(coords=coords, shape=(2, 2))
        
        n_rows, n_cols, positions = grid.resolve(['Q0', 'Q1', 'Q2'])  # Q2 missing
        
        assert n_rows == 2
        assert n_cols == 2
        assert 'Q0' in positions
        assert 'Q1' in positions
        assert 'Q2' not in positions


if __name__ == "__main__":
    pytest.main([__file__])
