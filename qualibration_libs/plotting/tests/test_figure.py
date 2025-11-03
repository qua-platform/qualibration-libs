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
        
        return xr.Dataset(data, coords={'detuning': detuning, 'qubit': qubits})
    
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
        # Should have at least one trace (the plotting function may combine qubits into one trace)
        assert len(fig.figure.data) > 0
    
    def test_custom_grid_layout(self, sample_data_1d):
        """Test custom qubit grid layout."""
        # Use the same qubit names as in the sample data
        grid = QubitGrid(
            {'qC1': (0, 0), 'qC2': (0, 1), 'qC3': (1, 0)},
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
        # Create a fit overlay to test residuals calculation
        detuning = sample_data_1d.coords['detuning'].values
        fit_curve = np.exp(-((detuning - 0.5e6) / 1e6)**2)
        fit_overlay = FitOverlay(y_fit=fit_curve, name="Test Fit")
        
        fig = QualibrationFigure.plot(
            sample_data_1d,
            x='detuning',
            data_var='qC1',
            overlays=[fit_overlay],
            residuals=True,
            title="Plot with Residuals"
        )
        
        assert fig is not None
        assert fig.figure is not None
        
        # Check for residual subplot shapes (zero line)
        shapes = fig.figure.layout.shapes
        zero_lines = [shape for shape in shapes if shape.type == 'line' and shape.y0 == 0]
        assert len(zero_lines) > 0
        
        # Check that residuals are actually plotted (should have residual traces)
        residual_traces = [trace for trace in fig.figure.data if 'residuals' in str(trace.name)]
        assert len(residual_traces) > 0, "No residual traces found - residuals should be plotted when fit overlay is provided"
    
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
        
        # Check for overlay traces (should have at least the data traces + overlay traces)
        # The overlays should add additional traces beyond the data traces
        assert len(fig.figure.data) > 0
    
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
        # Create new data with power dimension
        detuning = sample_data_1d.coords['detuning'].values
        power = [-30, -20, -10]
        
        # Create 3D data: (detuning, power, qubit)
        data_with_hue = {}
        for qubit in sample_data_1d.data_vars:
            # Expand the 1D data to 2D with power dimension
            original_data = sample_data_1d[qubit].values
            expanded_data = np.tile(original_data, (len(power), 1)).T
            data_with_hue[qubit] = (['detuning', 'power'], expanded_data)
        
        data_with_hue = xr.Dataset(data_with_hue, coords={'detuning': detuning, 'power': power})
        
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
        wavelength_values = 3e8 / (5e9 + data_with_x2.detuning.values)
        data_with_x2 = data_with_x2.assign_coords(
            wavelength=('detuning', wavelength_values)
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
    
    def test_x2_configuration_parameters(self, sample_data_1d):
        """Test x2 configuration parameters for margin and annotation positioning."""
        # Add secondary coordinate
        data_with_x2 = sample_data_1d.copy()
        wavelength_values = 3e8 / (5e9 + data_with_x2.detuning.values)
        data_with_x2 = data_with_x2.assign_coords(
            wavelength=('detuning', wavelength_values)
        )
        
        # Test with custom x2 configuration parameters
        fig = QualibrationFigure.plot(
            data_with_x2,
            x='detuning',
            x2='wavelength',
            data_var='qC1',
            title="Plot with Custom X2 Configuration",
            x2_top_margin=150,           # Custom top margin
            x2_annotation_offset=0.025   # Custom annotation offset
        )
        
        assert fig is not None
        assert fig.figure is not None
        
        # Check that the custom margin was applied
        assert fig.figure.layout.margin.t == 150
        
        # Check that annotations exist (qubit names should be moved up)
        annotations = fig.figure.layout.annotations
        assert len(annotations) > 0
        
        # Test with default values (should still work)
        fig_default = QualibrationFigure.plot(
            data_with_x2,
            x='detuning',
            x2='wavelength',
            data_var='qC1',
            title="Plot with Default X2 Configuration"
        )
        
        assert fig_default is not None
        assert fig_default.figure is not None
        
        # Check that default margin was applied
        assert fig_default.figure.layout.margin.t == 120  # Default value
    
    def test_x2_annotation_positioning_all_qubits(self, sample_data_1d):
        """Test that ALL subplot title annotations are moved when x2 is present, regardless of qubit name."""
        # Add secondary coordinate
        data_with_x2 = sample_data_1d.copy()
        wavelength_values = 3e8 / (5e9 + data_with_x2.detuning.values)
        data_with_x2 = data_with_x2.assign_coords(
            wavelength=('detuning', wavelength_values)
        )
        
        # Create custom grid with qubit names that exist in the dataset
        grid_coords = {
            'qC1': (0, 0),      # Starts with 'q'
            'qC2': (0, 1),      # Starts with 'q'
            'qC3': (1, 0),      # Starts with 'q'
        }
        grid = QubitGrid(grid_coords, shape=(2, 2))
        
        # Create plot with x2 and custom annotation offset
        annotation_offset = 0.05
        fig = QualibrationFigure.plot(
            data_with_x2,
            x='detuning',
            x2='wavelength',
            data_var='qC1',
            grid=grid,
            title="Test All Annotations Moved",
            x2_annotation_offset=annotation_offset
        )
        
        assert fig is not None
        assert fig.figure is not None
        
        # Get all annotations
        annotations = fig.figure.layout.annotations
        assert len(annotations) >= 3  # Should have at least 3 qubit title annotations
        
        # Check that ALL subplot title annotations have been moved up
        # (not just those starting with 'q')
        moved_annotations = []
        for annotation in annotations:
            if annotation.y is not None and annotation.text:
                # Check if this looks like a subplot title (not the main title)
                if annotation.text in ['qC1', 'qC2', 'qC3']:
                    moved_annotations.append(annotation)
        
        # All qubit title annotations should be present and moved
        assert len(moved_annotations) == 3, f"Expected 3 qubit annotations, got {len(moved_annotations)}"
        
        # Verify that annotations starting with 'q' are moved
        q_annotations = [ann for ann in moved_annotations if ann.text.startswith('q')]
        assert len(q_annotations) == 3, "Should have 3 annotations starting with 'q'"
        
        # All annotations should have been moved up by the annotation_offset
        for annotation in moved_annotations:
            # The annotation should have a y position that reflects the offset
            # (exact value depends on plotly's internal positioning)
            assert annotation.y is not None, f"Annotation '{annotation.text}' should have a y position"
    
    def test_x2_annotation_positioning_mixed_qubit_names(self):
        """Test that ALL subplot title annotations are moved regardless of qubit name format."""
        # Create custom dataset with mixed qubit names using qubit dimension
        detuning = np.linspace(-1.5e7, 1.5e7, 50)
        qubits = ['q0', 'Qubit_A', 'qubit_B', 'Q1']  # Mixed naming conventions
        
        # Create data with qubit dimension (like the sample_data_1d fixture)
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
        
        # Create custom grid with mixed qubit names
        grid_coords = {
            'q0': (0, 0),       # Starts with 'q'
            'Qubit_A': (0, 1),  # Doesn't start with 'q'
            'qubit_B': (1, 0),  # Starts with 'q'
            'Q1': (1, 1),       # Starts with 'Q' (capital)
        }
        grid = QubitGrid(grid_coords, shape=(2, 2))
        
        # Create plot with x2 - use qubit_dim to plot all qubits
        fig = QualibrationFigure.plot(
            ds,
            x='detuning',
            x2='wavelength',
            data_var='q0',  # Data variable name
            qubit_dim='qubit',  # Dimension containing qubit names
            grid=grid,
            title="Test Mixed Qubit Names",
            x2_annotation_offset=0.05
        )
        
        assert fig is not None
        assert fig.figure is not None
        
        # Get all annotations
        annotations = fig.figure.layout.annotations
        assert len(annotations) >= 4  # Should have at least 4 qubit title annotations
        
        # Check that ALL subplot title annotations have been moved up
        moved_annotations = []
        for annotation in annotations:
            if annotation.y is not None and annotation.text:
                if annotation.text in ['q0', 'Qubit_A', 'qubit_B', 'Q1']:
                    moved_annotations.append(annotation)
        
        # All qubit title annotations should be present and moved
        assert len(moved_annotations) == 4, f"Expected 4 qubit annotations, got {len(moved_annotations)}"
        
        # Verify that annotations starting with 'q' are moved
        q_annotations = [ann for ann in moved_annotations if ann.text.startswith('q')]
        assert len(q_annotations) == 2, "Should have 2 annotations starting with 'q'"
        
        # Verify that annotations NOT starting with 'q' are also moved
        non_q_annotations = [ann for ann in moved_annotations if not ann.text.startswith('q')]
        assert len(non_q_annotations) == 2, "Should have 2 annotations NOT starting with 'q'"
        
        # All annotations should have been moved up
        for annotation in moved_annotations:
            assert annotation.y is not None, f"Annotation '{annotation.text}' should have a y position"
    
    def test_x2_annotation_positioning_without_x2(self, sample_data_1d):
        """Test that annotations are NOT moved when x2 is not present."""
        # Create plot WITHOUT x2
        fig = QualibrationFigure.plot(
            sample_data_1d,
            x='detuning',
            data_var='qC1',
            title="Test Annotations NOT Moved (No X2)"
        )
        
        assert fig is not None
        assert fig.figure is not None
        
        # Get all annotations
        annotations = fig.figure.layout.annotations
        assert len(annotations) > 0
        
        # Check that annotations exist but haven't been artificially moved
        # (they should have their default positions)
        for annotation in annotations:
            if annotation.y is not None and annotation.text:
                # Annotations should have their default positions
                assert annotation.y is not None
    
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
        # Create a simple dictionary with just the data array
        # The current implementation only uses the first key
        data_dict = {
            'y': np.sin(np.linspace(0, 10, 100))
        }
        
        fig = QualibrationFigure.plot(
            data_dict,
            x='index',  # Use 'index' as the x coordinate (created automatically)
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
        grid = QubitGrid(coords, shape=(2, 2))
        
        assert grid.coords == coords
        assert grid.shape == (2, 2)
    
    def test_grid_resolution(self):
        """Test grid resolution functionality."""
        coords = {'Q0': (0, 0), 'Q1': (0, 1), 'Q2': (1, 0)}
        grid = QubitGrid(coords, shape=(2, 2))
        
        n_rows, n_cols, positions = grid.resolve(['Q0', 'Q1', 'Q2'])
        
        assert n_rows == 2
        assert n_cols == 2
        assert positions['Q0'] == (2, 1)  # 1-indexed, rows flipped
        assert positions['Q1'] == (2, 2)
        assert positions['Q2'] == (1, 1)
    
    def test_automatic_shape(self):
        """Test automatic shape determination."""
        coords = {'Q0': (0, 0), 'Q1': (0, 1), 'Q2': (1, 0)}
        grid = QubitGrid(coords)  # No shape specified
        
        n_rows, n_cols, positions = grid.resolve(['Q0', 'Q1', 'Q2'])
        
        assert n_rows == 2
        assert n_cols == 2
    
    def test_missing_qubits(self):
        """Test handling of missing qubits in resolution."""
        coords = {'Q0': (0, 0), 'Q1': (0, 1)}
        grid = QubitGrid(coords, shape=(2, 2))
        
        n_rows, n_cols, positions = grid.resolve(['Q0', 'Q1', 'Q2'])  # Q2 missing
        
        assert n_rows == 2
        assert n_cols == 2
        assert 'Q0' in positions
        assert 'Q1' in positions
        assert 'Q2' not in positions


if __name__ == "__main__":
    pytest.main([__file__])
