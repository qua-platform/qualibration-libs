# Testing and Demos for Qualibration Plotting Module

This document describes the comprehensive testing suite and demo scripts created for the qualibration plotting module.

## Overview

The testing and demo infrastructure includes:
- **Unit Tests**: Comprehensive test coverage for all plotting functionality
- **Demo Scripts**: Real-world examples using actual quantum device data
- **Test Data**: HDF5 files containing realistic quantum calibration measurements
- **Test Runner**: Automated testing and demo execution

## Test Data

The module includes 6 HDF5 test data files in `test_data/`:

### Raw Data Files
- **`ds_raw.h5`**: 1D frequency sweep data (8 qubits, 300 detuning points)
  - Data variables: `I`, `Q`, `IQ_abs`, `phase`
  - Coordinates: `detuning`, `qubit`, `full_freq`
  
- **`ds_raw_2.h5`**: 2D flux tuning data (2 qubits, 51 flux bias points, 150 detuning points)
  - Data variables: `I`, `Q`, `IQ_abs`, `phase`
  - Coordinates: `detuning`, `flux_bias`, `qubit`, `current`, `attenuated_current`
  
- **`ds_raw_3.h5`**: 3D power sweep data (4 qubits, 30 detuning points, 12 power points)
  - Data variables: `I`, `Q`, `IQ_abs`, `IQ_abs_norm`, `phase`, `rr_min_response*`, `below_threshold`
  - Coordinates: `detuning`, `power`, `qubit`, `full_freq`

### Fit Data Files
- **`ds_fit.h5`**: 1D fit results (8 qubits, 300 detuning points)
  - Data variables: `base_line`, `position`, `width`, `amplitude`
  - Coordinates: `detuning`, `qubit`, `res_freq`, `fwhm`, `success`
  
- **`ds_fit_2.h5`**: 2D flux tuning fit results (2 qubits, 51 flux bias points)
  - Data variables: `peak_freq`, `fit_results`
  - Coordinates: `flux_bias`, `fit_vals`, `qubit`, `idle_offset`, `flux_min`, `freq_shift`, `sweet_spot_frequency`, `success`
  
- **`ds_fit_3.h5`**: 3D power sweep fit results (4 qubits, 30 detuning points, 12 power points)
  - Data variables: Same as `ds_raw_3.h5` plus fit parameters
  - Coordinates: Same as `ds_raw_3.h5` plus `optimal_power`, `freq_shift`, `res_freq`, `success`

## Unit Tests

### Test Structure
```
tests/
├── __init__.py
├── test_figure.py      # QualibrationFigure functionality
├── test_overlays.py    # Overlay system tests
└── test_styles.py      # Styling and theming tests
```

### Test Coverage

#### `test_figure.py`
- **Basic plotting**: 1D and 2D plots
- **Multi-qubit layouts**: Automatic and custom grids
- **Residuals plots**: Subplot functionality
- **Overlay integration**: Reference lines, fits, annotations
- **Data input formats**: xarray, pandas, dictionaries
- **Error handling**: Invalid inputs and missing coordinates
- **Real data integration**: Loading and plotting actual HDF5 files

#### `test_overlays.py`
- **RefLine**: Vertical and horizontal reference lines
- **LineOverlay**: Custom line overlays
- **ScatterOverlay**: Scatter point overlays
- **TextBoxOverlay**: Text annotations
- **FitOverlay**: Fit curves and parameter display
- **Integration**: Multiple overlay types
- **Style overrides**: Custom styling options

#### `test_styles.py`
- **PlotTheme**: Theme configuration and customization
- **RcParams**: Runtime configuration parameters
- **Theme functions**: `set_theme`, `set_palette`, `theme_context`
- **Built-in palettes**: qualibrate, deep, muted
- **Layout application**: Theme application to plots
- **Context managers**: Temporary theme changes

### Running Tests

```bash
# Run all tests
python -m pytest qualibration_libs/plotting/tests/ -v

# Run specific test file
python -m pytest qualibration_libs/plotting/tests/test_figure.py -v

# Run with coverage
python -m pytest qualibration_libs/plotting/tests/ --cov=qualibration_libs.plotting
```

## Demo Scripts

### Demo Structure
```
demos/
├── __init__.py
├── simple_demo.py       # Basic functionality demo
├── basic_plots.py      # Fundamental plotting demos
└── advanced_plots.py   # Advanced plotting demos
```

### Demo Scripts

#### `simple_demo.py`
**Purpose**: Quick verification of basic functionality
**Features**:
- Load real HDF5 data
- Create basic 1D plots
- Test overlay functionality
- Test theme customization
- Multi-qubit plotting

**Usage**:
```bash
python qualibration_libs/plotting/demos/simple_demo.py
```

#### `basic_plots.py`
**Purpose**: Comprehensive demonstration of core plotting features
**Features**:
- 1D frequency sweep plots
- 2D heatmap plots
- Power sweep analysis
- Fit results visualization
- Overlay functionality
- Custom grid layouts
- Styling and theming

**Usage**:
```bash
python qualibration_libs/plotting/demos/basic_plots.py
```

#### `advanced_plots.py`
**Purpose**: Advanced plotting techniques and analysis
**Features**:
- Flux tuning analysis with 3D data
- Power optimization analysis
- Fit quality analysis and comparison
- Multi-qubit comparison with custom layouts
- Publication-ready plots
- Flux tuning fit analysis

**Usage**:
```bash
python qualibration_libs/plotting/demos/advanced_plots.py
```

## Test Runner

### `run_tests.py`
Automated test and demo execution script.

**Features**:
- Run all unit tests with pytest
- Execute demo scripts
- Provide comprehensive summary
- Error handling and reporting

**Usage**:
```bash
python qualibration_libs/plotting/run_tests.py
```

**Output**:
- Test results with pass/fail status
- Demo execution results
- Comprehensive summary report

## Data Analysis Examples

### 1D Frequency Sweep Analysis
```python
import qualibration_libs.plotting as qplot
import xarray as xr

# Load data
ds = xr.open_dataset('test_data/ds_raw.h5')

# Basic 1D plot
fig = qplot.QualibrationFigure.plot(
    ds,
    x='detuning',
    data_var='IQ_abs',
    title="IQ Magnitude vs Detuning"
)

# With residuals
fig = qplot.QualibrationFigure.plot(
    ds,
    x='detuning',
    data_var='IQ_abs',
    residuals=True,
    title="IQ Magnitude with Residuals"
)
```

### 2D Heatmap Analysis
```python
# 2D heatmap
fig = qplot.QualibrationFigure.plot(
    ds,
    x='detuning',
    y='flux_bias',
    data_var='IQ_abs',
    title="IQ Magnitude: Detuning vs Flux Bias"
)
```

### Multi-Qubit Analysis
```python
from qualibration_libs.plotting import QubitGrid

# Custom grid layout
grid = QubitGrid(
    coords={'qC1': (0, 0), 'qC2': (0, 1), 'qC3': (1, 0), 'qC4': (1, 1)},
    shape=(2, 2)
)

fig = qplot.QualibrationFigure.plot(
    ds,
    x='detuning',
    data_var='IQ_abs',
    grid=grid,
    title="Multi-Qubit Analysis"
)
```

### Overlay Integration
```python
from qualibration_libs.plotting.overlays import RefLine, FitOverlay

# Reference lines
overlays = [
    RefLine(x=0, name="Zero Detuning"),
    RefLine(x=1e6, name="1 MHz Offset")
]

# Fit overlay
fit_overlay = FitOverlay(
    y_fit=fit_curve,
    params={'center': 0.5e6, 'width': 1e6},
    formatter=lambda p: f"Center: {p['center']:.0f} Hz"
)

fig = qplot.QualibrationFigure.plot(
    ds,
    x='detuning',
    data_var='IQ_abs',
    overlays=overlays + [fit_overlay],
    title="Analysis with Overlays"
)
```

## Performance Considerations

### Data Loading
- HDF5 files are loaded on-demand
- xarray provides efficient data slicing
- Memory usage scales with data size

### Plotting Performance
- Plotly provides interactive plots
- Large datasets may require data reduction
- Subplot creation scales with qubit count

### Memory Usage
- Test data files: ~1-2 MB each
- Plot objects: ~10-50 MB depending on complexity
- Overlays add minimal memory overhead

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure package is installed
   pip install -e .
   ```

2. **Data Loading Issues**
   ```bash
   # Install required dependencies
   pip install netcdf4 h5py
   ```

3. **Plot Display Issues**
   ```python
   # Ensure plotly is properly configured
   import plotly.io as pio
   pio.renderers.default = "browser"  # or "notebook"
   ```

4. **Test Failures**
   ```bash
   # Run with verbose output
   python -m pytest qualibration_libs/plotting/tests/ -v -s
   ```

### Debug Mode
```python
# Enable debug output
import logging
logging.basicConfig(level=logging.DEBUG)

# Test data loading
ds = xr.open_dataset('test_data/ds_raw.h5')
print(f"Data shape: {dict(ds.dims)}")
print(f"Data variables: {list(ds.data_vars)}")
```

## Future Enhancements

### Planned Features
- Additional test data files
- Performance benchmarking
- Automated regression testing
- CI/CD integration
- Documentation generation

### Test Coverage Goals
- 100% function coverage
- Edge case testing
- Performance testing
- Integration testing

## Contributing

### Adding New Tests
1. Create test functions in appropriate test files
2. Follow naming convention: `test_<functionality>`
3. Include docstrings explaining test purpose
4. Use fixtures for common test data

### Adding New Demos
1. Create demo scripts in `demos/` directory
2. Include comprehensive docstrings
3. Handle errors gracefully
4. Provide clear output messages

### Test Data Guidelines
- Use realistic quantum device data
- Include various data dimensions
- Provide metadata and attributes
- Ensure data is properly formatted

## Conclusion

The testing and demo infrastructure provides comprehensive coverage of the qualibration plotting module, ensuring reliability and demonstrating capabilities with real quantum device data. The modular design allows for easy extension and maintenance.
