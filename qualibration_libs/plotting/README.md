# Qualibration Plotting Module

A comprehensive plotting library designed specifically for quantum calibration data visualization. Built on top of Plotly, this module provides specialized tools for creating publication-ready plots of quantum device data with support for multi-qubit layouts, overlays, and advanced styling.

## Table of Contents

- [Quick Start](#quick-start)
- [Core Components](#core-components)
- [Data Types and Input](#data-types-and-input)
- [Plotting Functions](#plotting-functions)
- [Styling and Themes](#styling-and-themes)
- [Grid Layouts](#grid-layouts)
- [Overlays](#overlays)
- [Advanced Features](#advanced-features)
- [Examples](#examples)
- [Quick Reference: Matplotlib → Qualibration Plotting](#quick-reference-matplotlib--qualibration-plotting)
- [Migration Process](#migration-process)
- [Testing and Demos](#testing-and-demos)
- [API Reference](#api-reference)

## Quick Start

```python
import qualibration_libs.plotting as qplot
import xarray as xr
import numpy as np

# Create sample data
data = xr.Dataset({
    'frequency': (['qubit', 'point'], np.random.rand(3, 10)),
    'amplitude': (['qubit', 'point'], np.random.rand(3, 10))
}, coords={
    'qubit': ['Q0', 'Q1', 'Q2'],
    'point': np.linspace(0, 1, 10)
})

# Create a simple plot
fig = qplot.QualibrationFigure.plot(
    data, 
    x='point', 
    data_var='amplitude',
    title="Quantum Device Calibration"
)
fig.figure.show()
```

## Core Components

### QualibrationFigure

The main plotting class that creates interactive Plotly figures with quantum-specific features.

**Key Features:**
- Multi-qubit subplot layouts
- Automatic axis labeling from data attributes
- Support for residuals plots
- Custom overlay system
- Theme integration

### QubitGrid

Manages the layout of qubits in subplot grids, supporting both automatic and custom positioning.

### Overlay System

Extensible system for adding reference lines, fits, annotations, and other visual elements to plots.

## Data Types and Input

The plotting module accepts several data formats:

### xarray.Dataset/DataArray
```python
# Preferred format - preserves metadata and coordinates
ds = xr.Dataset({
    'measurement': (['qubit', 'frequency'], data),
    'error': (['qubit', 'frequency'], errors)
}, coords={
    'qubit': ['Q0', 'Q1', 'Q2'],
    'frequency': np.linspace(4.5, 5.5, 100)
})
```

### pandas.DataFrame
```python
# Automatically converted to xarray
df = pd.DataFrame({
    'frequency': freqs,
    'amplitude': amps,
    'qubit': ['Q0'] * len(freqs)
})
```

### Dictionary
```python
# Simple dictionary format
data = {
    'x': np.linspace(0, 10, 100),
    'y': np.sin(np.linspace(0, 10, 100)),
    'qubit': ['Q0'] * 100
}
```

## Plotting Functions

### Basic 1D Plots

```python
# Simple line plot
fig = qplot.QualibrationFigure.plot(
    data,
    x='frequency',           # x-axis coordinate
    data_var='amplitude',    # data variable to plot
    title="Frequency Sweep"
)
```

### 2D Heatmaps

```python
# 2D heatmap plot
fig = qplot.QualibrationFigure.plot(
    data,
    x='frequency',          # x-axis coordinate
    y='power',              # y-axis coordinate  
    data_var='amplitude',   # z-axis data
    title="2D Parameter Scan"
)
```

### Multi-qubit Layouts

```python
# Automatic qubit layout
fig = qplot.QualibrationFigure.plot(
    data,
    x='frequency',
    data_var='amplitude',
    qubit_dim='qubit',      # dimension containing qubit names
    title="Multi-Qubit Calibration"
)
```

### Custom Grid Layout

```python
from qualibration_libs.plotting import QubitGrid

# Define custom qubit positions
grid = QubitGrid(
    coords={'Q0': (0, 0), 'Q1': (0, 1), 'Q2': (1, 0)},
    shape=(2, 2)  # 2x2 grid
)

fig = qplot.QualibrationFigure.plot(
    data,
    x='frequency',
    data_var='amplitude',
    grid=grid
)
```

### Residuals Plots

```python
# Add residuals subplot (requires FitOverlay for actual residual data)
from qualibration_libs.plotting.overlays import FitOverlay

# Create fit overlay with your fitted curve and parameters
# See FitOverlay API reference below for full parameter documentation
fit_overlay = FitOverlay(
    y_fit=fitted_curve,                    # Array of fitted y-values
    params={'f0': 5.0, 'Q': 1000, 'A': 0.5},  # Fit parameters for display
    formatter=lambda p: f"f₀ = {p['f0']:.3f} GHz, Q = {p['Q']:.0f}, A = {p['A']:.2f}",
    name="Resonance Fit"
)

fig = qplot.QualibrationFigure.plot(
    data,
    x='frequency',
    data_var='amplitude',
    overlays=[fit_overlay],
    residuals=True,         # Shows residuals = data - fit_curve in separate subplot
    title="Fit with Residuals"
)
```

**Note:** The `FitOverlay` creates both a visual fit curve and a text box displaying the fit parameters. See the [FitOverlay API reference](#fitoverlay) for complete parameter documentation.

## Styling and Themes

### Built-in Themes

```python
# Set global theme
from qualibration_libs.plotting.config import PlotTheme

qplot.set_theme(
    theme=PlotTheme(
        font_size=16,
        marker_size=8,
        line_width=3,
        show_grid=True
    )
)
```

### Color Palettes

```python
# Use built-in palettes
qplot.set_palette("qualibrate")  # Default quantum palette
qplot.set_palette("deep")        # Deep color palette
qplot.set_palette("muted")       # Muted color palette

# Custom palette
qplot.set_palette(["#FF0000", "#00FF00", "#0000FF"])
```

### Theme Context Manager

```python
# Temporary theme changes
from qualibration_libs.plotting.config import PlotTheme

with qplot.theme_context(
    theme=PlotTheme(font_size=20),
    palette="deep"
):
    fig = qplot.QualibrationFigure.plot(data, x='freq', data_var='amp')
```

### RC Parameters

```python
# Fine-grained control
qplot.set_theme(rc={
    'showlegend': True,
    'grid_opacity': 0.5
})
```

## Grid Layouts

### Automatic Layout

The module automatically creates subplot grids based on qubit data:

```python
# 3 qubits -> 1x3 or 3x1 layout automatically chosen
fig = qplot.QualibrationFigure.plot(data, x='freq', data_var='amp')
```

### Custom QubitGrid

```python
# Define specific qubit positions
grid = QubitGrid(
    coords={
        'Q0': (0, 0),  # Row 0, Col 0
        'Q1': (0, 1),  # Row 0, Col 1  
        'Q2': (1, 0),  # Row 1, Col 0
        'Q3': (1, 1)   # Row 1, Col 1
    },
    shape=(2, 2)  # 2x2 grid
)
```

## Overlays

### Reference Lines

```python
from qualibration_libs.plotting.overlays import RefLine

# Vertical reference line
ref_line = RefLine(x=5.0, name="Target Frequency")

fig = qplot.QualibrationFigure.plot(
    data,
    x='frequency',
    data_var='amplitude',
    overlays=[ref_line]
)
```

### Fit Overlays

```python
from qualibration_libs.plotting.overlays import FitOverlay

# Add fitted curve
fit_overlay = FitOverlay(
    y_fit=fitted_curve,
    params={'f0': 5.0, 'Q': 1000},
    formatter=lambda p: f"f₀ = {p['f0']:.3f} GHz, Q = {p['Q']:.0f}"
)

fig = qplot.QualibrationFigure.plot(
    data,
    x='frequency', 
    data_var='amplitude',
    overlays=[fit_overlay]
)
```

### Custom Overlays

```python
from qualibration_libs.plotting.overlays import Overlay

class CustomOverlay(Overlay):
    def add_to(self, fig, *, row, col, theme, **style):
        # Add custom plot elements
        fig.add_trace(go.Scatter(...), row=row, col=col)
```

### Per-Qubit Overlays

```python
# Different overlays for different qubits
overlays = {
    'Q0': [RefLine(x=5.0)],
    'Q1': [RefLine(x=5.1)],
    'Q2': [RefLine(x=5.2)]
}

fig = qplot.QualibrationFigure.plot(
    data,
    x='frequency',
    data_var='amplitude', 
    overlays=overlays
)
```

### Dynamic Overlays

```python
def overlay_func(qubit_name, qubit_data):
    # Generate overlays based on qubit data
    if qubit_name == 'Q0':
        return [RefLine(x=5.0)]
    return []

fig = qplot.QualibrationFigure.plot(
    data,
    x='frequency',
    data_var='amplitude',
    overlays=overlay_func
)
```

## Advanced Features

### Secondary X-Axis

```python
# Add secondary x-axis with different units
fig = qplot.QualibrationFigure.plot(
    data,
    x='frequency',      # Primary x-axis
    x2='wavelength',    # Secondary x-axis
    data_var='amplitude'
)
```

#### Customizing Secondary X-Axis Spacing

When using secondary x-axes, you can customize the spacing to prevent text overlap:

```python
# Custom spacing for secondary x-axis plots
fig = qplot.QualibrationFigure.plot(
    data,
    x='amp_prefactor',
    x2='amp_mV',        # Secondary x-axis
    data_var='state',
    title='Power Rabi Calibration',
    # Custom spacing parameters
    x2_top_margin=150,           # Increase top margin (default: 120)
    x2_annotation_offset=0.025   # Move qubit names higher (default: 0.08)
)

# Minimal spacing for compact plots
fig = qplot.QualibrationFigure.plot(
    data,
    x='frequency',
    x2='wavelength',
    data_var='amplitude',
    x2_top_margin=100,           # Smaller top margin
    x2_annotation_offset=0.01    # Smaller annotation offset
)
```

### Hue Dimension

```python
# Color-code by additional dimension
fig = qplot.QualibrationFigure.plot(
    data,
    x='frequency',
    data_var='amplitude',
    hue='power'         # Color by power level
)
```

Note: In multi-qubit/multi-subplot figures, the color cycle resets at the start of each subplot so hue colors are consistent panel-to-panel. Legend entries are deduplicated across the entire figure using `legendgroup`, so each hue value appears only once in the legend on the right.

See demo: `demos/legend_color_reset_demo.py` which saves `legend_color_reset_demo.html` for interactive preview.
### xarray Accessor

```python
# Use xarray accessor for convenient plotting
data.qplot.plot(x='frequency', data_var='amplitude')

# Register accessor globally
qplot.register_accessors()
```

## Examples

### Complete Calibration Plot

```python
import qualibration_libs.plotting as qplot
import xarray as xr
import numpy as np

# Create comprehensive calibration data
frequencies = np.linspace(4.5, 5.5, 100)
powers = np.linspace(-20, 0, 20)
qubits = ['Q0', 'Q1', 'Q2', 'Q3']

# Generate 2D data for each qubit
data = {}
for i, qubit in enumerate(qubits):
    # Create some realistic quantum data
    base_freq = 5.0 + i * 0.1
    response = np.exp(-((frequencies - base_freq) / 0.1)**2)
    power_response = np.outer(powers, response)
    
    data[qubit] = (['power', 'frequency'], power_response)

ds = xr.Dataset(data, coords={
    'frequency': frequencies,
    'power': powers,
    'qubit': qubits
})

# Create 2D heatmap plot
fig = qplot.QualibrationFigure.plot(
    ds,
    x='frequency',
    y='power', 
    data_var='Q0',  # Plot Q0 data
    title="Qubit Q0: Power vs Frequency Response"
)

# Add reference lines
from qualibration_libs.plotting.overlays import RefLine, LineOverlay

overlays = [
    RefLine(x=5.0, name="Target Frequency"),
    RefLine(y=-10, name="Optimal Power")
]

fig = qplot.QualibrationFigure.plot(
    ds,
    x='frequency',
    y='power',
    data_var='Q0',
    overlays=overlays,
    title="Qubit Q0 with Reference Lines"
)

fig.figure.show()
```

### Multi-Qubit Comparison

```python
# Compare all qubits in subplot grid
fig = qplot.QualibrationFigure.plot(
    ds,
    x='detuning',
    data_var='IQ_abs',  # Will plot all qubits
    title="Multi-Qubit IQ Magnitude"
)

# Custom grid layout
from qualibration_libs.plotting import QubitGrid

grid = QubitGrid(
    coords={'Q0': (0, 0), 'Q1': (0, 1), 'Q2': (1, 0), 'Q3': (1, 1)},
    shape=(2, 2)
)

fig = qplot.QualibrationFigure.plot(
    ds,
    x='frequency',
    data_var='Q0',
    grid=grid,
    title="2x2 Qubit Grid"
)
```

### Fit Analysis with Residuals

```python
# Generate data with known fit
x = np.linspace(0, 10, 50)
y_true = 2 * np.sin(x) + 0.5 * x
y_data = y_true + 0.1 * np.random.randn(50)

# Create dataset
ds = xr.Dataset({
    'data': (['x'], y_data),
    'fit': (['x'], y_true)
}, coords={'x': x})

# Fit overlay
from qualibration_libs.plotting.overlays import FitOverlay

fit_overlay = FitOverlay(
    y_fit=y_true,
    params={'amplitude': 2.0, 'frequency': 1.0, 'offset': 0.0},
    formatter=lambda p: f"A = {p['amplitude']:.2f}, f = {p['frequency']:.2f}"
)

fig = qplot.QualibrationFigure.plot(
    ds,
    x='x',
    data_var='data',
    overlays=[fit_overlay],
    residuals=True,
    title="Fit Analysis with Residuals"
)
```

## Quick Reference: Matplotlib → Qualibration Plotting

This section provides before/after examples for migrating from matplotlib to the qualibration plotting module.

### Import Changes

```python
# Before
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# After
import qualibration_libs.plotting as qplot
from qualibration_libs.plotting import QualibrationFigure, QubitGrid
from qualibration_libs.plotting.overlays import RefLine, FitOverlay, TextBoxOverlay
```

### Basic Plotting

```python
# Before
plt.plot(x, y)
plt.xlabel('X Label')
plt.ylabel('Y Label')
plt.title('Title')
plt.show()

# After
ds = xr.Dataset({'y': (['x'], y), 'x': x})
ds.x.attrs['long_name'] = 'X Label'
ds.y.attrs['long_name'] = 'Y Label'
fig = ds.qplot.plot(x='x', data_var='y', title='Title')
node.results["figures"] = {"Title": fig.figure}
```

### Scatter Plots

```python
# Before
plt.scatter(x, y, label='Data')
plt.legend()
plt.show()

# After
ds = xr.Dataset({'y': (['x'], y), 'x': x})
fig = ds.qplot.plot(x='x', data_var='y', title='Scatter Plot')
node.results["figures"] = {"Scatter": fig.figure}
```

### Multi-Subplot Layouts

```python
# Before
fig, axes = plt.subplots(2, 2)
for i, ax in enumerate(axes.flat):
    ax.plot(x, data[i])
    ax.set_title(f'Plot {i}')

# After
ds = xr.Dataset({
    'data': (['qubit', 'x'], data),
    'x': x
})
grid = QubitGrid(coords={f'q{i}': (i//2, i%2) for i in range(4)})
fig = QualibrationFigure.plot(ds, x='x', data_var='data', grid=grid)
node.results["figures"] = {"Multi-Plot": fig.figure}
```

### Reference Lines

```python
# Before
plt.axvline(x=threshold, color='r', linestyle='--', label='Threshold')
plt.axhline(y=baseline, color='g', linestyle=':', label='Baseline')

# After
overlays = [
    RefLine(x=threshold, name='Threshold', dash='dash'),
    RefLine(y=baseline, name='Baseline', dash='dot')
]
fig = ds.qplot.plot(x='x', data_var='y', overlays=overlays)
```

### Fit Overlays

```python
# Before
plt.plot(x, y_fit, 'r--', label=f'Fit: {params}')
plt.legend()

# After
fit_overlay = FitOverlay(
    y_fit=y_fit,
    params=params,
    formatter=lambda p: f"Fit: {p['param']:.2f}",
    name='Fit'
)
fig = ds.qplot.plot(x='x', data_var='y', overlays=[fit_overlay])
```

### 2D Heatmaps

```python
# Before
plt.imshow(Z, extent=[x.min(), x.max(), y.min(), y.max()])
plt.colorbar()

# After
ds = xr.Dataset({'Z': (['x', 'y'], Z), 'x': x, 'y': y})
fig = ds.qplot.plot(x='x', y='y', data_var='Z')
```

### Annotations

```python
# Before
plt.annotate('Peak', xy=(peak_x, peak_y), xytext=(peak_x+1, peak_y+1),
             arrowprops=dict(arrowstyle='->'))

# After
text_overlay = TextBoxOverlay(
    text=f'Peak at ({peak_x:.1f}, {peak_y:.1f})',
    anchor='top right'
)
fig = ds.qplot.plot(x='x', data_var='y', overlays=[text_overlay])
```

### Debug Plots

```python
# Before
plt.plot(x, y, label='Data')
plt.plot(x, y_fit, label='Fit')
plt.title('Debug Plot')
plt.legend()
plt.show()

# After
ds_debug = xr.Dataset({
    'data': (['x'], y),
    'fit': (['x'], y_fit),
    'x': x
})
fit_overlay = FitOverlay(y_fit=ds_debug.fit.values, name='Fit')
fig = ds_debug.qplot.plot(x='x', data_var='data', overlays=[fit_overlay], title='Debug Plot')
node.results["figures"] = {"Debug": fig.figure}
```

### Common Patterns

#### 1. Simple Line Plot
```python
# Data: x, y arrays
ds = xr.Dataset({'y': (['x'], y), 'x': x})
fig = ds.qplot.plot(x='x', data_var='y')
```

#### 2. Multi-Qubit Plot
```python
# Data: dict of {qubit: data} or xarray with qubit dimension
fig = ds.qplot.plot(x='x', data_var='y', qubit_dim='qubit')
```

#### 3. Plot with Fit
```python
# Data: x, y, y_fit arrays
ds = xr.Dataset({'y': (['x'], y), 'x': x})
fit_overlay = FitOverlay(y_fit=y_fit, params=params)
fig = ds.qplot.plot(x='x', data_var='y', overlays=[fit_overlay])
```

#### 4. 2D Heatmap
```python
# Data: 2D array Z with coordinates x, y
ds = xr.Dataset({'Z': (['x', 'y'], Z), 'x': x, 'y': y})
fig = ds.qplot.plot(x='x', y='y', data_var='Z')
```

## Migration Process

### Step-by-Step Migration from Matplotlib

#### Step 1: Identify Current Plotting Code
1. Search for `import matplotlib` or `plt.` in your code
2. Identify the plotting patterns used
3. Note the data structures being plotted

#### Step 2: Convert Data to xarray
1. Wrap your data in `xr.Dataset` or `xr.DataArray`
2. Add metadata attributes (`long_name`, `units`)
3. Ensure proper dimension names

**Data Conversion Examples:**

```python
import xarray as xr
import qualibration_libs.io as qio

# From NumPy arrays
x = np.linspace(0, 10, 100)
y = np.sin(x)
ds = xr.Dataset({
    'amplitude': (['time'], y),
    'time': x
})
ds.time.attrs['long_name'] = 'Time'
ds.time.attrs['units'] = 'ns'
ds.amplitude.attrs['long_name'] = 'Amplitude'
ds.amplitude.attrs['units'] = 'V'

# Save dataset using io.py
qio.save_dataset(ds, 'my_data.nc')

# Load existing dataset with automatic version upgrade
ds = qio.load_dataset('old_data.nc')
```

#### Step 3: Replace Plotting Calls
1. Use `ds.qplot.plot()` for simple cases
2. Use `QualibrationFigure.plot()` for complex cases
3. Add overlays for reference lines, fits, annotations

#### Step 4: Update Node Results
1. Store figures in `node.results["figures"]`
2. Use descriptive keys for figure names
3. Remove `plt.show()` calls

#### Step 5: Test and Refine
1. Test the new plots in QUAlibrate
2. Adjust styling if needed
3. Add overlays for better visualization

## API Reference

### QualibrationFigure

#### `plot(data, *, x, data_var=None, y=None, hue=None, x2=None, qubit_dim='qubit', qubit_names=None, grid=None, overlays=None, residuals=False, title=None, **style_overrides)`

Create a new plot from data.

**Parameters:**
- `data`: Input data (xarray, pandas, or dict)
- `x`: X-axis coordinate name
- `data_var`: Data variable to plot (if None, uses first data variable)
- `y`: Y-axis coordinate for 2D plots (creates heatmap)
- `hue`: Dimension to color-code by
- `x2`: Secondary x-axis coordinate
- `qubit_dim`: Dimension containing qubit names
- `qubit_names`: Explicit list of qubit names
- `grid`: QubitGrid for custom layout
- `overlays`: Overlay objects or dict/function
- `residuals`: Enable residuals subplot (requires FitOverlay for actual residual data)
- `title`: Plot title
- `**style_overrides`: Additional styling options

### QubitGrid

#### `__init__(coords, shape=None)`

**Parameters:**
- `coords`: Dict mapping qubit names to (row, col) tuples
- `shape`: Optional (n_rows, n_cols) tuple

### Overlay Classes

#### `RefLine(x=None, y=None, name=None, dash='dot', width=None)`
Reference line overlay.

#### `LineOverlay(x, y, name=None, dash='dash', width=None, show_legend=True)`
Custom line overlay.

#### `ScatterOverlay(x, y, name=None, marker_size=None)`
Scatter points overlay.

#### `TextBoxOverlay(text, anchor='top right')`
Text annotation overlay.

#### `FitOverlay(y_fit=None, params=None, formatter=None, name='fit', dash='dash', width=None)` {#fitoverlay}
Fit curve and parameter display overlay.

**Parameters:**
- `y_fit` (array, optional): Array of fitted y-values to plot as a curve
- `params` (dict, optional): Dictionary of fit parameters to display in text box
- `formatter` (callable, optional): Function that formats params dict into display text
- `name` (str): Name for the fit curve trace (default: 'fit')
- `dash` (str): Line style for fit curve (default: 'dash')
- `width` (float, optional): Line width for fit curve

**Features:**
- Creates a dashed line overlay showing the fitted curve
- Displays fit parameters in a text box when both `params` and `formatter` are provided
- Works with residuals plots to show fit quality

**Example:**
```python
fit_overlay = FitOverlay(
    y_fit=fitted_data,
    params={'f0': 5.0, 'Q': 1000, 'amplitude': 0.5},
    formatter=lambda p: f"f₀ = {p['f0']:.3f} GHz, Q = {p['Q']:.0f}",
    name="Resonance Fit"
)
```

### Styling Functions

#### `set_theme(theme=None, *, palette=None, rc=None)`
Set global plotting theme.

#### `set_palette(palette)`
Set color palette.

#### `theme_context(theme=None, *, palette=None, rc=None)`
Context manager for temporary theme changes.

### PlotTheme

Theme configuration with attributes:
- `font_size`: Base font size
- `title_size`: Title font size  
- `tick_label_size`: Tick label font size
- `marker_size`: Default marker size
- `line_width`: Default line width
- `show_grid`: Show grid lines
- `grid_opacity`: Grid line opacity
- `residuals_height_ratio`: Height ratio for residuals subplot
- `figure_bg`: Figure background color
- `paper_bg`: Paper background color
- `colorway`: Default color sequence

## Testing and Demos

### Running Tests

The plotting module includes comprehensive unit tests. Run them using:

```bash
# Run all tests
python qualibration_libs/plotting/run_tests.py

# Or run tests directly with pytest
pytest qualibration_libs/plotting/tests/ -v
```

### Running Demos

Several demo scripts are available to showcase the plotting capabilities:

```bash
# Run all demos
python qualibration_libs/plotting/run_tests.py

# Run individual demos
python qualibration_libs/plotting/demos/simple_demo.py
python qualibration_libs/plotting/demos/basic_plots.py
python qualibration_libs/plotting/demos/advanced_plots.py
python qualibration_libs/plotting/demos/fit_overlay_demo.py
python qualibration_libs/plotting/demos/residuals_demo.py
python qualibration_libs/plotting/demos/feature_verification.py
python qualibration_libs/plotting/demos/correct_raw_vs_fit_demo.py
python qualibration_libs/plotting/demos/real_fit_data_demo.py
```

### Demo Scripts Overview

- **`simple_demo.py`**: Basic functionality with real test data
- **`basic_plots.py`**: Fundamental plotting capabilities (1D, 2D, multi-qubit)
- **`advanced_plots.py`**: Complex scenarios (flux tuning, fit analysis)
- **`fit_overlay_demo.py`**: Fit overlays and parameter display
- **`residuals_demo.py`**: Comprehensive residuals functionality testing
- **`feature_verification.py`**: Advanced features and verification
- **`correct_raw_vs_fit_demo.py`**: Raw data vs fitted data comparison
- **`real_fit_data_demo.py`**: Real experimental data examples

### Test Data

The demos use test data files located in `qualibration_libs/plotting/test_data/`:
- `ds_raw.h5`, `ds_raw_2.h5`, `ds_raw_3.h5`: Raw measurement data
- `ds_fit.h5`, `ds_fit_2.h5`, `ds_fit_3.h5`: Fitted data with parameters

## Dependencies

- plotly
- xarray  
- numpy
- pandas (optional)
- pytest (for testing)

## License

This module is part of the qualibration-libs package. See the main package license for details.
