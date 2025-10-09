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
# Add residuals subplot
fig = qplot.QualibrationFigure.plot(
    data,
    x='frequency',
    data_var='amplitude',
    residuals=True,         # Enable residuals plot
    title="Fit with Residuals"
)
```

## Styling and Themes

### Built-in Themes

```python
# Set global theme
qplot.set_theme(
    theme=qplot.PlotTheme(
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
with qplot.theme_context(
    theme=qplot.PlotTheme(font_size=20),
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
    x='frequency',
    data_var='Q0',  # Will plot all qubits
    title="Multi-Qubit Frequency Response"
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
- `residuals`: Enable residuals subplot
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

#### `FitOverlay(y_fit=None, params=None, formatter=None, name='fit', dash='dash', width=None)`
Fit curve and parameter display overlay.

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

## Best Practices

1. **Use xarray for data**: Preserves metadata and enables automatic labeling
2. **Set meaningful attributes**: Use `long_name` and `units` in coordinates
3. **Leverage overlays**: Add reference lines, fits, and annotations
4. **Use theme contexts**: For temporary styling changes
5. **Custom grids**: For complex qubit layouts
6. **Residuals plots**: For fit quality assessment
7. **Consistent styling**: Use global themes for publication-ready plots

## Dependencies

- plotly
- xarray  
- numpy
- pandas (optional)

## License

This module is part of the qualibration-libs package. See the main package license for details.
