# Qualibration Plotting Module - Feature Status

## ✅ Working Features

All major plotting features are working correctly:

### 1. Multi-Qubit Subplot Grids ✅
- **Default layouts**: Automatic 1xN grid for N qubits
- **Custom grids**: User-defined qubit positions in any grid layout
- **Grid examples**: 1x8, 2x4, 4x2, 2x2, etc.
- **Subplot titles**: Each qubit gets its own subplot with title

**Example:**
```python
from qualibration_libs.plotting import QubitGrid

# Custom 2x4 grid
grid = QubitGrid(
    coords={'qC1': (0, 0), 'qC2': (0, 1), 'qC3': (0, 2), 'qC4': (0, 3),
            'qD1': (1, 0), 'qD2': (1, 1), 'qD3': (1, 2), 'qD4': (1, 3)},
    shape=(2, 4)
)

fig = qplot.QualibrationFigure.plot(
    ds,
    x='detuning',
    data_var='IQ_abs',
    grid=grid,
    title="Custom 2x4 Qubit Grid"
)
```

### 2. 2D Heatmap Plots ✅
- **2D data**: Direct heatmap from 2D datasets
- **3D to 2D**: Convert 3D data to 2D heatmaps
- **Multi-qubit heatmaps**: Compare multiple qubits in heatmap format
- **Automatic color scaling**: Plotly handles color mapping

**Example:**
```python
# 2D heatmap from 3D data
fig = qplot.QualibrationFigure.plot(
    ds_3d.isel(qubit=0),
    x='detuning',
    y='power',
    data_var='IQ_abs',
    title="2D Heatmap: Detuning vs Power"
)
```

### 3. Overlay System ✅
- **Reference lines**: Vertical and horizontal reference lines
- **Custom lines**: User-defined line overlays
- **Per-qubit overlays**: Different overlays for different qubits
- **Dynamic overlays**: Function-based overlay generation
- **Fit overlays**: Fit curves with parameter display

**Example:**
```python
from qualibration_libs.plotting.overlays import RefLine, LineOverlay

# Reference lines
overlays = [
    RefLine(x=0, name="Zero Detuning"),
    RefLine(x=1e6, name="1 MHz Offset")
]

# Custom line overlay
detuning = ds.coords['detuning'].values
fit_line = np.exp(-((detuning - center) / width)**2)
line_overlay = LineOverlay(x=detuning, y=fit_line, name="Fit")

fig = qplot.QualibrationFigure.plot(
    ds,
    x='detuning',
    data_var='IQ_abs',
    overlays=overlays + [line_overlay]
)
```

### 4. Theme Customization ✅
- **Custom themes**: Font sizes, colors, grid settings
- **Color palettes**: Built-in and custom color schemes
- **Theme contexts**: Temporary theme changes
- **RC parameters**: Fine-grained control

**Example:**
```python
# Custom theme
custom_theme = PlotTheme(
    font_size=18,
    marker_size=10,
    line_width=3,
    show_grid=True,
    grid_opacity=0.5
)
qplot.set_theme(theme=custom_theme)

# Custom palette
qplot.set_palette(["#FF0000", "#00FF00", "#0000FF"])

# Theme context
with qplot.theme_context(theme=PlotTheme(font_size=20), palette="deep"):
    fig = qplot.QualibrationFigure.plot(...)
```

### 5. Residuals Plots ✅
- **Automatic residuals**: Add residuals subplot below main plot
- **Zero reference line**: Horizontal line at y=0
- **Height ratio control**: Adjustable residuals subplot height
- **Multi-qubit residuals**: Residuals for each qubit

**Example:**
```python
fig = qplot.QualibrationFigure.plot(
    ds,
    x='detuning',
    data_var='IQ_abs',
    residuals=True,
    title="Plot with Residuals"
)
```

### 6. Data Input Formats ✅
- **xarray.Dataset/DataArray**: Preferred format with metadata
- **pandas.DataFrame**: Automatic conversion
- **Dictionary**: Simple key-value data
- **HDF5 files**: Direct loading with xarray

### 7. Advanced Features ✅
- **Secondary x-axis**: Dual x-axis with different units
- **Hue dimensions**: Color-coding by additional variables
- **xarray accessors**: Convenient `.qplot.plot()` syntax
- **Error handling**: Graceful handling of invalid inputs

## 🎯 Usage Examples

### Basic Multi-Qubit Plot
```python
import qualibration_libs.plotting as qplot
import xarray as xr

# Load data
ds = xr.open_dataset('test_data/ds_raw.h5')

# Create multi-qubit plot
fig = qplot.QualibrationFigure.plot(
    ds,
    x='detuning',
    data_var='IQ_abs',
    title="Multi-Qubit IQ Magnitude"
)
fig.figure.show()
```

### Custom Grid Layout
```python
from qualibration_libs.plotting import QubitGrid

# Define custom grid
grid = QubitGrid(
    coords={'qC1': (0, 0), 'qC2': (0, 1), 'qC3': (1, 0), 'qC4': (1, 1)},
    shape=(2, 2)
)

fig = qplot.QualibrationFigure.plot(
    ds,
    x='detuning',
    data_var='IQ_abs',
    grid=grid,
    title="2x2 Qubit Grid"
)
```

### 2D Heatmap
```python
# 2D heatmap from 3D data
fig = qplot.QualibrationFigure.plot(
    ds_3d.isel(qubit=0),
    x='detuning',
    y='power',
    data_var='IQ_abs',
    title="Power vs Detuning Heatmap"
)
```

### With Overlays
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
    residuals=True,
    title="Analysis with Overlays and Residuals"
)
```

## 🚀 Quick Start

### 1. Basic Plot
```python
import qualibration_libs.plotting as qplot
import xarray as xr

# Load your data
ds = xr.open_dataset('your_data.h5')

# Create plot
fig = qplot.QualibrationFigure.plot(
    ds,
    x='frequency',
    data_var='amplitude',
    title="Your Plot"
)
fig.figure.show()
```

### 2. Multi-Qubit Grid
```python
# Custom grid layout
from qualibration_libs.plotting import QubitGrid

grid = QubitGrid(
    coords={'Q0': (0, 0), 'Q1': (0, 1), 'Q2': (1, 0), 'Q3': (1, 1)},
    shape=(2, 2)
)

fig = qplot.QualibrationFigure.plot(
    ds,
    x='frequency',
    data_var='amplitude',
    grid=grid
)
```

### 3. 2D Heatmap
```python
# 2D heatmap
fig = qplot.QualibrationFigure.plot(
    ds,
    x='frequency',
    y='power',
    data_var='amplitude',
    title="2D Heatmap"
)
```

## 📊 Data Requirements

### For Multi-Qubit Plots
- Data must have a qubit dimension
- Each qubit gets its own subplot
- Automatic layout or custom grid

### For 2D Heatmaps
- Data must have 2 coordinate dimensions (x, y)
- Data variable for z-axis values
- Automatic color scaling

### For Overlays
- Compatible with all plot types
- Per-qubit or global overlays
- Custom styling options

## 🔧 Troubleshooting

### Common Issues

1. **Subplots not showing**: Ensure data has qubit dimension
2. **Heatmap not appearing**: Check that both x and y coordinates exist
3. **Overlays not visible**: Verify overlay coordinates match data range
4. **Theme not applied**: Check theme context and reset if needed

### Debug Tips

```python
# Check data structure
print(f"Data shape: {dict(ds.dims)}")
print(f"Coordinates: {list(ds.coords)}")
print(f"Data variables: {list(ds.data_vars)}")

# Check plot structure
print(f"Number of traces: {len(fig.figure.data)}")
print(f"Subplot titles: {[ann.text for ann in fig.figure.layout.annotations if ann.text]}")
```

## 📈 Performance

- **Data loading**: Efficient with xarray
- **Plotting**: Fast with Plotly
- **Memory usage**: Scales with data size
- **Interactive**: Full Plotly interactivity

## 🎉 Conclusion

All major plotting features are working correctly:
- ✅ Multi-qubit subplot grids
- ✅ 2D heatmap plots  
- ✅ Overlay functionality
- ✅ Theme customization
- ✅ Residuals plots
- ✅ Advanced features

The module is ready for production use with real quantum device data!
