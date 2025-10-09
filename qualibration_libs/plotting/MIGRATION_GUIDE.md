# Migration Guide: From Matplotlib to Qualibration Plotting

## Overview

This guide helps you migrate from matplotlib-based plotting to the new unified Qualibration plotting system. The new system provides interactive Plotly figures with consistent styling and better integration with QUAlibrate.

## Key Benefits of Migration

- **Interactive plots**: Hover, zoom, pan, toggle traces
- **Web-embeddable**: Works in QUAlibrate web UI
- **Consistent styling**: Unified appearance across all plots
- **Better integration**: Works seamlessly with xarray datasets
- **Multi-qubit layouts**: Automatic grid arrangements
- **Overlay system**: Easy addition of reference lines, fits, annotations

## Migration Patterns

### 1. Basic Plotting

#### Before (Matplotlib)
```python
import matplotlib.pyplot as plt
import numpy as np

# Simple line plot
x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.plot(x, y)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Sine Wave')
plt.show()
```

#### After (Qualibration Plotting)
```python
import xarray as xr
import qualibration_libs.plotting as qplot

# Create dataset
ds = xr.Dataset({
    'amplitude': (['time'], np.sin(x)),
    'time': x
})
ds.time.attrs['long_name'] = 'Time'
ds.amplitude.attrs['long_name'] = 'Amplitude'

# Plot with xarray accessor
fig = ds.qplot.plot(x='time', data_var='amplitude', title='Sine Wave')
# Store in node results
node.results["figures"] = {"Sine Wave": fig.figure}
```

### 2. Multi-Qubit Plots

#### Before (Matplotlib)
```python
import matplotlib.pyplot as plt

# Manual subplot creation
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for i, qubit in enumerate(qubits):
    row, col = i // 4, i % 4
    ax = axes[row, col]
    ax.plot(x, data[qubit])
    ax.set_title(f'Qubit {qubit}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude')
plt.tight_layout()
plt.show()
```

#### After (Qualibration Plotting)
```python
import qualibration_libs.plotting as qplot
from qualibration_libs.plotting import QubitGrid

# Create grid layout
grid = QubitGrid(coords={qubit: (i//4, i%4) for i, qubit in enumerate(qubits)})

# Plot all qubits at once
fig = qplot.QualibrationFigure.plot(
    ds,
    x='time',
    data_var='amplitude',
    qubit_dim='qubit',
    grid=grid,
    title='Multi-Qubit Analysis'
)
node.results["figures"] = {"Multi-Qubit": fig.figure}
```

### 3. Scatter Plots with Fits

#### Before (Matplotlib)
```python
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Plot data
plt.scatter(x, y, label='Data')

# Add fit
def fit_func(x, a, b):
    return a * x + b
popt, _ = curve_fit(fit_func, x, y)
y_fit = fit_func(x, *popt)
plt.plot(x, y_fit, 'r--', label=f'Fit: y = {popt[0]:.2f}x + {popt[1]:.2f}')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Data with Fit')
plt.legend()
plt.show()
```

#### After (Qualibration Plotting)
```python
import qualibration_libs.plotting as qplot
from qualibration_libs.plotting.overlays import FitOverlay

# Create dataset with fit data
ds = xr.Dataset({
    'y': (['x'], y),
    'x': x,
    'y_fit': (['x'], y_fit)
})

# Create fit overlay
fit_overlay = FitOverlay(
    y_fit=ds.y_fit.values,
    params={'slope': popt[0], 'intercept': popt[1]},
    formatter=lambda p: f"y = {p['slope']:.2f}x + {p['intercept']:.2f}",
    name='Fit'
)

# Plot with overlay
fig = qplot.QualibrationFigure.plot(
    ds,
    x='x',
    data_var='y',
    overlays=[fit_overlay],
    title='Data with Fit'
)
node.results["figures"] = {"Data with Fit": fig.figure}
```

### 4. 2D Heatmaps

#### Before (Matplotlib)
```python
import matplotlib.pyplot as plt
import numpy as np

# Create 2D data
X, Y = np.meshgrid(x, y)
Z = np.sin(X) * np.cos(Y)

# Plot heatmap
plt.imshow(Z, extent=[x.min(), x.max(), y.min(), y.max()], aspect='auto')
plt.colorbar(label='Intensity')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('2D Heatmap')
plt.show()
```

#### After (Qualibration Plotting)
```python
import qualibration_libs.plotting as qplot

# Create dataset
ds = xr.Dataset({
    'intensity': (['x', 'y'], Z),
    'x': x,
    'y': y
})

# Plot heatmap
fig = qplot.QualibrationFigure.plot(
    ds,
    x='x',
    y='y',
    data_var='intensity',
    title='2D Heatmap'
)
node.results["figures"] = {"2D Heatmap": fig.figure}
```

### 5. Reference Lines and Annotations

#### Before (Matplotlib)
```python
import matplotlib.pyplot as plt

# Plot data
plt.plot(x, y)

# Add reference lines
plt.axvline(x=threshold, color='r', linestyle='--', label='Threshold')
plt.axhline(y=baseline, color='g', linestyle=':', label='Baseline')

# Add annotation
plt.annotate('Peak', xy=(peak_x, peak_y), xytext=(peak_x+1, peak_y+1),
             arrowprops=dict(arrowstyle='->', color='red'))

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Data with References')
plt.legend()
plt.show()
```

#### After (Qualibration Plotting)
```python
import qualibration_libs.plotting as qplot
from qualibration_libs.plotting.overlays import RefLine, TextBoxOverlay

# Create overlays
overlays = [
    RefLine(x=threshold, name='Threshold', dash='dash'),
    RefLine(y=baseline, name='Baseline', dash='dot'),
    TextBoxOverlay(
        text=f'Peak at ({peak_x:.1f}, {peak_y:.1f})',
        anchor='top right'
    )
]

# Plot with overlays
fig = qplot.QualibrationFigure.plot(
    ds,
    x='x',
    data_var='y',
    overlays=overlays,
    title='Data with References'
)
node.results["figures"] = {"Data with References": fig.figure}
```

## Common Migration Scenarios

### Scenario 1: Debug Plots in Analysis Code

#### Before (in `qualibration_libs/analysis/fitting.py`)
```python
from matplotlib import pyplot as plt

# Debug plot when fit fails
plt.plot(x, decay_exp(x, a, offset, decay))
plt.plot(x, y)
plt.show()
```

#### After
```python
import qualibration_libs.plotting as qplot

# Debug plot when fit fails
ds_debug = xr.Dataset({
    'data': (['x'], y),
    'fit': (['x'], decay_exp(x, a, offset, decay)),
    'x': x
})

fig = qplot.QualibrationFigure.plot(
    ds_debug,
    x='x',
    data_var='data',
    overlays=[qplot.LineOverlay(x=x, y=decay_exp(x, a, offset, decay), name='Fit')],
    title='Fit Debug'
)
# Store in results for debugging
node.results["figures"] = {"Fit Debug": fig.figure}
```

### Scenario 2: Simulation Results

#### Before (in `qualibration_libs/runtime/simulate.py`)
```python
from matplotlib import pyplot as plt

# Plot simulated samples
fig, ax = plt.subplots(nrows=len(samples.keys()), sharex=True)
for i, con in enumerate(samples.keys()):
    plt.subplot(len(samples.keys()), 1, i + 1)
    samples[con].plot()
    plt.title(con)
plt.tight_layout()
```

#### After
```python
import qualibration_libs.plotting as qplot

# Convert samples to xarray
ds_samples = xr.Dataset({
    con: (['time'], samples[con].values) 
    for con in samples.keys()
})

# Plot with multi-qubit layout
fig = qplot.QualibrationFigure.plot(
    ds_samples,
    x='time',
    data_var=list(samples.keys())[0],  # Primary trace
    title='Simulated Samples'
)
node.results["figures"] = {"Simulated Samples": fig.figure}
```

## Step-by-Step Migration Process

### Step 1: Identify Current Plotting Code
1. Search for `import matplotlib` or `plt.` in your code
2. Identify the plotting patterns used
3. Note the data structures being plotted

### Step 2: Convert Data to xarray
1. Wrap your data in `xr.Dataset` or `xr.DataArray`
2. Add metadata attributes (`long_name`, `units`)
3. Ensure proper dimension names

### Step 3: Replace Plotting Calls
1. Use `ds.qplot.plot()` for simple cases
2. Use `QualibrationFigure.plot()` for complex cases
3. Add overlays for reference lines, fits, annotations

### Step 4: Update Node Results
1. Store figures in `node.results["figures"]`
2. Use descriptive keys for figure names
3. Remove `plt.show()` calls

### Step 5: Test and Refine
1. Test the new plots in QUAlibrate
2. Adjust styling if needed
3. Add overlays for better visualization

## Migration Checklist

- [ ] **Remove matplotlib imports**: Replace with `qualibration_libs.plotting`
- [ ] **Convert data to xarray**: Wrap data in `xr.Dataset`
- [ ] **Add metadata**: Set `long_name` and `units` attributes
- [ ] **Replace plotting calls**: Use `ds.qplot.plot()` or `QualibrationFigure.plot()`
- [ ] **Add overlays**: Use `RefLine`, `FitOverlay`, `TextBoxOverlay` as needed
- [ ] **Update node results**: Store in `node.results["figures"]`
- [ ] **Remove plt.show()**: Figures are automatically displayed
- [ ] **Test integration**: Verify plots work in QUAlibrate web UI

## Common Pitfalls and Solutions

### Pitfall 1: Forgetting to Store Figures
```python
# ❌ Wrong - figure not stored
fig = ds.qplot.plot(x='x', data_var='y')

# ✅ Correct - store in node results
fig = ds.qplot.plot(x='x', data_var='y')
node.results["figures"] = {"My Plot": fig.figure}
```

### Pitfall 2: Not Adding Metadata
```python
# ❌ Wrong - no metadata
ds = xr.Dataset({'y': (['x'], y), 'x': x})

# ✅ Correct - add metadata
ds = xr.Dataset({'y': (['x'], y), 'x': x})
ds.x.attrs['long_name'] = 'Time'
ds.x.attrs['units'] = 'ns'
ds.y.attrs['long_name'] = 'Amplitude'
ds.y.attrs['units'] = 'V'
```

### Pitfall 3: Complex Overlays
```python
# ❌ Wrong - trying to recreate matplotlib complexity
# Use the overlay system instead

# ✅ Correct - use overlays
overlays = [
    RefLine(x=threshold, name='Threshold'),
    FitOverlay(y_fit=fit_data, params=fit_params),
    TextBoxOverlay(text='Analysis Results')
]
```

## Benefits After Migration

1. **Interactive plots**: Users can zoom, pan, hover for details
2. **Consistent styling**: All plots look professional and unified
3. **Better integration**: Works seamlessly with QUAlibrate web UI
4. **Easier maintenance**: Centralized styling and theming
5. **More features**: Multi-qubit layouts, overlays, residuals
6. **Better performance**: Plotly is optimized for web display

## Getting Help

- **Documentation**: See `README.md` for complete API reference
- **Examples**: Check `demos/` directory for working examples
- **Tests**: Look at `tests/` for usage patterns
- **Styling**: See `config.py` for theme customization

## Next Steps

1. **Start with simple plots**: Begin with basic 1D plots
2. **Add complexity gradually**: Add overlays and multi-qubit layouts
3. **Customize styling**: Adjust themes and palettes as needed
4. **Share feedback**: Report issues and suggest improvements

The new plotting system provides a much better user experience and integrates seamlessly with QUAlibrate!
