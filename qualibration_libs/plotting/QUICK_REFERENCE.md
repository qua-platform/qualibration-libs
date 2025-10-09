# Quick Reference: Matplotlib → Qualibration Plotting

## Import Changes

```python
# Before
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# After
import qualibration_libs.plotting as qplot
from qualibration_libs.plotting import QualibrationFigure, QubitGrid
from qualibration_libs.plotting.overlays import RefLine, FitOverlay, TextBoxOverlay
```

## Basic Plotting

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

## Scatter Plots

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

## Multi-Subplot Layouts

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

## Reference Lines

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

## Fit Overlays

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

## 2D Heatmaps

```python
# Before
plt.imshow(Z, extent=[x.min(), x.max(), y.min(), y.max()])
plt.colorbar()

# After
ds = xr.Dataset({'Z': (['x', 'y'], Z), 'x': x, 'y': y})
fig = ds.qplot.plot(x='x', y='y', data_var='Z')
```

## Annotations

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

## Debug Plots

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

## Common Patterns

### 1. Simple Line Plot
```python
# Data: x, y arrays
ds = xr.Dataset({'y': (['x'], y), 'x': x})
fig = ds.qplot.plot(x='x', data_var='y')
```

### 2. Multi-Qubit Plot
```python
# Data: dict of {qubit: data} or xarray with qubit dimension
fig = ds.qplot.plot(x='x', data_var='y', qubit_dim='qubit')
```

### 3. Plot with Fit
```python
# Data: x, y, y_fit arrays
ds = xr.Dataset({'y': (['x'], y), 'x': x})
fit_overlay = FitOverlay(y_fit=y_fit, params=params)
fig = ds.qplot.plot(x='x', data_var='y', overlays=[fit_overlay])
```

### 4. 2D Heatmap
```python
# Data: 2D array Z with coordinates x, y
ds = xr.Dataset({'Z': (['x', 'y'], Z), 'x': x, 'y': y})
fig = ds.qplot.plot(x='x', y='y', data_var='Z')
```

## Essential Reminders

1. **Always store figures**: `node.results["figures"] = {"Name": fig.figure}`
2. **Add metadata**: Set `long_name` and `units` attributes
3. **Use overlays**: For reference lines, fits, annotations
4. **Remove plt.show()**: Figures display automatically
5. **Test in QUAlibrate**: Verify web UI integration

## Migration Checklist

- [ ] Replace `import matplotlib.pyplot as plt`
- [ ] Convert data to `xr.Dataset`
- [ ] Add metadata attributes
- [ ] Replace `plt.plot()` with `ds.qplot.plot()`
- [ ] Add overlays for reference lines/fits
- [ ] Store in `node.results["figures"]`
- [ ] Remove `plt.show()` calls
- [ ] Test in QUAlibrate web UI
