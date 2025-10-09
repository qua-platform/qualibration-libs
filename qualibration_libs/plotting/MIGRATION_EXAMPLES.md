# Migration Examples: Current Code → New Plotting

## Example 1: Analysis Debug Plots (`qualibration_libs/analysis/fitting.py`)

### Current Code (Matplotlib)
```python
from matplotlib import pyplot as plt

# In decay_exp fitting function
try:
    # ... fitting code ...
except RuntimeError:
    print("Fit failed:")
    print(f"{a=}, {offset=}, {decay=}")
    plt.plot(x, decay_exp(x, a, offset, decay))
    plt.plot(x, y)
    plt.show()
    # raise e
```

### Migrated Code (Qualibration Plotting)
```python
import qualibration_libs.plotting as qplot
from qualibration_libs.plotting.overlays import FitOverlay
import xarray as xr

# In decay_exp fitting function
try:
    # ... fitting code ...
except RuntimeError:
    print("Fit failed:")
    print(f"{a=}, {offset=}, {decay=}")
    
    # Create debug dataset
    ds_debug = xr.Dataset({
        'data': (['x'], y),
        'fit': (['x'], decay_exp(x, a, offset, decay)),
        'x': x
    })
    ds_debug.x.attrs['long_name'] = 'Time'
    ds_debug.data.attrs['long_name'] = 'Data'
    ds_debug.fit.attrs['long_name'] = 'Fit'
    
    # Create fit overlay
    fit_overlay = FitOverlay(
        y_fit=ds_debug.fit.values,
        params={'a': a, 'offset': offset, 'decay': decay},
        formatter=lambda p: f"a={p['a']:.3f}, offset={p['offset']:.3f}, decay={p['decay']:.3f}",
        name='Failed Fit'
    )
    
    # Create debug plot
    fig = qplot.QualibrationFigure.plot(
        ds_debug,
        x='x',
        data_var='data',
        overlays=[fit_overlay],
        title='Fit Debug - Decay Exp'
    )
    
    # Store in results for debugging
    if hasattr(node, 'results'):
        node.results["figures"] = {"Fit Debug - Decay Exp": fig.figure}
    # raise e
```

## Example 2: Simulation Results (`qualibration_libs/runtime/simulate.py`)

### Current Code (Matplotlib)
```python
from matplotlib import pyplot as plt

# Plot the simulated samples
samples = job.get_simulated_samples()
fig, ax = plt.subplots(nrows=len(samples.keys()), sharex=True)

for i, con in enumerate(samples.keys()):
    plt.subplot(len(samples.keys()), 1, i + 1)
    samples[con].plot()
    plt.title(con)
plt.tight_layout()
```

### Migrated Code (Qualibration Plotting)
```python
import qualibration_libs.plotting as qplot
import xarray as xr
import numpy as np

# Plot the simulated samples
samples = job.get_simulated_samples()

# Convert samples to xarray dataset
sample_data = {}
time_coords = None

for con, sample in samples.items():
    sample_data[con] = (['time'], sample.values)
    if time_coords is None:
        time_coords = sample.coords['time'].values

ds_samples = xr.Dataset(sample_data, coords={'time': time_coords})

# Add metadata
ds_samples.time.attrs['long_name'] = 'Time'
for con in samples.keys():
    ds_samples[con].attrs['long_name'] = f'{con} Signal'
    ds_samples[con].attrs['units'] = 'V'

# Create multi-trace plot
fig = qplot.QualibrationFigure.plot(
    ds_samples,
    x='time',
    data_var=list(samples.keys())[0],  # Primary trace
    title='Simulated Samples'
)

# Store in results
node.results["figures"] = {"Simulated Samples": fig.figure}
```

## Example 3: Oscillation Fit Debug

### Current Code (Matplotlib)
```python
# In oscillation_decay_exp fitting function
try:
    # ... fitting code ...
except RuntimeError:
    print(f"{a=}, {f=}, {phi=}, {offset=}, {decay=}")
    plt.plot(x, oscillation_decay_exp(x, a, f, phi, offset, decay))
    plt.plot(x, y)
    plt.show()
```

### Migrated Code (Qualibration Plotting)
```python
# In oscillation_decay_exp fitting function
try:
    # ... fitting code ...
except RuntimeError:
    print(f"{a=}, {f=}, {phi=}, {offset=}, {decay=}")
    
    # Create debug dataset
    ds_debug = xr.Dataset({
        'data': (['x'], y),
        'fit': (['x'], oscillation_decay_exp(x, a, f, phi, offset, decay)),
        'x': x
    })
    ds_debug.x.attrs['long_name'] = 'Time'
    ds_debug.data.attrs['long_name'] = 'Data'
    ds_debug.fit.attrs['long_name'] = 'Oscillation Fit'
    
    # Create fit overlay with parameters
    fit_overlay = FitOverlay(
        y_fit=ds_debug.fit.values,
        params={'a': a, 'f': f, 'phi': phi, 'offset': offset, 'decay': decay},
        formatter=lambda p: f"a={p['a']:.3f}, f={p['f']:.3f}, φ={p['phi']:.3f}, offset={p['offset']:.3f}, decay={p['decay']:.3f}",
        name='Oscillation Fit'
    )
    
    # Create debug plot
    fig = qplot.QualibrationFigure.plot(
        ds_debug,
        x='x',
        data_var='data',
        overlays=[fit_overlay],
        title='Oscillation Fit Debug'
    )
    
    # Store in results
    if hasattr(node, 'results'):
        node.results["figures"] = {"Oscillation Fit Debug": fig.figure}
```

## Example 4: Simple Oscillation Debug

### Current Code (Matplotlib)
```python
# In oscillation fitting function
try:
    # ... fitting code ...
except RuntimeError as e:
    print(f"{a=}, {f=}, {phi=}, {offset=}")
    plt.plot(x, oscillation(x, a, f, phi, offset))
    plt.plot(x, y)
    plt.show()
    raise e
```

### Migrated Code (Qualibration Plotting)
```python
# In oscillation fitting function
try:
    # ... fitting code ...
except RuntimeError as e:
    print(f"{a=}, {f=}, {phi=}, {offset=}")
    
    # Create debug dataset
    ds_debug = xr.Dataset({
        'data': (['x'], y),
        'fit': (['x'], oscillation(x, a, f, phi, offset)),
        'x': x
    })
    ds_debug.x.attrs['long_name'] = 'Time'
    ds_debug.data.attrs['long_name'] = 'Data'
    ds_debug.fit.attrs['long_name'] = 'Oscillation Fit'
    
    # Create fit overlay
    fit_overlay = FitOverlay(
        y_fit=ds_debug.fit.values,
        params={'a': a, 'f': f, 'phi': phi, 'offset': offset},
        formatter=lambda p: f"a={p['a']:.3f}, f={p['f']:.3f}, φ={p['phi']:.3f}, offset={p['offset']:.3f}",
        name='Oscillation Fit'
    )
    
    # Create debug plot
    fig = qplot.QualibrationFigure.plot(
        ds_debug,
        x='x',
        data_var='data',
        overlays=[fit_overlay],
        title='Oscillation Debug'
    )
    
    # Store in results
    if hasattr(node, 'results'):
        node.results["figures"] = {"Oscillation Debug": fig.figure}
    
    raise e
```

## Example 5: Multi-Qubit Analysis Plot

### Current Code (Matplotlib)
```python
# Manual multi-qubit plotting
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

### Migrated Code (Qualibration Plotting)
```python
import qualibration_libs.plotting as qplot
from qualibration_libs.plotting import QubitGrid
import xarray as xr

# Create dataset with qubit dimension
ds = xr.Dataset({
    'amplitude': (['qubit', 'time'], data),
    'time': x
})

# Add metadata
ds.time.attrs['long_name'] = 'Time'
ds.time.attrs['units'] = 'ns'
ds.amplitude.attrs['long_name'] = 'Amplitude'
ds.amplitude.attrs['units'] = 'V'

# Create grid layout
grid = QubitGrid(coords={qubit: (i//4, i%4) for i, qubit in enumerate(qubits)})

# Plot with multi-qubit layout
fig = qplot.QualibrationFigure.plot(
    ds,
    x='time',
    data_var='amplitude',
    qubit_dim='qubit',
    grid=grid,
    title='Multi-Qubit Analysis'
)

# Store in results
node.results["figures"] = {"Multi-Qubit Analysis": fig.figure}
```

## Migration Benefits

### Before (Matplotlib)
- Static plots
- Manual subplot management
- Inconsistent styling
- No web integration
- Debug plots not saved

### After (Qualibration Plotting)
- Interactive plots with hover/zoom
- Automatic multi-qubit layouts
- Consistent professional styling
- Web-embeddable in QUAlibrate
- Debug plots saved in results
- Better error visualization
- Parameter display in overlays

## Key Migration Steps

1. **Replace imports**: `matplotlib.pyplot` → `qualibration_libs.plotting`
2. **Convert data**: Wrap in `xr.Dataset` with metadata
3. **Replace plotting**: `plt.plot()` → `ds.qplot.plot()`
4. **Add overlays**: Use `FitOverlay`, `RefLine`, `TextBoxOverlay`
5. **Store results**: `node.results["figures"] = {"Name": fig.figure}`
6. **Remove plt.show()**: Figures display automatically

## Testing Migration

After migration, test that:
- [ ] Plots display correctly in QUAlibrate web UI
- [ ] Interactive features work (hover, zoom, pan)
- [ ] Debug plots are saved in results
- [ ] Styling is consistent with other plots
- [ ] Multi-qubit layouts work properly
- [ ] Overlays display correctly
