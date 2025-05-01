# qualibration_libs.data

This directory provides tools related to data acquisition and processing

## XarrayDataFetcher

The `XarrayDataFetcher` class is a powerful tool for iteratively fetching data from a running or completed QM job.
Automatically structures the fetched data into an `xarray.Dataset`, using provided coordinate axes.
Handles different data shapes, including cases with a leading "qubit" dimension.
Supports live data acquisition during job execution via iteration.

To begin, sweep axes need to be provided. This is a dictionary where the keys are the axis names, and the values
are the arrays. The arrays can be numpy arrays, though xarray is preferred as it allows additional parameters such as units to be passed.
Note that the order of the axes matters as the first entry should match the outermost loop, etc.

**Example:**

```python
import xarray as xr
import numpy as np
from qm import qua
from qualang_tools.loops import from_array
from qualibration_libs.data import XarrayDataFetcher, progress_counter

# Define
axes = {
    "qubits": xr.DataArray(["q0", "q1"]),
    "num_pi_pulses": xr.DataArray([1, 2, 3]),
    "amplitudes": xr.DataArray([0.1, 0.2, 0.3]),
}
```

After defining the sweep axes, they can be used in a QUA program and executed

```python
with program as prog:
    num_pi = qua.declare(int)
    amplitudes = qua.declare(float)

    for qubit in qubits:  # Here we assume qubits is already defined somewhere
        with qua.for_(*from_array(num_pi, axes["num_pi_pulses"])):
            with qua.for_(*from_array(amplitudes, axes["amplitudes"])):
                # Perform QUA pulse operations here

qmm = QuantumMachinesManager()
qm = qmm.open_qm(config)
job = qm.execute(prog)
```

Once the program is running, the `XarrayDataFetcher` can be used to aquire data:

```python
data_fetcher = XarrayDataFetcher(job, axes)

#
for dataset in data_fetcher:
    progress_counter(data_fetcher["n"], n_avg, start_time=data_fetcher.t_start)
```

The main routine is `for dataset in data_fetcher:`, which is a for loop that keeps providing an updated Xarray dataset using the coordinates from the sweep axes and the data fetched so far.
Additionally, specific data variables can be accessed through `data_fetcher["variable_name"]`.

Once the job has completed, the data fetcher's for loop runs one more time, returning the full dataset. This dataset can then be used for post-processing.
