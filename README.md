# qualibration-libs

[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

**Core utility library for the QUAlibrate quantum calibration platform.**

## Introduction

`qualibration-libs` provides a collection of essential tools and utility functions used internally by the [QUAlibrate](https://qua-platform.github.io/qualibrate/) software. These libraries facilitate data handling, processing, plotting, and interaction with quantum hardware configurations, primarily within the context of QUAlibrate's calibration nodes and graphs.

## What is QUAlibrate?

[QUAlibrate](https://qua-platform.github.io/qualibrate/) is an advanced, open-source calibration platform designed to streamline the calibration process for quantum computers, particularly those using OPX controllers. It enables users to create, manage, and execute calibration routines, abstracting hardware complexities and allowing focus on the quantum system itself. Key components include:

- **Calibration Nodes:** Reusable scripts for specific calibration tasks.
- **Calibration Graphs:** Directed acyclic graphs linking nodes for adaptive routines.
- **Web Interface:** A GUI for executing and monitoring calibrations.
- **QUAM Integration:** Leverages the Quantum Abstract Machine (QUAM) for a persistent digital model of the quantum setup.

## Features

This package includes the following modules:

### `batchable_list`

Provides `BatchableList`, a list-like data structure that allows elements to be grouped into batches for potentially parallel processing or execution.

### `plot_utils`

Contains utility functions for generating plots related to quantum experiments. Includes functions for plotting simulated waveform samples (`plot_simulator_output`), tools for visualizing data on qubit grids (`QubitGrid`, `QubitPairGrid`), and helpers for plotting spectra (`plot_spectrum`) and active reset attempt distributions (`plot_active_reset_attempts`).

### `power_tools`

Functions to precisely set and retrieve the output power (in dBm) for specific operations on `MWChannel` and `IQChannel` components defined in QUAM. Handles adjustments to full-scale power or gain/amplitude to achieve target power levels.

### `qua_datasets`

Offers functions for processing and manipulating `xarray.Dataset` objects commonly resulting from QUA experiments.
Includes utilities for:

- Converting raw I/Q data to Volts (`convert_IQ_to_V`).
- Calculating and adding amplitude (`IQ_abs`) and phase (`phase`) data variables (`add_amplitude_and_phase`).
- Applying transformations like modulus, angle calculation (with optional unwrapping), slope subtraction, and phase unrotation (`apply_modulus`, `apply_angle`, `subtract_slope`, `unrotate_phase`).
- Generating histograms for integer data (`integer_histogram`).
- Converting datasets to long-format pandas DataFrames (`to_long_dataframe`).

### `save_utils`

Provides `fetch_results_as_xarray` to simplify fetching multiple result handles from a QM job into a structured `xarray.Dataset`.

### `trackable_object`

Introduces the `TrackableObject` class and the `tracked_updates` context manager. Allows temporary modification of object attributes (including nested objects and dictionaries) while keeping track of original values. Supports automatic or manual reverting of changes, useful for temporarily altering configurations (like QUAM states) during calibration steps.

Note that this class may be deprecated in the near future in favor of something natively implemented in QUAlibrate/QUAM.

### `xarray_data_fetcher`

Provides the `XarrayDataFetcher` class, a powerful tool for iteratively fetching data from a running or completed QM job. Automatically structures the fetched data into an `xarray.Dataset`, using provided coordinate axes. Handles different data shapes, including cases with a leading "qubit" dimension. Supports live data acquisition during job execution via iteration.

**Example:**

```python
import xarray as xr
import numpy as np
from qualang_tools.results import XarrayDataFetcher, progress_counter

axes = {
    "qubits": xr.DataArray(["q0", "q1"]),
    "num_pi_pulses": xr.DataArray([1, 2, 3]),
    "amplitudes": xr.DataArray([0.1, 0.2, 0.3]),
}

with program as prog:
    # QUA program with n_avg averaging iterations

qmm = QuantumMachinesManager()
qm = qmm.open_qm(config)
job = qm.execute(prog)

data_fetcher = XarrayDataFetcher(job, axes)
for dataset in data_fetcher:
    progress_counter(data_fetcher["n"], n_avg, start_time=data_fetcher.t_start)
```

## Installation

This package is typically installed as a dependency of the `qua-libs` superconducting calibrations. If you need to install it separately:

```bash
pip install git+https://github.com/qua-platform/qualibration-libs.git
```

## Related Packages

- [**QUAlibrate**](https://github.com/qua-platform/qualibrate): The main quantum calibration platform.
- [**QUAM**](https://github.com/qua-platform/quam): The Quantum Abstract Machine (QUAM) library for representing quantum hardware.
- [**quam-builder**](https://github.com/qua-platform/quam-builder): Tools for building QUAM configurations.
- [**qua-libs**](https://github.com/qua-platform/qua-libs): A library providing calibration scripts. The folder `Quantum-Control-Applications-QUAM` contains the calibration scripts using the QUAlibrate framework.

## License

`qualibration-libs` is licensed under the BSD 3-Clause License. See the LICENSE file for details.
