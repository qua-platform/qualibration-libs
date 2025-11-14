# qualibration-libs

Utility library supporting calibration nodes and graphs for the QUAlibration graphs platform.

## Introduction

`qualibration-libs` provides a collection of essential tools and utility functions designed to support the calibration nodes and graphs found within the [qualibration-graphs](https://github.com/qua-platform/qua-libs/tree/main/qualibration_graphs) repository.
These libraries facilitate data handling, processing, analysis, plotting, execution management, and interaction with quantum hardware configurations defined using QUAM.

While not a core component of the QUAlibrate platform itself, this library is a key dependency for running many of the example calibration routines provided in `qualibration_graphs`.

## What is qualibration-graphs?

QUAlibration-graphs provides a comprehensive library for calibrating qubits using the Quantum Orchestration Platform (QOP), QUAM, and QUAlibrate.
It includes configurable experiment nodes, analysis routines, and tools for managing the quantum system state ([QUAM](https://qua-platform.github.io/quam/)).

This library is built upon [QUAlibrate](https://qua-platform.github.io/qualibrate/), an advanced, open-source software framework designed specifically for the automated calibration of Quantum Processing Units (QPUs).
QUAlibrate provides tools to create, manage, and execute calibration routines efficiently.
The configurable experiment nodes, analysis routines, and state management tools included here are designed to integrate seamlessly with the QUAlibrate ecosystem.

- **Calibration Nodes:** Reusable scripts for specific calibration tasks.
- **Calibration Graphs:** Directed acyclic graphs linking nodes for adaptive routines.
- **Web Interface:** A GUI for executing and monitoring calibrations.
- **QUAM Integration:** Leverages the Quantum Abstract Machine (QUAM) for a persistent digital model of the quantum setup.

## Features

This package includes the following modules:

- **`analysis`**: Provides tools for data analysis, including:
  - `fitting`: Functions like `fit_decay_exp`, `fit_oscillation_decay_exp`, `fit_resonator` for curve fitting common experimental results using models defined in `analysis.models`.
  - `feature_detection`: Functions like `peaks_dips` to find peaks/dips in data and `extract_dominant_frequencies` using FFT used as helpers in the fitting routines.
  - `models`: Defines physical models (e.g., `lorentzian_peak`, `oscillation`, `decay_exp`, `S21_abs`, `S21_single`) used for fitting.
- **`config`**: Contains functions to setup the QUAlibration-graphs environment:
  - `setup_qualibrate`: Framework to set up the QUAlibrate config interactively.
- **`core`**: Includes fundamental utilities used at the core of the calibration nodes:
  - `batchable_list`: Provides `BatchableList`, a list-like data structure that allows elements to be grouped into batches for potentially parallel processing or execution.
  - `trackable_object`: Introduces the `TrackableObject` class and the `tracked_updates` context manager. Allows temporary modification of object attributes (including nested objects and dictionaries) while keeping track of original values. Supports automatic or manual reverting of changes, useful for temporarily altering configurations (like QUAM states) during calibration steps. _(Note: May be deprecated in the future)._
- **`data`**: Handles data fetching and processing:
  - `fetcher`: Provides `fetch_results_as_xarray` to simplify fetching multiple result handles from a QM job into a structured `xarray.Dataset`. Includes the `XarrayDataFetcher` class for iteratively fetching data from QM jobs and structuring it into an `xarray.Dataset` with coordinate axes, supporting live updates. Details about the `XarrayDataFetcher` can be found in the [data folder](./qualibration_libs/data/README.md).
  - `processing`: Offers functions for processing `xarray.Dataset` objects commonly resulting from QUA experiments. Includes utilities for converting raw I/Q data to Volts (`convert_IQ_to_V`), calculating and adding amplitude (`IQ_abs`) and phase (`phase`) data variables (`add_amplitude_and_phase`), and so on.
- **`runtime`**: Set of functions called during the runtime (execution/simulation) of an experiment:
  - `simulate`: Contains `simulate_and_plot` for simulating QUA programs, plotting analog samples, and optionally generating interactive waveform reports using `qm.waveform_report`.
- **`parameters`**: Defines Pydantic parameter structures used in Qualibration nodes:
  - `common`: Includes `CommonNodeParameters` for simulation settings, timeouts, loading historical data, etc.
  - `experiment`: Defines `QubitsExperimentNodeParameters` (qubit selection, multiplexing, state discrimination, reset types) and helper functions like `get_qubits` and `make_batchable_list_from_multiplexed` to manage qubit lists based on parameters.
  - `sweep`: Provides the `IdleTimeNodeParameters` for sweeping the idle time in Ramsey or T1 experiments for instance, as well as `get_idle_times_in_clock_cycles` for generating linear or logarithmic sweep arrays (e.g., for Ramsey/T1 experiments).
- **`plotting`**: Offers utilities for visualizing experiment results:
  - `grids`: Contains `QubitGrid` and `QubitPairGrid` classes for creating grid-based plots reflecting the physical layout of qubits or qubit pairs, along with the `grid_iter` helper.

## Installation

This package is typically installed as a dependency when setting up the `qua-libs` superconducting calibrations. If you need to install it separately:

```bash
pip install git+https://github.com/qua-platform/qualibration-libs.git
```

## Development

### Running Tests

This project uses pytest for testing. To run the test suite:

```bash
# Install test dependencies (if not already installed)
pip install pytest

# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest qualibration_libs/tests/test_exceptions.py

# Run tests from a specific directory
pytest qualibration_libs/tests/
```

### Code Quality

The library uses enhanced exception handling to provide helpful error messages. When exceptions occur, error messages will list available options (up to 10 items) to help diagnose issues quickly. All exceptions preserve the original exception chain using `raise ... from e` to maintain full stack traces for debugging.

## Related Packages

- **QUAlibrate:** The main quantum calibration platform. [Link: [https://github.com/qua-platform/qualibrate](https://github.com/qua-platform/qualibrate)]
- **QUAM:** The Quantum Abstract Machine (QUAM) library for representing quantum hardware. [Link: [https://github.com/qua-platform/quam](https://github.com/qua-platform/quam)]
- **quam-builder:** Tools for building QUAM configurations. [Link: [https://github.com/qua-platform/quam-builder](https://github.com/qua-platform/quam-builder)]
- **qua-libs:** A library providing calibration scripts and examples. The folder `qualibration_graph` contains calibration scripts using the QUAlibrate framework, which depend on `qualibration-libs`. [Link: [https://github.com/qua-platform/qua-libs/qualibration_graphs](https://github.com/qua-platform/qua-libs/qualibration_graphs)]

## License

`qualibration-libs` is licensed under the BSD 3-Clause License. See the LICENSE file for details.
