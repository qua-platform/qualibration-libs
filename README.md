# qualibration-libs

Utility library supporting calibration nodes and graphs for the QUAlibrate platform.

## Introduction

`qualibration-libs` provides a collection of essential tools and utility functions designed to support the calibration nodes and graphs found within the [qua-libs](https://github.com/qua-platform/qua-libs) repository, particularly those utilizing the QUAlibrate platform. These libraries facilitate data handling, processing, analysis, plotting, execution management, and interaction with quantum hardware configurations defined using QUAM.

While not a core component of the QUAlibrate platform itself, this library is a key dependency for running many of the example calibration routines provided in `qua-libs`.

## What is QUAlibrate?

QUAlibrate is an advanced, open-source calibration platform designed to streamline the calibration process for quantum computers, particularly those using OPX controllers. It enables users to create, manage, and execute calibration routines, abstracting hardware complexities and allowing focus on the quantum system itself. Key components include:

- **Calibration Nodes:** Reusable scripts for specific calibration tasks.
- **Calibration Graphs:** Directed acyclic graphs linking nodes for adaptive routines.
- **Web Interface:** A GUI for executing and monitoring calibrations.
- **QUAM Integration:** Leverages the Quantum Abstract Machine (QUAM) for a persistent digital model of the quantum setup.

## Features

This package includes the following modules:

- **`analysis`**: Provides tools for data analysis, including:
  - `fitting`: Functions like `fit_decay_exp`, `fit_oscillation_decay_exp`, `fit_oscillation`, `fit_resonator_purcell`, `fit_resonator` for curve fitting common experimental results using models defined in `analysis.models`.
  - `feature_detection`: Functions like `peaks_dips` to find peaks/dips in data and `extract_dominant_frequencies` using FFT.
  - `models`: Defines physical models (e.g., `lorentzian_peak`, `oscillation`, `decay_exp`, `S21_abs`, `S21_single`) used for fitting.
- **`config`**: Contains setup scripts:
  - `setup`: Includes `create_qualibrate_config` for interactive configuration of the Qualibration environment.
- **`core`**: Includes fundamental utilities:
  - `batchable_list`: Provides `BatchableList`, a list-like data structure that allows elements to be grouped into batches for potentially parallel processing or execution.
  - `trackable_object`: Introduces the `TrackableObject` class and the `tracked_updates` context manager. Allows temporary modification of object attributes (including nested objects and dictionaries) while keeping track of original values. Supports automatic or manual reverting of changes, useful for temporarily altering configurations (like QUAM states) during calibration steps. _(Note: May be deprecated in the future)._
- **`data`**: Handles data fetching and processing:
  - `fetcher`: Provides `fetch_results_as_xarray` to simplify fetching multiple result handles from a QM job into a structured `xarray.Dataset`. Includes the `XarrayDataFetcher` class for iteratively fetching data from QM jobs and structuring it into an `xarray.Dataset` with coordinate axes, supporting live updates. Also contains `fetch_dataset` for combined fetching and processing based on node parameters and sweep axes.
  - `processing`: Offers functions for processing `xarray.Dataset` objects commonly resulting from QUA experiments. Includes utilities for converting raw I/Q data to Volts (`convert_IQ_to_V`), calculating and adding amplitude (`IQ_abs`) and phase (`phase`) data variables (`add_amplitude_and_phase`), applying transformations like modulus (`apply_modulus`), angle calculation (`apply_angle`), slope subtraction (`subtract_slope`), and phase unrotation (`unrotate_phase`), generating histograms (`integer_histogram`), and converting datasets to long-format pandas DataFrames (`to_long_dataframe`).
- **`execute`**: Manages the execution flow of experiments:
  - `progress`: Provides `print_progress_bar` for displaying live progress during job execution using `qualang_tools.results.progress_counter`.
  - `simulate`: Contains `simulate_and_plot` for simulating QUA programs, plotting analog samples, and optionally generating interactive waveform reports using `qm.waveform_report`.
- **`hardware`**: Interfaces with hardware components:
  - `power`: Functions to precisely set and retrieve the output power (in dBm) for specific operations on `MWChannel` (`set_output_power_mw_channel`, `get_output_power_mw_channel`) and `IQChannel` (`set_output_power_iq_channel`, `get_output_power_iq_channel`) components defined in QUAM. Handles adjustments to full-scale power or gain/amplitude to achieve target power levels using helpers like `calculate_voltage_scaling_factor`.
- **`parameters`**: Defines Pydantic parameter structures used in Qualibration nodes:
  - `common`: Includes `CommonNodeParameters` for simulation settings, timeouts, loading historical data, etc.
  - `experiment`: Defines `QubitsExperimentNodeParameters` (qubit selection, multiplexing, state discrimination, reset types) and helper functions like `get_qubits` and `make_batchable_list_from_multiplexed` to manage qubit lists based on parameters.
  - `sweep`: Provides `get_idle_times_in_clock_cycles` for generating linear or logarithmic sweep arrays (e.g., for Ramsey/T1 experiments).
- **`plotting`**: Offers utilities for visualizing experiment results:
  - `grids`: Contains `QubitGrid` and `QubitPairGrid` classes for creating grid-based plots reflecting the physical layout of qubits or qubit pairs, along with the `grid_iter` helper.
  - `standard_plots`: Includes helpers for plotting signal spectra (`plot_spectrum`) and active reset attempt distributions (`plot_active_reset_attempts`).

## Installation

This package is typically installed as a dependency when setting up the `qua-libs` superconducting calibrations. If you need to install it separately:

```bash
pip install git+https://github.com/qua-platform/qualibration-libs.git
```

## Related Packages

- **QUAlibrate:** The main quantum calibration platform. [Link: [https://github.com/qua-platform/qualibrate](https://github.com/qua-platform/qualibrate)]
- **QUAM:** The Quantum Abstract Machine (QUAM) library for representing quantum hardware. [Link: placeholder_quam_url]
- **quam-builder:** Tools for building QUAM configurations. [Link: placeholder_quam_builder_url]
- **qua-libs:** A library providing calibration scripts and examples. The folder `Quantum-Control-Applications-QUAM` contains calibration scripts using the QUAlibrate framework, which depend on `qualibration-libs`. [Link: https://github.com/qua-platform/qua-libs]

## License

`qualibration-libs` is licensed under the BSD 3-Clause License. See the LICENSE file for details.
