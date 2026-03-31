# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)

## [Unreleased]

## [0.3.0] - 2026-03-31
### Added
- Added support for Python 3.13.
- 
### Fixed
- parameters: Use qualibrate.core.parameters instead of removed qualibrate.parameters (fixes ModuleNotFoundError with current qualibrate API).
- analysis/fitting - allows the fit to converge even if the exponential is inverted.
- analysis/fitting - fit oscillations in a more robust manner.  

### Changed
- Raise minimum qualibrate to 1.0.2 and require Python >=3.10 to align with qualibrate.
- data/fetcher: Switch result fetching from qm-qua dependent `qm_qua.QmJob.fetch_all` to qualang_tools dependent `qualang_tools.results.fetching_tool`.
- data/fetcher: Use `fetching_tool` to retrieve acquisition metadata (e.g., `is_processing()` and `get_start_times()`).
- data/fetcher: Apply `ignore_handles` filtering in `__init__` to reduce overhead in `retrieve_latest_data`.

## [0.2.1] - 2025-11-14
### Added
- Add TwoQubitExperimentNodeParameters class
- New dependencies: lmfit, matplotlib, numpy, qm, quam-builder, qualibrate, scipy, xarray
- Test suite (using pytest) with 39 tests covering exception handling and core functionality
- Github action that runs the test suite during pull request workflows

### Changed
- Enhanced exception messages throughout the library to provide more helpful context
- Updated BatchableList.repr method to show batch configuration
- Updated analysis modules : extracted qiskit-experiment guess.py into package.

### Removed
- Removed qiskit-experiment dependency

## [0.2.0] - 2025-08-06
### Changed
- Transfer the `power_tools` to quam-builder to remove its dependency to qualibration-libs.

## [0.1.0] - 2025-05-07
### Added
- First release for the Superconducting QUAlibration graph.

[Unreleased]: https://github.com/qua-platform/qualibration-libs/compare/v0.3.0...HEAD
[0.2.1]: https://github.com/qua-platform/qualibration-libs/releases/tag/v0.3.0
[0.2.1]: https://github.com/qua-platform/qualibration-libs/releases/tag/v0.2.1
[0.2.0]: https://github.com/qua-platform/qualibration-libs/releases/tag/v0.2.0
[0.1.0]: https://github.com/qua-platform/qualibration-libs/releases/tag/v0.1.0
