# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)

## [Unreleased]

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

[Unreleased]: https://github.com/qua-platform/qualibration-libs/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/qua-platform/qualibration-libs/releases/tag/v0.2.1
[0.2.0]: https://github.com/qua-platform/qualibration-libs/releases/tag/v0.2.0
[0.1.0]: https://github.com/qua-platform/qualibration-libs/releases/tag/v0.1.0
