# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)

## [Unreleased]

### Added
- New dependencies: lmfit, matplotlib, numpy, qm, quam-builder, qualibrate, scipy, xarray

### Changed
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

[Unreleased]: https://github.com/qua-platform/qualibration-libs/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/qua-platform/qualibration-libs/releases/tag/v0.2.0
[0.1.0]: https://github.com/qua-platform/qualibration-libs/releases/tag/v0.1.0
