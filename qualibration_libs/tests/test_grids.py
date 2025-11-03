"""Tests for exception handling in plotting/grids.py."""

import pytest
import xarray as xr
import numpy as np
from qualibration_libs.plotting.grids import QubitGrid


class TestQubitGridInit:
    """Test exception handling in QubitGrid.__init__."""

    def test_valid_qubit_format(self):
        # Create a dataset with properly formatted qubit names
        ds = xr.Dataset(
            coords={"qubit": ["q-0,0", "q-1,0", "q-0,1"]}
        )

        grid = QubitGrid(ds, size=3)
        assert grid.fig is not None
        assert len(grid.axes) > 0

    def test_invalid_qubit_format_raises_valueerror(self):
        # Create a dataset with qubit names that contain non-numeric characters after cleaning
        # This will cause int() to raise ValueError
        ds = xr.Dataset(
            coords={"qubit": ["q-abc,def", "q-ghi,jkl"]}
        )

        with pytest.raises(ValueError) as exc_info:
            QubitGrid(ds, size=3)

        error_msg = str(exc_info.value)
        assert "Error parsing qubit grid locations" in error_msg
        assert "Expected format like 'q-1,2'" in error_msg
        assert exc_info.value.__cause__ is not None

    def test_valid_grid_names_parameter(self):
        # Test with explicit grid_names parameter
        ds = xr.Dataset(
            coords={"qubit": ["q1", "q2"]}
        )

        grid = QubitGrid(ds, grid_names=["0,0", "1,0"], size=3)
        assert grid.fig is not None
        assert len(grid.axes) > 0

    def test_single_qubit_dataset(self):
        # Test with a single qubit
        ds = xr.Dataset(
            coords={"qubit": ["q-0,0"]}
        )

        grid = QubitGrid(ds, size=3)
        assert grid.fig is not None
        assert len(grid.axes) > 0
