"""Tests for exception handling in data/processing.py."""

import pytest
import numpy as np
import xarray as xr
from qualibration_libs.data.processing import (
    add_amplitude_and_phase,
    subtract_slope,
)


class TestAddAmplitudeAndPhase:
    """Test exception handling in add_amplitude_and_phase function."""

    def test_missing_I_variable(self):
        """Test that missing 'I' variable raises helpful KeyError."""
        ds = xr.Dataset({"Q": xr.DataArray([1, 2, 3], dims=["time"])})

        with pytest.raises(KeyError) as exc_info:
            add_amplitude_and_phase(ds, dim="time")

        error_msg = str(exc_info.value)
        assert "'I' and/or 'Q' not found" in error_msg
        assert "Available variables: 'Q'" in error_msg
        assert exc_info.value.__cause__ is not None

    def test_both_I_Q_present_works(self):
        """Test that having both I and Q works normally."""
        ds = xr.Dataset({
            "I": xr.DataArray([1, 2, 3], dims=["time"]),
            "Q": xr.DataArray([4, 5, 6], dims=["time"])
        })

        result = add_amplitude_and_phase(ds, dim="time", unwrap_flag=False)
        assert "IQ_abs" in result
        assert "phase" in result


class TestSubtractSlope:
    """Test exception handling in subtract_slope function."""

    def test_invalid_dimension(self):
        """Test that invalid dimension raises helpful KeyError."""
        da = xr.DataArray([1, 2, 3, 4], dims=["time"])

        with pytest.raises(KeyError) as exc_info:
            subtract_slope(da, dim="invalid_dim")

        error_msg = str(exc_info.value)
        assert "Dimension 'invalid_dim' not found" in error_msg
        assert "Available dimensions: 'time'" in error_msg
        assert exc_info.value.__cause__ is not None

    def test_valid_dimension_works(self):
        """Test that valid dimension works normally."""
        da = xr.DataArray([1, 2, 3, 4], dims=["time"], coords={"time": [0, 1, 2, 3]})

        result = subtract_slope(da, dim="time")
        assert result is not None
        assert result.dims == ("time",)