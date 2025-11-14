"""Tests for exception handling in analysis/feature_detection.py."""

import pytest
import numpy as np
import xarray as xr
from qualibration_libs.analysis.feature_detection import peaks_dips


class TestPeaksDips:
    """Test exception handling in peaks_dips function."""

    def test_valid_dimension_works(self):
        # Create a simple peak signal
        x = np.linspace(0, 10, 100)
        y = np.exp(-((x - 5)**2) / 2) + 0.1 * np.random.randn(100)
        da = xr.DataArray(y, dims=["x"], coords={"x": x})

        result = peaks_dips(da, dim="x")
        assert "amplitude" in result
        assert "position" in result
        assert "width" in result
        assert "base_line" in result

    def test_invalid_dimension_fails(self):
        # Create a simple peak signal
        x = np.linspace(0, 10, 100)
        y = np.exp(-((x - 5)**2) / 2) + 0.1 * np.random.randn(100)
        da = xr.DataArray(y, dims=["x"], coords={"x": x})

        with pytest.raises(KeyError) as exc_info:
            peaks_dips(da, dim="y")

        error_msg = str(exc_info.value)
        assert "Coordinate 'y' not found in DataArray." in error_msg
        assert "Available coordinates: 'x'" in error_msg
