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
        y = np.exp(-((x - 5) ** 2) / 2) + 0.1 * np.random.randn(100)
        da = xr.DataArray(y, dims=["x"], coords={"x": x})

        result = peaks_dips(da, dim="x")
        assert "amplitude" in result
        assert "position" in result
        assert "width" in result
        assert "base_line" in result

    def test_invalid_dimension_fails(self):
        # Create a simple peak signal
        x = np.linspace(0, 10, 100)
        y = np.exp(-((x - 5) ** 2) / 2) + 0.1 * np.random.randn(100)
        da = xr.DataArray(y, dims=["x"], coords={"x": x})

        with pytest.raises(KeyError) as exc_info:
            peaks_dips(da, dim="y")

        error_msg = str(exc_info.value)
        assert "Coordinate 'y' not found in DataArray." in error_msg
        assert "Available coordinates: 'x'" in error_msg

    def test_small_peak_found_alongside_large_peak_in_batch(self):
        """Regression test: in a multiplexed sweep (e.g. one entry per qubit), a
        clean but small-amplitude peak must still be found even when another entry
        in the same batch has a much larger amplitude.

        The noise/prominence threshold must be estimated per-slice along the extra
        dimension (e.g. per qubit), not pooled across the whole array - otherwise a
        loud entry's noise floor sets the threshold for a quiet entry's real,
        visible peak and it gets dropped.
        """
        rng = np.random.default_rng(0)
        x = np.linspace(0, 10, 200)

        # "loud" entry: amplitude ~1.0, noise std ~0.02
        y_loud = np.exp(-((x - 5) ** 2) / (2 * 0.3**2)) + 0.02 * rng.standard_normal(200)
        # "quiet" entry: amplitude ~0.005, but still ~10x its own local noise (std ~5e-4)
        y_quiet = 0.005 * np.exp(-((x - 3) ** 2) / (2 * 0.3**2)) + 0.0005 * rng.standard_normal(200)

        da = xr.DataArray(
            np.stack([y_loud, y_quiet]),
            dims=["batch", "x"],
            coords={"batch": ["loud", "quiet"], "x": x},
        )

        result = peaks_dips(da, dim="x")

        assert not np.isnan(result.position.sel(batch="loud").item())
        assert not np.isnan(result.position.sel(batch="quiet").item()), (
            "quiet entry's clearly visible peak was missed - noise threshold was "
            "likely pooled across the batch instead of computed per-slice"
        )
        assert result.position.sel(batch="quiet").item() == pytest.approx(3, abs=0.3)
