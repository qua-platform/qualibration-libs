"""Tests for the xarray-native lmfit fits in analysis/lmfit_fits.py."""

import numpy as np
import pytest
import xarray as xr

from qualibration_libs.analysis import fit_gaussian2d, fit_line, fit_lorentzian

QUBITS = ["q1", "q2"]


def _lorentzian(x, center, sigma, amplitude, c):
    return amplitude / np.pi * sigma / ((x - center) ** 2 + sigma**2) + c


def _lorentzian_da(centers, sigma=2.0, amplitude=40.0, c=1.0, noise=0.0, seed=0):
    x = np.linspace(-20, 20, 201)
    rng = np.random.default_rng(seed)
    data = np.stack([_lorentzian(x, ctr, sigma, amplitude, c) for ctr in centers])
    data = data + noise * rng.standard_normal(data.shape)
    return xr.DataArray(data, dims=["qubit", "x"], coords={"qubit": QUBITS[: len(centers)], "x": x})


class TestFitLine:
    def test_recovers_slope_and_intercept_per_qubit(self):
        x = np.linspace(0, 10, 30)
        da = xr.DataArray(
            np.stack([2.0 * x + 1.0, -3.0 * x + 5.0]),
            dims=["qubit", "x"],
            coords={"qubit": QUBITS, "x": x},
        )
        ds = fit_line(da, "x")
        np.testing.assert_allclose(ds.slope, [2.0, -3.0], rtol=1e-6)
        np.testing.assert_allclose(ds.intercept, [1.0, 5.0], rtol=1e-6)
        assert ds.best_fit.dims == ("qubit", "x")

    def test_fix_intercept_value(self):
        x = np.linspace(0, 10, 30)
        da = xr.DataArray(2.0 * x, dims=["x"], coords={"x": x})
        ds = fit_line(da, "x", fix_intercept_value=0.0)
        assert float(ds.intercept) == 0.0
        assert float(ds.intercept_err) == 0.0

    def test_custom_x_with_extra_dim(self):
        # x axis that differs per qubit, as for absolute frequency vs detuning
        detuning = np.linspace(0, 10, 30)
        offsets = xr.DataArray([100.0, 200.0], dims=["qubit"], coords={"qubit": QUBITS})
        full = offsets + xr.DataArray(detuning, dims=["x"], coords={"x": detuning})
        da = 4.0 * full
        ds = fit_line(da, "x", x=full)
        np.testing.assert_allclose(ds.slope, [4.0, 4.0], rtol=1e-6)


class TestFitLorentzian:
    def test_recovers_parameters_per_qubit(self):
        ds = fit_lorentzian(_lorentzian_da([-3.0, 5.0], noise=0.01), "x")
        np.testing.assert_allclose(ds.center, [-3.0, 5.0], atol=0.05)
        np.testing.assert_allclose(np.abs(ds.sigma), 2.0, atol=0.05)
        assert set(ds.data_vars) >= {"center", "center_err", "redchi", "rsquared", "best_fit"}

    def test_negative_dip(self):
        da = -_lorentzian_da([1.0], noise=0.01) + 50.0
        ds = fit_lorentzian(da, "x", negative=True)
        assert abs(float(ds.center.item()) - 1.0) < 0.05

    def test_fixed_center_per_qubit_and_weights(self):
        da = _lorentzian_da([-3.0, 5.0], noise=0.01)
        centers = xr.DataArray([-3.0, 5.0], dims=["qubit"], coords={"qubit": QUBITS})
        weights = xr.ones_like(da)
        ds = fit_lorentzian(da, "x", weights=weights, fixed_values={"center": centers})
        np.testing.assert_array_equal(ds.center, [-3.0, 5.0])
        assert (ds.center_err == 0.0).all()
        assert np.isfinite(ds.sigma_err).all()

    def test_suffix(self):
        ds = fit_lorentzian(_lorentzian_da([0.0]), "x", suffix="_g")
        assert "center_g" in ds and "best_fit_g" in ds and "center" not in ds

    def test_failed_fit_gives_nan_not_exception(self):
        da = _lorentzian_da([0.0, 0.0])
        da[1] = np.nan
        ds = fit_lorentzian(da, "x")
        assert np.isfinite(ds.center.sel(qubit="q1"))
        assert np.isnan(ds.center.sel(qubit="q2"))
        assert ds.best_fit.sel(qubit="q2").isnull().all()


class TestFitGaussian2d:
    @staticmethod
    def _gaussian_da(cx, cy, sx, sy, amplitude=100.0, c=2.0):
        x, y = np.linspace(-5, 5, 31), np.linspace(-4, 6, 41)  # different lengths
        X, Y = np.meshgrid(x, y, indexing="ij")
        z = (
            amplitude
            / (2 * np.pi * sx * sy)
            * np.exp(-((X - cx) ** 2) / (2 * sx**2) - (Y - cy) ** 2 / (2 * sy**2))
            + c
        )
        return xr.DataArray(z, dims=["x", "y"], coords={"x": x, "y": y})

    def test_recovers_centers_and_widths(self):
        ds = fit_gaussian2d(self._gaussian_da(0.5, 1.0, 1.0, 1.5), ("x", "y"))
        assert abs(float(ds.centerx) - 0.5) < 1e-3
        assert abs(float(ds.centery) - 1.0) < 1e-3
        assert abs(abs(float(ds.sigmax)) - 1.0) < 1e-3
        assert abs(abs(float(ds.sigmay)) - 1.5) < 1e-3
        assert ds.best_fit.dims == ("x", "y")

    def test_weights_accepted(self):
        da = self._gaussian_da(0.0, 0.0, 1.0, 1.0)
        ds = fit_gaussian2d(da, ("x", "y"), weights=1.0 / np.sqrt(np.maximum(da, 1.0)))
        assert abs(float(ds.centerx)) < 1e-3
