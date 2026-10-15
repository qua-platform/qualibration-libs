"""xarray-native fits built on lmfit models.

Each public ``fit_*`` takes a DataArray and the dimension(s) to fit along, runs one lmfit fit per
remaining coordinate (e.g. per qubit) and returns an ``xr.Dataset`` with, for every fitted parameter
``p``: ``p`` and ``p_err`` (its standard error), plus ``redchi``, ``rsquared`` and ``best_fit``.
A failed fit gives NaN for that slice instead of raising.

To add a fit, write a small ``_<name>_1d`` function that returns the lmfit ``ModelResult`` for one
slice and a public wrapper that calls ``_fit_xr`` with it, as the fits below do.
"""

from collections.abc import Callable, Sequence

import lmfit
import numpy as np
import xarray as xr
from lmfit.model import ModelResult
from lmfit.models import ConstantModel, Gaussian2dModel, LinearModel, LorentzianModel

__all__ = ["fit_gaussian2d", "fit_line", "fit_lorentzian"]


def _fix_values(params: lmfit.Parameters, fixed_values: dict[str, float] | None) -> None:
    """Fix parameters to the given values so the fit cannot vary them."""
    for name, value in (fixed_values or {}).items():
        if value is not None and name in params:
            params[name].set(value=value, vary=False)


def _fit_xr(
    fit_1d: Callable[..., ModelResult],
    da: xr.DataArray,
    dims: Sequence[str],
    coords: Sequence[xr.DataArray],
    param_names: Sequence[str],
    weights: xr.DataArray | None = None,
    fixed_values: dict[str, float | xr.DataArray] | None = None,
    suffix: str = "",
    **fit_kwargs,
) -> xr.Dataset:
    """Run ``fit_1d(*coords, data, weights, fixed_values, **fit_kwargs)`` on every slice of ``da``.

    ``fit_1d`` receives plain numpy arrays (``weights`` may be None, ``fixed_values`` is a dict of
    floats) and returns an lmfit ``ModelResult``. ``coords`` holds the x axis of each fitted dim and
    may carry extra dims (e.g. a per-qubit frequency axis). ``weights`` must have the dims of
    ``da``; each ``fixed_values`` entry is a scalar or a DataArray without the fitted dims.
    ``suffix`` is appended to every output name (e.g. ``"_g"`` to fit two states side by side).
    """
    dims = list(dims)
    fixed_values = {k: v for k, v in (fixed_values or {}).items() if v is not None}
    n_coords = len(coords)

    inputs = [*coords, da]
    core_dims = [*([d] for d in dims), dims]
    if weights is not None:
        inputs.append(weights)
        core_dims.append(dims)
    inputs.extend(fixed_values.values())
    core_dims.extend([] for _ in fixed_values)

    n_out = 2 * len(param_names) + 2
    nan = float("nan")

    def run(*arrays):
        xs, y, rest = arrays[:n_coords], arrays[n_coords], list(arrays[n_coords + 1 :])
        w = rest.pop(0) if weights is not None else None
        fixed = dict(zip(fixed_values, rest))
        try:
            result = fit_1d(*xs, y, w, fixed, **fit_kwargs)
            values = []
            for name in param_names:
                stderr = result.params[name].stderr
                values += [
                    float(result.params[name].value),
                    nan if stderr is None else float(stderr),
                ]
            return (*values, float(result.redchi), float(result.rsquared), result.best_fit)
        except Exception:  # noqa: BLE001 -- any fit failure degrades to NaN, not a crash
            return (*([nan] * n_out), np.full(np.shape(y), nan))

    outputs = xr.apply_ufunc(
        run,
        *inputs,
        input_core_dims=core_dims,
        output_core_dims=[[]] * n_out + [dims],
        vectorize=True,
        output_dtypes=[float] * (n_out + 1),
    )

    names = [n for p in param_names for n in (p, f"{p}_err")] + ["redchi", "rsquared", "best_fit"]
    return xr.Dataset({f"{n}{suffix}": out for n, out in zip(names, outputs)})


def _line_1d(x, y, weights, fixed, fix_intercept_value=None) -> ModelResult:
    model = LinearModel()
    params = model.guess(y, x=x)
    if fix_intercept_value is not None:
        params["intercept"].set(value=fix_intercept_value, vary=False)
    return model.fit(y, params, x=x, weights=weights)


def fit_line(
    da: xr.DataArray,
    dim: str,
    x: xr.DataArray | None = None,
    fix_intercept_value: float | None = None,
    weights: xr.DataArray | None = None,
    suffix: str = "",
) -> xr.Dataset:
    """Linear fit ``y = slope * x + intercept`` along ``dim``.

    ``x`` defaults to the ``dim`` coordinate. ``weights`` is 1/sigma per point; a zero weight
    drops that point. Outputs: ``slope``, ``intercept`` (+ ``_err``).
    """
    x = da[dim] if x is None else x
    return _fit_xr(
        _line_1d,
        da,
        [dim],
        [x],
        ["slope", "intercept"],
        weights=weights,
        suffix=suffix,
        fix_intercept_value=fix_intercept_value,
    )


def _lorentzian_1d(x, y, weights, fixed, negative=False) -> ModelResult:
    lorentzian, offset = LorentzianModel(), ConstantModel()
    offset_guess = np.max(y) if negative else np.min(y)
    params = lorentzian.guess(y, x=x, negative=negative) + offset.make_params(c=offset_guess)
    _fix_values(params, fixed)
    return (lorentzian + offset).fit(y, params, x=x, weights=weights)


def fit_lorentzian(
    da: xr.DataArray,
    dim: str,
    x: xr.DataArray | None = None,
    negative: bool = False,
    weights: xr.DataArray | None = None,
    fixed_values: dict[str, float | xr.DataArray] | None = None,
    suffix: str = "",
) -> xr.Dataset:
    """Lorentzian plus constant offset along ``dim``; ``negative=True`` fits a dip.

    ``x`` defaults to the ``dim`` coordinate. ``weights`` is 1/sigma per point, ``fixed_values``
    holds parameters to keep fixed, e.g. ``{"center": da_center}``.
    Outputs: ``center``, ``sigma``, ``amplitude``, ``c`` (+ ``_err``).
    """
    x = da[dim] if x is None else x
    return _fit_xr(
        _lorentzian_1d,
        da,
        [dim],
        [x],
        ["center", "sigma", "amplitude", "c"],
        weights=weights,
        fixed_values=fixed_values,
        suffix=suffix,
        negative=negative,
    )


def _gaussian2d_1d(x, y, z, weights, fixed, negative=False) -> ModelResult:
    gaussian, offset = Gaussian2dModel(), ConstantModel()
    X, Y = np.meshgrid(x, y, indexing="ij")
    params = gaussian.guess(z.flatten(), X.flatten(), Y.flatten(), negative=negative)
    params += offset.make_params()
    for name, axis in (("centerx", x), ("centery", y)):  # keep the centers inside the axes
        params[name].min, params[name].max = np.min(axis), np.max(axis)
    _fix_values(params, fixed)
    return (gaussian + offset).fit(data=z, x=X, y=Y, params=params, weights=weights)


def fit_gaussian2d(
    da: xr.DataArray,
    dims: tuple[str, str],
    x: xr.DataArray | None = None,
    y: xr.DataArray | None = None,
    negative: bool = False,
    weights: xr.DataArray | None = None,
    suffix: str = "",
) -> xr.Dataset:
    """Axis-aligned 2D Gaussian plus constant offset over ``dims = (dim_x, dim_y)``.

    The axes ``x`` and ``y`` default to the coordinates of ``dims``; pass DataArrays to use axes
    that differ per slice (e.g. histogram bin centers, with dims ``dim_x`` / ``dim_y`` plus the
    slice dims). ``weights`` has the dims of ``da`` (e.g. Poisson
    ``1/sqrt(counts)`` for histograms). ``best_fit`` has the dims of ``dims``.
    Outputs: ``centerx``, ``centery``, ``sigmax``, ``sigmay``, ``amplitude``, ``c`` (+ ``_err``).
    """
    return _fit_xr(
        _gaussian2d_1d,
        da,
        dims,
        [da[dims[0]] if x is None else x, da[dims[1]] if y is None else y],
        ["centerx", "centery", "sigmax", "sigmay", "amplitude", "c"],
        weights=weights,
        suffix=suffix,
        negative=negative,
    )
