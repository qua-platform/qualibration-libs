from __future__ import annotations

from typing import Union, Mapping, Any, Iterable

import pandas as pd
import xarray as xr

DataLike = Union[xr.Dataset, xr.DataArray, pd.DataFrame, Mapping[str, Any]]
SeriesLike = Union[xr.DataArray, pd.Series, Iterable[float]]
