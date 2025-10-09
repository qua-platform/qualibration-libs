from __future__ import annotations

import xarray as xr

from .figure import QualibrationFigure


@xr.register_dataset_accessor("qplot")
@xr.register_dataarray_accessor("qplot")
class XrQualPlotAccessor:
    def __init__(self, xrobj):
        self._obj = xrobj

    def plot(self, **kwargs) -> QualibrationFigure:
        return QualibrationFigure.plot(self._obj, **kwargs)
