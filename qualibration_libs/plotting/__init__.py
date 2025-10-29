from .grid import QubitGrid, grid_iter
from .figure import QualibrationFigure
from .api import set_theme, set_palette, theme_context
from .accessors import XrQualPlotAccessor
from .config import with_palette


def register_accessors() -> str:
    _ = XrQualPlotAccessor  # keep linter quiet; side-effect already applied
    return "qplot"


__all__ = [
    "QubitGrid",
    "grid_iter",
    "QualibrationFigure",
    "set_theme",
    "set_palette",
    "theme_context",
    "with_palette",
    "register_accessors",
]