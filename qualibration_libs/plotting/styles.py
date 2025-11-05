from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Tuple, Optional, Iterable

from .config import PlotTheme
from . import config as _config

_PALETTES: Dict[str, Tuple[str, ...]] = {
    "qualibrate": PlotTheme().colorway,
    "deep": (
        "#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b2", "#937860", "#da8bc3", "#8c8c8c", "#ccb974", "#64b5cd"),
    "muted": (
        "#4878d0", "#ee854a", "#6acc64", "#d65f5f", "#b47cc7", "#82c6e2", "#d5bb67", "#8c8c8c", "#ff9da6", "#9d755d"),
}


def set_theme(
    theme: Optional[PlotTheme] = None, *,
    palette: Optional[str | Iterable[str]] = None,
    rc: Optional[dict] = None
) -> None:
    if theme is not None:
        # Mutate existing theme object in-place so all modules holding a reference see updates
        for k, v in theme.__dict__.items():
            setattr(_config.CURRENT_THEME, k, v)
    if palette is not None:
        if isinstance(palette, str):
            _config.CURRENT_PALETTE = _PALETTES.get(palette, PlotTheme().colorway)
        else:
            _config.CURRENT_PALETTE = tuple(palette)
    if rc is not None:
        _config.CURRENT_RC.values.update(rc)


def set_palette(palette: str | Iterable[str]) -> None:
    if isinstance(palette, str):
        _config.CURRENT_PALETTE = _PALETTES.get(palette, PlotTheme().colorway)
    else:
        _config.CURRENT_PALETTE = tuple(palette)


@contextmanager
def theme_context(
    theme: Optional[PlotTheme] = None, *,
    palette: Optional[str | Iterable[str]] = None,
    rc: Optional[dict] = None
):
    # Snapshot current theme values to restore later
    orig_theme = PlotTheme(**_config.CURRENT_THEME.__dict__)
    orig_palette = _config.CURRENT_PALETTE
    orig_rc = dict(_config.CURRENT_RC.values)
    try:
        set_theme(theme, palette=palette, rc=rc)
        yield
    finally:
        # Restore theme values in-place
        for k, v in orig_theme.__dict__.items():
            setattr(_config.CURRENT_THEME, k, v)
        _config.CURRENT_PALETTE = orig_palette
        _config.CURRENT_RC.values.clear()
        _config.CURRENT_RC.values.update(orig_rc)
