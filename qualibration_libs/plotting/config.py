from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, Any


@dataclass
class PlotTheme:
    font_size: int = 14
    title_size: int = 16
    tick_label_size: int = 12
    marker_size: int = 6
    line_width: int = 2
    show_grid: bool = True
    grid_opacity: float = 0.25
    residuals_height_ratio: float = 0.35
    figure_bg: str = "white"
    paper_bg: str = "white"
    colorway: Tuple[str, ...] = (
        "#4c78a8",
        "#f58518",
        "#e45756",
        "#72b7b2",
        "#54a24b",
        "#eeca3b",
        "#b279a2",
        "#ff9da6",
        "#9d755d",
        "#bab0ac",
    )


@dataclass
class RcParams:
    values: Dict[str, Any] = field(default_factory=dict)


CURRENT_THEME = PlotTheme()
CURRENT_RC = RcParams()
CURRENT_PALETTE: Optional[Tuple[str, ...]] = None


def apply_theme_to_layout(layout: Any) -> None:
    t = CURRENT_THEME
    if hasattr(layout, "update"):
        layout.update(
            template="plotly_white",
            paper_bgcolor=t.paper_bg,
            plot_bgcolor=t.figure_bg,
            font={"size": t.font_size},
        )
        if CURRENT_PALETTE:
            layout.update(colorway=list(CURRENT_PALETTE))
    else:
        layout.setdefault("template", "plotly_white")
        layout.setdefault("paper_bgcolor", t.paper_bg)
        layout.setdefault("plot_bgcolor", t.figure_bg)
        layout.setdefault("font", {"size": t.font_size})
        if CURRENT_PALETTE:
            layout.setdefault("colorway", list(CURRENT_PALETTE))