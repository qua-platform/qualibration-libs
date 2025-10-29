from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, Any, Callable, Union
from functools import wraps


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


def with_palette(palette: Union[str, Tuple[str, ...], list[str]]) -> Callable:
    """Decorator to temporarily set a color palette for plotting functions.
    
    This decorator temporarily changes the global color palette during the execution
    of a plotting function and restores the original palette afterward.
    
    Args:
        palette: Color palette to use. Can be:
            - A string name of a predefined palette (e.g., 'viridis', 'plasma', 'tab10')
            - A tuple/list of color strings (e.g., ('#ff0000', '#00ff00', '#0000ff'))
            - A list of color strings
    
    Examples:
        >>> @with_palette('viridis')
        ... def plot_data(data):
        ...     return QualibrationFigure.plot(data, x='x', data_var='y')
        
        >>> @with_palette(['#ff0000', '#00ff00', '#0000ff'])
        ... def plot_custom(data):
        ...     return QualibrationFigure.plot(data, x='x', data_var='y')
        
        >>> @with_palette(('red', 'green', 'blue'))
        ... def plot_named_colors(data):
        ...     return QualibrationFigure.plot(data, x='x', data_var='y')
    
    Note:
        The decorator preserves the original palette state, so nested calls with
        different palettes work correctly.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            global CURRENT_PALETTE
            # Store the current palette
            original_palette = CURRENT_PALETTE
            
            try:
                # Set the new palette
                if isinstance(palette, str):
                    # Handle predefined palette names
                    palette_map = {
                        'viridis': ('#440154', '#482777', '#3f4a8a', '#31678e', '#26838f', '#1f9d8a', '#6cce5a', '#b6de2b', '#fee825'),
                        'plasma': ('#0d0887', '#46039f', '#7201a8', '#9c179e', '#bd3786', '#d8576b', '#ed7953', '#fb9f3a', '#f0f921'),
                        'tab10': ('#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'),
                        'tab20': ('#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5', '#8c564b', '#c49c94', '#e377c2', '#f7b6d3', '#7f7f7f', '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5'),
                        'set1': ('#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf', '#999999'),
                        'set2': ('#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f', '#e5c494', '#b3b3b3'),
                        'set3': ('#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3', '#fdb462', '#b3de69', '#fccde5', '#d9d9d9', '#bc80bd', '#ccebc5', '#ffed6f'),
                        'pastel1': ('#fbb4ae', '#b3cde3', '#ccebc5', '#decbe4', '#fed9a6', '#ffffcc', '#e5d8bd', '#fddaec', '#f2f2f2'),
                        'pastel2': ('#b3e2cd', '#fdcdac', '#cbd5e8', '#f4cae4', '#e6f5c9', '#fff2ae', '#f1e2cc', '#cccccc'),
                        'dark2': ('#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02', '#a6761d', '#666666'),
                        'paired': ('#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928'),
                        'accent': ('#7fc97f', '#beaed4', '#fdc086', '#ffff99', '#386cb0', '#f0027f', '#bf5b17', '#666666'),
                        'spectral': ('#9e0142', '#d53e4f', '#f46d43', '#fdae61', '#fee08b', '#ffffbf', '#e6f598', '#abdda4', '#66c2a5', '#3288bd', '#5e4fa2'),
                        'coolwarm': ('#3b4cc0', '#6884d1', '#8fa3e0', '#b5c2ed', '#d9e0f7', '#f0f2fa', '#f7f8fc', '#fefefe', '#fef7f7', '#fce8e8', '#f5d0d0', '#e8b8b8', '#d6a0a0', '#c08888', '#a87070', '#905858', '#784040', '#602828', '#481010', '#300000'),
                        'rdylbu': ('#a50026', '#d73027', '#f46d43', '#fdae61', '#fee090', '#ffffbf', '#e0f3f8', '#abd9e9', '#74add1', '#4575b4', '#313695'),
                        'rdylgn': ('#a50026', '#d73027', '#f46d43', '#fdae61', '#fee08b', '#ffffbf', '#d9ef8b', '#a6d96a', '#66bd63', '#1a9641', '#006837'),
                        'rdbu': ('#67001f', '#b2182b', '#d6604d', '#f4a582', '#fddbc7', '#f7f7f7', '#d1e5f0', '#92c5de', '#4393c3', '#2166ac', '#053061'),
                        'piyg': ('#8e0152', '#c51b7d', '#de77ae', '#f1b6da', '#fde0ef', '#f7f7f7', '#e6f5d0', '#b8e186', '#7fbc41', '#4d9221', '#276419'),
                        'prgn': ('#40004b', '#762a83', '#9970ab', '#c2a5cf', '#e7d4e8', '#f7f7f7', '#d9f0d3', '#a6dba0', '#5aae61', '#1b7837', '#00441b'),
                        'brbg': ('#543005', '#8c510a', '#bf812d', '#dfc27d', '#f6e8c3', '#f5f5f5', '#c7eae5', '#80cdc1', '#35978f', '#01665e', '#003c30'),
                        'puor': ('#7f3b08', '#b35806', '#e08214', '#fdb863', '#fee0b6', '#f7f7f7', '#d8daeb', '#b2abd2', '#8073ac', '#542788', '#2d004b'),
                        'rdgy': ('#67001f', '#b2182b', '#d6604d', '#f4a582', '#fddbc7', '#ffffff', '#e0e0e0', '#bababa', '#878787', '#4d4d4d', '#1a1a1a'),
                        'rdylgn': ('#a50026', '#d73027', '#f46d43', '#fdae61', '#fee08b', '#ffffbf', '#d9ef8b', '#a6d96a', '#66bd63', '#1a9641', '#006837'),
                        'terrain': ('#00a2ed', '#00c4ed', '#00e6ed', '#00ffed', '#00ffcc', '#00ffaa', '#00ff88', '#00ff66', '#00ff44', '#00ff22', '#00ff00', '#22ff00', '#44ff00', '#66ff00', '#88ff00', '#aaff00', '#ccff00', '#eeff00', '#ffff00', '#ffcc00', '#ff9900', '#ff6600', '#ff3300', '#ff0000'),
                        'ocean': ('#000080', '#0000aa', '#0000d4', '#0000ff', '#1a1aff', '#3333ff', '#4d4dff', '#6666ff', '#8080ff', '#9999ff', '#b3b3ff', '#ccccff', '#e6e6ff', '#ffffff'),
                        'rainbow': ('#ff0000', '#ff8000', '#ffff00', '#80ff00', '#00ff00', '#00ff80', '#00ffff', '#0080ff', '#0000ff', '#8000ff', '#ff00ff', '#ff0080'),
                    }
                    if palette.lower() in palette_map:
                        CURRENT_PALETTE = palette_map[palette.lower()]
                    else:
                        raise ValueError(f"Unknown palette name: {palette}. Available palettes: {list(palette_map.keys())}")
                else:
                    # Handle tuple/list of colors
                    CURRENT_PALETTE = tuple(palette) if isinstance(palette, list) else palette
                
                # Call the function with the new palette
                return func(*args, **kwargs)
                
            finally:
                # Restore the original palette
                CURRENT_PALETTE = original_palette
                
        return wrapper
    return decorator