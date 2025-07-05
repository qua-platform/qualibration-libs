"""
Centralized visual standards for quantum calibration plotting.

This module provides standardized colors, dimensions, styling, and formatting
constants to ensure consistent visual appearance across all quantum experiments.
Based on analysis of existing plotting implementations to maintain exact compatibility.
"""

from typing import Dict, Any

# ===== COLOR SCHEMES =====
class Colors:
    """Standard color palette for quantum calibration plots."""
    
    # Primary data colors
    RAW_DATA = "#1f77b4"        # Blue - used for all raw experimental data
    FIT_LINE = "#FF0000"        # Red - used for all fitted curves
    OPTIMAL_MARKER = "#FF00FF"  # Magenta - for optimal/sweet spot markers
    SECONDARY_FIT = "#800080"   # Purple - for secondary fit elements (flux minimum)
    
    # Heatmap colors
    HEATMAP_COLORSCALE = "Viridis"  # Standard colorscale for all heatmaps
    BINARY_COLORSCALE = "binary"    # For binary analysis heatmaps


class LineStyles:
    """Standard line styling for different plot elements."""
    
    # Raw data styling
    RAW_LINE_WIDTH = 1.0
    RAW_MARKER_SIZE = 4
    RAW_ALPHA = 0.5  # For Power Rabi plots
    
    # Fit line styling  
    FIT_LINE_WIDTH = 2.0
    FIT_LINE_DASH = "dash"
    FIT_LINE_STYLE = "--"  # Matplotlib equivalent
    
    # Overlay line styling
    OVERLAY_LINE_WIDTH = 2.5
    OVERLAY_LINE_DASH = "dash"
    OVERLAY_LINE_STYLE = "dashed"  # Matplotlib equivalent
    
    # Marker styling
    MARKER_SIZE = 15
    MARKER_SYMBOL = "x"


# ===== FIGURE DIMENSIONS =====
class FigureDimensions:
    """Standard figure dimensions and layout settings."""
    
    # Matplotlib dimensions (inches)
    MATPLOTLIB_WIDTH = 15
    MATPLOTLIB_HEIGHT = 9
    
    # Plotly dimensions (pixels)
    PLOTLY_WIDTH = 1500
    PLOTLY_HEIGHT = 900
    PLOTLY_MIN_WIDTH = 1000
    
    # Subplot dimensions
    SUBPLOT_WIDTH = 400
    SUBPLOT_HEIGHT = 400
    
    # Matplotlib grid sizing
    MATPLOTLIB_SUBPLOT_SIZE = 3


class SubplotSpacing:
    """Standard spacing configurations for different plot types."""
    
    # Standard spacing
    STANDARD_HORIZONTAL = 0.1   # 10%
    STANDARD_VERTICAL = 0.2     # 20%
    
    # Heatmap spacing (needs room for colorbars)
    HEATMAP_HORIZONTAL = 0.15   # 15%
    HEATMAP_VERTICAL = 0.12     # 12%
    
    # Flux plot spacing (needs extra room for overlays)
    FLUX_HORIZONTAL = 0.25      # 25%
    FLUX_VERTICAL = 0.12        # 12%


class Margins:
    """Standard margin settings for figures."""
    
    PLOTLY_MARGINS = {
        "l": 60,   # left
        "r": 60,   # right  
        "t": 80,   # top
        "b": 60    # bottom
    }


# ===== COLORBAR SPECIFICATIONS =====
class ColorbarConfig:
    """Standard colorbar positioning and styling."""
    
    # Position settings
    X_OFFSET = 0.03           # Distance from subplot right edge
    WIDTH = 0.02              # 2% of figure width
    HEIGHT_RATIO = 0.90       # 90% of subplot height
    
    # Styling settings
    THICKNESS = 14            # pixels
    TICKS = "outside"
    TICKLABELPOSITION = "outside"
    DEFAULT_TITLE = "|IQ|"
    
    # Z-axis scaling
    ZMIN_PERCENTILE = 2.0     # Robust minimum (2nd percentile)
    ZMAX_PERCENTILE = 98.0    # Robust maximum (98th percentile)


# ===== AXIS FORMATTING =====
class AxisLabels:
    """Standard axis labels and units for different measurements."""
    
    # Frequency labels
    RF_FREQUENCY_GHZ = "RF frequency [GHz]"
    FREQUENCY_GHZ = "Frequency (GHz)"
    DETUNING_MHZ = "Detuning [MHz]"
    
    # Amplitude labels  
    IQ_AMPLITUDE_MV = "R = √(I² + Q²) [mV]"
    IQ_AMPLITUDE_MV_MATPLOTLIB = r"$R=\sqrt{I^2 + Q^2}$ [mV]"  # LaTeX for matplotlib
    ROTATED_I_MV = "Rotated I quadrature [mV]"
    PULSE_AMPLITUDE_MV = "Pulse amplitude [mV]"
    
    # Power and flux labels
    POWER_DBM = "Power (dBm)"
    FLUX_BIAS_V = "Flux bias [V]"
    CURRENT_A = "Current (A)"
    
    # State labels
    QUBIT_STATE = "Qubit state"
    AMPLITUDE_PREFACTOR = "Amplitude prefactor"


class NumberFormatting:
    """Standard number formatting for different quantities."""
    
    # Frequency formatting
    FREQUENCY_GHZ_PRECISION = "{:.3f}"  # 3 decimal places for GHz
    FREQUENCY_GHZ_HIGH_PRECISION = "{:.4f}"  # 4 decimal places for precise freq
    DETUNING_MHZ_PRECISION = "{:.2f}"   # 2 decimal places for MHz
    
    # Power and amplitude formatting
    POWER_DBM_PRECISION = "{:.2f}"      # 2 decimal places for dBm
    AMPLITUDE_MV_PRECISION = "{:.3f}"   # 3 decimal places for mV
    
    # Current formatting
    CURRENT_A_PRECISION = "{:.6f}"      # 6 decimal places for current


# ===== HOVER TEMPLATES =====
class HoverTemplates:
    """Standard hover template formats for interactive plots."""
    
    # Resonator spectroscopy hover
    RESONATOR_SPECTROSCOPY = (
        "<b>Freq</b>: %{x:.4f} GHz<br>"
        "<b>Detuning</b>: %{customdata[0]:.2f} MHz<br>"
        "<b>Amplitude</b>: %{y:.3f} mV<extra></extra>"
    )
    
    # Power Rabi hover
    POWER_RABI = (
        "Amplitude: %{x:.3f} mV<br>"
        "Prefactor: %{customdata[0]:.3f}<br>"
        "%{y:.3f} mV<extra></extra>"
    )
    
    # Heatmap hover (generic)
    HEATMAP_GENERIC = (
        "Freq [GHz]: %{x:.3f}<br>"
        "Power [dBm]: %{y:.2f}<br>"
        "Detuning [MHz]: %{customdata:.2f}<br>"
        "|IQ|: %{z:.3f}<extra>%{text}</extra>"
    )


# ===== DUAL AXIS CONFIGURATION =====
class DualAxisConfig:
    """Configuration for secondary axes (top x-axis)."""
    
    AXIS_OFFSET = 100           # Plotly axis numbering offset
    SIDE = "top"                # Position of secondary axis
    SHOW_GRID = False           # Disable grid on secondary axes


# ===== FONT AND TYPOGRAPHY =====
class Typography:
    """Font and text styling settings."""
    
    ANNOTATION_FONT_SIZE = 16   # pixels for Plotly annotations
    SUBPLOT_TITLE_SIZE = 16     # pixels for subplot titles


# ===== LEGEND CONFIGURATION =====
class LegendConfig:
    """Legend display settings."""
    
    SHOW_LEGEND = False         # Generally disabled to avoid clutter
    SHOW_LEGEND_PER_TRACE = False  # Individual trace legends disabled


# ===== UNIT CONVERSION CONSTANTS =====
class UnitConversions:
    """Standard unit conversion factors."""
    
    HZ_TO_GHZ = 1e-9           # Convert Hz to GHz
    HZ_TO_MHZ = 1e-6           # Convert Hz to MHz  
    V_TO_MV = 1e3              # Convert V to mV
    GHZ_TO_HZ = 1e9            # Convert GHz to Hz
    MHZ_TO_HZ = 1e6            # Convert MHz to Hz
    MV_TO_V = 1e-3             # Convert mV to V


# ===== CONVENIENCE FUNCTIONS =====
def get_standard_plotly_style() -> Dict[str, Any]:
    """Get standard Plotly figure layout settings."""
    return {
        "width": FigureDimensions.PLOTLY_WIDTH,
        "height": FigureDimensions.PLOTLY_HEIGHT,
        "margin": Margins.PLOTLY_MARGINS,
        "showlegend": LegendConfig.SHOW_LEGEND,
    }


def get_standard_matplotlib_size() -> tuple:
    """Get standard Matplotlib figure size."""
    return (FigureDimensions.MATPLOTLIB_WIDTH, FigureDimensions.MATPLOTLIB_HEIGHT)


def get_raw_data_style() -> Dict[str, Any]:
    """Get standard styling for raw data traces."""
    return {
        "color": Colors.RAW_DATA,
        "width": LineStyles.RAW_LINE_WIDTH,
    }


def get_fit_line_style() -> Dict[str, Any]:
    """Get standard styling for fit line traces."""
    return {
        "color": Colors.FIT_LINE,
        "width": LineStyles.FIT_LINE_WIDTH,
        "dash": LineStyles.FIT_LINE_DASH,
    }


def get_optimal_marker_style() -> Dict[str, Any]:
    """Get standard styling for optimal point markers."""
    return {
        "symbol": LineStyles.MARKER_SYMBOL,
        "color": Colors.OPTIMAL_MARKER,
        "size": LineStyles.MARKER_SIZE,
    }


def get_heatmap_style() -> Dict[str, Any]:
    """Get standard styling for heatmap traces."""
    return {
        "colorscale": Colors.HEATMAP_COLORSCALE,
        "showscale": True,
        "colorbar": {
            "title": ColorbarConfig.DEFAULT_TITLE,
            "thickness": ColorbarConfig.THICKNESS,
            "ticks": ColorbarConfig.TICKS,
            "ticklabelposition": ColorbarConfig.TICKLABELPOSITION,
        }
    }