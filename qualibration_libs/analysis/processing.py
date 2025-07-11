
import logging
import re

import numpy as np
import xarray as xr


def convert_power_strings_to_numeric(power_values):
    """Convert power string values to numeric values.

    Args:
        power_values: A list or array of power values, which may be strings.

    Returns:
        An array of numeric power values.
    """
    if len(power_values) == 0:
        return np.array([])

    if not isinstance(power_values[0], str):
        return np.array(power_values, dtype=float)

    try:
        numeric_powers = []
        for p in power_values:
            # Use regex to extract number (including negative sign and decimal point)
            match = re.search(r"-?\d+\.?\d*", str(p))
            if match:
                numeric_powers.append(float(match.group()))
            else:
                # Fallback: try to extract any numeric characters
                numeric_part = "".join(c for c in str(p) if c.isdigit() or c in ".-")
                if numeric_part and numeric_part != "." and numeric_part != "-":
                    numeric_powers.append(float(numeric_part))
                else:
                    # Last resort: use index
                    numeric_powers.append(float(len(numeric_powers)))
        return np.array(numeric_powers)
    except (ValueError, TypeError):
        # Fallback: use indices
        return np.arange(len(power_values), dtype=float)


def calculate_sweep_span(fit: xr.Dataset) -> float:
    """Calculate sweep span for relative comparisons."""
    if "detuning" in fit.dims:
        return float(fit.coords["detuning"].max() - fit.coords["detuning"].min())
    if "full_freq" in fit.dims:
        return float(fit.coords["full_freq"].max() - fit.coords["full_freq"].min())
    return 0.0


def extract_qubit_signal(fit: xr.Dataset, qubit: str) -> np.ndarray:
    """Extract signal data for a specific qubit."""
    if hasattr(fit, "I") and hasattr(fit.I, "sel"):
        return fit.I.sel(qubit=qubit).values
    elif hasattr(fit, "state") and hasattr(fit.state, "sel"):
        return fit.state.sel(qubit=qubit).values
    else:
        logging.warning(
            f"Could not find 'I' or 'state' data for qubit {qubit}. Returning zeros."
        )
        return np.zeros((2, 2))