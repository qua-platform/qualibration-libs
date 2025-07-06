
import re

import numpy as np


def convert_power_strings_to_numeric(power_values):
    """Convert power string values to numeric values.

    Parameters
    ----------
    power_values : list or np.ndarray
        A list or array of power values, which may be strings.

    Returns
    -------
    np.ndarray
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