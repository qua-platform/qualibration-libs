"""Utility functions for enhanced exception handling."""

from typing import Any, Sequence, Union


__all__ = ["format_available_items"]


def format_available_items(
    items: Union[Sequence[Any], dict],
    max_items: int = 10,
    item_type: str = "keys"
) -> str:
    """
    Format a list of available items for error messages.

    Parameters
    ----------
    items : Sequence or dict
        The items to format. If dict, will use keys().
    max_items : int
        Maximum number of items to include before truncating (default: 10)
    item_type : str
        Descriptive name for the items (e.g., "keys", "qubits", "dimensions")

    Returns
    -------
    str
        Formatted string like "Available keys: 'key1', 'key2', 'key3', ..."
    """
    item_keys = list(items.keys()) if isinstance(items, dict) else list(items)
    if item_keys:
        items_str = "'" + "', '".join(str(item) for item in item_keys[:max_items])
        items_str += "', ..." if len(item_keys) > max_items else "'."
    else:
        items_str = "None."

    return f"Available {item_type}: {items_str}"
