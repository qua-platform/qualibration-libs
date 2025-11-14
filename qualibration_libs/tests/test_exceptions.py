"""Tests for exception handling utilities."""

import pytest
from qualibration_libs.core.exceptions import format_available_items


class TestFormatAvailableItems:
    """Test the format_available_items utility function."""

    def test_format_dict(self):
        """Test formatting a dictionary."""
        result = format_available_items({'a': 1, 'b': 2, 'c': 3}, item_type="keys")
        assert "Available keys:" in result
        assert "'a'" in result and "'b'" in result and "'c'" in result
        assert result.endswith("'.")

    def test_format_list(self):
        """Test formatting a list."""
        result = format_available_items(['x', 'y', 'z'], item_type="items")
        assert "Available items:" in result
        assert "'x'" in result and "'y'" in result and "'z'" in result

    def test_truncation_at_10_items(self):
        """Test that long lists are truncated at 10 items."""
        result = format_available_items(range(15), item_type="indices")
        assert "..." in result
        assert "'0'" in result and "'9'" in result
        assert "'10'" not in result and "'14'" not in result

    def test_empty_dict(self):
        """Test formatting an empty list."""
        result = format_available_items({}, item_type="keys")
        assert result == "Available keys: None."

    def test_single_item(self):
        """Test formatting a single item."""
        result = format_available_items({'only': 'you'}, item_type="items")
        assert result == "Available items: 'only'."

    def test_exactly_10_items(self):
        """Test with exactly 10 items (boundary case)."""
        result = format_available_items(range(10), item_type="numbers")
        assert "..." not in result
        assert "'9'" in result
        assert result.endswith("'.")
