"""Tests for exception handling in analysis/guess.py."""

import pytest
import numpy as np
from unittest.mock import patch
from qualibration_libs.analysis.guess import exp_decay


class TestExpDecay:
    """Test exception handling in exp_decay function."""

    def test_valid_exponential_data(self):
        x = np.linspace(0, 10, 50)
        y = np.exp(-0.5 * x) + 0.01

        decay_rate = exp_decay(x, y)
        assert isinstance(decay_rate, float)
        assert decay_rate < 0  # Decay should be negative

    def test_nan_in_x_values_raises_exception(self):
        # Test with NaN in x array (not filtered by y > 0 check)
        # NaN values in x will cause polyfit to fail
        x = np.array([np.nan, np.nan, 1.0, 2.0])
        y = np.array([0.5, 0.6, 0.7, -1.0])

        with pytest.raises(ValueError) as exc_info:
            exp_decay(x, y)

        error_msg = str(exc_info.value)
        assert "Failed to fit exponential decay" in error_msg
        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, (ValueError, np.linalg.LinAlgError))

    def test_polyfit_linalgerror_is_caught_and_reraised(self):
        # Test that LinAlgError from polyfit is caught and reraised with context (wasn't able to trigger it without mocking)
        x = np.linspace(0, 10, 10)
        y = np.exp(-0.5 * x) + 0.01

        with patch('numpy.polyfit') as mock_polyfit:
            mock_polyfit.side_effect = np.linalg.LinAlgError("Mock linalg error")

            with pytest.raises(ValueError) as exc_info:
                exp_decay(x, y)

            error_msg = str(exc_info.value)
            assert "Failed to fit exponential decay" in error_msg
            assert exc_info.value.__cause__ is not None
