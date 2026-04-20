"""Tests for exception handling in data/fetcher.py."""

import numpy as np
import pytest
import xarray as xr
from unittest.mock import MagicMock, Mock, patch

from qualibration_libs.data.fetcher import XarrayDataFetcher


class TestXarrayDataFetcherGetItem:
    """Test exception handling in XarrayDataFetcher.__getitem__."""

    def test_valid_key_access(self):
        mock_job = Mock()
        mock_job.result_handles.keys.return_value = []

        fetcher = XarrayDataFetcher(job=mock_job, axes=None)
        result = fetcher["dataset"]
        assert result is not None

    def test_invalid_key_raises_keyerror(self):
        mock_job = Mock()
        mock_job.result_handles.keys.return_value = []

        fetcher = XarrayDataFetcher(job=mock_job, axes=None)

        with pytest.raises(KeyError) as exc_info:
            _ = fetcher["invalid_key"]

        error_msg = str(exc_info.value)
        assert "Data key 'invalid_key' not found in XarrayDataFetcher" in error_msg
        assert "Available keys: 'dataset'" in error_msg
        assert exc_info.value.__cause__ is not None


@patch("qualibration_libs.data.fetcher.fetching_tool")
def test_zero_dim_shot_counter_with_swept_iq_does_not_raise(mock_fetching_tool):
    """Regression: n_st.save('n') yields 0-D 'n' while I*/Q* are 1-D along detuning."""
    mock_result = MagicMock()
    mock_result.fetch_all.return_value = [
        np.array(42, dtype=np.int64),
        np.linspace(0, 1, 5),
        np.linspace(1, 0, 5),
    ]
    mock_result.is_processing.return_value = False
    mock_result.get_start_time.return_value = 0.0
    mock_fetching_tool.return_value = mock_result

    mock_job = Mock()
    mock_job.result_handles.keys.return_value = ["n", "I1", "Q1"]

    axes = {
        "qubit": xr.DataArray(["q0"]),
        "detuning": xr.DataArray(np.linspace(-1e6, 1e6, 5), attrs={"units": "Hz"}),
    }
    fetcher = XarrayDataFetcher(job=mock_job, axes=axes)
    fetcher.retrieve_latest_data()
    fetcher.update_dataset()

    assert fetcher.get("n") == 42
    assert "I" in fetcher.dataset.data_vars
    assert "Q" in fetcher.dataset.data_vars
    assert fetcher.dataset["I"].shape == (1, 5)
    assert fetcher.dataset["Q"].shape == (1, 5)


@patch("qualibration_libs.data.fetcher.fetching_tool")
def test_zero_dim_n_two_qubits(mock_fetching_tool):
    mock_result = MagicMock()
    mock_result.fetch_all.return_value = [
        np.array(100, dtype=np.int32),
        np.arange(3.0),
        np.arange(3.0, 6.0),
        np.arange(6.0, 9.0),
        np.arange(9.0, 12.0),
    ]
    mock_result.is_processing.return_value = False
    mock_result.get_start_time.return_value = 0.0
    mock_fetching_tool.return_value = mock_result

    mock_job = Mock()
    mock_job.result_handles.keys.return_value = ["n", "I1", "Q1", "I2", "Q2"]

    axes = {
        "qubit": xr.DataArray(["q0", "q1"]),
        "detuning": xr.DataArray(np.zeros(3)),
    }
    fetcher = XarrayDataFetcher(job=mock_job, axes=axes)
    fetcher.retrieve_latest_data()
    fetcher.update_dataset()

    assert fetcher.get("n") == 100
    assert fetcher.dataset["I"].shape == (2, 3)
    assert fetcher.dataset["Q"].shape == (2, 3)
