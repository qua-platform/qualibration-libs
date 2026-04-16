"""Tests for exception handling in data/fetcher.py."""

import pytest
from unittest.mock import Mock
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
