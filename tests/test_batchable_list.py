"""Tests for exception handling in core/batchable_list.py."""

import pytest
from qualibration_libs.core.batchable_list import BatchableList


class TestBatchableListGetItem:
    """Test exception handling in BatchableList.__getitem__."""

    def test_valid_index_access(self):
        items = ["a", "b", "c"]
        batch_groups = [[0], [1], [2]]
        bl = BatchableList(items, batch_groups)

        assert bl[0] == "a"
        assert bl[1] == "b"
        assert bl[2] == "c"

    def test_negative_index_access(self):
        items = ["a", "b", "c"]
        batch_groups = [[0], [1], [2]]
        bl = BatchableList(items, batch_groups)

        assert bl[-1] == "c"
        assert bl[-2] == "b"

    def test_invalid_index_raises_indexerror(self):
        items = ["a", "b", "c"]
        batch_groups = [[0], [1], [2]]
        bl = BatchableList(items, batch_groups)

        with pytest.raises(IndexError) as exc_info:
            _ = bl[10]

        error_msg = str(exc_info.value)
        assert "Index 10 out of range for BatchableList with 3 items" in error_msg
        assert exc_info.value.__cause__ is not None


class TestBatchableListBatch:
    """Test exception handling in BatchableList.batch."""

    def test_valid_batch_groups(self):
        items = ["a", "b", "c", "d"]
        batch_groups = [[0, 2], [1, 3]]
        bl = BatchableList(items, batch_groups)

        result = bl.batch()
        assert len(result) == 2
        assert result[0] == {0: "a", 2: "c"}
        assert result[1] == {1: "b", 3: "d"}

    def test_single_batch_multiplexed(self):
        items = ["a", "b", "c"]
        batch_groups = [[0, 1, 2]]
        bl = BatchableList(items, batch_groups)

        result = bl.batch()
        assert len(result) == 1
        assert result[0] == {0: "a", 1: "b", 2: "c"}
