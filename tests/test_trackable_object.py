"""Tests for exception handling in core/trackable_object.py."""

import pytest
from qualibration_libs.core.trackable_object import TrackableObject


class TestTrackableObjectGetAttr:
    """Test exception handling in TrackableObject.__getattr__."""

    def test_valid_attribute_access(self):
        class MockObj:
            def __init__(self):
                self.value = 42
                self.name = "test"

        obj = MockObj()
        tracked = TrackableObject(obj)

        assert tracked.value == 42
        assert tracked.name == "test"

    def test_invalid_attribute_raises_attributeerror(self):
        class MockObj:
            def __init__(self):
                self.value = 42
                self.name = "test"

        obj = MockObj()
        tracked = TrackableObject(obj)

        with pytest.raises(AttributeError) as exc_info:
            _ = tracked.invalid_attr

        error_msg = str(exc_info.value)
        assert "Attribute 'invalid_attr' not found in tracked object of type 'MockObj'" in error_msg
        assert "Available attributes: 'name', 'value'" in error_msg
        assert exc_info.value.__cause__ is not None


class TestTrackableObjectGetItem:
    """Test exception handling in TrackableObject.__getitem__."""

    def test_valid_dict_key_access(self):
        obj = {"key1": "value1", "key2": "value2"}
        tracked = TrackableObject(obj)

        assert tracked["key1"]._obj == "value1"
        assert tracked["key2"]._obj == "value2"

    def test_invalid_dict_key_raises_keyerror(self):
        obj = {"key1": "value1", "key2": "value2"}
        tracked = TrackableObject(obj)

        with pytest.raises(KeyError) as exc_info:
            _ = tracked["invalid_key"]

        error_msg = str(exc_info.value)
        assert "Key 'invalid_key' not found in tracked object" in error_msg
        assert "Available keys: 'key1', 'key2'" in error_msg
        assert exc_info.value.__cause__ is not None

    def test_valid_list_index_access(self):
        obj = ["a", "b", "c"]
        tracked = TrackableObject(obj)

        assert tracked[0]._obj == "a"
        assert tracked[1]._obj == "b"
        assert tracked[2]._obj == "c"

    def test_invalid_list_index_raises_indexerror(self):
        obj = ["a", "b", "c"]
        tracked = TrackableObject(obj)

        with pytest.raises(IndexError) as exc_info:
            _ = tracked[10]

        error_msg = str(exc_info.value)
        assert "Index 10 out of range for tracked object with length 3." in error_msg
        assert exc_info.value.__cause__ is not None


class TestTrackableObjectTracking:
    """Test that tracking functionality still works after adding exception handling."""

    def test_attribute_modification_tracking(self):
        class MockObj:
            def __init__(self):
                self.value = 42

        obj = MockObj()
        tracked = TrackableObject(obj)

        tracked.value = 100
        assert obj.value == 100

        tracked.revert_changes()
        assert obj.value == 42
