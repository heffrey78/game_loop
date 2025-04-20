"""Basic test module to verify that pytest is working properly."""

import pytest


def test_basic_assertion():
    """A simple test to verify pytest is working."""
    assert True


def test_basic_math():
    """A simple test for basic arithmetic operations."""
    assert 1 + 1 == 2
    assert 2 * 2 == 4


def test_basic_exception():
    """Test that an exception is raised appropriately."""
    with pytest.raises(ValueError):
        int("not a number")
