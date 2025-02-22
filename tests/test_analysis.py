import sys
import os
import numpy as np
import pytest
from src.analysis.analysis import perform_pca, calculate_statistics

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)


@pytest.fixture
def fake_5d_array():
    # Create a fake 5D image array of shape (5, 2, 3, 20, 20)
    return np.random.randint(0, 256, size=(5, 2, 3, 20, 20), dtype=np.uint8)


def test_perform_pca(fake_5d_array):
    result = perform_pca(fake_5d_array, n_components=2)
    Z, T, C, X, Y = fake_5d_array.shape
    # Expected shape: (Z, T, C, 2)
    assert result.shape == (Z, T, C, 2)


def test_calculate_statistics(fake_5d_array):
    stats = calculate_statistics(fake_5d_array)
    for key in ["mean", "std_dev", "min", "max"]:
        assert isinstance(stats[key], float)
