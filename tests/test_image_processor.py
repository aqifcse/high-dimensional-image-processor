import os
import sys
import numpy as np
import pytest
from skimage import io
from src.image_processing.manipulation import ImageManipulator


sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)


@pytest.fixture
def fake_5d_image(tmp_path):
    # Create a random fake 5D image of shape (10, 3, 2, 50, 50)
    image_array = np.random.randint(
        0, 256, size=(10, 3, 2, 50, 50), dtype=np.uint8
    )
    file_path = tmp_path / "fake_image.tif"
    io.imsave(str(file_path), image_array)
    return str(file_path)


def test_load_image(fake_5d_image):
    manipulator = ImageManipulator(fake_5d_image)
    assert manipulator.image.ndim == 5


def test_extract_slice(fake_5d_image):
    manipulator = ImageManipulator(fake_5d_image)
    slice_data = manipulator.extract_slice(z=0, time=1, channel=1)
    assert isinstance(slice_data, np.ndarray)
    assert slice_data.ndim == 2


def test_apply_pca(fake_5d_image):
    manipulator = ImageManipulator(fake_5d_image)
    pca_result = manipulator.apply_pca(n_components=3)
    Z, T, C, X, Y = manipulator.image.shape
    # Expected shape: (Z, T, C, 3)
    assert pca_result.shape == (Z, T, C, 3)


def test_calculate_statistics(fake_5d_image):
    manipulator = ImageManipulator(fake_5d_image)
    stats = manipulator.calculate_statistics()
    for key in ["mean", "std_dev", "min", "max"]:
        assert isinstance(stats[key], float)


def test_apply_segmentation(fake_5d_image):
    manipulator = ImageManipulator(fake_5d_image)
    seg_result = manipulator.apply_segmentation()
    # Segmentation uses the first slice [0, 0] of
    # shape (Channel, 50, 50) and if fewer than 3 channels,
    # it returns the first channel, so resulting array is (50, 50)
    assert isinstance(seg_result, np.ndarray)
    assert seg_result.ndim == 2
    assert seg_result.shape == (50, 50)
