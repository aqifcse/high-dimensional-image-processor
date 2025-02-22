import sys
import os
import shutil
import numpy as np
import pytest
from fastapi.testclient import TestClient
from skimage import io
from src.main import app

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

client = TestClient(app)


@pytest.fixture
def fake_5d_file(tmp_path):
    # Create a random fake 5D image of shape (10, 3, 2, 50, 50)
    image_array = np.random.randint(
        0, 256, size=(10, 3, 2, 50, 50),
        dtype=np.uint8
    )
    file_path = tmp_path / "fake_image.tif"
    io.imsave(str(file_path), image_array)

    # Ensure the uploads directory exists
    uploads_dir = os.path.join("uploads")
    os.makedirs(uploads_dir, exist_ok=True)

    # Copy the file to uploads (so endpoints can find it)
    dest_path = os.path.join(uploads_dir, file_path.name)
    shutil.copy(str(file_path), dest_path)

    return (open(str(file_path), "rb").read(), file_path.name)


def test_upload(fake_5d_file):
    file_content, filename = fake_5d_file
    response = client.post(
        "/upload/",
        files={"file": (filename, file_content, "image/tiff")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == filename
    # Check file exists
    assert os.path.isfile(os.path.join("uploads", filename))


def test_metadata(fake_5d_file):
    _, filename = fake_5d_file
    response = client.get(f"/metadata/{filename}")
    assert response.status_code == 200
    data = response.json()
    assert "metadata" in data


def test_extract_slice(fake_5d_file):
    _, filename = fake_5d_file
    response = client.post(
        f"/extract-slice/?filename={filename}&z=0&time=1&channel=1"
    )
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data.get("slice_data"), list)


def test_analyze(fake_5d_file):
    _, filename = fake_5d_file
    response = client.post(f"/analyze/?filename={filename}")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data.get("analysis_results"), list)


def test_statistics(fake_5d_file):
    _, filename = fake_5d_file
    response = client.get(f"/statistics/{filename}")
    assert response.status_code == 200
    data = response.json()
    stats = data.get("statistics", {})
    for key in ["mean", "std_dev", "min", "max"]:
        assert isinstance(stats.get(key), float)


def test_segment(fake_5d_file):
    _, filename = fake_5d_file
    response = client.post(f"/segment/?filename={filename}")
    assert response.status_code == 200
    data = response.json()
    seg_img = data.get("segmented_image")
    # For a fake image of shape (10, 3, 2, 50, 50)
    # segmentation uses the first channel of slice [0,0]
    # so the expected segmentation result shape
    # is (50,50) (converted to a list)
    assert isinstance(seg_img, list)
