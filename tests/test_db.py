import os
import sqlite3
import pytest
from src.db import DB_NAME, TABLE_NAME, init_db, insert_metadata, get_metadata


@pytest.fixture(scope="function", autouse=True)
def use_temp_db(tmp_path, monkeypatch):
    """
    Use a temporary database file for testing.
    """
    temp_db = tmp_path / "temp_data.db"
    # Override DB_NAME in the module with our temporary database path.
    monkeypatch.setattr("src.db.DB_NAME", str(temp_db))
    # Ensure a fresh DB is used for each test.
    if temp_db.exists():
        os.remove(temp_db)
    init_db()
    yield
    if temp_db.exists():
        os.remove(temp_db)


def test_init_db_creates_table():
    """
    Verify that init_db creates the required table.
    """
    # init_db has been called in fixture
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (TABLE_NAME,) # noqa
    )
    result = cursor.fetchone()
    conn.close()
    assert result is not None


def test_insert_and_get_metadata_with_analysis():
    filename = "test_image.tif"
    dimensions = (10, 3, 2, 50, 50)
    analysis = {"mean": 100.0, "std_dev": 15.0}
    insert_metadata(filename, dimensions, analysis)
    meta = get_metadata(filename)
    assert meta is not None
    # JSON encoding converts tuple to list, so we compare with list.
    assert meta["dimensions"] == list(dimensions)
    assert meta["analysis"] == analysis


def test_insert_and_get_metadata_without_analysis():
    filename = "test_image2.tif"
    dimensions = (5, 2, 1, 30, 30)
    insert_metadata(filename, dimensions)
    meta = get_metadata(filename)
    assert meta is not None
    assert meta["dimensions"] == list(dimensions)
    assert meta["analysis"] is None


def test_get_metadata_nonexistent():
    meta = get_metadata("nonexistent.tif")
    assert meta is None
