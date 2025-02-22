import os
import sys
from src.main import app


sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)


def test_app_loaded():
    # Update the expected title to match what is actually set in your app.
    # If your app.title is "FastAPI", then use that value.
    assert app.title == "FastAPI"
