from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Dict
import os
import numpy as np
import json
from src.image_processing.manipulation import ImageManipulator
from src.analysis.analysis import perform_pca, calculate_statistics
from src.db import insert_metadata, get_metadata  # new import

router = APIRouter()


@router.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        os.makedirs("uploads", exist_ok=True)
        file_location = os.path.join("uploads", file.filename)
        with open(file_location, "wb") as f:
            f.write(contents)
        # After saving, load the image and store its dimensions in the DB.
        manipulator = ImageManipulator(file_location)
        dimensions = manipulator.image.shape
        # Initially, no analysis saved.
        insert_metadata(file.filename, dimensions)
        return {"filename": file.filename, "message": "Upload successful"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metadata/{filename}")
async def get_metadata_endpoint(filename: str) -> Dict:
    meta = get_metadata(filename)
    if meta is None:
        # fallback: return basic info
        return {"filename": filename, "metadata": "No metadata stored"}
    return {"filename": filename, "metadata": meta}


@router.post("/extract-slice/")
async def extract_image_slice(
    filename: str,
    z: int = None,
    time: int = None,
    channel: int = None
):
    manipulator = ImageManipulator(os.path.join("uploads", filename))
    slice_data = manipulator.extract_slice(z=z, time=time, channel=channel)
    if isinstance(slice_data, np.ndarray):
        slice_data = slice_data.tolist()
    return {"filename": filename, "slice_data": slice_data}


@router.post("/analyze/")
async def analyze_uploaded_image(filename: str) -> Dict:
    manipulator = ImageManipulator(os.path.join("uploads", filename))
    analysis_results = perform_pca(manipulator.image)
    # Store analysis results in the DB alongside dimensions.
    dimensions = manipulator.image.shape
    insert_metadata(
        filename,
        dimensions,
        analysis=json.loads(
            json.dumps(
                analysis_results.tolist()
            )
        )
    )
    return {
        "filename": filename,
        "analysis_results": analysis_results.tolist()
    }


@router.get("/statistics/{filename}")
async def get_image_statistics(filename: str) -> Dict:
    manipulator = ImageManipulator(os.path.join("uploads", filename))
    stats = calculate_statistics(manipulator.image)
    return {"filename": filename, "statistics": stats}


@router.post("/segment/")
async def segment_image(filename: str):
    manipulator = ImageManipulator(os.path.join("uploads", filename))
    segmented_image = manipulator.apply_segmentation()
    return {
        "filename": filename,
        "segmented_image": segmented_image.tolist()
    }
