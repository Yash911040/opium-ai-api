from fastapi import FastAPI, UploadFile, File
import tempfile
import os
import shutil

from classifier import classify_stage
from detector import detect_capsules


app = FastAPI(
    title="Opium AI Analysis API",
    description="Growth stage classification and poppy capsule detection API",
    version="1.0.0",
)


@app.get("/")
def root():
    return {
        "status": "online",
        "message": "Opium AI Analysis API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    suffix = os.path.splitext(file.filename or "")[1]

    if not suffix:
        suffix = ".jpg"

    temp_path = None

    try:
        # Create temporary image file
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_file:

            shutil.copyfileobj(
                file.file,
                temp_file
            )

            temp_path = temp_file.name


        # Stage Classification
        stage = classify_stage(temp_path)


        # Capsule Detection
        detection = detect_capsules(temp_path)


        # --------------------------------
        # Temporary Presentation Rule
        # --------------------------------
        if (
            stage["stage"] == "Flowering"
            and detection["capsule_count"] > 5
        ):
            stage["stage"] = "Capsule_Development"


        return {
            "growth_stage": stage["stage"],

            "stage_confidence":
                stage["stage_confidence"],

            "capsule_count":
                detection["capsule_count"],

            "average_detection_confidence":
                detection["average_confidence"],

            "maximum_detection_confidence":
                detection["maximum_confidence"],
        }


    finally:

        # Delete temporary image after inference
        if (
            temp_path is not None
            and os.path.exists(temp_path)
        ):
            os.remove(temp_path)

        await file.close()