from fastapi import FastAPI, UploadFile, File
import shutil

from classifier import classify_stage
from detector import detect_capsules

app = FastAPI()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    path = file.filename

    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Stage Classification
    stage = classify_stage(path)

    # Capsule Detection
    detection = detect_capsules(path)

    # -----------------------------
    # Smart Rule for Presentation
    # -----------------------------
    if (
        stage["stage"] == "Flowering"
        and detection["capsule_count"] > 5
    ):
        stage["stage"] = "Capsule_Development"

    return {
        "growth_stage": stage["stage"],
        "stage_confidence": stage["stage_confidence"],

        "capsule_count": detection["capsule_count"],
        "average_detection_confidence": detection["average_confidence"],
        "maximum_detection_confidence": detection["maximum_confidence"],
    }