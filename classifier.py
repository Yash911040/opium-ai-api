from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent

model = YOLO(
    str(BASE_DIR / "models" / "stage_classifier.pt")
)

def classify_stage(image_path):
    results = model.predict(
        source=image_path,
        verbose=False
    )

    probs = results[0].probs
    stage = results[0].names[probs.top1]
    confidence = float(probs.top1conf)

    return {
        "stage": stage,
        "stage_confidence": round(confidence, 3)
    }