from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent

model = YOLO(
    str(BASE_DIR / "models" / "capsule_detector.pt")
)


def detect_capsules(image_path, conf=0.6):

    results = model.predict(
        source=image_path,
        conf=conf,
        save=False,
        verbose=False
    )

    boxes = results[0].boxes

    capsule_count = len(boxes)

    if capsule_count > 0:
        confidences = boxes.conf.cpu().numpy()

        avg_conf = float(confidences.mean())
        max_conf = float(confidences.max())
    else:
        avg_conf = 0.0
        max_conf = 0.0

    return {
        "capsule_count": capsule_count,
        "average_confidence": round(avg_conf, 3),
        "maximum_confidence": round(max_conf, 3)
    }