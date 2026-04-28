from ultralytics import YOLO
from typing import Dict

MODEL_WEIGHTS = {
    'nano': 'yolo11n.pt',
    'small': 'yolo11s.pt',
    'medium': 'yolo11m.pt',
    'large': 'yolo11l.pt',
    'xlarge': 'yolo11x.pt',
}


def load_models() -> Dict[str, YOLO]:
    """Load all YOLO model variants."""
    return {size: YOLO(path) for size, path in MODEL_WEIGHTS.items()}


def load_model(model_size_or_path: str) -> YOLO:
    """Load a YOLO model by size alias or direct path."""
    if model_size_or_path in MODEL_WEIGHTS:
        return YOLO(MODEL_WEIGHTS[model_size_or_path])
    return YOLO(model_size_or_path)
