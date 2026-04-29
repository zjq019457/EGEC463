import argparse
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from ultralytics import YOLO

matplotlib.use("Agg")
import matplotlib.pyplot as plt


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def read_label_classes(label_path: Path) -> list[int]:
    if not label_path.exists():
        return []
    classes = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if parts:
            classes.append(int(float(parts[0])))
    return classes


def find_positive_images(images_dir: Path, labels_dir: Path, limit: int) -> list[Path]:
    images = []
    for image_path in sorted(images_dir.glob("*")):
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label_path = labels_dir / f"{image_path.stem}.txt"
        if 1 in read_label_classes(label_path):
            images.append(image_path)
        if len(images) >= limit:
            break
    return images


def positive_class_index(model: YOLO) -> int:
    names = model.names
    for index, name in names.items():
        if name == "positive":
            return int(index)
    raise ValueError(f"Could not find 'positive' in classifier names: {names}")


def positive_probability(model: YOLO, image, positive_index: int, imgsz: int) -> float:
    result = model(image, imgsz=imgsz, verbose=False)[0]
    return float(result.probs.data[positive_index])


def occlusion_heatmap(model: YOLO, image_path: Path, positive_index: int, imgsz: int, grid: int) -> tuple[np.ndarray, float]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    baseline = positive_probability(model, image, positive_index, imgsz)
    heatmap = np.zeros((grid, grid), dtype=np.float32)
    fill_value = image.mean(axis=(0, 1), keepdims=True).astype(image.dtype)
    height, width = image.shape[:2]

    for row in range(grid):
        for col in range(grid):
            y1 = int(round(row * height / grid))
            y2 = int(round((row + 1) * height / grid))
            x1 = int(round(col * width / grid))
            x2 = int(round((col + 1) * width / grid))
            occluded = image.copy()
            occluded[y1:y2, x1:x2] = fill_value
            prob = positive_probability(model, occluded, positive_index, imgsz)
            heatmap[row, col] = max(0.0, baseline - prob)

    return heatmap, baseline


def save_overlay(image_path: Path, heatmap: np.ndarray, baseline_prob: float, output_path: Path):
    image = cv2.imread(str(image_path))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)
    if resized.max() > 0:
        resized = resized / resized.max()

    plt.figure(figsize=(6, 6))
    plt.imshow(image_rgb)
    plt.imshow(resized, cmap="inferno", alpha=0.45)
    plt.title(f"{image_path.name} | positive prob={baseline_prob:.3f}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Create classifier occlusion heatmaps for positive MRI samples.")
    parser.add_argument("--weights", default="runs/classification/brain_tumor_cls_nano/weights/best.pt")
    parser.add_argument("--images-dir", default="datasets/brain-tumor/images/val")
    parser.add_argument("--labels-dir", default="datasets/brain-tumor/labels/val")
    parser.add_argument("--output-dir", default="runs/interpretability_heatmaps")
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--grid", type=int, default=7)
    parser.add_argument("--samples", type=int, default=3)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)
    positive_index = positive_class_index(model)
    image_paths = find_positive_images(Path(args.images_dir), Path(args.labels_dir), args.samples)

    for image_path in image_paths:
        heatmap, baseline = occlusion_heatmap(model, image_path, positive_index, args.imgsz, args.grid)
        output_path = output_dir / f"{image_path.stem}_occlusion_heatmap.png"
        save_overlay(image_path, heatmap, baseline, output_path)
        print(f"Saved heatmap: {output_path}")

    print(f"Saved {len(image_paths)} heatmaps to: {output_dir}")


if __name__ == "__main__":
    main()
