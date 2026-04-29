import argparse
import csv
import html
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib
from ultralytics import YOLO

matplotlib.use("Agg")
import matplotlib.pyplot as plt


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class DetectionRecord:
    image_path: Path
    label_path: Path
    max_positive_conf: float
    missed_boxes: list[dict]


def read_yolo_labels(label_path: Path) -> list[dict]:
    if not label_path.exists():
        return []

    boxes = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls_id, cx, cy, width, height = parts
        boxes.append(
            {
                "class_id": int(float(cls_id)),
                "cx": float(cx),
                "cy": float(cy),
                "width": float(width),
                "height": float(height),
                "area": float(width) * float(height),
            }
        )
    return boxes


def collect_validation_pairs(images_dir: Path, labels_dir: Path) -> list[tuple[Path, Path]]:
    pairs = []
    for image_path in sorted(images_dir.glob("*")):
        if image_path.suffix.lower() in IMAGE_SUFFIXES:
            pairs.append((image_path, labels_dir / f"{image_path.stem}.txt"))
    return pairs


def normalized_box_to_pixels(box: dict, image_shape: tuple[int, int, int]) -> tuple[int, int, int, int]:
    height, width = image_shape[:2]
    box_width = box["width"] * width
    box_height = box["height"] * height
    x_center = box["cx"] * width
    y_center = box["cy"] * height
    x1 = max(0, int(round(x_center - box_width / 2)))
    y1 = max(0, int(round(y_center - box_height / 2)))
    x2 = min(width - 1, int(round(x_center + box_width / 2)))
    y2 = min(height - 1, int(round(y_center + box_height / 2)))
    return x1, y1, x2, y2


def location_bucket(value: float) -> str:
    if value < 0.33:
        return "low"
    if value > 0.66:
        return "high"
    return "mid"


def analyze_box_pixels(image, box: dict) -> dict:
    x1, y1, x2, y2 = normalized_box_to_pixels(box, image.shape)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    roi = gray[y1 : y2 + 1, x1 : x2 + 1]
    whole_std = float(gray.std()) or 1.0

    if roi.size == 0:
        return {
            "roi_mean": 0.0,
            "roi_std": 0.0,
            "relative_contrast": 0.0,
            "pixel_width": 0,
            "pixel_height": 0,
        }

    return {
        "roi_mean": float(roi.mean()),
        "roi_std": float(roi.std()),
        "relative_contrast": float(roi.std() / whole_std),
        "pixel_width": int(x2 - x1 + 1),
        "pixel_height": int(y2 - y1 + 1),
    }


def max_positive_conf(model: YOLO, image_path: Path, positive_class: int, min_conf: float, imgsz: int) -> float:
    result = model(str(image_path), conf=min_conf, imgsz=imgsz, verbose=False)[0]
    boxes = result.boxes
    if boxes is None or not len(boxes):
        return 0.0

    best = 0.0
    for cls_id, conf in zip(boxes.cls.tolist(), boxes.conf.tolist()):
        if int(cls_id) == positive_class and conf > best:
            best = float(conf)
    return best


def find_false_negatives(
    model: YOLO,
    pairs: list[tuple[Path, Path]],
    positive_class: int,
    threshold: float,
    min_conf: float,
    imgsz: int,
) -> list[DetectionRecord]:
    records = []
    for index, (image_path, label_path) in enumerate(pairs, start=1):
        labels = read_yolo_labels(label_path)
        positive_boxes = [box for box in labels if box["class_id"] == positive_class]
        if not positive_boxes:
            continue

        positive_conf = max_positive_conf(model, image_path, positive_class, min_conf, imgsz)
        if positive_conf >= threshold:
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            continue

        enriched_boxes = []
        for box in positive_boxes:
            enriched = dict(box)
            enriched.update(analyze_box_pixels(image, box))
            enriched["x_bucket"] = location_bucket(box["cx"])
            enriched["y_bucket"] = location_bucket(box["cy"])
            enriched_boxes.append(enriched)

        records.append(DetectionRecord(image_path, label_path, positive_conf, enriched_boxes))
        if index % 50 == 0 or index == len(pairs):
            print(f"Checked {index}/{len(pairs)} images")

    return records


def summarize_records(records: list[DetectionRecord]) -> dict:
    all_boxes = [box for record in records for box in record.missed_boxes]
    if not all_boxes:
        return {
            "fn_images": 0,
            "fn_boxes": 0,
            "mean_area": 0.0,
            "median_area": 0.0,
            "mean_relative_contrast": 0.0,
            "small_box_count": 0,
        }

    areas = sorted(box["area"] for box in all_boxes)
    mid = len(areas) // 2
    median_area = areas[mid] if len(areas) % 2 else (areas[mid - 1] + areas[mid]) / 2
    return {
        "fn_images": len(records),
        "fn_boxes": len(all_boxes),
        "mean_area": sum(areas) / len(areas),
        "median_area": median_area,
        "mean_relative_contrast": sum(box["relative_contrast"] for box in all_boxes) / len(all_boxes),
        "small_box_count": sum(1 for box in all_boxes if box["area"] < 0.02),
    }


def write_csv_outputs(records: list[DetectionRecord], output_dir: Path):
    rows = []
    for record in records:
        for box in record.missed_boxes:
            rows.append(
                {
                    "image": str(record.image_path),
                    "label": str(record.label_path),
                    "max_positive_conf": record.max_positive_conf,
                    "cx": box["cx"],
                    "cy": box["cy"],
                    "width": box["width"],
                    "height": box["height"],
                    "area": box["area"],
                    "pixel_width": box["pixel_width"],
                    "pixel_height": box["pixel_height"],
                    "roi_mean": box["roi_mean"],
                    "roi_std": box["roi_std"],
                    "relative_contrast": box["relative_contrast"],
                    "x_bucket": box["x_bucket"],
                    "y_bucket": box["y_bucket"],
                }
            )

    with (output_dir / "false_negative_boxes.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image",
                "label",
                "max_positive_conf",
                "cx",
                "cy",
                "width",
                "height",
                "area",
                "pixel_width",
                "pixel_height",
                "roi_mean",
                "roi_std",
                "relative_contrast",
                "x_bucket",
                "y_bucket",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def draw_false_negative_visuals(records: list[DetectionRecord], output_dir: Path, max_images: int):
    visuals_dir = output_dir / "fn_visuals"
    visuals_dir.mkdir(parents=True, exist_ok=True)
    visual_rows = []

    for record in records[:max_images]:
        image = cv2.imread(str(record.image_path))
        if image is None:
            continue

        for box in record.missed_boxes:
            x1, y1, x2, y2 = normalized_box_to_pixels(box, image.shape)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                image,
                f"missed positive {record.max_positive_conf:.3f}",
                (x1, max(14, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

        output_path = visuals_dir / f"fn_{record.image_path.stem}.jpg"
        cv2.imwrite(str(output_path), image)
        visual_rows.append({"image": record.image_path, "visual": output_path, "conf": record.max_positive_conf})

    return visual_rows


def plot_area_histogram(records: list[DetectionRecord], output_dir: Path):
    areas = [box["area"] for record in records for box in record.missed_boxes]
    plt.figure(figsize=(7, 4))
    plt.hist(areas, bins=12, color="#3d6fb6", edgecolor="white")
    plt.xlabel("Normalized GT box area")
    plt.ylabel("Missed positive boxes")
    plt.tight_layout()
    plt.savefig(output_dir / "fn_box_area_histogram.png", dpi=160)
    plt.close()


def write_html_report(output_dir: Path, summary: dict, threshold: float, visual_rows: list[dict]):
    visual_cards = "\n".join(
        "<figure>"
        f"<img src='{html.escape(Path(row['visual']).relative_to(output_dir).as_posix())}' alt='FN case'>"
        f"<figcaption>{html.escape(Path(row['image']).name)} | max positive conf={row['conf']:.3f}</figcaption>"
        "</figure>"
        for row in visual_rows
    )
    (output_dir / "false_negative_report.html").write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>False Negative Analysis</title>
  <style>
    body {{ font-family: sans-serif; margin: 32px; color: #172026; background: #f7f7f4; }}
    .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin: 20px 0; }}
    .metric {{ background: white; border: 1px solid #ddd; border-radius: 6px; padding: 12px; }}
    .metric strong {{ display: block; font-size: 24px; margin-top: 4px; }}
    img {{ max-width: 100%; border: 1px solid #ddd; background: white; }}
    .cases {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; }}
    figure {{ margin: 0; background: white; border: 1px solid #ddd; border-radius: 6px; padding: 10px; }}
    figcaption {{ font-size: 13px; margin-top: 8px; }}
  </style>
</head>
<body>
  <h1>False Negative Analysis</h1>
  <p>Missed positive images at threshold {threshold:.2f}. Red boxes are ground-truth positive regions missed by the detector.</p>
  <section class="summary">
    <div class="metric">FN images<strong>{summary['fn_images']}</strong></div>
    <div class="metric">FN boxes<strong>{summary['fn_boxes']}</strong></div>
    <div class="metric">Median box area<strong>{summary['median_area']:.3f}</strong></div>
    <div class="metric">Mean relative contrast<strong>{summary['mean_relative_contrast']:.3f}</strong></div>
    <div class="metric">Small boxes &lt;2%<strong>{summary['small_box_count']}</strong></div>
  </section>
  <h2>Missed Box Area</h2>
  <img src="fn_box_area_histogram.png" alt="FN area histogram">
  <h2>FN Examples</h2>
  <div class="cases">{visual_cards}</div>
</body>
</html>
""",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description="Analyze detector false negatives for positive tumor cases.")
    parser.add_argument("--weights", default="runs/detect/train/weights/best.pt")
    parser.add_argument("--images-dir", default="datasets/brain-tumor/images/val")
    parser.add_argument("--labels-dir", default="datasets/brain-tumor/labels/val")
    parser.add_argument("--output-dir", default="runs/false_negative_analysis")
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--min-conf", type=float, default=0.01)
    parser.add_argument("--positive-class", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--max-images", type=int, default=16)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = collect_validation_pairs(Path(args.images_dir), Path(args.labels_dir))
    model = YOLO(args.weights)

    print(f"Analyzing false negatives at threshold={args.threshold}")
    records = find_false_negatives(model, pairs, args.positive_class, args.threshold, args.min_conf, args.imgsz)
    summary = summarize_records(records)
    write_csv_outputs(records, output_dir)
    visual_rows = draw_false_negative_visuals(records, output_dir, args.max_images)
    plot_area_histogram(records, output_dir)
    write_html_report(output_dir, summary, args.threshold, visual_rows)

    print(
        f"FN images={summary['fn_images']} | FN boxes={summary['fn_boxes']} | "
        f"median area={summary['median_area']:.4f} | small boxes={summary['small_box_count']}"
    )
    print(f"Saved report: {output_dir / 'false_negative_report.html'}")
    print(f"Saved box table: {output_dir / 'false_negative_boxes.csv'}")


if __name__ == "__main__":
    main()
