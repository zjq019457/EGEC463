import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def read_yolo_labels(path: Path) -> list[dict]:
    if not path.exists():
        return []

    boxes = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        parts = line.split()
        if len(parts) != 5:
            boxes.append({"invalid": True, "line": line_number, "raw": line})
            continue
        cls_id, cx, cy, width, height = parts
        boxes.append(
            {
                "invalid": False,
                "class_id": int(float(cls_id)),
                "cx": float(cx),
                "cy": float(cy),
                "width": float(width),
                "height": float(height),
                "area": float(width) * float(height),
            }
        )
    return boxes


def collect_split(root: Path, split: str) -> tuple[list[dict], list[dict]]:
    images_dir = root / "images" / split
    labels_dir = root / "labels" / split
    image_rows = []
    box_rows = []

    for image_path in sorted(images_dir.glob("*")):
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue

        label_path = labels_dir / f"{image_path.stem}.txt"
        image = cv2.imread(str(image_path))
        readable = image is not None
        height = image.shape[0] if readable else 0
        width = image.shape[1] if readable else 0
        boxes = read_yolo_labels(label_path)
        valid_boxes = [box for box in boxes if not box.get("invalid")]
        invalid_boxes = [box for box in boxes if box.get("invalid")]

        classes = sorted({box["class_id"] for box in valid_boxes})
        image_rows.append(
            {
                "split": split,
                "image": str(image_path),
                "label": str(label_path),
                "readable": readable,
                "width": width,
                "height": height,
                "has_label": label_path.exists(),
                "box_count": len(valid_boxes),
                "invalid_label_lines": len(invalid_boxes),
                "classes": ",".join(str(cls_id) for cls_id in classes),
                "image_level_class": "positive" if 1 in classes else "negative",
            }
        )

        for box in valid_boxes:
            row = {key: value for key, value in box.items() if key != "invalid"}
            row.update({"split": split, "image": str(image_path), "label": str(label_path)})
            box_rows.append(row)

    return image_rows, box_rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(image_rows: list[dict], box_rows: list[dict]) -> list[dict]:
    by_split_images = defaultdict(list)
    by_split_boxes = defaultdict(list)
    for row in image_rows:
        by_split_images[row["split"]].append(row)
    for row in box_rows:
        by_split_boxes[row["split"]].append(row)

    summaries = []
    for split in sorted(by_split_images):
        images = by_split_images[split]
        boxes = by_split_boxes[split]
        image_level_counts = Counter(row["image_level_class"] for row in images)
        box_class_counts = Counter(row["class_id"] for row in boxes)
        areas = [row["area"] for row in boxes]
        small_boxes = sum(1 for area in areas if area < 0.02)
        summaries.append(
            {
                "split": split,
                "images": len(images),
                "readable_images": sum(1 for row in images if row["readable"]),
                "missing_labels": sum(1 for row in images if not row["has_label"]),
                "invalid_label_lines": sum(row["invalid_label_lines"] for row in images),
                "image_negative": image_level_counts["negative"],
                "image_positive": image_level_counts["positive"],
                "box_negative": box_class_counts[0],
                "box_positive": box_class_counts[1],
                "boxes": len(boxes),
                "mean_box_area": sum(areas) / len(areas) if areas else 0.0,
                "small_boxes_lt_2pct": small_boxes,
            }
        )
    return summaries


def plot_class_distribution(summary_rows: list[dict], output_dir: Path):
    splits = [row["split"] for row in summary_rows]
    negative = [row["image_negative"] for row in summary_rows]
    positive = [row["image_positive"] for row in summary_rows]

    x_positions = range(len(splits))
    plt.figure(figsize=(7, 4))
    plt.bar(x_positions, negative, label="negative")
    plt.bar(x_positions, positive, bottom=negative, label="positive")
    plt.xticks(list(x_positions), splits)
    plt.ylabel("Images")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "image_class_distribution.png", dpi=160)
    plt.close()


def plot_box_area(box_rows: list[dict], output_dir: Path):
    areas = [row["area"] for row in box_rows]
    plt.figure(figsize=(7, 4))
    plt.hist(areas, bins=24, color="#3d6fb6", edgecolor="white")
    plt.xlabel("Normalized box area")
    plt.ylabel("Boxes")
    plt.tight_layout()
    plt.savefig(output_dir / "box_area_histogram.png", dpi=160)
    plt.close()


def write_markdown(path: Path, summary_rows: list[dict]):
    lines = [
        "# Data Audit Summary",
        "",
        "| Split | Images | Positive images | Negative images | Boxes | Positive boxes | Small boxes <2% | Missing labels | Invalid label lines |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['split']} | {row['images']} | {row['image_positive']} | {row['image_negative']} | "
            f"{row['boxes']} | {row['box_positive']} | {row['small_boxes_lt_2pct']} | "
            f"{row['missing_labels']} | {row['invalid_label_lines']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Audit image and YOLO label quality/statistics.")
    parser.add_argument("--dataset-root", default="datasets/brain-tumor")
    parser.add_argument("--output-dir", default="runs/data_audit")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_rows = []
    box_rows = []
    for split in ["train", "val"]:
        split_images, split_boxes = collect_split(dataset_root, split)
        image_rows.extend(split_images)
        box_rows.extend(split_boxes)

    summary_rows = summarize(image_rows, box_rows)
    write_csv(
        output_dir / "image_audit.csv",
        image_rows,
        [
            "split",
            "image",
            "label",
            "readable",
            "width",
            "height",
            "has_label",
            "box_count",
            "invalid_label_lines",
            "classes",
            "image_level_class",
        ],
    )
    write_csv(
        output_dir / "box_audit.csv",
        box_rows,
        ["split", "image", "label", "class_id", "cx", "cy", "width", "height", "area"],
    )
    write_csv(
        output_dir / "data_audit_summary.csv",
        summary_rows,
        [
            "split",
            "images",
            "readable_images",
            "missing_labels",
            "invalid_label_lines",
            "image_negative",
            "image_positive",
            "box_negative",
            "box_positive",
            "boxes",
            "mean_box_area",
            "small_boxes_lt_2pct",
        ],
    )
    write_markdown(output_dir / "data_audit_summary.md", summary_rows)
    plot_class_distribution(summary_rows, output_dir)
    plot_box_area(box_rows, output_dir)

    for row in summary_rows:
        print(
            f"{row['split']}: images={row['images']} positive={row['image_positive']} "
            f"negative={row['image_negative']} boxes={row['boxes']} small_boxes={row['small_boxes_lt_2pct']}"
        )
    print(f"Saved audit to: {output_dir}")


if __name__ == "__main__":
    main()
