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
class ImageScore:
    image_path: Path
    label_path: Path
    actual_positive: bool
    max_positive_conf: float
    max_any_conf: float
    top_class: int | None
    top_conf: float


def parse_thresholds(raw: str) -> list[float]:
    thresholds = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            thresholds.append(float(item))
    if not thresholds:
        raise ValueError("At least one threshold is required.")
    return sorted(set(thresholds))


def read_label_classes(label_path: Path) -> list[int]:
    if not label_path.exists():
        return []

    classes = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if parts:
            classes.append(int(float(parts[0])))
    return classes


def collect_validation_images(images_dir: Path, labels_dir: Path) -> list[tuple[Path, Path]]:
    pairs = []
    for image_path in sorted(images_dir.glob("*")):
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label_path = labels_dir / f"{image_path.stem}.txt"
        pairs.append((image_path, label_path))
    return pairs


def score_images(model: YOLO, pairs: list[tuple[Path, Path]], positive_class: int, min_conf: float, imgsz: int) -> list[ImageScore]:
    scores = []
    for index, (image_path, label_path) in enumerate(pairs, start=1):
        labels = read_label_classes(label_path)
        actual_positive = positive_class in labels

        result = model(str(image_path), conf=min_conf, imgsz=imgsz, verbose=False)[0]
        boxes = result.boxes
        max_positive_conf = 0.0
        max_any_conf = 0.0
        top_class = None

        if boxes is not None and len(boxes):
            confs = boxes.conf.tolist()
            classes = [int(value) for value in boxes.cls.tolist()]
            for cls_id, conf in zip(classes, confs):
                if conf > max_any_conf:
                    max_any_conf = float(conf)
                    top_class = cls_id
                if cls_id == positive_class and conf > max_positive_conf:
                    max_positive_conf = float(conf)

        scores.append(
            ImageScore(
                image_path=image_path,
                label_path=label_path,
                actual_positive=actual_positive,
                max_positive_conf=max_positive_conf,
                max_any_conf=max_any_conf,
                top_class=top_class,
                top_conf=max_any_conf,
            )
        )

        if index % 50 == 0 or index == len(pairs):
            print(f"Scored {index}/{len(pairs)} images")

    return scores


def compute_threshold_metrics(scores: list[ImageScore], thresholds: list[float]) -> list[dict]:
    rows = []
    for threshold in thresholds:
        tp = fp = tn = fn = 0
        for score in scores:
            predicted_positive = score.max_positive_conf >= threshold
            if score.actual_positive and predicted_positive:
                tp += 1
            elif score.actual_positive and not predicted_positive:
                fn += 1
            elif not score.actual_positive and predicted_positive:
                fp += 1
            else:
                tn += 1

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "threshold": threshold,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "specificity": specificity,
                "f1": f1,
            }
        )
    return rows


def choose_threshold(metrics: list[dict]) -> dict:
    return max(metrics, key=lambda row: (row["f1"], row["recall"], row["specificity"]))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_metric_curves(metrics: list[dict], output_path: Path):
    thresholds = [row["threshold"] for row in metrics]
    plt.figure(figsize=(8, 5))
    for key in ["precision", "recall", "specificity", "f1"]:
        plt.plot(thresholds, [row[key] for row in metrics], marker="o", label=key)
    plt.xlabel("Positive confidence threshold")
    plt.ylabel("Score")
    plt.ylim(0, 1.02)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_confusion_matrix(row: dict, output_path: Path):
    matrix = [[row["tn"], row["fp"]], [row["fn"], row["tp"]]]
    labels = [["TN", "FP"], ["FN", "TP"]]

    plt.figure(figsize=(5, 4))
    plt.imshow(matrix, cmap="Blues")
    plt.xticks([0, 1], ["Pred negative", "Pred positive"])
    plt.yticks([0, 1], ["Actual negative", "Actual positive"])
    plt.title(f"Threshold {row['threshold']:.2f}")

    max_value = max(max(values) for values in matrix) or 1
    for y, values in enumerate(matrix):
        for x, value in enumerate(values):
            color = "white" if value > max_value / 2 else "black"
            plt.text(x, y, f"{labels[y][x]}\n{value}", ha="center", va="center", color=color)

    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def classify_cases(scores: list[ImageScore], threshold: float) -> dict[str, list[ImageScore]]:
    cases = {"tp": [], "fp": [], "tn": [], "fn": []}
    for score in scores:
        predicted_positive = score.max_positive_conf >= threshold
        if score.actual_positive and predicted_positive:
            cases["tp"].append(score)
        elif score.actual_positive:
            cases["fn"].append(score)
        elif predicted_positive:
            cases["fp"].append(score)
        else:
            cases["tn"].append(score)

    cases["fp"].sort(key=lambda item: item.max_positive_conf, reverse=True)
    cases["fn"].sort(key=lambda item: item.max_positive_conf, reverse=True)
    cases["tp"].sort(key=lambda item: item.max_positive_conf, reverse=True)
    cases["tn"].sort(key=lambda item: item.max_positive_conf)
    return cases


def save_case_visuals(
    model: YOLO,
    cases: dict[str, list[ImageScore]],
    threshold: float,
    output_dir: Path,
    imgsz: int,
    max_per_group: int,
) -> list[dict]:
    visuals_dir = output_dir / "case_visuals"
    visuals_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for case_name in ["fp", "fn", "tp", "tn"]:
        for score in cases[case_name][:max_per_group]:
            result = model(str(score.image_path), conf=threshold, imgsz=imgsz, verbose=False)[0]
            plotted = result.plot(conf=True, labels=True)
            output_path = visuals_dir / f"{case_name}_{score.image_path.stem}.jpg"
            cv2.imwrite(str(output_path), plotted)
            rows.append(
                {
                    "case": case_name,
                    "image": str(score.image_path),
                    "visual": str(output_path),
                    "actual_positive": score.actual_positive,
                    "max_positive_conf": score.max_positive_conf,
                    "top_class": "" if score.top_class is None else score.top_class,
                    "top_conf": score.top_conf,
                }
            )

    return rows


def write_html_report(output_dir: Path, summary: dict, metrics: list[dict], visual_rows: list[dict]):
    report_path = output_dir / "screening_report.html"
    metric_rows = "\n".join(
        "<tr>"
        f"<td>{row['threshold']:.2f}</td>"
        f"<td>{row['precision']:.3f}</td>"
        f"<td>{row['recall']:.3f}</td>"
        f"<td>{row['specificity']:.3f}</td>"
        f"<td>{row['f1']:.3f}</td>"
        f"<td>{row['tp']}</td><td>{row['fp']}</td><td>{row['tn']}</td><td>{row['fn']}</td>"
        "</tr>"
        for row in metrics
    )
    visual_cards = "\n".join(
        "<figure>"
        f"<img src='{html.escape(Path(row['visual']).relative_to(output_dir).as_posix())}' alt='{html.escape(row['case'])}'>"
        f"<figcaption>{html.escape(row['case'].upper())}: {html.escape(Path(row['image']).name)} "
        f"(positive_conf={row['max_positive_conf']:.3f})</figcaption>"
        "</figure>"
        for row in visual_rows
    )

    report_path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Brain Tumor Screening Report</title>
  <style>
    body {{ font-family: sans-serif; margin: 32px; color: #172026; background: #f7f7f4; }}
    h1, h2 {{ margin-bottom: 8px; }}
    .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin: 20px 0; }}
    .metric {{ background: white; border: 1px solid #ddd; padding: 12px; border-radius: 6px; }}
    .metric strong {{ display: block; font-size: 24px; margin-top: 4px; }}
    table {{ border-collapse: collapse; width: 100%; background: white; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    img {{ max-width: 100%; border: 1px solid #ddd; background: white; }}
    .charts {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; margin: 20px 0; }}
    .cases {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; }}
    figure {{ margin: 0; background: white; border: 1px solid #ddd; padding: 10px; border-radius: 6px; }}
    figcaption {{ font-size: 13px; margin-top: 8px; }}
  </style>
</head>
<body>
  <h1>Brain Tumor Screening Report</h1>
  <p>Image-level positive screening from YOLO detections. Positive means at least one class-1 detection passes the threshold.</p>
  <section class="summary">
    <div class="metric">Recommended threshold<strong>{summary['threshold']:.2f}</strong></div>
    <div class="metric">Precision<strong>{summary['precision']:.3f}</strong></div>
    <div class="metric">Recall<strong>{summary['recall']:.3f}</strong></div>
    <div class="metric">F1<strong>{summary['f1']:.3f}</strong></div>
  </section>
  <h2>Curves</h2>
  <div class="charts">
    <img src="threshold_curves.png" alt="Threshold curves">
    <img src="confusion_matrix.png" alt="Confusion matrix">
  </div>
  <h2>Threshold Sweep</h2>
  <table>
    <thead>
      <tr><th>Threshold</th><th>Precision</th><th>Recall</th><th>Specificity</th><th>F1</th><th>TP</th><th>FP</th><th>TN</th><th>FN</th></tr>
    </thead>
    <tbody>{metric_rows}</tbody>
  </table>
  <h2>Selected Cases</h2>
  <div class="cases">{visual_cards}</div>
</body>
</html>
""",
        encoding="utf-8",
    )
    return report_path


def main():
    parser = argparse.ArgumentParser(description="Build an image-level screening report from YOLO predictions.")
    parser.add_argument("--weights", default="runs/detect/train/weights/best.pt", help="Path to YOLO weights.")
    parser.add_argument("--images-dir", default="datasets/brain-tumor/images/val", help="Validation image directory.")
    parser.add_argument("--labels-dir", default="datasets/brain-tumor/labels/val", help="Validation label directory.")
    parser.add_argument("--output-dir", default="runs/screening_report", help="Where to save report outputs.")
    parser.add_argument("--positive-class", type=int, default=1, help="Class ID treated as positive/tumor.")
    parser.add_argument("--imgsz", type=int, default=320, help="Inference image size.")
    parser.add_argument(
        "--thresholds",
        default="0.01,0.03,0.05,0.08,0.10,0.15,0.20,0.25,0.30,0.40,0.50",
        help="Comma-separated confidence thresholds to evaluate.",
    )
    parser.add_argument("--max-per-group", type=int, default=3, help="Visual examples per TP/FP/TN/FN group.")
    args = parser.parse_args()

    weights = Path(args.weights)
    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    thresholds = parse_thresholds(args.thresholds)
    pairs = collect_validation_images(images_dir, labels_dir)
    if not pairs:
        raise SystemExit(f"No validation images found in {images_dir}")

    print(f"Loading model: {weights}")
    print(f"Scoring {len(pairs)} images")
    model = YOLO(weights)
    scores = score_images(model, pairs, args.positive_class, min(thresholds), args.imgsz)

    metrics = compute_threshold_metrics(scores, thresholds)
    recommended = choose_threshold(metrics)
    cases = classify_cases(scores, recommended["threshold"])

    score_rows = [
        {
            "image": str(score.image_path),
            "label": str(score.label_path),
            "actual_positive": score.actual_positive,
            "max_positive_conf": score.max_positive_conf,
            "max_any_conf": score.max_any_conf,
            "top_class": "" if score.top_class is None else score.top_class,
            "top_conf": score.top_conf,
        }
        for score in scores
    ]

    write_csv(
        output_dir / "image_scores.csv",
        score_rows,
        ["image", "label", "actual_positive", "max_positive_conf", "max_any_conf", "top_class", "top_conf"],
    )
    write_csv(
        output_dir / "threshold_metrics.csv",
        metrics,
        ["threshold", "tp", "fp", "tn", "fn", "precision", "recall", "specificity", "f1"],
    )
    plot_metric_curves(metrics, output_dir / "threshold_curves.png")
    plot_confusion_matrix(recommended, output_dir / "confusion_matrix.png")
    visual_rows = save_case_visuals(model, cases, recommended["threshold"], output_dir, args.imgsz, args.max_per_group)
    write_csv(
        output_dir / "selected_cases.csv",
        visual_rows,
        ["case", "image", "visual", "actual_positive", "max_positive_conf", "top_class", "top_conf"],
    )
    report_path = write_html_report(output_dir, recommended, metrics, visual_rows)

    print(
        "Recommended threshold "
        f"{recommended['threshold']:.2f}: precision={recommended['precision']:.3f}, "
        f"recall={recommended['recall']:.3f}, specificity={recommended['specificity']:.3f}, "
        f"f1={recommended['f1']:.3f}"
    )
    print(f"Saved report: {report_path}")
    print(f"Saved metrics: {output_dir / 'threshold_metrics.csv'}")
    print(f"Saved per-image scores: {output_dir / 'image_scores.csv'}")


if __name__ == "__main__":
    main()
