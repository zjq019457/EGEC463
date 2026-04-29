import argparse
import csv
import html
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import torch
from ultralytics import YOLO

matplotlib.use("Agg")
import matplotlib.pyplot as plt


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class SampleScore:
    image_path: Path
    actual_positive: bool
    classifier_positive_prob: float
    detector_positive_conf: float


def read_label_classes(label_path: Path) -> list[int]:
    if not label_path.exists():
        return []

    classes = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if parts:
            classes.append(int(float(parts[0])))
    return classes


def label_name(label_path: Path, positive_class: int) -> str:
    classes = read_label_classes(label_path)
    return "positive" if positive_class in classes else "negative"


def link_or_skip(source: Path, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        return
    try:
        destination.symlink_to(source.resolve())
    except OSError:
        destination.write_bytes(source.read_bytes())


def prepare_classification_dataset(source_root: Path, output_root: Path, positive_class: int) -> dict:
    counts = {}
    for split in ["train", "val"]:
        split_counts = {"negative": 0, "positive": 0}
        images_dir = source_root / "images" / split
        labels_dir = source_root / "labels" / split

        for image_path in sorted(images_dir.glob("*")):
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            class_name = label_name(labels_dir / f"{image_path.stem}.txt", positive_class)
            destination = output_root / split / class_name / image_path.name
            link_or_skip(image_path, destination)
            split_counts[class_name] += 1

        counts[split] = split_counts

    return counts


def train_classifier(data_dir: Path, model_name: str, epochs: int, imgsz: int, batch: int, project: str, run_name: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(model_name)
    results = model.train(
        data=str(data_dir),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=run_name,
        workers=0,
    )
    save_dir = Path(getattr(results, "save_dir", Path(project) / run_name))
    if hasattr(model, "trainer") and getattr(model.trainer, "save_dir", None):
        save_dir = Path(model.trainer.save_dir)
    return save_dir / "weights" / "best.pt", save_dir


def collect_validation_pairs(source_root: Path) -> list[tuple[Path, Path]]:
    pairs = []
    images_dir = source_root / "images" / "val"
    labels_dir = source_root / "labels" / "val"
    for image_path in sorted(images_dir.glob("*")):
        if image_path.suffix.lower() in IMAGE_SUFFIXES:
            pairs.append((image_path, labels_dir / f"{image_path.stem}.txt"))
    return pairs


def classifier_positive_prob(model: YOLO, image_path: Path, imgsz: int) -> float:
    result = model(str(image_path), imgsz=imgsz, verbose=False)[0]
    names = result.names
    positive_index = None
    for index, name in names.items():
        if name == "positive":
            positive_index = int(index)
            break
    if positive_index is None:
        raise ValueError(f"Classifier class names do not include 'positive': {names}")
    return float(result.probs.data[positive_index])


def detector_positive_conf(model: YOLO, image_path: Path, positive_class: int, min_conf: float, imgsz: int) -> float:
    result = model(str(image_path), conf=min_conf, imgsz=imgsz, verbose=False)[0]
    boxes = result.boxes
    if boxes is None or not len(boxes):
        return 0.0

    best = 0.0
    for cls_id, conf in zip(boxes.cls.tolist(), boxes.conf.tolist()):
        if int(cls_id) == positive_class and conf > best:
            best = float(conf)
    return best


def score_validation_set(
    classifier_weights: Path,
    detector_weights: Path,
    source_root: Path,
    positive_class: int,
    classifier_imgsz: int,
    detector_imgsz: int,
    detector_min_conf: float,
) -> list[SampleScore]:
    classifier = YOLO(classifier_weights)
    detector = YOLO(detector_weights)
    pairs = collect_validation_pairs(source_root)
    scores = []

    for index, (image_path, label_path) in enumerate(pairs, start=1):
        actual_positive = label_name(label_path, positive_class) == "positive"
        cls_prob = classifier_positive_prob(classifier, image_path, classifier_imgsz)
        det_conf = detector_positive_conf(detector, image_path, positive_class, detector_min_conf, detector_imgsz)
        scores.append(SampleScore(image_path, actual_positive, cls_prob, det_conf))
        if index % 50 == 0 or index == len(pairs):
            print(f"Scored {index}/{len(pairs)} validation images")

    return scores


def confusion_from_predictions(actual: list[bool], predicted: list[bool]) -> dict:
    tp = fp = tn = fn = 0
    for actual_positive, predicted_positive in zip(actual, predicted):
        if actual_positive and predicted_positive:
            tp += 1
        elif actual_positive:
            fn += 1
        elif predicted_positive:
            fp += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
    }


def evaluate_strategies(scores: list[SampleScore], classifier_thresholds: list[float], detector_threshold: float) -> list[dict]:
    actual = [score.actual_positive for score in scores]
    rows = []

    detector_pred = [score.detector_positive_conf >= detector_threshold for score in scores]
    detector_metrics = confusion_from_predictions(actual, detector_pred)
    rows.append({"strategy": "detector_only", "classifier_threshold": "", "detector_threshold": detector_threshold, **detector_metrics})

    for threshold in classifier_thresholds:
        classifier_pred = [score.classifier_positive_prob >= threshold for score in scores]
        classifier_metrics = confusion_from_predictions(actual, classifier_pred)
        rows.append(
            {
                "strategy": "classifier_only",
                "classifier_threshold": threshold,
                "detector_threshold": "",
                **classifier_metrics,
            }
        )

        ensemble_pred = [
            score.detector_positive_conf >= detector_threshold or score.classifier_positive_prob >= threshold
            for score in scores
        ]
        ensemble_metrics = confusion_from_predictions(actual, ensemble_pred)
        rows.append(
            {
                "strategy": "classifier_or_detector",
                "classifier_threshold": threshold,
                "detector_threshold": detector_threshold,
                **ensemble_metrics,
            }
        )

    return rows


def choose_best(rows: list[dict], strategy: str) -> dict:
    candidates = [row for row in rows if row["strategy"] == strategy]
    return max(candidates, key=lambda row: (row["f1"], row["recall"], row["specificity"]))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_strategy_comparison(rows: list[dict], output_path: Path):
    best_rows = [
        next(row for row in rows if row["strategy"] == "detector_only"),
        choose_best(rows, "classifier_only"),
        choose_best(rows, "classifier_or_detector"),
    ]
    labels = ["Detector", "Classifier", "Combined"]
    x_positions = range(len(labels))

    plt.figure(figsize=(8, 5))
    for metric in ["precision", "recall", "specificity", "f1"]:
        values = [row[metric] for row in best_rows]
        plt.plot(x_positions, values, marker="o", label=metric)
    plt.xticks(list(x_positions), labels)
    plt.ylim(0, 1.02)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def write_html_report(output_dir: Path, rows: list[dict], counts: dict, classifier_weights: Path, detector_weights: Path):
    detector = next(row for row in rows if row["strategy"] == "detector_only")
    classifier = choose_best(rows, "classifier_only")
    combined = choose_best(rows, "classifier_or_detector")
    selected = [detector, classifier, combined]

    cards = "\n".join(
        "<div class='metric'>"
        f"<span>{html.escape(row['strategy'])}</span>"
        f"<strong>F1 {row['f1']:.3f}</strong>"
        f"<small>P {row['precision']:.3f} | R {row['recall']:.3f} | S {row['specificity']:.3f} | FN {row['fn']}</small>"
        "</div>"
        for row in selected
    )
    table_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(str(row['strategy']))}</td>"
        f"<td>{row['classifier_threshold']}</td>"
        f"<td>{row['detector_threshold']}</td>"
        f"<td>{row['precision']:.3f}</td>"
        f"<td>{row['recall']:.3f}</td>"
        f"<td>{row['specificity']:.3f}</td>"
        f"<td>{row['f1']:.3f}</td>"
        f"<td>{row['tp']}</td><td>{row['fp']}</td><td>{row['tn']}</td><td>{row['fn']}</td>"
        "</tr>"
        for row in rows
    )

    (output_dir / "classification_assist_report.html").write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Classification Assist Report</title>
  <style>
    body {{ font-family: sans-serif; margin: 32px; color: #172026; background: #f7f7f4; }}
    .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; margin: 20px 0; }}
    .metric {{ background: white; border: 1px solid #ddd; border-radius: 6px; padding: 12px; }}
    .metric span {{ display: block; font-size: 13px; text-transform: uppercase; color: #57636f; }}
    .metric strong {{ display: block; font-size: 24px; margin: 6px 0; }}
    .metric small {{ display: block; line-height: 1.5; }}
    table {{ border-collapse: collapse; width: 100%; background: white; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    img {{ max-width: 100%; border: 1px solid #ddd; background: white; }}
  </style>
</head>
<body>
  <h1>Classification Assist Report</h1>
  <p>Image-level classifier trained from detection labels, then combined with the detector by OR logic.</p>
  <p>Classifier weights: {html.escape(str(classifier_weights))}<br>Detector weights: {html.escape(str(detector_weights))}</p>
  <p>Dataset counts: train positive={counts['train']['positive']}, train negative={counts['train']['negative']}, val positive={counts['val']['positive']}, val negative={counts['val']['negative']}</p>
  <section class="summary">{cards}</section>
  <h2>Comparison</h2>
  <img src="strategy_comparison.png" alt="Strategy comparison">
  <h2>All Thresholds</h2>
  <table>
    <thead>
      <tr><th>Strategy</th><th>Cls threshold</th><th>Det threshold</th><th>Precision</th><th>Recall</th><th>Specificity</th><th>F1</th><th>TP</th><th>FP</th><th>TN</th><th>FN</th></tr>
    </thead>
    <tbody>{table_rows}</tbody>
  </table>
</body>
</html>
""",
        encoding="utf-8",
    )


def parse_thresholds(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate a classifier-assisted screening model.")
    parser.add_argument("--source-root", default="datasets/brain-tumor")
    parser.add_argument("--classification-root", default="datasets/brain-tumor-cls")
    parser.add_argument("--positive-class", type=int, default=1)
    parser.add_argument("--classifier-base", default="yolo11n-cls.pt")
    parser.add_argument("--classifier-weights", default=None)
    parser.add_argument("--detector-weights", default="runs/detect/train/weights/best.pt")
    parser.add_argument("--output-dir", default="runs/classification_assist")
    parser.add_argument("--project", default="runs/classification")
    parser.add_argument("--run-name", default="brain_tumor_cls_nano")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--detector-imgsz", type=int, default=320)
    parser.add_argument("--detector-threshold", type=float, default=0.25)
    parser.add_argument("--detector-min-conf", type=float, default=0.01)
    parser.add_argument(
        "--classifier-thresholds",
        default="0.05,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90",
    )
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    source_root = Path(args.source_root)
    classification_root = Path(args.classification_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Preparing classification dataset")
    counts = prepare_classification_dataset(source_root, classification_root, args.positive_class)
    print(f"Dataset counts: {counts}")

    if args.skip_train:
        if not args.classifier_weights:
            raise SystemExit("--classifier-weights is required when --skip-train is used")
        classifier_weights = Path(args.classifier_weights)
        train_save_dir = classifier_weights.parent.parent
    else:
        print("Training classifier")
        classifier_weights, train_save_dir = train_classifier(
            classification_root,
            args.classifier_base,
            args.epochs,
            args.imgsz,
            args.batch,
            args.project,
            args.run_name,
        )

    print(f"Classifier weights: {classifier_weights}")
    print("Evaluating classifier + detector")
    scores = score_validation_set(
        classifier_weights,
        Path(args.detector_weights),
        source_root,
        args.positive_class,
        args.imgsz,
        args.detector_imgsz,
        args.detector_min_conf,
    )

    score_rows = [
        {
            "image": str(score.image_path),
            "actual_positive": score.actual_positive,
            "classifier_positive_prob": score.classifier_positive_prob,
            "detector_positive_conf": score.detector_positive_conf,
        }
        for score in scores
    ]
    write_csv(
        output_dir / "classification_scores.csv",
        score_rows,
        ["image", "actual_positive", "classifier_positive_prob", "detector_positive_conf"],
    )

    rows = evaluate_strategies(scores, parse_thresholds(args.classifier_thresholds), args.detector_threshold)
    write_csv(
        output_dir / "strategy_metrics.csv",
        rows,
        [
            "strategy",
            "classifier_threshold",
            "detector_threshold",
            "tp",
            "fp",
            "tn",
            "fn",
            "precision",
            "recall",
            "specificity",
            "f1",
        ],
    )
    plot_strategy_comparison(rows, output_dir / "strategy_comparison.png")
    write_html_report(output_dir, rows, counts, classifier_weights, Path(args.detector_weights))

    detector = next(row for row in rows if row["strategy"] == "detector_only")
    classifier = choose_best(rows, "classifier_only")
    combined = choose_best(rows, "classifier_or_detector")
    print(
        f"Detector only: F1={detector['f1']:.3f}, recall={detector['recall']:.3f}, "
        f"specificity={detector['specificity']:.3f}, FN={detector['fn']}"
    )
    print(
        f"Best classifier: threshold={classifier['classifier_threshold']}, F1={classifier['f1']:.3f}, "
        f"recall={classifier['recall']:.3f}, specificity={classifier['specificity']:.3f}, FN={classifier['fn']}"
    )
    print(
        f"Best combined: cls_threshold={combined['classifier_threshold']}, F1={combined['f1']:.3f}, "
        f"recall={combined['recall']:.3f}, specificity={combined['specificity']:.3f}, FN={combined['fn']}"
    )
    print(f"Saved report: {output_dir / 'classification_assist_report.html'}")
    print(f"Training outputs: {train_save_dir}")


if __name__ == "__main__":
    main()
