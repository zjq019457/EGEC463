import argparse
import csv
from pathlib import Path


def best_detection_row(results_csv: Path) -> dict | None:
    if not results_csv.exists():
        return None
    with results_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None
    best = max(rows, key=lambda row: float(row["metrics/mAP50(B)"]))
    best_map50_95 = max(float(row["metrics/mAP50-95(B)"]) for row in rows)
    return {
        "experiment": results_csv.parent.as_posix(),
        "epoch": int(float(best["epoch"])),
        "precision": float(best["metrics/precision(B)"]),
        "recall": float(best["metrics/recall(B)"]),
        "map50": float(best["metrics/mAP50(B)"]),
        "map50_95": best_map50_95,
        "artifact_dir": results_csv.parent.as_posix(),
    }


def read_classification_strategy(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = []
    for strategy in ["detector_only", "classifier_only", "classifier_or_detector"]:
        candidates = [row for row in rows if row["strategy"] == strategy]
        if not candidates:
            continue
        best = max(
            candidates,
            key=lambda row: (float(row["f1"]), float(row["recall"]), float(row["specificity"])),
        )
        selected.append(
            {
                "strategy": strategy,
                "classifier_threshold": best["classifier_threshold"],
                "detector_threshold": best["detector_threshold"],
                "precision": float(best["precision"]),
                "recall": float(best["recall"]),
                "specificity": float(best["specificity"]),
                "f1": float(best["f1"]),
                "fn": int(best["fn"]),
                "fp": int(best["fp"]),
            }
        )
    return selected


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(output_dir: Path, detection_rows: list[dict], strategy_rows: list[dict]):
    lines = [
        "# Experiment Summary",
        "",
        "## Detection Experiments",
        "",
        "| Experiment | Best epoch | Precision | Recall | mAP50 | mAP50-95 | Artifacts |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in detection_rows:
        lines.append(
            f"| {row['experiment']} | {row['epoch']} | {row['precision']:.3f} | "
            f"{row['recall']:.3f} | {row['map50']:.3f} | {row['map50_95']:.3f} | "
            f"{row['artifact_dir']} |"
        )

    lines.extend(
        [
            "",
            "## Screening Strategies",
            "",
            "| Strategy | Classifier threshold | Detector threshold | Precision | Recall | Specificity | F1 | FN | FP |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in strategy_rows:
        lines.append(
            f"| {row['strategy']} | {row['classifier_threshold']} | {row['detector_threshold']} | "
            f"{row['precision']:.3f} | {row['recall']:.3f} | {row['specificity']:.3f} | "
            f"{row['f1']:.3f} | {row['fn']} | {row['fp']} |"
        )

    lines.extend(
        [
            "",
            "## Existing Ultralytics Evaluation Artifacts",
            "",
            "- `PR_curve.png`: precision-recall curve",
            "- `confusion_matrix.png`: confusion matrix",
            "- `confusion_matrix_normalized.png`: normalized confusion matrix",
            "- `results.csv`: per-epoch precision, recall, mAP50, and mAP50-95",
            "",
        "Note: full nano/small/medium 640px comparisons are complete. Multi-run stability and cross-validation still require additional runs.",
        ]
    )
    (output_dir / "experiment_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Summarize completed detector and screening experiments.")
    parser.add_argument("--output-dir", default="runs/experiment_summary")
    parser.add_argument(
        "--detection-results",
        nargs="*",
        default=[
            "runs/detect/train/results.csv",
            "runs/detect/train2/results.csv",
            "runs/experiments/medical_focus_small_640/results.csv",
        ],
    )
    parser.add_argument(
        "--strategy-metrics",
        default="runs/classification_assist_nano_5e/strategy_metrics.csv",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    detection_rows = [
        row
        for row in (best_detection_row(Path(path)) for path in args.detection_results)
        if row is not None
    ]
    strategy_rows = read_classification_strategy(Path(args.strategy_metrics))

    write_csv(
        output_dir / "detection_experiment_summary.csv",
        detection_rows,
        ["experiment", "epoch", "precision", "recall", "map50", "map50_95", "artifact_dir"],
    )
    write_csv(
        output_dir / "screening_strategy_summary.csv",
        strategy_rows,
        [
            "strategy",
            "classifier_threshold",
            "detector_threshold",
            "precision",
            "recall",
            "specificity",
            "f1",
            "fn",
            "fp",
        ],
    )
    write_markdown(output_dir, detection_rows, strategy_rows)
    print(f"Saved experiment summary to: {output_dir}")


if __name__ == "__main__":
    main()
