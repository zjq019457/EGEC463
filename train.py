import argparse
import csv
import locale
from copy import deepcopy
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import matplotlib
import torch
from ultralytics import YOLO

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EXPERIMENTS = {
    "baseline": {
        "description": "Original lightweight baseline: YOLO11n + 320px + stronger augmentations.",
        "model_size": "nano",
        "epochs": 10,
        "batch_size": 16,
        "image_size": 320,
        "prediction_conf": 0.25,
        "project": "runs/experiments",
        "run_name": "baseline_nano_320",
        "augmentation": {
            "scale": 0.9,
            "mosaic": 0.9,
            "mixup": 0.2,
            "copy_paste": 0.4,
            "rotation": 10,
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
        },
    },
    "medical_focus": {
        "description": "Medical-image variant: higher resolution with gentler augmentations and a slightly larger model.",
        "model_size": "small",
        "epochs": 5,
        "batch_size": 8,
        "image_size": 640,
        "prediction_conf": 0.05,
        "project": "runs/experiments",
        "run_name": "medical_focus_small_640",
        "augmentation": {
            "scale": 0.2,
            "mosaic": 0.1,
            "mixup": 0.0,
            "copy_paste": 0.0,
            "rotation": 5,
            "hsv_h": 0.0,
            "hsv_s": 0.0,
            "hsv_v": 0.0,
        },
    },
}


def build_experiment_config(name: str, epoch_override: int | None = None) -> dict:
    if name not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {name}")

    config = deepcopy(EXPERIMENTS[name])
    if epoch_override is not None:
        config["epochs"] = epoch_override
        config["run_name"] = f"{config['run_name']}_{epoch_override}e"
    return config


def summarize_results(results_csv: Path) -> dict | None:
    if not results_csv.exists():
        return None

    with results_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        return None

    best_row = max(rows, key=lambda row: float(row["metrics/mAP50(B)"]))
    return {
        "epoch": int(float(best_row["epoch"])),
        "precision": float(best_row["metrics/precision(B)"]),
        "recall": float(best_row["metrics/recall(B)"]),
        "map50": float(best_row["metrics/mAP50(B)"]),
        "map50_95": float(best_row["metrics/mAP50-95(B)"]),
    }


class BrainTumorDetector:
    def __init__(self, config: dict):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        locale.getpreferredencoding = lambda: "UTF-8"
        self.model_weights = {
            "nano": "yolo11n.pt",
            "small": "yolo11s.pt",
            "medium": "yolo11m.pt",
            "large": "yolo11l.pt",
            "xlarge": "yolo11x.pt",
        }

    def load_data(self):
        """Download the dataset config if it is not present yet."""
        self.data_path = Path("brain-tumor.yaml")
        if not self.data_path.exists():
            urlretrieve(
                "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/datasets/brain-tumor.yaml",
                self.data_path,
            )

    def train_model(self):
        """Train the selected YOLO variant for the active experiment."""
        model = YOLO(self.model_weights[self.config["model_size"]])
        training_args = {
            "data": str(self.data_path),
            "epochs": self.config["epochs"],
            "batch": self.config["batch_size"],
            "imgsz": self.config["image_size"],
            "device": self.device,
            "project": self.config["project"],
            "name": self.config["run_name"],
            "scale": self.config["augmentation"]["scale"],
            "mosaic": self.config["augmentation"]["mosaic"],
            "mixup": self.config["augmentation"]["mixup"],
            "copy_paste": self.config["augmentation"]["copy_paste"],
            "degrees": self.config["augmentation"]["rotation"],
            "hsv_h": self.config["augmentation"]["hsv_h"],
            "hsv_s": self.config["augmentation"]["hsv_s"],
            "hsv_v": self.config["augmentation"]["hsv_v"],
        }

        results = model.train(**training_args)
        save_dir = Path(getattr(results, "save_dir", Path(training_args["project"]) / training_args["name"]))
        if hasattr(model, "trainer") and getattr(model.trainer, "save_dir", None):
            save_dir = Path(model.trainer.save_dir)
        return model, results, save_dir

    def get_validation_pairs(self, limit=3):
        """Collect validation image/label pairs from the downloaded dataset."""
        images_dir = Path("datasets/brain-tumor/images/val")
        labels_dir = Path("datasets/brain-tumor/labels/val")

        if not images_dir.exists() or not labels_dir.exists():
            return []

        pairs = []
        for img_path in sorted(images_dir.glob("*")):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue

            label_path = labels_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                pairs.append((img_path, label_path))
            if len(pairs) >= limit:
                break

        return pairs

    def evaluate_model(self, model_path: Path, validation_images, output_dir: Path):
        """Save side-by-side prediction images and print per-sample box scores."""
        model = YOLO(model_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        for img_path, label_path in validation_images:
            img_path = Path(img_path)
            label_path = Path(label_path)

            if not img_path.exists():
                print(f"Skipping missing image: {img_path}")
                continue

            image = cv2.imread(str(img_path))
            if image is None:
                print(f"Skipping unreadable image: {img_path}")
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = model(str(img_path), conf=self.config["prediction_conf"], verbose=False)
            boxes = results[0].boxes

            box_count = 0 if boxes is None else len(boxes)
            print(f"Sample: {img_path.name} | boxes={box_count} | label={label_path.name}")
            if boxes is not None and len(boxes):
                print(f"  confidences={boxes.conf.tolist()}")
                print(f"  classes={boxes.cls.tolist()}")

            output_path = output_dir / f"{img_path.stem}_comparison.png"
            plotted = results[0].plot(conf=True, labels=True)
            self._plot_comparison(image, cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB), str(img_path), output_path)

    def _plot_comparison(self, original, prediction, title, output_path: Path):
        """Save original vs prediction visualization."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(original)
        ax1.set_title("Original Image")
        ax1.axis("off")

        ax2.imshow(prediction)
        ax2.set_title("Model Prediction")
        ax2.axis("off")

        plt.suptitle(title)
        plt.tight_layout()
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
        plt.close(fig)


def print_summary(label: str, summary: dict | None):
    if not summary:
        print(f"{label}: no summary available")
        return

    print(
        f"{label}: best epoch={summary['epoch']} | "
        f"precision={summary['precision']:.4f} | "
        f"recall={summary['recall']:.4f} | "
        f"mAP50={summary['map50']:.4f} | "
        f"mAP50-95={summary['map50_95']:.4f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate brain tumor YOLO experiments.")
    parser.add_argument(
        "--experiment",
        default="medical_focus",
        choices=sorted(EXPERIMENTS.keys()),
        help="Experiment recipe to run.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Optional epoch override.")
    parser.add_argument("--samples", type=int, default=3, help="How many validation samples to visualize.")
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training and only summarize/evaluate an existing weights file.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to an existing weights file. Useful with --skip-train.",
    )
    args = parser.parse_args()

    config = build_experiment_config(args.experiment, args.epochs)
    print(f"Running experiment: {args.experiment}")
    print(f"Description: {config['description']}")
    print(
        f"Setup: model={config['model_size']} | epochs={config['epochs']} | "
        f"batch={config['batch_size']} | imgsz={config['image_size']} | device=auto"
    )

    detector = BrainTumorDetector(config)
    detector.load_data()
    if args.skip_train:
        if args.weights:
            best_model_path = Path(args.weights)
            save_dir = best_model_path.parent.parent
        else:
            save_dir = Path(config["project"]) / config["run_name"]
            best_model_path = save_dir / "weights" / "best.pt"
    else:
        _, _, save_dir = detector.train_model()
        best_model_path = save_dir / "weights" / "best.pt"

    current_summary = summarize_results(save_dir / "results.csv")
    if current_summary:
        print_summary("Current experiment", current_summary)

    baseline_summary = summarize_results(Path("runs/detect/train/results.csv"))
    if args.experiment != "baseline" and baseline_summary:
        print_summary("Existing baseline", baseline_summary)

    validation_images = detector.get_validation_pairs(limit=args.samples)
    if not validation_images:
        print("No validation image/label pairs found. Skipping post-training visualization.")
        return

    if not best_model_path.exists():
        print(f"Best model not found at {best_model_path}. Skipping post-training visualization.")
        return

    output_dir = save_dir / "sample_predictions"
    detector.evaluate_model(best_model_path, validation_images, output_dir)
    print(f"Saved comparison images to: {output_dir}")


if __name__ == "__main__":
    main()
