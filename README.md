# Brain Tumor Detection using YOLOv11

This project implements a brain tumor detection system using YOLOv11 object detection models. It provides a complete pipeline for training and evaluating models on medical imaging data.

## Method Documentation

- English method README: [`METHOD_README.md`](./METHOD_README.md)
- Chinese method README: [`METHOD_README.zh-CN.md`](./METHOD_README.zh-CN.md)

## Latest Results

The strongest detector-only model is `medical_nano_640`, which improves mAP50 from `0.508` to `0.540` and mAP50-95 from `0.284` to `0.373` compared with the original YOLO11n 320px baseline.

For screening, combining the classifier with the best detector reduces false negatives from `48` to `2` and improves recall from `0.407` to `0.975`, with the tradeoff of more false positives.

## Reproducibility Shortcuts

After activating the environment, common workflows can be run with `make` or `run.sh`:

```bash
make audit
make screening
make fn
make heatmap
make summary
```

Or:

```bash
./run.sh all
```

## Features

- Support for multiple YOLO model architectures (nano to xlarge)
- Configurable data augmentation pipeline
- Training progress visualization
- Model evaluation tools
- Validation on medical images

## Conda Environment

This repository now includes an [`environment.yml`](./environment.yml) file for a CPU-ready setup.

### 1. Create the environment

```bash
conda env create -f environment.yml
conda activate brain-tumor-yolo
```

If you prefer `mamba`, you can use:

```bash
mamba env create -f environment.yml
mamba activate brain-tumor-yolo
```

### 2. Verify the installation

```bash
python -c "import torch, ultralytics, cv2, yaml, click, matplotlib; print('torch:', torch.__version__)"
```

### 3. Run training

```bash
python train.py
```

## GPU Note

The provided environment is CPU-first so it is easier to reproduce on most machines.

If you want to use an NVIDIA GPU, install a CUDA-enabled PyTorch build after activating the environment, for example:

```bash
conda activate brain-tumor-yolo
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Then confirm CUDA is available:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Usage

1. Clone the repository
2. Create and activate the Conda environment
3. Run the training script or package CLI

```bash
python train.py
# or
python -m brain_tumor_detector.cli --config config/default.yaml
```

## Screening Report

After training, you can build an image-level screening report from any YOLO weights file. The report sweeps positive-class confidence thresholds and saves metrics, plots, and representative TP/FP/TN/FN cases.

Run the baseline report:

```bash
python screening_report.py \
  --weights runs/detect/train/weights/best.pt \
  --output-dir runs/screening_report_baseline
```

Run the medical-focus experiment report:

```bash
python screening_report.py \
  --weights runs/experiments/medical_focus_small_640/weights/best.pt \
  --imgsz 640 \
  --output-dir runs/screening_report_medical_focus
```

Open the generated HTML report:

```bash
xdg-open runs/screening_report_baseline/screening_report.html
```

Important outputs:

- `screening_report.html`: visual report
- `threshold_metrics.csv`: precision, recall, specificity, and F1 for each threshold
- `image_scores.csv`: per-image positive confidence scores
- `case_visuals/`: selected false positives, false negatives, true positives, and true negatives

This report is for model debugging and research workflow only. It is not a clinical diagnosis tool.

## False Negative Analysis

For screening-style projects, false negatives are especially important because they are missed positive cases. This script finds positive validation images that the detector misses at a chosen threshold, then summarizes missed box size, location, contrast, and visual examples.

```bash
python false_negative_analysis.py \
  --weights runs/detect/train/weights/best.pt \
  --threshold 0.25 \
  --output-dir runs/fn_analysis_baseline_t025
```

Important outputs:

- `false_negative_report.html`: visual FN report
- `false_negative_boxes.csv`: missed positive box size, position, and contrast table
- `fn_visuals/`: validation images with missed positive boxes drawn in red

## Classification Assist

Detection is useful for locating tumors, but an image-level classifier can help answer a simpler screening question: does this MRI look positive or negative? The classification assistant converts the detection dataset into a classification dataset, trains a YOLO classification model, and compares three strategies:

- detector only
- classifier only
- classifier OR detector

Train and evaluate the classifier-assisted workflow:

```bash
python classification_assist.py \
  --epochs 5 \
  --imgsz 224 \
  --batch 16 \
  --output-dir runs/classification_assist_nano_5e
```

Evaluate again using an existing classifier weight:

```bash
python classification_assist.py \
  --skip-train \
  --classifier-weights runs/classification/brain_tumor_cls_nano/weights/best.pt \
  --output-dir runs/classification_assist_nano_5e
```

Important outputs:

- `classification_assist_report.html`: classifier-vs-detector-vs-combined report
- `strategy_metrics.csv`: precision, recall, specificity, F1, and FN count for each strategy
- `classification_scores.csv`: per-image classifier positive probability and detector positive confidence

## Configuration

The training configuration can be modified in `train.py` or `config/default.yaml`. Key parameters include:

- Epochs
- Batch size
- Image size
- Data augmentation settings
- Model architecture selection

## Model Architectures

Available YOLO models:
- YOLOv11n (nano)
- YOLOv11s (small)
- YOLOv11m (medium)
- YOLOv11l (large)
- YOLOv11x (xlarge)

## Results

Training results and model weights are saved in the `runs/brain_tumor_detection` directory.
