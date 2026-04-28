# Brain Tumor Screening Method

This project extends a YOLO11 brain tumor detector into a screening-oriented workflow. The goal is not only to draw tumor boxes, but also to reduce missed positive MRI images.

The final workflow has three parts:

- YOLO detector: locates `negative` and `positive` regions in MRI images.
- False negative analysis: finds positive validation images missed by the detector and summarizes why they are difficult.
- Classification assist: trains an image-level `positive` / `negative` classifier and combines it with the detector.

This project is for model development and research workflow only. It is not a clinical diagnosis tool.

## Method

### 1. YOLO Detection Baseline

The original detector uses YOLO11 object detection. It predicts bounding boxes and class labels for MRI images.

At a detector confidence threshold of `0.25`, the baseline detector had:

| Strategy | Precision | Recall | Specificity | F1 | False negatives |
| --- | ---: | ---: | ---: | ---: | ---: |
| Detector only | 0.384 | 0.407 | 0.627 | 0.395 | 48 |

This means the detector was conservative at the normal threshold and missed many positive cases.

### 2. False Negative Analysis

For medical screening, false negatives are especially important because they are missed positive images. The FN analysis script identifies positive validation cases where the detector does not produce a positive detection above the selected threshold.

At threshold `0.25`, the baseline detector missed:

- 48 positive images
- 53 positive boxes
- 48 of the missed boxes were smaller than 2% of the image area
- Median missed box area was `0.0081`

This shows a clear failure pattern: the detector often misses small tumor regions.

### 3. Classification Assist

Detection is good for localization, but image-level classification is often better for the simpler screening question: does this MRI look positive or negative?

The classification assist workflow:

1. Converts YOLO detection labels into image-level labels.
2. Trains a YOLO11 classification model on `positive` and `negative` folders.
3. Evaluates three strategies:

- detector only
- classifier only
- classifier OR detector

The combined rule is:

```text
predict positive if:
  detector positive confidence >= detector threshold
  OR classifier positive probability >= classifier threshold
```

## Improvement

The classifier was trained for a short 5-epoch CPU experiment using `YOLO11n-cls`, `imgsz=224`, and `batch=16`.

| Strategy | Thresholds | Precision | Recall | Specificity | F1 | False negatives |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Detector only | detector=0.25 | 0.384 | 0.407 | 0.627 | 0.395 | 48 |
| Classifier only | classifier=0.30 | 0.528 | 0.704 | 0.641 | 0.603 | 24 |
| Classifier OR detector | classifier=0.30, detector=0.25 | 0.443 | 0.864 | 0.380 | 0.586 | 11 |

Main improvement:

- False negatives dropped from `48` to `11`.
- Recall improved from `0.407` to `0.864`.
- F1 improved from `0.395` to `0.586` for the combined screening strategy.

Tradeoff:

- Specificity dropped from `0.627` to `0.380`.
- The combined model catches more positive cases, but it also creates more false positives.

This is a common screening tradeoff: the system becomes more sensitive, but less selective.

## How to Run

### 1. Create the Conda environment

```bash
conda env create -f environment.yml
conda activate brain-tumor-yolo
```

### 2. Train or reuse the detector

Train the detector:

```bash
python train.py
```

Or reuse an existing detector weight:

```text
runs/detect/train/weights/best.pt
```

### 3. Run threshold screening report

```bash
python screening_report.py \
  --weights runs/detect/train/weights/best.pt \
  --output-dir runs/screening_report_baseline
```

Open the report:

```bash
xdg-open runs/screening_report_baseline/screening_report.html
```

### 4. Run false negative analysis

```bash
python false_negative_analysis.py \
  --weights runs/detect/train/weights/best.pt \
  --threshold 0.25 \
  --output-dir runs/fn_analysis_baseline_t025
```

Open the FN report:

```bash
xdg-open runs/fn_analysis_baseline_t025/false_negative_report.html
```

### 5. Train and evaluate the classifier-assisted model

```bash
python classification_assist.py \
  --epochs 5 \
  --imgsz 224 \
  --batch 16 \
  --output-dir runs/classification_assist_nano_5e
```

Open the combined report:

```bash
xdg-open runs/classification_assist_nano_5e/classification_assist_report.html
```

### 6. Re-run evaluation using an existing classifier

```bash
python classification_assist.py \
  --skip-train \
  --classifier-weights runs/classification/brain_tumor_cls_nano/weights/best.pt \
  --output-dir runs/classification_assist_nano_5e
```

## Important Outputs

### Screening Report

- `runs/screening_report_baseline/screening_report.html`
- `runs/screening_report_baseline/threshold_metrics.csv`
- `runs/screening_report_baseline/image_scores.csv`

### False Negative Analysis

- `runs/fn_analysis_baseline_t025/false_negative_report.html`
- `runs/fn_analysis_baseline_t025/false_negative_boxes.csv`
- `runs/fn_analysis_baseline_t025/fn_visuals/`

### Classification Assist

- `runs/classification/brain_tumor_cls_nano/weights/best.pt`
- `runs/classification_assist_nano_5e/classification_assist_report.html`
- `runs/classification_assist_nano_5e/strategy_metrics.csv`
- `runs/classification_assist_nano_5e/classification_scores.csv`

## What to Improve Next

The FN analysis shows that small tumor regions are the main weakness. The next detector-focused improvement should target small-object recall:

- Train YOLO with `imgsz=640`.
- Use lighter medical-image augmentation.
- Train for more epochs, such as 50 or 100.
- Try crop-based training around small positive boxes.
- Use the classifier as a screening gate and YOLO as a localization model.
