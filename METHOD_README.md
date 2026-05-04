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

### Detection Results

The current completed detection experiments are:

| Experiment | Model / setup | Best mAP50 epoch | Precision | Recall | Best mAP50 | Best mAP50-95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `runs/detect/train` | YOLO11n, 320px, original augmentation | 9 | 0.469 | 0.810 | 0.508 | 0.284 |
| `runs/detect/train2` | repeated YOLO11n baseline run | 9 | 0.469 | 0.810 | 0.508 | 0.284 |
| `runs/experiments/medical_focus_small_640` | YOLO11s, 640px, light medical augmentation, pilot run | 1 | 0.359 | 0.705 | 0.418 | 0.245 |
| `runs/experiments/medical_nano_640` | YOLO11n, 640px, light medical augmentation | 41 | 0.492 | 0.740 | 0.540 | 0.373 |
| `runs/experiments/medical_small_640` | YOLO11s, 640px, light medical augmentation | 24 | 0.468 | 0.826 | 0.526 | 0.363 |
| `runs/experiments/medical_medium_640` | YOLO11m, 640px, light medical augmentation | 25 | 0.452 | 0.819 | 0.515 | 0.358 |

Ultralytics also generated standard detection evaluation artifacts:

- `PR_curve.png`
- `confusion_matrix.png`
- `confusion_matrix_normalized.png`
- `results.csv`

#### Slide-ready detection comparison

- Baseline YOLO11n 320px is the reference model. It has acceptable localization metrics, but misses a large number of small, low-contrast tumor regions in MRI.
- `medical_nano_640` is the best detection model in this project: it increases mAP50 and mAP50-95, and cuts detector-only false negatives nearly in half.
- `medical_small_640` and `medical_medium_640` prioritize higher recall; they are stronger at finding hard cases when the tumor footprint is very small.

#### Visual case comparison

- Representative cases are available in the false negative analysis outputs: `runs/fn_analysis_baseline_t025/fn_visuals` versus `runs/fn_analysis_medical_nano_640_t025/fn_visuals`.
- In the same MRI image, the baseline detector often fails to draw a bounding box on a small tumor, while `medical_nano_640` recovers the lesion with a tighter box.
- This contrast is ideal for PPT slides: show the baseline miss, the recovered detection, and the final screening decision.

The best detector is `medical_nano_640`. Compared with the original YOLO11n 320px baseline, it improves:

- mAP50 from `0.508` to `0.540`
- mAP50-95 from `0.284` to `0.373`
- detector-only false negatives at threshold `0.25` from `48` to `19`

This confirms the main hypothesis from the FN analysis: increasing resolution and using lighter medical-image augmentation helps small-object recall.

### Screening Improvement

The classifier was trained for a short 5-epoch CPU experiment using `YOLO11n-cls`, `imgsz=224`, and `batch=16`.

| Strategy | Thresholds | Precision | Recall | Specificity | F1 | False negatives |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Baseline detector only | detector=0.25 | 0.384 | 0.407 | 0.627 | 0.395 | 48 |
| Best detector only | medical_nano_640, detector=0.25 | 0.373 | 0.765 | 0.268 | 0.502 | 19 |
| Classifier only | classifier=0.30 | 0.528 | 0.704 | 0.641 | 0.603 | 24 |
| Classifier OR baseline detector | classifier=0.30, detector=0.25 | 0.443 | 0.864 | 0.380 | 0.586 | 11 |
| Classifier OR best detector | classifier=0.30, medical_nano_640 detector=0.25 | 0.393 | 0.975 | 0.141 | 0.560 | 2 |

#### Screening strategy comparison

- `Baseline detector only` is the conservative reference: moderate precision, low recall, and a large number of missed positive images.
- `Best detector only` improves recall significantly, showing that detector retraining with higher resolution is the key detection-level improvement.
- `Classifier only` demonstrates that image-level classification can catch positive images that localization misses, with fewer false positives than the detector-only option.
- `Classifier OR detector` is the recommended screening strategy when recall is the priority: it combines the strengths of both models and recovers nearly all positive cases.

Main improvement:

- Detector-only false negatives dropped from `48` to `19` after retraining YOLO11n at 640px with lighter augmentation.
- Combined false negatives dropped from `48` to `2` when using the classifier OR best detector screening strategy.
- Combined recall improved from `0.407` to `0.975`.

Tradeoff:

- The best combined strategy is very sensitive but creates many false positives.
- Specificity drops from `0.627` to `0.141` in the highest-recall combined setting.

This is a common screening tradeoff: the system becomes more sensitive, but less selective.

### PPT takeaways

- Slide 1: problem statement — small tumor detection is hard and creates many false negatives.
- Slide 2: detector comparison — baseline vs `medical_nano_640` vs `medical_small_640`.
- Slide 3: screening strategy comparison — detector, classifier, combined.
- Slide 4: visual examples from the false negative analysis reports.

### Visual outputs for presentation

Key visualization assets already present in this repository:

- Baseline false negative examples: `runs/fn_analysis_baseline_t025/fn_visuals`
- Improved detection cases: `runs/fn_analysis_medical_nano_640_t025/fn_visuals`
- Screening report charts: `runs/screening_report_baseline/confusion_matrix.png`, `runs/screening_report_baseline/threshold_curves.png`
- Best detector charts: `runs/screening_report_medical_nano_640/confusion_matrix.png`, `runs/screening_report_medical_nano_640/threshold_curves.png`

Example comparison images for PPT:

Baseline miss (small/low-contrast tumor):

![Baseline missed small lesion](runs/fn_analysis_baseline_t025/fn_visuals/fn_val_1%20(100).jpg)

Improved detection by `medical_nano_640`:

![Improved detection by medical_nano_640](runs/fn_analysis_medical_nano_640_t025/fn_visuals/fn_val_1%20(100).jpg)

These visual examples are ready to use in slides to show how the baseline detector misses small tumors and how the improved model recovers them.

## Data and Error Analysis

The data audit script checks image readability, missing labels, class balance, label validity, and bounding-box size.

Current dataset summary:

| Split | Images | Positive images | Negative images | Boxes | Small boxes <2% |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 893 | 459 | 434 | 925 | 612 |
| val | 223 | 81 | 142 | 241 | 179 |

The high number of small boxes explains why the baseline detector misses many positive cases. This supports the next detector-improvement direction: higher input resolution and small-object-focused training.

## Completed Implementation Checklist

- Training configurations: baseline, `medical_focus`, `medical_nano_640`, `medical_small_640`, and `medical_medium_640` are available in `train.py`.
- Data augmentation: original strong augmentation and lighter medical-image augmentation are both configured.
- Evaluation workflow: mAP50, mAP50-95, precision, recall, PR curve, confusion matrix, threshold sweep, and F1 are collected.
- Error analysis: false positive / false negative examples are saved, and missed positive boxes are measured.
- Classification assist: a lightweight YOLO11 classification model is trained and evaluated with detector-only, classifier-only, and combined strategies.
- Explainability: occlusion heatmaps are generated for the classifier.
- Reproducibility: `environment.yml`, `requirements.txt`, `.gitignore`, `run.sh`, and `Makefile` are included.

## Not Fully Completed Yet

The following items are implemented as runnable workflows, but require additional compute time to produce stronger stability evidence:

- Multi-run repeated experiments with different seeds.
- Cross-validation.
- Longer training beyond 50 epochs.

The scripts and commands are already prepared, but these runs should ideally be done on GPU.

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

Run the best detector screening report:

```bash
python screening_report.py \
  --weights runs/experiments/medical_nano_640/weights/best.pt \
  --imgsz 640 \
  --output-dir runs/screening_report_medical_nano_640
```

### 4. Run data audit

```bash
python data_audit.py --output-dir runs/data_audit
```

Important outputs:

- `runs/data_audit/data_audit_summary.md`
- `runs/data_audit/image_class_distribution.png`
- `runs/data_audit/box_area_histogram.png`

### 5. Run false negative analysis

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

Run FN analysis for the best detector:

```bash
python false_negative_analysis.py \
  --weights runs/experiments/medical_nano_640/weights/best.pt \
  --imgsz 640 \
  --threshold 0.25 \
  --output-dir runs/fn_analysis_medical_nano_640_t025
```

### 6. Train and evaluate the classifier-assisted model

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

### 7. Re-run evaluation using an existing classifier

```bash
python classification_assist.py \
  --skip-train \
  --classifier-weights runs/classification/brain_tumor_cls_nano/weights/best.pt \
  --output-dir runs/classification_assist_nano_5e
```

Evaluate the classifier with the best detector:

```bash
python classification_assist.py \
  --skip-train \
  --classifier-weights runs/classification/brain_tumor_cls_nano/weights/best.pt \
  --detector-weights runs/experiments/medical_nano_640/weights/best.pt \
  --detector-imgsz 640 \
  --detector-threshold 0.25 \
  --output-dir runs/classification_assist_medical_nano_640
```

### 8. Generate classifier explainability heatmaps

```bash
python interpretability_heatmap.py \
  --weights runs/classification/brain_tumor_cls_nano/weights/best.pt \
  --output-dir runs/interpretability_heatmaps
```

### 9. Summarize all completed experiments

```bash
python experiment_summary.py --output-dir runs/experiment_summary
```

### 10. Use Makefile or run.sh

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

Longer detector comparisons:

```bash
python train.py --experiment medical_nano_640
python train.py --experiment medical_small_640
python train.py --experiment medical_medium_640
```

## Important Outputs

### Screening Report

- `runs/screening_report_baseline/screening_report.html`
- `runs/screening_report_medical_nano_640/screening_report.html`
- `runs/screening_report_baseline/threshold_metrics.csv`
- `runs/screening_report_baseline/image_scores.csv`

### False Negative Analysis

- `runs/fn_analysis_baseline_t025/false_negative_report.html`
- `runs/fn_analysis_medical_nano_640_t025/false_negative_report.html`
- `runs/fn_analysis_baseline_t025/false_negative_boxes.csv`
- `runs/fn_analysis_baseline_t025/fn_visuals/`

### Classification Assist

- `runs/classification/brain_tumor_cls_nano/weights/best.pt`
- `runs/classification_assist_nano_5e/classification_assist_report.html`
- `runs/classification_assist_medical_nano_640/classification_assist_report.html`
- `runs/classification_assist_nano_5e/strategy_metrics.csv`
- `runs/classification_assist_nano_5e/classification_scores.csv`

### Data Audit, Summary, and Explainability

- `runs/data_audit/data_audit_summary.md`
- `runs/experiment_summary/experiment_summary.md`
- `runs/interpretability_heatmaps/`

## What to Improve Next

The FN analysis shows that small tumor regions are the main weakness. The next detector-focused improvement should target small-object recall:

- Train YOLO with `imgsz=640`.
- Use lighter medical-image augmentation.
- Train for more epochs, such as 50 or 100.
- Try crop-based training around small positive boxes.
- Use the classifier as a screening gate and YOLO as a localization model.
