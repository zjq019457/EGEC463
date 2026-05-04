# Brain Tumor Screening Method

This project extends a YOLO11 brain tumor detector into a screening-oriented workflow. The goal is not only to draw tumor boxes, but also to reduce missed positive MRI images.

The final workflow has three parts:

- YOLO detector: locates `negative` and `positive` regions in MRI images.
- False negative analysis: finds positive validation images missed by the detector and summarizes why they are difficult.
- Classification assist: trains an image-level `positive` / `negative` classifier and combines it with the detector.

This project is for model development and research workflow only. It is not a clinical diagnosis tool.

## Presentation Report: Brain Tumor Detection and Screening

### Slide 1: Research Goal

- Objective: turn a YOLO11 detection model into a screening workflow that minimizes missed positive MRI images.
- Key outcome: reduce false negatives while preserving reasonable precision.
- Why this matters: in brain tumor screening, a missed positive image can delay diagnosis and treatment.

### Slide 2: Dataset and challenge

- Dataset contains MRI images labeled with tumor bounding boxes and image-level positive / negative status.
- Challenge: many tumor regions are small, low-contrast, or partially visible.
- Data statistics:
  - Train: 893 images, 459 positive, 434 negative, 925 boxes
  - Val: 223 images, 81 positive, 142 negative, 241 boxes
  - Small boxes (<2% image area): 612 train, 179 val
- Consequence: baseline detection struggles on tiny tumor regions.

### Slide 3: Baseline detection performance

- Model: YOLO11n trained at 320px with original augmentation.
- Performance at detector threshold 0.25:
  - Precision: 0.384
  - Recall: 0.407
  - Specificity: 0.627
  - F1: 0.395
  - False negatives: 48
- Interpretation: the model is too conservative for screening, missing nearly half of positive images.

### Slide 4: False negative failure mode

- In analysis, the baseline detector missed 48 positive images and 53 positive boxes.
- Most missed boxes are tiny:
  - 48 / 53 missed boxes are smaller than 2% of image area
  - median missed box area is 0.0081
- Visual evidence shows small, low-contrast lesions where the model fails to output any detection.

### Slide 5: Improved detection experiments

| Experiment | Model | Input size | Focus |
| --- | --- | --- | --- |
| `runs/detect/train` | YOLO11n | 320px | baseline |
| `runs/detect/train2` | YOLO11n | 320px | repeat baseline |
| `runs/experiments/medical_focus_small_640` | YOLO11s | 640px | pilot small-object focus |
| `runs/experiments/medical_nano_640` | YOLO11n | 640px | light medical augmentation |
| `runs/experiments/medical_small_640` | YOLO11s | 640px | recall-focused |
| `runs/experiments/medical_medium_640` | YOLO11m | 640px | balanced recall |

- Best detector: `medical_nano_640`.
- Improvement over baseline:
  - mAP50: from 0.508 to 0.540
  - mAP50-95: from 0.284 to 0.373
  - False negatives at 0.25: from 48 to 19

### Slide 6: Detector comparison summary

- `baseline 320px`: stable but misses small tumors.
- `medical_nano_640`: best overall, more accurate localization and fewer misses.
- `medical_small_640` / `medical_medium_640`: trade precision for recall, better at hard small-tumor cases.
- The main technical insight: higher input resolution + lighter medical augmentation improves small-object recall.

### Slide 7: Screening strategy design

- Strategy 1: detector only
- Strategy 2: classifier only
- Strategy 3: detector OR classifier

Combined rule:
```text
predict positive if:
  detector positive confidence >= detector threshold
  OR classifier positive probability >= classifier threshold
```

- Classifier acts as a second opinion for image-level positivity.
- This is useful when detection may miss a lesion but image evidence still suggests disease.

### Slide 8: Screening results comparison

| Strategy | Precision | Recall | Specificity | F1 | False negatives |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline detector only | 0.384 | 0.407 | 0.627 | 0.395 | 48 |
| Best detector only | 0.373 | 0.765 | 0.268 | 0.502 | 19 |
| Classifier only | 0.528 | 0.704 | 0.641 | 0.603 | 24 |
| Classifier OR baseline detector | 0.443 | 0.864 | 0.380 | 0.586 | 11 |
| Classifier OR best detector | 0.393 | 0.975 | 0.141 | 0.560 | 2 |

- Best screening result: classifier OR best detector, recall = 0.975.
- Tradeoff: specificity drops from 0.627 to 0.141 in the highest-recall setting.
- Conclusion: for screening, recall is prioritized, and the combined model recovers nearly all positives.

### Slide 9: Visual examples for slides

Use these ready-made assets in presentation slides:

- Baseline false negative examples: `runs/fn_analysis_baseline_t025/fn_visuals`
- Improved detection cases: `runs/fn_analysis_medical_nano_640_t025/fn_visuals`
- Baseline screening case visuals: `runs/screening_report_baseline/case_visuals`
- Best detector screening case visuals: `runs/screening_report_medical_nano_640/case_visuals`
- Confusion matrix and threshold curves:
  - `runs/screening_report_baseline/confusion_matrix.png`
  - `runs/screening_report_baseline/threshold_curves.png`
  - `runs/screening_report_medical_nano_640/confusion_matrix.png`
  - `runs/screening_report_medical_nano_640/threshold_curves.png`

#### Slide example 1: baseline miss vs improved detect

Baseline missed small lesion:

![Baseline missed small lesion](runs/fn_analysis_baseline_t025/fn_visuals/fn_val_1%20(100).jpg)

Improved detection by `medical_nano_640`:

![Improved detection by medical_nano_640](runs/fn_analysis_medical_nano_640_t025/fn_visuals/fn_val_1%20(100).jpg)

#### Slide example 2: screening case comparison

Baseline screening false negative:

![Baseline screening false negative](runs/screening_report_baseline/case_visuals/fn_val_1%20(170).jpg)

Best detector screening true positive:

![Best detector screening true positive](runs/screening_report_medical_nano_640/case_visuals/tp_val_1%20(26).jpg)

### Slide 10: Key takeaways

- Small tumor regions are the main failure mode for baseline detection.
- Higher-resolution detection models reduce false negatives significantly.
- A classifier-assisted screening pipeline is effective for recovering missed positives.
- The combined strategy is the best choice for a screening setting, with recall prioritized over specificity.
- Recommended next steps: more epochs, more seeds, crop-based training for small boxes, and stronger medical augmentation.

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
