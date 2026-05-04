# Brain Tumor YOLO Screening Method Report

> Format: Markdown presentation report. Each section is written like a slide, with the slide message, visuals, and speaker notes.
>
> Scope: this project is for model development, experiment reporting, and research presentation. It is not a clinical diagnosis tool.

## One-Slide Summary

This project extends a YOLO11 brain tumor detector into a screening-oriented workflow. The goal is not just to draw tumor boxes, but to reduce missed positive MRI images.

The main experimental result is:

| Comparison | Baseline YOLO11n 320px | Best detector YOLO11n 640px | Classifier OR best detector |
| --- | ---: | ---: | ---: |
| Image-level recall | 0.407 | 0.765 | 0.975 |
| Image-level precision | 0.384 | 0.373 | 0.393 |
| Image-level specificity | 0.627 | 0.268 | 0.141 |
| False negatives | 48 | 19 | 2 |
| Best use | Baseline reference | Better detector | High-recall screening |

The main interpretation is that the baseline detector misses many small tumor regions. Higher input resolution and lighter medical-image augmentation reduce missed positives. Adding an image-level classifier as a second opinion recovers nearly all positives, but increases false positives and lowers specificity.

---

## Slide 01 | From Detection to Screening

**Main Message**

- Topic: YOLO-based detection and screening for brain tumor MRI images.
- Goal: turn a detector into a screening workflow.
- Screening priority: reduce false negatives first, then discuss false positives and localization quality.

**Speaker Notes**

Standard object detection asks whether boxes are localized correctly. Medical screening asks a more safety-sensitive question: did the model miss a positive image? This project keeps YOLO localization, then adds false negative analysis, threshold sweeps, and image-level classifier assistance.

**Pipeline**

```mermaid
flowchart LR
  A[Brain MRI images] --> B[YOLO detector]
  B --> C[Image-level screening score]
  C --> D[Threshold sweep]
  D --> E[FN / FP / TP / TN case review]
  A --> F[Image-level classifier]
  B --> G[Detector OR Classifier]
  F --> G
  G --> H[High-recall screening decision]
```

**Presentation Cue**

The classifier does not replace YOLO. YOLO localizes suspicious regions, while the classifier provides an image-level second opinion.

---

## Slide 02 | Why mAP Is Not Enough

**Main Message**

- YOLO mAP, precision, and recall are detection-level metrics.
- Screening needs image-level positive / negative decisions.
- The same detector can look reasonable under box metrics but still miss too many positive images.

**Visuals: validation labels and baseline predictions**

| Ground truth labels | Baseline predictions |
| --- | --- |
| ![Baseline validation labels](runs/detect/train/val_batch0_labels.jpg) | ![Baseline validation predictions](runs/detect/train/val_batch0_pred.jpg) |
| ![Baseline validation labels batch 1](runs/detect/train/val_batch1_labels.jpg) | ![Baseline validation predictions batch 1](runs/detect/train/val_batch1_pred.jpg) |

**Speaker Notes**

Detection metrics answer whether boxes and classes are predicted correctly. Screening metrics answer whether each true positive image is flagged. In this project, an image is predicted positive if at least one class-1 positive detection passes the threshold.

---

## Slide 03 | Dataset and Class Balance

**Main Message**

- Train split: 893 images. Validation split: 223 images.
- Train positives and negatives are close in count; validation has more negatives.
- Many tumor boxes are very small, which directly explains the false-negative problem.

**Dataset Audit**

| Split | Images | Positive images | Negative images | Boxes | Positive boxes | Small boxes <2% | Missing labels | Invalid label lines |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 893 | 459 | 434 | 925 | 488 | 612 | 15 | 0 |
| val | 223 | 81 | 142 | 241 | 87 | 179 | 0 | 0 |

**Visuals: audit outputs**

| Image-level class distribution | Bounding-box area distribution |
| --- | --- |
| ![Image class distribution](runs/data_audit/image_class_distribution.png) | ![Box area histogram](runs/data_audit/box_area_histogram.png) |

**Speaker Notes**

The dataset is not extremely class-imbalanced, but the validation set has more negative images, so specificity matters. The stronger signal is box size: 612 train boxes and 179 validation boxes are smaller than 2% of image area. Small objects lose detail after resizing, especially at 320px.

---

## Slide 04 | Label Distribution and Small Objects

**Main Message**

- Label plots help inspect class, box size, and box position.
- Correlograms show relationships among center position, width, height, and class.
- A dataset dominated by small boxes can create a conservative detector.

**Visuals: baseline label distribution**

| Labels overview | Labels correlogram |
| --- | --- |
| ![Baseline labels](runs/detect/train/labels.jpg) | ![Baseline labels correlogram](runs/detect/train/labels_correlogram.jpg) |

**Visuals: 640px medical experiment label distribution**

| Labels overview | Labels correlogram |
| --- | --- |
| ![Medical nano labels](runs/experiments/medical_nano_640/labels.jpg) | ![Medical nano labels correlogram](runs/experiments/medical_nano_640/labels_correlogram.jpg) |

**Speaker Notes**

These plots are not final performance evidence, but they explain why the model struggles. Small boxes need more input resolution. Low-contrast tumor boundaries also make strong augmentation risky because it can distort already subtle image evidence.

---

## Slide 05 | Baseline Detector Setup

**Main Message**

- Baseline model: YOLO11n.
- Input size: 320px.
- Augmentation: original stronger augmentation.
- Purpose: provide a reference point for later improvements.

**Training examples**

| Train batch 0 | Train batch 1 | Train batch 2 |
| --- | --- | --- |
| ![Baseline train batch 0](runs/detect/train/train_batch0.jpg) | ![Baseline train batch 1](runs/detect/train/train_batch1.jpg) | ![Baseline train batch 2](runs/detect/train/train_batch2.jpg) |

**Speaker Notes**

YOLO11n is fast and lightweight. The weakness is that 320px input can compress small MRI lesions into very few pixels. That makes the baseline useful as an engineering starting point, but not strong enough for screening where missed positives are costly.

---

## Slide 06 | Baseline Detection Results

**Main Message**

- Ultralytics detection evaluation: best epoch = 9.
- Detection-level metrics: precision 0.469, recall 0.810, mAP50 0.508, mAP50-95 0.284.
- The detector learned useful localization, but image-level screening must be checked separately.

**Visuals: training and detection curves**

| Results | PR curve |
| --- | --- |
| ![Baseline training results](runs/detect/train/results.png) | ![Baseline PR curve](runs/detect/train/PR_curve.png) |

| F1 curve | Confusion matrix |
| --- | --- |
| ![Baseline F1 curve](runs/detect/train/F1_curve.png) | ![Baseline confusion matrix](runs/detect/train/confusion_matrix.png) |

**Speaker Notes**

The baseline is not useless. It has detection ability. The problem appears when detection outputs are converted into image-level screening decisions and the false negatives are counted.

---

## Slide 07 | Baseline Image-Level Screening

**Main Message**

- At threshold 0.25, baseline image-level recall is only 0.407.
- Out of 81 true positive validation images, only 33 are recovered.
- The model misses 48 positive images.

**Screening Result**

| Threshold | TP | FP | TN | FN | Precision | Recall | Specificity | F1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.25 | 33 | 53 | 89 | 48 | 0.384 | 0.407 | 0.627 | 0.395 |

**Visuals: baseline screening report**

| Image-level confusion matrix | Threshold curves |
| --- | --- |
| ![Baseline screening confusion matrix](runs/screening_report_baseline/confusion_matrix.png) | ![Baseline screening threshold curves](runs/screening_report_baseline/threshold_curves.png) |

**Speaker Notes**

This is the key baseline failure. The detector is conservative enough to preserve some specificity, but that conservatism produces too many missed positive images for a screening workflow.

---

## Slide 08 | Baseline Cases: TP, FN, FP, TN

**Main Message**

- Aggregate metrics need image-level examples.
- TP shows successful positive detection.
- FN shows the safety-critical failure mode.
- FP shows suspicious negative regions.
- TN shows normal negative exclusion.

**Visuals: representative baseline cases**

| True positive | False negative | False positive | True negative |
| --- | --- | --- | --- |
| ![Baseline TP](<runs/screening_report_baseline/case_visuals/tp_val_1 (14).jpg>) | ![Baseline FN](<runs/screening_report_baseline/case_visuals/fn_val_1 (170).jpg>) | ![Baseline FP](<runs/screening_report_baseline/case_visuals/fp_val_1 (34).jpg>) | ![Baseline TN](<runs/screening_report_baseline/case_visuals/tn_val_1 (39).jpg>) |
| ![Baseline TP 2](<runs/screening_report_baseline/case_visuals/tp_val_1 (76).jpg>) | ![Baseline FN analysis case](<runs/fn_analysis_baseline_t025/fn_visuals/fn_val_1 (100).jpg>) | ![Baseline FP 2](<runs/screening_report_baseline/case_visuals/fp_val_1 (10).jpg>) | ![Baseline TN 2](<runs/screening_report_baseline/case_visuals/tn_val_1 (202).jpg>) |

**Speaker Notes**

False negatives matter most here. False positives can often be reviewed downstream, but false negatives can leave a positive image unflagged.

---

## Slide 09 | False Negative Analysis

**Main Message**

- Baseline misses 48 positive images at threshold 0.25.
- These images contain 53 positive boxes.
- 48 of 53 missed boxes are smaller than 2% of image area.
- Median missed-box area is 0.008.

**Visual: missed-box area distribution**

![Baseline false negative box area histogram](runs/fn_analysis_baseline_t025/fn_box_area_histogram.png)

**Speaker Notes**

The false-negative analysis script selects true positive validation images where the detector fails to output a sufficiently confident positive box. Red boxes mark ground-truth positive regions that were missed. The conclusion becomes specific: the baseline mainly misses small tumor regions.

---

## Slide 10 | Baseline False Negative Gallery

**Main Message**

- Many missed lesions are small, low-contrast, or hard to separate from surrounding tissue.
- These visuals are ready to use in presentation slides.
- Red boxes are ground-truth tumor regions missed by the detector.

**Visuals: baseline false negative examples**

| FN 1 | FN 100 | FN 102 | FN 105 |
| --- | --- | --- | --- |
| ![FN val 1](<runs/fn_analysis_baseline_t025/fn_visuals/fn_val_1 (1).jpg>) | ![FN val 100](<runs/fn_analysis_baseline_t025/fn_visuals/fn_val_1 (100).jpg>) | ![FN val 102](<runs/fn_analysis_baseline_t025/fn_visuals/fn_val_1 (102).jpg>) | ![FN val 105](<runs/fn_analysis_baseline_t025/fn_visuals/fn_val_1 (105).jpg>) |

| FN 110 | FN 117 | FN 138 | FN 146 |
| --- | --- | --- | --- |
| ![FN val 110](<runs/fn_analysis_baseline_t025/fn_visuals/fn_val_1 (110).jpg>) | ![FN val 117](<runs/fn_analysis_baseline_t025/fn_visuals/fn_val_1 (117).jpg>) | ![FN val 138](<runs/fn_analysis_baseline_t025/fn_visuals/fn_val_1 (138).jpg>) | ![FN val 146](<runs/fn_analysis_baseline_t025/fn_visuals/fn_val_1 (146).jpg>) |

**Speaker Notes**

These examples connect the metric failure to visual evidence. The detector often misses very small or weakly contrasted lesions, which is exactly the failure mode a screening workflow should reduce.

---

## Slide 11 | Improvement Plan: 640px and Medical Augmentation

**Main Message**

- Increase input size from 320px to 640px.
- Use lighter medical-image augmentation.
- Compare nano, small, and medium model scales.
- Optimize for fewer false negatives, not only higher precision.

**Experiment Design**

| Experiment | Model | Input size | Purpose |
| --- | --- | ---: | --- |
| `runs/detect/train` | YOLO11n | 320 | baseline |
| `runs/detect/train2` | YOLO11n | 320 | repeated baseline |
| `runs/experiments/medical_focus_small_640` | YOLO11s | 640 | small-object pilot |
| `runs/experiments/medical_nano_640` | YOLO11n | 640 | light medical augmentation, best overall |
| `runs/experiments/medical_small_640` | YOLO11s | 640 | recall-oriented |
| `runs/experiments/medical_medium_640` | YOLO11m | 640 | medium model scale |

**Visuals: 640px sample predictions**

| Sample 1 | Sample 10 | Sample 100 |
| --- | --- | --- |
| ![Medical nano sample 1](<runs/experiments/medical_nano_640/sample_predictions/val_1 (1)_comparison.png>) | ![Medical nano sample 10](<runs/experiments/medical_nano_640/sample_predictions/val_1 (10)_comparison.png>) | ![Medical nano sample 100](<runs/experiments/medical_nano_640/sample_predictions/val_1 (100)_comparison.png>) |

**Speaker Notes**

The resolution change is straightforward: a small lesion keeps more pixels at 640px than at 320px. Medical augmentation is kept lighter because aggressive color or geometry transforms can damage subtle MRI cues.

---

## Slide 12 | Best Detector: medical_nano_640

**Main Message**

- `medical_nano_640` is the best overall detector in the completed experiments.
- Best epoch = 41.
- mAP50 improves from 0.508 to 0.540.
- mAP50-95 improves from 0.284 to 0.373.

**Detection Experiment Summary**

| Experiment | Best epoch | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline 320px | 9 | 0.469 | 0.810 | 0.508 | 0.284 |
| medical_nano_640 | 41 | 0.492 | 0.740 | 0.540 | 0.373 |
| medical_small_640 | 24 | 0.468 | 0.826 | 0.526 | 0.363 |
| medical_medium_640 | 25 | 0.452 | 0.819 | 0.515 | 0.358 |

**Visuals: medical_nano_640 training and validation**

| Results | PR curve |
| --- | --- |
| ![Medical nano results](runs/experiments/medical_nano_640/results.png) | ![Medical nano PR curve](runs/experiments/medical_nano_640/PR_curve.png) |

| F1 curve | Confusion matrix |
| --- | --- |
| ![Medical nano F1 curve](runs/experiments/medical_nano_640/F1_curve.png) | ![Medical nano confusion matrix](runs/experiments/medical_nano_640/confusion_matrix.png) |

**Speaker Notes**

Detection-level recall is not the only story. The best detector improves localization quality and, more importantly for screening, reduces image-level false negatives from 48 to 19.

---

## Slide 13 | Other 640px Detectors

**Main Message**

- Bigger is not automatically better on this dataset.
- `medical_small_640` has strong recall, but lower overall balance than nano.
- `medical_medium_640` did not outperform nano in this run.
- The current recommended detector is `medical_nano_640`.

**Visuals: small and medium training results**

| Medical small results | Medical medium results |
| --- | --- |
| ![Medical small results](runs/experiments/medical_small_640/results.png) | ![Medical medium results](runs/experiments/medical_medium_640/results.png) |

| Medical small PR | Medical medium PR |
| --- | --- |
| ![Medical small PR curve](runs/experiments/medical_small_640/PR_curve.png) | ![Medical medium PR curve](runs/experiments/medical_medium_640/PR_curve.png) |

**Speaker Notes**

With limited medical data, larger models may need longer training, more seeds, and stronger validation before they clearly beat a smaller model. The nano 640px model currently offers the best balance between performance and compute cost.

---

## Slide 14 | Best Detector Image-Level Screening

**Main Message**

- At threshold 0.25, best detector image-level recall is 0.765.
- False negatives drop from 48 to 19.
- False positives rise to 104 and specificity drops to 0.268.
- The detector becomes more sensitive, but less specific.

**Screening Result**

| Detector | Threshold | TP | FP | TN | FN | Precision | Recall | Specificity | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline 320px | 0.25 | 33 | 53 | 89 | 48 | 0.384 | 0.407 | 0.627 | 0.395 |
| medical_nano_640 | 0.25 | 62 | 104 | 38 | 19 | 0.373 | 0.765 | 0.268 | 0.502 |

**Visuals: best detector screening report**

| Confusion matrix | Threshold curves |
| --- | --- |
| ![Medical nano screening confusion matrix](runs/screening_report_medical_nano_640/confusion_matrix.png) | ![Medical nano screening threshold curves](runs/screening_report_medical_nano_640/threshold_curves.png) |

**Speaker Notes**

This is the screening tradeoff. Recovering more positives usually increases false positives. For a first-stage screening tool, that can be acceptable if downstream review catches false alarms.

---

## Slide 15 | Best Detector Cases

**Main Message**

- The best detector recovers more positive images.
- It still has remaining false negatives.
- False positives often come from negative structures that resemble suspicious regions.

**Visuals: medical_nano_640 screening cases**

| True positive | False negative | False positive | True negative |
| --- | --- | --- | --- |
| ![Medical nano TP](<runs/screening_report_medical_nano_640/case_visuals/tp_val_1 (26).jpg>) | ![Medical nano FN](<runs/screening_report_medical_nano_640/case_visuals/fn_val_1 (14).jpg>) | ![Medical nano FP](<runs/screening_report_medical_nano_640/case_visuals/fp_val_1 (22).jpg>) | ![Medical nano TN](<runs/screening_report_medical_nano_640/case_visuals/tn_val_1 (11).jpg>) |
| ![Medical nano TP 2](<runs/screening_report_medical_nano_640/case_visuals/tp_val_1 (69).jpg>) | ![Medical nano FN 2](<runs/screening_report_medical_nano_640/case_visuals/fn_val_1 (146).jpg>) | ![Medical nano FP 2](<runs/screening_report_medical_nano_640/case_visuals/fp_val_1 (196).jpg>) | ![Medical nano TN 2](<runs/screening_report_medical_nano_640/case_visuals/tn_val_1 (113).jpg>) |

**Speaker Notes**

When presenting the best model, include both success and failure cases. The false negatives show what still needs improvement, while false positives show what a later review stage must filter.

---

## Slide 16 | Remaining Misses After Improvement

**Main Message**

- `medical_nano_640` reduces missed positives, but does not remove them.
- 14 of 19 remaining missed boxes are still smaller than 2% of image area.
- Median missed-box area is about 0.010.
- Small targets remain the hardest failure mode.

**Visuals: best detector false negatives**

| FN area histogram | FN 100 | FN 138 |
| --- | --- | --- |
| ![Medical nano FN area histogram](runs/fn_analysis_medical_nano_640_t025/fn_box_area_histogram.png) | ![Medical nano FN 100](<runs/fn_analysis_medical_nano_640_t025/fn_visuals/fn_val_1 (100).jpg>) | ![Medical nano FN 138](<runs/fn_analysis_medical_nano_640_t025/fn_visuals/fn_val_1 (138).jpg>) |

| FN 14 | FN 146 | FN 156 |
| --- | --- | --- |
| ![Medical nano FN 14](<runs/fn_analysis_medical_nano_640_t025/fn_visuals/fn_val_1 (14).jpg>) | ![Medical nano FN 146](<runs/fn_analysis_medical_nano_640_t025/fn_visuals/fn_val_1 (146).jpg>) | ![Medical nano FN 156](<runs/fn_analysis_medical_nano_640_t025/fn_visuals/fn_val_1 (156).jpg>) |

**Speaker Notes**

This slide keeps the conclusion honest. The best detector is better, but the remaining misses still cluster around small, subtle regions. Next detector improvements should focus on small-object recall.

---

## Slide 17 | Why Add an Image-Level Classifier

**Main Message**

- The detector localizes tumors but may miss a positive image if box confidence is low.
- The classifier predicts image-level positive / negative status.
- The classifier can act as a second opinion.
- The combined rule is OR: if either model says positive, the image is flagged.

**Combined Rule**

```text
predict positive if:
  detector positive confidence >= detector threshold
  OR classifier positive probability >= classifier threshold
```

**Visuals: classifier outputs**

| Classifier results | Classifier confusion matrix |
| --- | --- |
| ![Classifier training results](runs/classification/brain_tumor_cls_nano/results.png) | ![Classifier confusion matrix](runs/classification/brain_tumor_cls_nano/confusion_matrix.png) |

| Validation labels | Validation predictions |
| --- | --- |
| ![Classifier val labels](runs/classification/brain_tumor_cls_nano/val_batch0_labels.jpg) | ![Classifier val predictions](runs/classification/brain_tumor_cls_nano/val_batch0_pred.jpg) |

**Speaker Notes**

The classifier does not provide tumor boxes, so it should not replace the detector. Its role is to recover image-level positive evidence when the detector is uncertain or misses a small lesion.

---

## Slide 18 | Screening Strategy Comparison

**Main Message**

- detector only: use the detector alone.
- classifier only: use the image-level classifier alone.
- classifier OR detector: flag positive if either model is positive.
- The OR strategy reduces false negatives but lowers specificity.

**Baseline detector + classifier**

| Strategy | Classifier threshold | Detector threshold | Precision | Recall | Specificity | F1 | FN | FP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| detector only |  | 0.25 | 0.384 | 0.407 | 0.627 | 0.395 | 48 | 53 |
| classifier only | 0.30 |  | 0.528 | 0.704 | 0.641 | 0.603 | 24 | 51 |
| classifier OR detector | 0.30 | 0.25 | 0.443 | 0.864 | 0.380 | 0.586 | 11 | 88 |

**Best detector + classifier**

| Strategy | Classifier threshold | Detector threshold | Precision | Recall | Specificity | F1 | FN | FP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| detector only |  | 0.25 | 0.373 | 0.765 | 0.268 | 0.502 | 19 | 104 |
| classifier only | 0.30 |  | 0.528 | 0.704 | 0.641 | 0.603 | 24 | 51 |
| classifier OR detector | 0.30 | 0.25 | 0.393 | 0.975 | 0.141 | 0.560 | 2 | 122 |

**Visuals: strategy comparison**

| Baseline detector strategies | Best detector strategies |
| --- | --- |
| ![Baseline strategy comparison](runs/classification_assist_nano_5e/strategy_comparison.png) | ![Medical nano strategy comparison](runs/classification_assist_medical_nano_640/strategy_comparison.png) |

**Speaker Notes**

If the goal is balanced classification F1, classifier-only is strong. If the goal is screening recall, classifier OR detector is strongest. With the best detector, the OR rule reduces false negatives from 19 to 2.

---

## Slide 19 | Threshold Selection Tradeoff

**Main Message**

- Lower thresholds usually increase recall and false positives.
- Higher thresholds usually improve specificity but increase missed positives.
- Screening usually selects a high-recall operating point.

**Baseline detector threshold sweep**

| Threshold | TP | FP | TN | FN | Precision | Recall | Specificity | F1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.01 | 80 | 140 | 2 | 1 | 0.364 | 0.988 | 0.014 | 0.532 |
| 0.10 | 56 | 93 | 49 | 25 | 0.376 | 0.691 | 0.345 | 0.487 |
| 0.25 | 33 | 53 | 89 | 48 | 0.384 | 0.407 | 0.627 | 0.395 |
| 0.50 | 6 | 22 | 120 | 75 | 0.214 | 0.074 | 0.845 | 0.110 |

**Best detector threshold sweep**

| Threshold | TP | FP | TN | FN | Precision | Recall | Specificity | F1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.01 | 73 | 117 | 25 | 8 | 0.384 | 0.901 | 0.176 | 0.539 |
| 0.10 | 66 | 113 | 29 | 15 | 0.369 | 0.815 | 0.204 | 0.508 |
| 0.25 | 62 | 104 | 38 | 19 | 0.373 | 0.765 | 0.268 | 0.502 |
| 0.50 | 58 | 82 | 60 | 23 | 0.414 | 0.716 | 0.423 | 0.525 |

**Speaker Notes**

There is no universal best threshold. If missing a positive is the highest risk, choose a lower threshold or an OR strategy. If review resources are limited, add specificity constraints.

---

## Slide 20 | Classifier Explainability: Occlusion Heatmaps

**Main Message**

- Occlusion heatmaps show which regions influence classifier probability.
- If masking a region changes the positive probability, that region matters to the classifier.
- This is model-behavior analysis, not clinical explanation.

**Visuals: occlusion heatmaps**

| Heatmap 1 | Heatmap 100 | Heatmap 102 |
| --- | --- | --- |
| ![Occlusion heatmap val 1](<runs/interpretability_heatmaps/val_1 (1)_occlusion_heatmap.png>) | ![Occlusion heatmap val 100](<runs/interpretability_heatmaps/val_1 (100)_occlusion_heatmap.png>) | ![Occlusion heatmap val 102](<runs/interpretability_heatmaps/val_1 (102)_occlusion_heatmap.png>) |

**Speaker Notes**

Heatmaps help check whether the classifier is sensitive to plausible image regions or irrelevant background. They do not prove medical reasoning, but they are useful for debugging and presentation.

---

## Slide 21 | Recommended Workflow

**Main Message**

- Stage 1: use `medical_nano_640` detector to produce positive confidence and boxes.
- Stage 2: use the image-level classifier to produce positive probability.
- Stage 3: apply the OR rule for high-recall screening.
- Stage 4: review positive images manually or with a stricter second-stage model.

**Recommended Configuration**

| Module | Recommended setting | Reason |
| --- | --- | --- |
| Detector | `runs/experiments/medical_nano_640/weights/best.pt` | Reduces image-level FN from 48 to 19 and improves mAP50-95 |
| Detector threshold | 0.25 | Main operating point in the reports |
| Classifier | `runs/classification/brain_tumor_cls_nano/weights/best.pt` | Provides image-level second opinion |
| Classifier threshold | 0.30 | Balanced recall and specificity setting |
| Combined rule | classifier OR detector | Reduces FN to 2 and reaches recall 0.975 |

**Speaker Notes**

The recommended workflow is not simply the highest F1 setting. It matches the screening preference: reduce missed positives first, then handle false positives through downstream review.

---

## Slide 22 | Completed Work

**Implemented**

- Data audit: readability, missing labels, class balance, box areas.
- Detector training: baseline, medical focus, medical nano/small/medium at 640px.
- Detector evaluation: precision, recall, mAP50, mAP50-95, PR/F1/P/R curves, confusion matrices.
- Image-level screening: threshold sweeps, TP/FP/TN/FN counts, case visualizations.
- False negative analysis: missed images, missed-box areas, relative contrast, example visuals.
- Classification assist: image-level classifier plus classifier-only / detector-only / OR strategy comparison.
- Explainability: classifier occlusion heatmaps.
- Reproducibility: `Makefile`, `run.sh`, `requirements.txt`, and `environment.yml`.

**Important Visual Outputs**

| Type | Representative files |
| --- | --- |
| Data audit | `runs/data_audit/image_class_distribution.png`, `runs/data_audit/box_area_histogram.png` |
| Baseline detector | `runs/detect/train/results.png`, `runs/detect/train/PR_curve.png` |
| Best detector | `runs/experiments/medical_nano_640/results.png`, `runs/experiments/medical_nano_640/confusion_matrix.png` |
| Baseline screening | `runs/screening_report_baseline/threshold_curves.png`, `runs/screening_report_baseline/case_visuals/` |
| Best screening | `runs/screening_report_medical_nano_640/threshold_curves.png`, `runs/screening_report_medical_nano_640/case_visuals/` |
| FN analysis | `runs/fn_analysis_baseline_t025/fn_visuals/`, `runs/fn_analysis_medical_nano_640_t025/fn_visuals/` |
| Classification assist | `runs/classification_assist_medical_nano_640/strategy_comparison.png` |
| Heatmaps | `runs/interpretability_heatmaps/` |

---

## Slide 23 | Remaining Limitations and Risks

**Still Needs More Compute or Data**

- Repeated runs with multiple random seeds.
- Cross-validation.
- Longer training, such as 100 epochs.
- External test set validation.
- Generalization evaluation across MRI sequences or acquisition devices.

**Current Risks**

- Dataset size is limited, so results may depend on seed and split.
- The highest-recall OR strategy has low specificity and many false positives.
- Occlusion heatmaps explain model sensitivity, not clinical meaning.
- Without an external clinical test set, this cannot be described as clinically deployable.

**Speaker Notes**

This slide keeps the report honest. The project has a complete experimental loop, but it is still a research workflow, not a finished medical product.

---

## Slide 24 | Reproduction Commands

**Create environment**

```bash
conda env create -f environment.yml
conda activate brain-tumor-yolo
```

**Data audit**

```bash
python data_audit.py --output-dir runs/data_audit
```

**Train detectors**

```bash
python train.py --experiment baseline
python train.py --experiment medical_nano_640
python train.py --experiment medical_small_640
python train.py --experiment medical_medium_640
```

**Generate screening reports**

```bash
python screening_report.py \
  --weights runs/detect/train/weights/best.pt \
  --output-dir runs/screening_report_baseline

python screening_report.py \
  --weights runs/experiments/medical_nano_640/weights/best.pt \
  --imgsz 640 \
  --output-dir runs/screening_report_medical_nano_640
```

**False negative analysis**

```bash
python false_negative_analysis.py \
  --weights runs/detect/train/weights/best.pt \
  --threshold 0.25 \
  --output-dir runs/fn_analysis_baseline_t025

python false_negative_analysis.py \
  --weights runs/experiments/medical_nano_640/weights/best.pt \
  --imgsz 640 \
  --threshold 0.25 \
  --output-dir runs/fn_analysis_medical_nano_640_t025
```

**Classifier assist and heatmaps**

```bash
python classification_assist.py \
  --skip-train \
  --classifier-weights runs/classification/brain_tumor_cls_nano/weights/best.pt \
  --detector-weights runs/experiments/medical_nano_640/weights/best.pt \
  --detector-imgsz 640 \
  --detector-threshold 0.25 \
  --output-dir runs/classification_assist_medical_nano_640

python interpretability_heatmap.py \
  --weights runs/classification/brain_tumor_cls_nano/weights/best.pt \
  --output-dir runs/interpretability_heatmaps
```

**One-command helpers**

```bash
make audit
make screening
make fn
make classify
make heatmap
make summary

./run.sh all
```

---

## Slide 25 | Final Takeaways

**Takeaway 1: the baseline problem is false negatives.**

At threshold 0.25, baseline image-level recall is only 0.407 and it misses 48 positive images. The FN analysis shows that most missed boxes are very small.

**Takeaway 2: the 640px medical detector reduces missed positives.**

`medical_nano_640` reduces image-level FN from 48 to 19 and improves mAP50-95 from 0.284 to 0.373.

**Takeaway 3: classifier OR detector is best for high-recall screening.**

The best combined strategy reaches recall 0.975 and reduces FN to 2. The tradeoff is lower specificity at 0.141, so downstream review is needed.

**Takeaway 4: the next work is clear.**

Improve small-object recall and control false positives through longer training, multiple seeds, crop-based small-lesion training, external testing, and a stronger second-stage review strategy.
