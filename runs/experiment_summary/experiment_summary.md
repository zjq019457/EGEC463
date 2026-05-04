# Experiment Summary

## Detection Experiments

| Experiment | Best epoch | Precision | Recall | mAP50 | mAP50-95 | Artifacts |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| runs/detect/train | 9 | 0.469 | 0.810 | 0.508 | 0.284 | runs/detect/train |
| runs/detect/train2 | 9 | 0.469 | 0.810 | 0.508 | 0.284 | runs/detect/train2 |
| runs/experiments/medical_focus_small_640 | 1 | 0.359 | 0.705 | 0.418 | 0.245 | runs/experiments/medical_focus_small_640 |
| runs/experiments/medical_nano_640 | 41 | 0.492 | 0.740 | 0.540 | 0.373 | runs/experiments/medical_nano_640 |
| runs/experiments/medical_small_640 | 24 | 0.468 | 0.826 | 0.526 | 0.363 | runs/experiments/medical_small_640 |
| runs/experiments/medical_medium_640 | 25 | 0.452 | 0.819 | 0.515 | 0.358 | runs/experiments/medical_medium_640 |

## Screening Strategies

| Strategy | Classifier threshold | Detector threshold | Precision | Recall | Specificity | F1 | FN | FP |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| detector_only |  | 0.25 | 0.373 | 0.765 | 0.268 | 0.502 | 19 | 104 |
| classifier_only | 0.3 |  | 0.528 | 0.704 | 0.641 | 0.603 | 24 | 51 |
| classifier_or_detector | 0.3 | 0.25 | 0.393 | 0.975 | 0.141 | 0.560 | 2 | 122 |

## Existing Ultralytics Evaluation Artifacts

- `PR_curve.png`: precision-recall curve
- `confusion_matrix.png`: confusion matrix
- `confusion_matrix_normalized.png`: normalized confusion matrix
- `results.csv`: per-epoch precision, recall, mAP50, and mAP50-95

Note: full nano/small/medium 640px comparisons are complete. Multi-run stability and cross-validation still require additional runs.
