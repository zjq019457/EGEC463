#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-all}"

case "$MODE" in
  audit)
    python data_audit.py --output-dir runs/data_audit
    ;;
  train)
    python train.py --experiment baseline
    ;;
  train-medical-nano)
    python train.py --experiment medical_nano_640
    ;;
  train-medical-small)
    python train.py --experiment medical_small_640
    ;;
  train-medical-medium)
    python train.py --experiment medical_medium_640
    ;;
  screening)
    python screening_report.py --weights runs/detect/train/weights/best.pt --output-dir runs/screening_report_baseline
    ;;
  fn)
    python false_negative_analysis.py --weights runs/detect/train/weights/best.pt --threshold 0.25 --output-dir runs/fn_analysis_baseline_t025
    ;;
  classify)
    python classification_assist.py --epochs 5 --imgsz 224 --batch 16 --output-dir runs/classification_assist_nano_5e
    ;;
  heatmap)
    python interpretability_heatmap.py --weights runs/classification/brain_tumor_cls_nano/weights/best.pt --output-dir runs/interpretability_heatmaps
    ;;
  summary)
    python experiment_summary.py --output-dir runs/experiment_summary
    ;;
  all)
    python data_audit.py --output-dir runs/data_audit
    python screening_report.py --weights runs/detect/train/weights/best.pt --output-dir runs/screening_report_baseline
    python false_negative_analysis.py --weights runs/detect/train/weights/best.pt --threshold 0.25 --output-dir runs/fn_analysis_baseline_t025
    python classification_assist.py --skip-train --classifier-weights runs/classification/brain_tumor_cls_nano/weights/best.pt --output-dir runs/classification_assist_nano_5e
    python interpretability_heatmap.py --weights runs/classification/brain_tumor_cls_nano/weights/best.pt --output-dir runs/interpretability_heatmaps
    python experiment_summary.py --output-dir runs/experiment_summary
    ;;
  *)
    echo "Usage: ./run.sh [audit|train|train-medical-nano|train-medical-small|train-medical-medium|screening|fn|classify|heatmap|summary|all]"
    exit 1
    ;;
esac
