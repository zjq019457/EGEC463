# 脑肿瘤筛查方法说明

这个项目把原来的 YOLO11 脑肿瘤检测模型扩展成了一个更偏“筛查”的工作流。目标不只是画出肿瘤框，而是尽量减少漏掉 positive MRI 图片的情况。

最终方法包含三部分：

- YOLO 检测器：在 MRI 图像中定位 `negative` 和 `positive` 区域。
- False negative 专项分析：找出检测器漏掉的 positive 图片，并分析这些样本为什么难。
- 分类模型辅助：训练一个 image-level 的 `positive` / `negative` 分类器，再和 YOLO 检测器组合使用。

这个项目只用于模型开发和研究流程，不是临床诊断工具。

## 方法介绍

### 1. YOLO 检测 baseline

原始模型使用 YOLO11 做目标检测。它会预测 MRI 图像中的检测框和类别标签。

在检测阈值 `0.25` 下，baseline 检测器结果如下：

| 策略 | Precision | Recall | Specificity | F1 | 漏检数量 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 只用检测器 | 0.384 | 0.407 | 0.627 | 0.395 | 48 |

这说明检测器在常规阈值下比较保守，会漏掉很多 positive 病例。

### 2. False negative 专项分析

医学筛查里最需要关注的是 false negative，因为它表示真实 positive 被模型漏掉。FN 分析脚本会找出验证集中真实为 positive、但检测器没有给出足够高置信度 positive 检测框的图片。

在阈值 `0.25` 下，baseline 检测器漏掉了：

- 48 张 positive 图片
- 53 个 positive 框
- 其中 48 个漏检框面积小于整张图的 2%
- 漏检框面积中位数是 `0.0081`

这说明 baseline 的主要问题很明确：模型容易漏掉小肿瘤区域。

### 3. 分类模型辅助

检测模型适合定位肿瘤，但如果问题是“这张 MRI 有没有肿瘤”，image-level 分类模型往往更直接。

分类辅助流程做了这些事：

1. 把 YOLO 检测标签转换成图片级分类标签。
2. 训练一个 YOLO11 分类模型，类别是 `positive` 和 `negative`。
3. 比较三种策略：

- 只用检测器
- 只用分类器
- 分类器 OR 检测器

组合规则是：

```text
如果满足任一条件，就判断为 positive：
  检测器 positive 置信度 >= 检测阈值
  或 分类器 positive 概率 >= 分类阈值
```

## 改进效果

### 检测模型实验结果

当前已经完成的检测实验如下：

| 实验 | 模型 / 设置 | 最佳 mAP50 epoch | Precision | Recall | 最佳 mAP50 | 最佳 mAP50-95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `runs/detect/train` | YOLO11n，320px，原始增强 | 9 | 0.469 | 0.810 | 0.508 | 0.284 |
| `runs/detect/train2` | 重复 YOLO11n baseline | 9 | 0.469 | 0.810 | 0.508 | 0.284 |
| `runs/experiments/medical_focus_small_640` | YOLO11s，640px，医学轻增强，pilot 实验 | 1 | 0.359 | 0.705 | 0.418 | 0.245 |
| `runs/experiments/medical_nano_640` | YOLO11n，640px，医学轻增强 | 41 | 0.492 | 0.740 | 0.540 | 0.373 |
| `runs/experiments/medical_small_640` | YOLO11s，640px，医学轻增强 | 24 | 0.468 | 0.826 | 0.526 | 0.363 |
| `runs/experiments/medical_medium_640` | YOLO11m，640px，医学轻增强 | 25 | 0.452 | 0.819 | 0.515 | 0.358 |

Ultralytics 已经生成了标准检测评估文件：

- `PR_curve.png`
- `confusion_matrix.png`
- `confusion_matrix_normalized.png`
- `results.csv`

目前最好的检测器是 `medical_nano_640`。相比原始 YOLO11n 320px baseline，它带来了这些提升：

- mAP50 从 `0.508` 提升到 `0.540`
- mAP50-95 从 `0.284` 提升到 `0.373`
- 在检测阈值 `0.25` 下，detector-only 漏检从 `48` 降到 `19`

这验证了前面 FN 分析得到的判断：提高输入分辨率并使用更轻的医学图像增强，有助于提升小目标召回。

### 筛查策略改进结果

分类器使用 `YOLO11n-cls`，在 CPU 上短训练了 5 个 epoch，设置为 `imgsz=224`、`batch=16`。

| 策略 | 阈值 | Precision | Recall | Specificity | F1 | 漏检数量 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Baseline 只用检测器 | detector=0.25 | 0.384 | 0.407 | 0.627 | 0.395 | 48 |
| 最佳检测器单独使用 | medical_nano_640, detector=0.25 | 0.373 | 0.765 | 0.268 | 0.502 | 19 |
| 只用分类器 | classifier=0.30 | 0.528 | 0.704 | 0.641 | 0.603 | 24 |
| 分类器 OR baseline 检测器 | classifier=0.30, detector=0.25 | 0.443 | 0.864 | 0.380 | 0.586 | 11 |
| 分类器 OR 最佳检测器 | classifier=0.30, medical_nano_640 detector=0.25 | 0.393 | 0.975 | 0.141 | 0.560 | 2 |

主要提升：

- YOLO 检测器本体改进后，detector-only 漏检从 `48` 降到 `19`。
- 使用“分类器 OR 最佳检测器”后，漏检进一步从 `48` 降到 `2`。
- 最高召回组合策略的 recall 从 `0.407` 提升到 `0.975`。

代价：

- 最高召回组合策略非常敏感，但误报较多。
- Specificity 从 `0.627` 降到 `0.141`。

这是筛查任务里常见的取舍：模型更敏感，但选择性会下降。

## 数据与错误分析

数据审计脚本会检查图片是否可读、标签是否缺失、类别分布、标签合法性和检测框大小。

当前数据统计如下：

| 数据集 | 图片数 | Positive 图片 | Negative 图片 | 检测框数 | 小框数量 <2% |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 893 | 459 | 434 | 925 | 612 |
| val | 223 | 81 | 142 | 241 | 179 |

可以看到，小框数量非常多。这和 false negative 分析结果一致：baseline 的主要问题是小目标召回不够。

## 已实现内容清单

- 训练配置：`train.py` 中已经包含 baseline、`medical_focus`、`medical_nano_640`、`medical_small_640`、`medical_medium_640`。
- 数据增强：已经包含原始强增强和医学图像轻增强两类设置。
- 评估流程：已经包含 mAP50、mAP50-95、precision、recall、PR curve、confusion matrix、阈值扫描和 F1。
- 错误分析：已经保存 FP/FN 案例，并对漏检 positive 框做面积、位置、对比度分析。
- 分类辅助模型：已经训练轻量 YOLO11 分类器，并比较 detector only、classifier only、combined 三种策略。
- 可解释性：已经加入分类器 occlusion heatmap。
- 工程复现：已经加入 `environment.yml`、`requirements.txt`、`.gitignore`、`run.sh` 和 `Makefile`。

## 尚未完整完成的内容

下面这些已经有可运行入口，但还需要更多计算资源来证明结果稳定性：

- 多随机种子重复实验。
- 交叉验证。
- 超过 50 epoch 的更长检测器训练。

## 如何运行

### 1. 创建 Conda 环境

```bash
conda env create -f environment.yml
conda activate brain-tumor-yolo
```

### 2. 训练或复用检测器

训练检测器：

```bash
python train.py
```

也可以直接复用已有检测器权重：

```text
runs/detect/train/weights/best.pt
```

### 3. 生成阈值筛查报告

```bash
python screening_report.py \
  --weights runs/detect/train/weights/best.pt \
  --output-dir runs/screening_report_baseline
```

打开报告：

```bash
xdg-open runs/screening_report_baseline/screening_report.html
```

生成最佳检测器的筛查报告：

```bash
python screening_report.py \
  --weights runs/experiments/medical_nano_640/weights/best.pt \
  --imgsz 640 \
  --output-dir runs/screening_report_medical_nano_640
```

### 4. 运行数据审计

```bash
python data_audit.py --output-dir runs/data_audit
```

重要输出：

- `runs/data_audit/data_audit_summary.md`
- `runs/data_audit/image_class_distribution.png`
- `runs/data_audit/box_area_histogram.png`

### 5. 运行 false negative 分析

```bash
python false_negative_analysis.py \
  --weights runs/detect/train/weights/best.pt \
  --threshold 0.25 \
  --output-dir runs/fn_analysis_baseline_t025
```

打开 FN 报告：

```bash
xdg-open runs/fn_analysis_baseline_t025/false_negative_report.html
```

运行最佳检测器的 FN 分析：

```bash
python false_negative_analysis.py \
  --weights runs/experiments/medical_nano_640/weights/best.pt \
  --imgsz 640 \
  --threshold 0.25 \
  --output-dir runs/fn_analysis_medical_nano_640_t025
```

### 6. 训练并评估分类辅助模型

```bash
python classification_assist.py \
  --epochs 5 \
  --imgsz 224 \
  --batch 16 \
  --output-dir runs/classification_assist_nano_5e
```

打开组合模型报告：

```bash
xdg-open runs/classification_assist_nano_5e/classification_assist_report.html
```

### 7. 使用已有分类器重新评估

```bash
python classification_assist.py \
  --skip-train \
  --classifier-weights runs/classification/brain_tumor_cls_nano/weights/best.pt \
  --output-dir runs/classification_assist_nano_5e
```

使用已有分类器和最佳检测器重新评估：

```bash
python classification_assist.py \
  --skip-train \
  --classifier-weights runs/classification/brain_tumor_cls_nano/weights/best.pt \
  --detector-weights runs/experiments/medical_nano_640/weights/best.pt \
  --detector-imgsz 640 \
  --detector-threshold 0.25 \
  --output-dir runs/classification_assist_medical_nano_640
```

### 8. 生成分类器可解释性热力图

```bash
python interpretability_heatmap.py \
  --weights runs/classification/brain_tumor_cls_nano/weights/best.pt \
  --output-dir runs/interpretability_heatmaps
```

### 9. 汇总所有已完成实验

```bash
python experiment_summary.py --output-dir runs/experiment_summary
```

### 10. 使用 Makefile 或 run.sh 一键复现

```bash
make audit
make screening
make fn
make heatmap
make summary
```

也可以运行：

```bash
./run.sh all
```

更长的检测模型对比实验：

```bash
python train.py --experiment medical_nano_640
python train.py --experiment medical_small_640
python train.py --experiment medical_medium_640
```

## 重要输出文件

### 阈值筛查报告

- `runs/screening_report_baseline/screening_report.html`
- `runs/screening_report_medical_nano_640/screening_report.html`
- `runs/screening_report_baseline/threshold_metrics.csv`
- `runs/screening_report_baseline/image_scores.csv`

### False negative 分析

- `runs/fn_analysis_baseline_t025/false_negative_report.html`
- `runs/fn_analysis_medical_nano_640_t025/false_negative_report.html`
- `runs/fn_analysis_baseline_t025/false_negative_boxes.csv`
- `runs/fn_analysis_baseline_t025/fn_visuals/`

### 分类辅助模型

- `runs/classification/brain_tumor_cls_nano/weights/best.pt`
- `runs/classification_assist_nano_5e/classification_assist_report.html`
- `runs/classification_assist_medical_nano_640/classification_assist_report.html`
- `runs/classification_assist_nano_5e/strategy_metrics.csv`
- `runs/classification_assist_nano_5e/classification_scores.csv`

### 数据审计、实验汇总和可解释性

- `runs/data_audit/data_audit_summary.md`
- `runs/experiment_summary/experiment_summary.md`
- `runs/interpretability_heatmaps/`

## 下一步怎么继续改进

FN 分析说明，小肿瘤区域是当前检测器的主要弱点。下一步如果想提升 YOLO 检测器本体，应该重点提升小目标召回：

- 使用 `imgsz=640` 训练 YOLO。
- 使用更轻的医学图像增强。
- 把训练轮数提高到 50 或 100。
- 尝试围绕小 positive 框做 crop-based 训练。
- 保留分类器作为筛查入口，让 YOLO 负责定位。
