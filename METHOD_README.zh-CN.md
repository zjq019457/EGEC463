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

分类器使用 `YOLO11n-cls`，在 CPU 上短训练了 5 个 epoch，设置为 `imgsz=224`、`batch=16`。

| 策略 | 阈值 | Precision | Recall | Specificity | F1 | 漏检数量 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 只用检测器 | detector=0.25 | 0.384 | 0.407 | 0.627 | 0.395 | 48 |
| 只用分类器 | classifier=0.30 | 0.528 | 0.704 | 0.641 | 0.603 | 24 |
| 分类器 OR 检测器 | classifier=0.30, detector=0.25 | 0.443 | 0.864 | 0.380 | 0.586 | 11 |

主要提升：

- 漏检数量从 `48` 降到 `11`。
- Recall 从 `0.407` 提升到 `0.864`。
- 组合筛查策略的 F1 从 `0.395` 提升到 `0.586`。

代价：

- Specificity 从 `0.627` 降到 `0.380`。
- 组合模型能找回更多 positive，但误报也会增加。

这是筛查任务里常见的取舍：模型更敏感，但选择性会下降。

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

### 4. 运行 false negative 分析

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

### 5. 训练并评估分类辅助模型

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

### 6. 使用已有分类器重新评估

```bash
python classification_assist.py \
  --skip-train \
  --classifier-weights runs/classification/brain_tumor_cls_nano/weights/best.pt \
  --output-dir runs/classification_assist_nano_5e
```

## 重要输出文件

### 阈值筛查报告

- `runs/screening_report_baseline/screening_report.html`
- `runs/screening_report_baseline/threshold_metrics.csv`
- `runs/screening_report_baseline/image_scores.csv`

### False negative 分析

- `runs/fn_analysis_baseline_t025/false_negative_report.html`
- `runs/fn_analysis_baseline_t025/false_negative_boxes.csv`
- `runs/fn_analysis_baseline_t025/fn_visuals/`

### 分类辅助模型

- `runs/classification/brain_tumor_cls_nano/weights/best.pt`
- `runs/classification_assist_nano_5e/classification_assist_report.html`
- `runs/classification_assist_nano_5e/strategy_metrics.csv`
- `runs/classification_assist_nano_5e/classification_scores.csv`

## 下一步怎么继续改进

FN 分析说明，小肿瘤区域是当前检测器的主要弱点。下一步如果想提升 YOLO 检测器本体，应该重点提升小目标召回：

- 使用 `imgsz=640` 训练 YOLO。
- 使用更轻的医学图像增强。
- 把训练轮数提高到 50 或 100。
- 尝试围绕小 positive 框做 crop-based 训练。
- 保留分类器作为筛查入口，让 YOLO 负责定位。
