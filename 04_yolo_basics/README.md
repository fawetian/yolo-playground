# 04 - YOLO 基础 🎯

## 本模块内容

学习 YOLO (You Only Look Once) 目标检测框架的基础使用。

## 子目录

| 目录 | 内容 | 预计时间 |
|-----|------|---------|
| `01_intro/` | YOLO 入门 | 2 天 |
| `02_detection/` | 目标检测 | 3 天 |
| `03_segmentation/` | 实例分割 | 2 天 |
| `04_classification/` | 图像分类 | 1 天 |
| `05_pose_estimation/` | 姿态估计 | 2 天 |

## 学习目标

完成本模块后，你将掌握：

- [ ] 理解 YOLO 的工作原理
- [ ] 使用预训练模型进行目标检测
- [ ] 理解检测结果的数据结构
- [ ] 使用 YOLO 进行分割、分类、姿态估计
- [ ] 在 Apple Silicon 上使用 MPS 加速

## YOLO 模型系列

| 模型 | 参数量 | 速度 | 精度 |
|-----|--------|------|------|
| YOLO11n | 最小 | 最快 | 一般 |
| YOLO11s | 小 | 快 | 较好 |
| YOLO11m | 中 | 中 | 好 |
| YOLO11l | 大 | 慢 | 很好 |
| YOLO11x | 最大 | 最慢 | 最好 |

## 快速开始

```bash
conda activate yolo

# 运行第一个 YOLO 示例
python 01_intro/01_yolo_quickstart.py
```

## macOS MPS 加速

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
# Apple Silicon 使用 MPS 加速
results = model("image.jpg", device="mps")
```

## 下一步

完成后进入 `05_yolo_training/` 学习自定义模型训练。

