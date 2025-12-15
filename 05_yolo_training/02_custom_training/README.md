# 02_custom_training - 自定义训练 🎓

## 学习目标

- 使用自定义数据集训练模型
- 理解训练参数
- 监控训练过程

## 训练流程

### 1. 准备数据集
确保数据集格式正确，参考 `01_dataset_prep/`

### 2. 创建配置文件
```yaml
# data.yaml
path: /path/to/dataset
train: train/images
val: val/images
nc: 2
names: ['class1', 'class2']
```

### 3. 开始训练
```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")  # 预训练模型
results = model.train(
    data="data.yaml",
    epochs=100,
    device="mps",
    batch=8,
)
```

## 常用训练参数

| 参数 | 说明 | 建议值 |
|-----|------|--------|
| `epochs` | 训练轮数 | 100-300 |
| `batch` | 批次大小 | 8-16 (MPS) |
| `imgsz` | 图像尺寸 | 640 |
| `lr0` | 初始学习率 | 0.01 |
| `patience` | 早停耐心值 | 50 |
| `device` | 设备 | "mps" |

## 训练输出

训练结果保存在 `runs/train/` 目录：
```
runs/train/exp/
├── weights/
│   ├── best.pt      # 最佳模型
│   └── last.pt      # 最后模型
├── results.csv      # 训练指标
├── confusion_matrix.png
└── ...
```

## 恢复训练

```python
# 从中断处继续
model = YOLO("runs/train/exp/weights/last.pt")
model.train(resume=True)
```

## 验证模型

```python
model = YOLO("runs/train/exp/weights/best.pt")
metrics = model.val()

print(f"mAP50: {metrics.box.map50:.3f}")
print(f"mAP50-95: {metrics.box.map:.3f}")
```

## 文件列表

| 文件 | 内容 |
|-----|------|
| `train_custom_model.py` | 训练脚本模板 |

## 运行

```bash
conda activate yolo
python train_custom_model.py
```

