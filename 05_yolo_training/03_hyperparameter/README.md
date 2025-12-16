# 03_hyperparameter - 超参数调优 ⚙️

## 学习目标

- 理解关键超参数的作用
- 使用超参数搜索
- 优化模型性能

## 关键超参数

### 学习率相关
| 参数 | 说明 | 默认值 |
|-----|------|--------|
| `lr0` | 初始学习率 | 0.01 |
| `lrf` | 最终学习率比例 | 0.01 |
| `momentum` | SGD 动量 | 0.937 |
| `weight_decay` | 权重衰减 | 0.0005 |

### 数据增强
| 参数 | 说明 | 默认值 |
|-----|------|--------|
| `hsv_h` | 色相增强 | 0.015 |
| `hsv_s` | 饱和度增强 | 0.7 |
| `hsv_v` | 明度增强 | 0.4 |
| `degrees` | 旋转角度 | 0.0 |
| `translate` | 平移比例 | 0.1 |
| `scale` | 缩放比例 | 0.5 |
| `mosaic` | Mosaic 增强 | 1.0 |
| `mixup` | MixUp 增强 | 0.0 |

### 损失权重
| 参数 | 说明 | 默认值 |
|-----|------|--------|
| `box` | 边界框损失权重 | 7.5 |
| `cls` | 分类损失权重 | 0.5 |
| `dfl` | DFL 损失权重 | 1.5 |

## 超参数调优

### 手动调优
```python
model.train(
    data="data.yaml",
    epochs=100,
    lr0=0.001,       # 尝试更小的学习率
    mosaic=0.5,      # 减少 mosaic
    degrees=10,      # 添加旋转增强
)
```

### 自动超参数搜索
```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
result = model.tune(
    data="data.yaml",
    epochs=30,
    iterations=100,
    device="mps",
)
```

## 调优建议

### 过拟合
- 增加数据增强
- 减小模型大小
- 增加 dropout
- 使用早停

### 欠拟合
- 增加训练轮数
- 使用更大的模型
- 增加学习率

### 训练不稳定
- 减小学习率
- 增加 warmup 轮数
- 减小 batch size

## 示例脚本

| 脚本 | 功能 |
|-----|-----|
| `01_custom_cfg.py` | 使用自定义超参数 (学习率、优化器等) 进行训练 |

## 运行

```bash
python 01_custom_cfg.py
```

## 待创建文件

- `01_hyperparameter_guide.py` - 超参数指南
- `02_auto_tuning.py` - 自动调优示例
