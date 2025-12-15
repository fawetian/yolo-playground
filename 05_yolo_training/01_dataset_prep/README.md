# 01_dataset_prep - 数据集准备 📁

## 学习目标

- 理解 YOLO 数据集格式
- 使用标注工具标注图像
- 划分训练集/验证集

## YOLO 数据集结构

```
dataset/
├── train/
│   ├── images/
│   │   ├── img001.jpg
│   │   └── img002.jpg
│   └── labels/
│       ├── img001.txt
│       └── img002.txt
├── val/
│   ├── images/
│   └── labels/
└── data.yaml
```

## 标注格式

每行一个目标：
```
<class_id> <x_center> <y_center> <width> <height>
```

所有坐标归一化到 0-1：
```
x_center = (x1 + x2) / 2 / image_width
y_center = (y1 + y2) / 2 / image_height
width = (x2 - x1) / image_width
height = (y2 - y1) / image_height
```

## data.yaml 配置

```yaml
path: /path/to/dataset
train: train/images
val: val/images

nc: 2  # 类别数量
names:
  0: cat
  1: dog
```

## 推荐标注工具

### LabelMe (推荐)
```bash
conda activate yolo
pip install labelme
labelme
```

### 在线工具
- CVAT: https://www.cvat.ai/
- Roboflow: https://roboflow.com/

## 文件列表

| 文件 | 内容 |
|-----|------|
| `dataset_format.md` | 详细格式说明 |

## 练习

1. 收集 50+ 张图片
2. 使用 LabelMe 标注
3. 转换为 YOLO 格式
4. 按 8:2 划分训练/验证集

## 数据增强建议

Ultralytics 自动进行数据增强，也可以手动配置：
```yaml
# 在训练时指定
augment: True
hsv_h: 0.015  # 色相
hsv_s: 0.7    # 饱和度
hsv_v: 0.4    # 明度
degrees: 10   # 旋转
translate: 0.1
scale: 0.5
flipud: 0.5   # 垂直翻转
fliplr: 0.5   # 水平翻转
```

