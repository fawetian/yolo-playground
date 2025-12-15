# 01_intro - YOLO 入门 🚀

## 学习目标

- 理解 YOLO 的基本概念
- 安装和配置 Ultralytics
- 运行第一个目标检测

## YOLO 是什么

**YOLO (You Only Look Once)** 是一种实时目标检测算法：
- 将图像划分为网格
- 每个网格预测边界框和类别
- 单次前向传播完成检测（因此叫 "只看一次"）

## 核心 API

### 加载模型
```python
from ultralytics import YOLO

# 加载预训练模型
model = YOLO("yolo11n.pt")  # 自动下载

# 查看模型信息
print(model.names)  # 可检测的类别
```

### 推理
```python
# 图像推理
results = model("image.jpg")

# 指定设备
results = model("image.jpg", device="mps")  # Apple Silicon

# 批量推理
results = model(["img1.jpg", "img2.jpg"])
```

### 解析结果
```python
result = results[0]

for box in result.boxes:
    # 边界框
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    
    # 置信度
    confidence = box.conf[0].item()
    
    # 类别
    class_id = int(box.cls[0].item())
    class_name = model.names[class_id]
```

### 可视化
```python
# 获取带标注的图像
annotated = result.plot()
cv2.imshow("Detection", annotated)
```

## 文件列表

| 文件 | 内容 |
|-----|------|
| `01_yolo_quickstart.py` | YOLO 快速入门 |

## 练习

1. 下载不同大小的模型，对比推理速度
2. 用你自己的图片进行检测
3. 调整置信度阈值，观察检测结果变化

## 运行

```bash
conda activate yolo
python 01_yolo_quickstart.py
```

