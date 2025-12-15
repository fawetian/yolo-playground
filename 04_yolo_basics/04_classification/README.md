# 04_classification - 图像分类 🏷️

## 学习目标

- 理解图像分类与目标检测的区别
- 使用 YOLO 分类模型
- 处理分类结果

## 图像分类 vs 目标检测

| 任务 | 输出 | 问题 |
|-----|------|------|
| 分类 | 整图类别 | "这是什么？" |
| 检测 | 多个目标位置 | "图中有什么，在哪？" |

## 核心 API

### 加载分类模型
```python
from ultralytics import YOLO

# 分类模型以 -cls 结尾
model = YOLO("yolo11n-cls.pt")
```

### 推理
```python
results = model("image.jpg", device="mps")
result = results[0]

# 获取分类结果
probs = result.probs
```

### 访问分类结果
```python
# Top-1 预测
top1_idx = probs.top1
top1_conf = probs.top1conf.item()
top1_name = result.names[top1_idx]

print(f"预测: {top1_name} ({top1_conf:.2%})")

# Top-5 预测
top5_idx = probs.top5
top5_conf = probs.top5conf.tolist()

for idx, conf in zip(top5_idx, top5_conf):
    print(f"  {result.names[idx]}: {conf:.2%}")
```

### 获取所有概率
```python
all_probs = probs.data.cpu().numpy()
# all_probs[i] 是类别 i 的概率
```

## 可视化
```python
annotated = result.plot()
cv2.imshow("Classification", annotated)
```

## 实际应用

### 批量图片分类
```python
from pathlib import Path

results = model(list(Path("images").glob("*.jpg")))

for result in results:
    img_path = result.path
    top_class = result.names[result.probs.top1]
    confidence = result.probs.top1conf.item()
    print(f"{img_path}: {top_class} ({confidence:.2%})")
```

## 待创建文件

- `01_classification_basic.py` - 分类基础
- `02_batch_classification.py` - 批量分类

