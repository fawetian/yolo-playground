# 03_segmentation - 实例分割 🎭

## 学习目标

- 理解实例分割与目标检测的区别
- 使用 YOLO 分割模型
- 处理分割掩码

## 实例分割 vs 目标检测

| 任务 | 输出 | 用途 |
|-----|------|------|
| 目标检测 | 边界框 | 定位物体 |
| 实例分割 | 边界框 + 掩码 | 精确轮廓 |

## 核心 API

### 加载分割模型
```python
from ultralytics import YOLO

# 分割模型以 -seg 结尾
model = YOLO("yolo11n-seg.pt")
```

### 推理
```python
results = model("image.jpg", device="mps")
result = results[0]

# 获取掩码
masks = result.masks
```

### 访问掩码数据
```python
if result.masks is not None:
    # 所有掩码的二进制数据
    masks_data = result.masks.data.cpu().numpy()
    
    # 原始图像尺寸的掩码
    masks_orig = result.masks.orig_shape
    
    # 每个目标的掩码
    for i, mask in enumerate(masks_data):
        # mask 是一个 0-1 的数组
        binary_mask = (mask > 0.5).astype(np.uint8) * 255
```

### 提取分割区域
```python
# 获取特定目标的分割区域
mask = masks_data[0]
mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))
mask_binary = (mask_resized > 0.5).astype(np.uint8)

# 应用掩码
segmented = cv2.bitwise_and(img, img, mask=mask_binary)
```

### 可视化
```python
# 自动绘制分割结果
annotated = result.plot()
cv2.imshow("Segmentation", annotated)
```

## 实际应用

### 背景移除
```python
# 假设检测到人物在索引 0
person_mask = masks_data[0]
person_mask = cv2.resize(person_mask, (w, h))

# 创建透明背景
rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
rgba[:, :, 3] = (person_mask * 255).astype(np.uint8)
```

## 文件列表

| 文件 | 内容 |
|-----|------|
| `01_segmentation_basic.py` | 分割基础 - 模型加载、推理、结果解析 |
| `02_mask_processing.py` | 掩码处理 - 尺寸对齐、目标提取、轮廓检测 |
| `03_background_removal.py` | 背景移除 - 透明背景、背景替换、边缘羽化 |

## 运行

```bash
conda activate yolo
python 01_segmentation_basic.py
python 02_mask_processing.py
python 03_background_removal.py
```

