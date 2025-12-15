# 02_detection - 目标检测 🔍

## 学习目标

- 深入理解目标检测的输出
- 批量检测与参数配置
- 结果过滤与后处理

## 检测参数

```python
results = model(
    source,            # 输入源
    conf=0.25,         # 置信度阈值
    iou=0.45,          # NMS IoU 阈值
    classes=[0, 2],    # 只检测特定类别
    imgsz=640,         # 输入图像尺寸
    device="mps",      # 设备
    verbose=False      # 静默模式
)
```

## 输入源类型

```python
# 图像文件
results = model("image.jpg")

# 目录
results = model("images/")

# URL
results = model("https://example.com/image.jpg")

# 视频
results = model("video.mp4")

# 摄像头
results = model(0)

# NumPy 数组
results = model(cv2_image)
```

## 结果数据结构

```python
result = results[0]

# 原始图像
result.orig_img

# 边界框
result.boxes.xyxy   # [x1, y1, x2, y2]
result.boxes.xywh   # [x, y, w, h]
result.boxes.conf   # 置信度
result.boxes.cls    # 类别 ID

# 转换为 numpy
boxes_np = result.boxes.xyxy.cpu().numpy()
```

## 结果过滤

```python
boxes = result.boxes

# 按置信度过滤
high_conf = boxes[boxes.conf > 0.7]

# 按类别过滤
persons = boxes[boxes.cls == 0]

# 按面积过滤
areas = (boxes.xyxy[:, 2] - boxes.xyxy[:, 0]) * \
        (boxes.xyxy[:, 3] - boxes.xyxy[:, 1])
large = boxes[areas > 10000]
```

## 文件列表

| 文件 | 内容 |
|-----|------|
| `01_batch_detection.py` | 批量检测示例 |

## 练习

1. 对一个视频进行检测并保存结果
2. 实现只检测 "person" 和 "car" 的过滤器
3. 统计一张图片中各类别的数量

## 运行

```bash
conda activate yolo
python 01_batch_detection.py
```

