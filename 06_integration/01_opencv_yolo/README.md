# 01_opencv_yolo - OpenCV + YOLO 集成 🔗

## 学习目标

- 在 OpenCV 工作流中使用 YOLO
- 自定义检测结果可视化
- 结果后处理

## 基本集成

```python
import cv2
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

# 读取图像
img = cv2.imread("image.jpg")

# YOLO 检测（直接接受 OpenCV 图像）
results = model(img, device="mps")

# 获取结果并可视化
annotated = results[0].plot()
cv2.imshow("Detection", annotated)
```

## 自定义可视化

```python
result = results[0]

for box in result.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
    conf = box.conf[0].item()
    cls_id = int(box.cls[0].item())
    label = f"{model.names[cls_id]} {conf:.0%}"
    
    # 绘制边界框
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # 绘制标签背景
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(img, (x1, y1-h-10), (x1+w, y1), (0, 255, 0), -1)
    
    # 绘制标签文字
    cv2.putText(img, label, (x1, y1-5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
```

## 实时检测流程

```python
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model(frame, device="mps", verbose=False)
    annotated = results[0].plot()
    
    cv2.imshow("Realtime", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## 文件列表

| 文件 | 内容 |
|-----|------|
| `realtime_detection.py` | 实时检测完整示例 |

## 运行

```bash
conda activate yolo
python realtime_detection.py
```

