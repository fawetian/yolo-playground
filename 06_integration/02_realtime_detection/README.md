# 02_realtime_detection - 实时检测系统 ⚡

## 学习目标

- 优化实时检测性能
- 添加检测统计功能
- 实现帧率控制

## 性能优化技巧

### 1. 选择合适的模型
```python
# 速度优先
model = YOLO("yolo11n.pt")

# 精度优先
model = YOLO("yolo11m.pt")
```

### 2. 调整输入尺寸
```python
# 降低分辨率提高速度
results = model(frame, imgsz=320)
```

### 3. 跳帧处理
```python
frame_count = 0
skip_frames = 2  # 每 3 帧检测一次

while True:
    ret, frame = cap.read()
    frame_count += 1
    
    if frame_count % (skip_frames + 1) == 0:
        results = model(frame)
        last_result = results[0]
    
    # 使用上次的检测结果绘制
    annotated = last_result.plot()
```

### 4. 使用追踪而非检测
```python
# 使用内置追踪器
results = model.track(frame, persist=True)
```

## 检测统计

```python
from collections import defaultdict

class_counts = defaultdict(int)

for box in result.boxes:
    cls_id = int(box.cls[0].item())
    class_counts[model.names[cls_id]] += 1

print(f"检测到: {dict(class_counts)}")
```

## FPS 计算

```python
import time

fps_start = time.time()
frame_count = 0

while True:
    frame_count += 1
    
    # 每秒更新一次 FPS
    if frame_count % 30 == 0:
        fps = 30 / (time.time() - fps_start)
        fps_start = time.time()
        print(f"FPS: {fps:.1f}")
```

## 录制检测视频

```python
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 30, (width, height))

while True:
    ret, frame = cap.read()
    results = model(frame)
    annotated = results[0].plot()
    
    out.write(annotated)
    cv2.imshow("Recording", annotated)

out.release()
```

## 待创建文件

- `01_optimized_detection.py` - 优化的实时检测
- `02_detection_statistics.py` - 检测统计
- `03_video_recording.py` - 视频录制

