# 03_video_analysis - 视频分析 📊

## 学习目标

- 分析视频中的检测结果
- 生成统计报告
- 目标追踪与计数

## 视频分析流程

```
视频输入 → 逐帧检测 → 数据收集 → 统计分析 → 报告生成
```

## 目标计数

### 使用检测
```python
from collections import defaultdict

frame_stats = []

cap = cv2.VideoCapture("video.mp4")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model(frame)
    counts = defaultdict(int)
    
    for box in results[0].boxes:
        cls_name = model.names[int(box.cls[0].item())]
        counts[cls_name] += 1
    
    frame_stats.append(counts)
```

### 使用追踪 (更准确)
```python
from collections import defaultdict

unique_ids = defaultdict(set)

cap = cv2.VideoCapture("video.mp4")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 使用追踪获取唯一 ID
    results = model.track(frame, persist=True)
    
    if results[0].boxes.id is not None:
        for box in results[0].boxes:
            cls_name = model.names[int(box.cls[0].item())]
            track_id = int(box.id[0].item())
            unique_ids[cls_name].add(track_id)

# 统计唯一目标数量
for cls_name, ids in unique_ids.items():
    print(f"{cls_name}: {len(ids)} 个唯一目标")
```

## 越线计数

```python
# 定义计数线
line_y = 300

crossed_ids = set()
count = 0

while True:
    results = model.track(frame, persist=True)
    
    for box in results[0].boxes:
        if box.id is None:
            continue
        
        track_id = int(box.id[0].item())
        _, y1, _, y2 = box.xyxy[0].tolist()
        center_y = (y1 + y2) / 2
        
        # 检测是否越过计数线
        if center_y > line_y and track_id not in crossed_ids:
            crossed_ids.add(track_id)
            count += 1
    
    # 绘制计数线
    cv2.line(frame, (0, line_y), (width, line_y), (0, 255, 0), 2)
    cv2.putText(frame, f"Count: {count}", (10, 50), ...)
```

## 生成报告

```python
import pandas as pd
import matplotlib.pyplot as plt

# 创建 DataFrame
df = pd.DataFrame(frame_stats)
df.index.name = 'frame'

# 保存 CSV
df.to_csv("detection_report.csv")

# 生成图表
df.plot(kind='line', figsize=(12, 6))
plt.title("Detection Over Time")
plt.xlabel("Frame")
plt.ylabel("Count")
plt.savefig("detection_chart.png")
```

## 待创建文件

- `01_video_analysis.py` - 视频分析基础
- `02_object_counting.py` - 目标计数
- `03_report_generation.py` - 报告生成

