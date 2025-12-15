# 02_video_processing - 视频处理 🎞️

## 学习目标

- 对视频帧进行处理
- 保存处理后的视频
- 视频格式转换

## 核心 API

### 保存视频
```python
# 创建 VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 编码器
out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

# 写入帧
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 处理帧
    processed = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)  # 转回 3 通道
    
    out.write(processed)

out.release()
```

### 常用编码器 (macOS)

| FourCC | 格式 | 说明 |
|--------|------|------|
| `mp4v` | .mp4 | MPEG-4，兼容性好 |
| `avc1` | .mp4 | H.264，质量好 |
| `XVID` | .avi | 广泛支持 |

### 实时帧处理
```python
while True:
    ret, frame = cap.read()
    
    # 添加时间戳
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, timestamp, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # 添加帧率
    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(frame, fps_text, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Processed", frame)
```

## 批量视频处理

```python
from pathlib import Path

video_dir = Path("videos")
for video_path in video_dir.glob("*.mp4"):
    process_video(video_path)
```

## 待创建文件

- `01_video_save.py` - 视频保存
- `02_frame_processing.py` - 帧处理
- `03_video_effects.py` - 视频特效

