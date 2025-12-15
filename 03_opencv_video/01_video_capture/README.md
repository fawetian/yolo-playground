# 01_video_capture - 视频捕获 📹

## 学习目标

- 从摄像头读取视频流
- 从视频文件读取帧
- 获取视频属性

## 核心 API

### 打开视频源
```python
# 摄像头 (0 为默认摄像头)
cap = cv2.VideoCapture(0)

# 视频文件
cap = cv2.VideoCapture("video.mp4")

# 网络流
cap = cv2.VideoCapture("rtsp://...")
```

### 读取帧
```python
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### 获取视频属性
```python
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
```

### 设置属性
```python
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
```

### 跳转到指定帧
```python
cap.set(cv2.CAP_PROP_POS_FRAMES, 100)  # 跳转到第 100 帧
cap.set(cv2.CAP_PROP_POS_MSEC, 5000)   # 跳转到第 5 秒
```

## macOS 摄像头权限

```python
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("请在系统设置中授予摄像头权限")
    print("系统设置 → 隐私与安全性 → 摄像头")
```

## 待创建文件

- `01_camera_capture.py` - 摄像头捕获
- `02_video_file.py` - 视频文件读取
- `03_video_properties.py` - 视频属性获取

