# 03_optical_flow - 光流与运动检测 🌊

## 学习目标

- 理解光流的概念
- 使用背景减除检测运动
- 使用光流追踪物体

## 背景减除

### 创建背景减除器
```python
# MOG2 (推荐)
bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=16,
    detectShadows=True
)

# KNN
bg_subtractor = cv2.createBackgroundSubtractorKNN()
```

### 应用背景减除
```python
while True:
    ret, frame = cap.read()
    
    # 获取前景掩码
    fg_mask = bg_subtractor.apply(frame)
    
    # 形态学处理去噪
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    
    cv2.imshow("Foreground", fg_mask)
```

## 光流

### 稀疏光流 (Lucas-Kanade)
```python
# 检测特征点
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, maxCorners=100, 
                              qualityLevel=0.3, minDistance=7)

# 计算光流
p1, status, err = cv2.calcOpticalFlowPyrLK(
    old_gray, new_gray, p0, None,
    winSize=(15, 15),
    maxLevel=2
)

# 绘制轨迹
for i, (new, old) in enumerate(zip(p1, p0)):
    if status[i]:
        a, b = new.ravel()
        c, d = old.ravel()
        cv2.line(frame, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
        cv2.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)
```

### 稠密光流 (Farneback)
```python
flow = cv2.calcOpticalFlowFarneback(
    old_gray, new_gray, None,
    pyr_scale=0.5, levels=3, winsize=15,
    iterations=3, poly_n=5, poly_sigma=1.2, flags=0
)

# 转换为极坐标 (方向和大小)
magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

# 可视化
hsv = np.zeros_like(frame)
hsv[..., 0] = angle * 180 / np.pi / 2  # 色相表示方向
hsv[..., 1] = 255
hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
```

## 实际应用

### 运动检测报警
```python
# 计算运动区域面积
contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
total_area = sum(cv2.contourArea(c) for c in contours)

if total_area > threshold:
    print("检测到运动!")
```

## 待创建文件

- `01_background_subtraction.py` - 背景减除
- `02_sparse_optical_flow.py` - 稀疏光流
- `03_dense_optical_flow.py` - 稠密光流

