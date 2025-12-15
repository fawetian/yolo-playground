# 04_contours - 轮廓检测 🔍

## 学习目标

- 检测和绘制轮廓
- 计算轮廓属性（面积、周长、边界框等）
- 轮廓近似和凸包
- 轮廓匹配

## 核心 API

### 轮廓检测
```python
# 输入必须是二值图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# 检测轮廓
contours, hierarchy = cv2.findContours(
    binary, 
    cv2.RETR_EXTERNAL,     # 检索模式
    cv2.CHAIN_APPROX_SIMPLE # 近似方法
)
```

### 检索模式
| 模式 | 说明 |
|-----|------|
| `RETR_EXTERNAL` | 只检测最外层轮廓 |
| `RETR_LIST` | 检测所有轮廓，无层级 |
| `RETR_TREE` | 检测所有轮廓，有完整层级 |

### 绘制轮廓
```python
cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
# -1 表示绘制所有轮廓
```

### 轮廓属性
```python
for cnt in contours:
    # 面积
    area = cv2.contourArea(cnt)
    
    # 周长
    perimeter = cv2.arcLength(cnt, True)
    
    # 边界矩形
    x, y, w, h = cv2.boundingRect(cnt)
    
    # 最小外接矩形 (可旋转)
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    
    # 最小外接圆
    (cx, cy), radius = cv2.minEnclosingCircle(cnt)
    
    # 轮廓质心
    M = cv2.moments(cnt)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
```

### 轮廓近似
```python
epsilon = 0.02 * cv2.arcLength(cnt, True)
approx = cv2.approxPolyDP(cnt, epsilon, True)
```

### 凸包
```python
hull = cv2.convexHull(cnt)
```

## 实际应用

### 物体计数
```python
# 检测轮廓
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 过滤小轮廓
min_area = 100
filtered = [c for c in contours if cv2.contourArea(c) > min_area]

print(f"检测到 {len(filtered)} 个物体")
```

## 待创建文件

- `01_find_contours.py` - 轮廓检测基础
- `02_contour_properties.py` - 轮廓属性计算
- `03_shape_detection.py` - 形状识别

