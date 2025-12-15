# 04_drawing - 绘图操作 ✏️

## 学习目标

- 绘制基本图形（线、矩形、圆、多边形）
- 添加文字
- 在检测结果上绘制标注

## 核心 API

### 绘制线条
```python
cv2.line(img, pt1, pt2, color, thickness)
# pt1, pt2: 起点终点 (x, y)
# color: BGR 颜色 (B, G, R)
# thickness: 线宽
```

### 绘制矩形
```python
cv2.rectangle(img, pt1, pt2, color, thickness)
# thickness=-1 表示填充
```

### 绘制圆
```python
cv2.circle(img, center, radius, color, thickness)
```

### 绘制多边形
```python
pts = np.array([[10,5], [20,30], [70,20]], np.int32)
cv2.polylines(img, [pts], isClosed=True, color=(0,255,0), thickness=2)
cv2.fillPoly(img, [pts], color=(0,255,0))  # 填充
```

### 添加文字
```python
cv2.putText(img, text, org, fontFace, fontScale, color, thickness)
# org: 文字左下角坐标
# fontFace: cv2.FONT_HERSHEY_SIMPLEX 等
# fontScale: 字体大小比例
```

## 常用颜色 (BGR)

```python
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
YELLOW = (0, 255, 255)
```

## 实际应用

绘制检测框示例：
```python
def draw_bbox(img, x1, y1, x2, y2, label, color=(0, 255, 0)):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, label, (x1, y1-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
```

## 待创建文件

- `01_draw_shapes.py` - 基本图形绘制
- `02_draw_text.py` - 文字添加
- `03_draw_detection.py` - 检测结果可视化

