# 02_morphology - 形态学操作 🔲

## 学习目标

- 理解形态学操作的原理
- 掌握腐蚀、膨胀、开/闭运算
- 应用于图像分割和噪点去除

## 核心概念

形态学操作基于图像形状进行处理，通常用于二值图像。

### 基本操作

| 操作 | 效果 | 用途 |
|-----|------|------|
| 腐蚀 | 缩小白色区域 | 去除小白点噪声 |
| 膨胀 | 扩大白色区域 | 填充小黑点/连接断开区域 |
| 开运算 | 先腐蚀后膨胀 | 去除白色噪点 |
| 闭运算 | 先膨胀后腐蚀 | 填充黑色空洞 |
| 梯度 | 膨胀 - 腐蚀 | 提取边缘 |
| 顶帽 | 原图 - 开运算 | 提取亮细节 |
| 黑帽 | 闭运算 - 原图 | 提取暗细节 |

## 核心 API

### 结构元素
```python
# 矩形
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# 椭圆
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# 十字形
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
```

### 腐蚀与膨胀
```python
eroded = cv2.erode(img, kernel, iterations=1)
dilated = cv2.dilate(img, kernel, iterations=1)
```

### 形态学变换
```python
# 开运算
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

# 闭运算
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

# 梯度
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

# 顶帽
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)

# 黑帽
blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
```

## 实际应用

### 去除文档图像噪点
```python
# 1. 转灰度并二值化
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# 2. 开运算去除噪点
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
```

## 待创建文件

- `01_erosion_dilation.py` - 腐蚀与膨胀
- `02_morphology_operations.py` - 形态学变换

