# 01_filters - 图像滤波 🌫️

## 学习目标

- 理解图像滤波/卷积的原理
- 掌握常用滤波器的使用场景
- 对比不同滤波器的效果

## 滤波器类型

| 滤波器 | 特点 | 使用场景 |
|-------|------|---------|
| 均值滤波 | 简单快速 | 一般性模糊 |
| 高斯滤波 | 效果自然 | 最常用的模糊方法 |
| 中值滤波 | 保留边缘 | 去除椒盐噪声 |
| 双边滤波 | 保边去噪 | 人像美颜 |

## 核心 API

### 均值模糊
```python
blur = cv2.blur(img, (5, 5))  # ksize: 核大小
```

### 高斯模糊
```python
gaussian = cv2.GaussianBlur(img, (5, 5), 0)
# sigmaX=0 自动计算标准差
```

### 中值滤波
```python
median = cv2.medianBlur(img, 5)  # ksize 必须是奇数
```

### 双边滤波
```python
bilateral = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
# d: 邻域直径
# sigmaColor: 颜色空间标准差
# sigmaSpace: 坐标空间标准差
```

### 自定义卷积核
```python
kernel = np.array([[-1,-1,-1],
                   [-1, 9,-1],
                   [-1,-1,-1]])  # 锐化核
sharpened = cv2.filter2D(img, -1, kernel)
```

## 文件列表

| 文件 | 内容 |
|-----|------|
| `01_blur_filters.py` | 模糊滤波器对比 |

## 练习

1. 对带噪声的图像分别应用不同滤波器，对比效果
2. 尝试不同的核大小，观察模糊程度的变化
3. 使用自定义卷积核实现图像锐化

## 运行

```bash
conda activate yolo
python 01_blur_filters.py
```

