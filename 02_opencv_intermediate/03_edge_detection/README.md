# 03_edge_detection - 边缘检测 📐

## 学习目标

- 理解边缘检测的原理（梯度）
- 掌握 Canny、Sobel、Laplacian 算法
- 选择合适的边缘检测方法

## 边缘检测方法对比

| 方法 | 原理 | 特点 |
|-----|------|------|
| Canny | 多阶段算法 | 最常用，效果好 |
| Sobel | 一阶导数 | 可分别检测 x/y 方向 |
| Scharr | Sobel 改进 | 更精确的梯度 |
| Laplacian | 二阶导数 | 检测所有方向，对噪声敏感 |

## 核心 API

### Canny 边缘检测
```python
# threshold1: 低阈值, threshold2: 高阈值
# 经验: threshold2 = 2~3 * threshold1
edges = cv2.Canny(img, 50, 150)

# 带高斯模糊预处理
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, 50, 150)
```

### Sobel 算子
```python
# dx=1: 水平方向梯度（检测垂直边缘）
# dy=1: 垂直方向梯度（检测水平边缘）
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

# 转换为可显示格式
sobel_x = cv2.convertScaleAbs(sobel_x)
sobel_y = cv2.convertScaleAbs(sobel_y)

# 合并
sobel = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)
```

### Laplacian 算子
```python
laplacian = cv2.Laplacian(gray, cv2.CV_64F)
laplacian = cv2.convertScaleAbs(laplacian)
```

## Canny 参数选择

```python
# 自动计算阈值（基于中值）
median = np.median(gray)
lower = int(max(0, 0.7 * median))
upper = int(min(255, 1.3 * median))
edges = cv2.Canny(gray, lower, upper)
```

## 文件列表

| 文件 | 内容 |
|-----|------|
| `01_edge_detection.py` | 边缘检测方法对比 |

## 练习

1. 对比不同阈值对 Canny 结果的影响
2. 分别用 Sobel 检测水平和垂直边缘
3. 在边缘检测前后分别加高斯模糊，对比效果

## 运行

```bash
conda activate yolo
python 01_edge_detection.py
```

