# 02_image_operations - 图像基本操作 ✂️

## 学习目标

- 图像裁剪 (ROI)
- 图像缩放
- 图像旋转
- 图像翻转
- 图像拼接

## 核心 API

### 裁剪 (使用 NumPy 切片)
```python
# roi = img[y1:y2, x1:x2]
roi = img[100:300, 200:400]
```

### 缩放
```python
# 指定尺寸
resized = cv2.resize(img, (width, height))

# 按比例
resized = cv2.resize(img, None, fx=0.5, fy=0.5)

# 插值方法
# - cv2.INTER_NEAREST: 最近邻 (最快)
# - cv2.INTER_LINEAR: 双线性 (默认，适合放大)
# - cv2.INTER_AREA: 区域 (适合缩小)
# - cv2.INTER_CUBIC: 双三次 (高质量)
```

### 翻转
```python
flipped = cv2.flip(img, flipCode)
# flipCode: 0=垂直, 1=水平, -1=同时
```

### 旋转
```python
# 90° 倍数
rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

# 任意角度
center = (w//2, h//2)
matrix = cv2.getRotationMatrix2D(center, angle=45, scale=1.0)
rotated = cv2.warpAffine(img, matrix, (w, h))
```

### 拼接
```python
import numpy as np
h_concat = np.hstack([img1, img2])  # 水平
v_concat = np.vstack([img1, img2])  # 垂直
```

## 文件列表

| 文件 | 内容 |
|-----|------|
| `01_basic_operations.py` | 基本图像操作演示 |

## 练习

1. 从图像中裁剪感兴趣区域
2. 将图像缩小 50%，再放大回原尺寸，观察质量损失
3. 创建一张图像的四方向拼接（原图、水平翻转、垂直翻转、180°旋转）

## 运行

```bash
conda activate yolo
python 01_basic_operations.py
```

