# 01 - OpenCV 基础 📷

## 本模块内容

学习 OpenCV 的基础操作，掌握图像处理的核心概念。

## 子目录

| 目录 | 内容 | 预计时间 |
|-----|------|---------|
| `01_image_io/` | 图像读取与保存 | 2 天 |
| `02_image_operations/` | 图像基本操作 | 2 天 |
| `03_color_spaces/` | 颜色空间转换 | 1 天 |
| `04_drawing/` | 绘图操作 | 1 天 |
| `05_roi_mask/` | ROI 与掩码 | 1 天 |

## 学习目标

完成本模块后，你将掌握：

- [ ] 使用 `cv2.imread()` 和 `cv2.imwrite()` 读写图像
- [ ] 理解图像的 NumPy 数组表示
- [ ] 进行图像裁剪、缩放、旋转、翻转
- [ ] 在 BGR、RGB、HSV、灰度之间转换
- [ ] 在图像上绘制矩形、圆、线条、文字
- [ ] 使用 ROI 和掩码进行区域操作

## 核心概念

### 图像本质
```python
import cv2
img = cv2.imread("image.jpg")
# img 是一个 NumPy 数组: shape = (height, width, channels)
# OpenCV 使用 BGR 颜色顺序（不是 RGB）
```

### 像素访问
```python
# 访问像素: img[y, x] 或 img[y, x, channel]
pixel = img[100, 200]       # BGR 值
blue = img[100, 200, 0]     # 蓝色通道
```

## 运行示例

```bash
conda activate yolo
python 01_image_io/01_read_image.py
```

## 下一步

完成后进入 `02_opencv_intermediate/` 学习进阶内容。

