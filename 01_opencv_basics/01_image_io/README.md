# 01_image_io - 图像读写 📁

## 学习目标

- 使用 `cv2.imread()` 读取图像
- 使用 `cv2.imwrite()` 保存图像
- 使用 `cv2.imshow()` 显示图像
- 理解图像的 NumPy 数组结构

## 核心 API

### 读取图像
```python
import cv2

# 读取彩色图像 (默认)
img = cv2.imread("image.jpg", cv2.IMREAD_COLOR)

# 读取灰度图像
img_gray = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)

# 读取包含 alpha 通道
img_alpha = cv2.imread("image.png", cv2.IMREAD_UNCHANGED)
```

### 保存图像
```python
# 基本保存
cv2.imwrite("output.jpg", img)

# JPEG 质量设置 (0-100)
cv2.imwrite("output.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])

# PNG 压缩级别 (0-9)
cv2.imwrite("output.png", img, [cv2.IMWRITE_PNG_COMPRESSION, 5])
```

### 显示图像
```python
cv2.imshow("Window Title", img)
cv2.waitKey(0)          # 等待按键
cv2.destroyAllWindows()
```

## 文件列表

| 文件 | 内容 |
|-----|------|
| `01_read_image.py` | 图像读取基础 |
| `02_save_image.py` | 图像保存与格式 |

## 练习

1. 读取一张图像，打印其形状和数据类型
2. 将彩色图像转换为灰度并保存
3. 对比 JPEG 不同质量参数的文件大小

## 运行

```bash
conda activate yolo
python 01_read_image.py
python 02_save_image.py
```

