# 05_roi_mask - ROI 与掩码 🎭

## 学习目标

- 使用 ROI（感兴趣区域）
- 创建和应用掩码
- 位运算操作
- 图像融合

## 核心概念

### ROI (Region of Interest)
```python
# 使用切片提取 ROI
roi = img[y1:y2, x1:x2]

# 修改 ROI
img[y1:y2, x1:x2] = new_value
```

### 掩码 (Mask)
```python
# 创建掩码（与图像同尺寸的二值图像）
mask = np.zeros(img.shape[:2], dtype=np.uint8)
cv2.circle(mask, (cx, cy), radius, 255, -1)

# 应用掩码
result = cv2.bitwise_and(img, img, mask=mask)
```

## 位运算

```python
# 与运算 - 保留两者都有的部分
result = cv2.bitwise_and(img1, img2)

# 或运算 - 合并
result = cv2.bitwise_or(img1, img2)

# 异或运算 - 不同的部分
result = cv2.bitwise_xor(img1, img2)

# 非运算 - 反转
result = cv2.bitwise_not(img)
```

## 图像融合

```python
# 加权融合
# dst = α * img1 + β * img2 + γ
blended = cv2.addWeighted(img1, 0.7, img2, 0.3, 0)
```

## 实际应用

### 给图像添加 Logo
```python
# 1. 读取主图和 Logo
img = cv2.imread("main.jpg")
logo = cv2.imread("logo.png")

# 2. 定义 Logo 放置位置 (ROI)
rows, cols = logo.shape[:2]
roi = img[0:rows, 0:cols]

# 3. 创建 Logo 掩码
gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
_, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)

# 4. 融合
bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
fg = cv2.bitwise_and(logo, logo, mask=mask)
dst = cv2.add(bg, fg)

# 5. 放回原图
img[0:rows, 0:cols] = dst
```

## 待创建文件

- `01_roi_operations.py` - ROI 操作
- `02_mask_operations.py` - 掩码操作
- `03_image_blending.py` - 图像融合

