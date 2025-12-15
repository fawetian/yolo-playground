# 05_feature_detection - 特征检测 🎯

## 学习目标

- 理解特征点的概念
- 使用 Harris、SIFT、ORB 检测特征
- 特征匹配与图像配准

## 特征检测器对比

| 检测器 | 特点 | 速度 | 准确度 |
|-------|------|------|--------|
| Harris | 角点检测 | 快 | 一般 |
| SIFT | 尺度不变 | 慢 | 高 |
| SURF | SIFT 加速版 | 中 | 高 |
| ORB | 快速、免费 | 快 | 中 |
| AKAZE | 非线性尺度 | 中 | 高 |

## 核心 API

### Harris 角点
```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)

# blockSize: 角点检测窗口大小
# ksize: Sobel 算子孔径
# k: Harris 检测器自由参数
dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

# 标记角点
img[dst > 0.01 * dst.max()] = [0, 0, 255]
```

### SIFT 特征
```python
sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(gray, None)

# 绘制关键点
img_kp = cv2.drawKeypoints(img, keypoints, None, 
                           flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
```

### ORB 特征
```python
orb = cv2.ORB_create(nfeatures=500)
keypoints, descriptors = orb.detectAndCompute(gray, None)
```

### 特征匹配
```python
# 暴力匹配
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(desc1, desc2)
matches = sorted(matches, key=lambda x: x.distance)

# FLANN 匹配 (更快)
index_params = dict(algorithm=1, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(desc1, desc2, k=2)

# 绘制匹配
img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None)
```

### 比率测试 (Lowe's ratio test)
```python
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)
```

## 实际应用

### 图像拼接 (Homography)
```python
# 获取匹配点坐标
src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])

# 计算单应性矩阵
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# 透视变换
result = cv2.warpPerspective(img1, H, (w, h))
```

## 待创建文件

- `01_harris_corner.py` - Harris 角点检测
- `02_sift_orb.py` - SIFT/ORB 特征检测
- `03_feature_matching.py` - 特征匹配

