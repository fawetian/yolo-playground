"""
图像滤波 - 模糊处理
==================

学习目标:
- 均值模糊
- 高斯模糊
- 中值滤波
- 双边滤波
"""

import cv2
import numpy as np
from pathlib import Path


def main():
    # 创建带噪声的示例图像
    img = create_noisy_image()
    
    print("=" * 50)
    print("🌫️ 图像模糊/平滑处理")
    print("=" * 50)
    
    # ==========================================
    # 1. 均值模糊 (Box Blur)
    # ==========================================
    
    print("\n1️⃣ 均值模糊")
    print("  原理: 用邻域像素的平均值替代中心像素")
    
    # ksize: 卷积核大小 (必须是奇数)
    blur_3 = cv2.blur(img, (3, 3))
    blur_7 = cv2.blur(img, (7, 7))
    blur_15 = cv2.blur(img, (15, 15))
    
    print("  核大小越大，模糊效果越强")
    
    # ==========================================
    # 2. 高斯模糊 (Gaussian Blur)
    # ==========================================
    
    print("\n2️⃣ 高斯模糊")
    print("  原理: 用高斯函数加权的邻域平均值")
    print("  特点: 中心权重大，边缘权重小，效果更自然")
    
    # ksize: 核大小, sigmaX: 标准差 (0表示自动计算)
    gaussian_3 = cv2.GaussianBlur(img, (3, 3), 0)
    gaussian_7 = cv2.GaussianBlur(img, (7, 7), 0)
    gaussian_15 = cv2.GaussianBlur(img, (15, 15), 0)
    
    # ==========================================
    # 3. 中值滤波 (Median Filter)
    # ==========================================
    
    print("\n3️⃣ 中值滤波")
    print("  原理: 用邻域像素的中值替代中心像素")
    print("  特点: 对椒盐噪声效果特别好")
    
    # ksize: 必须是奇数
    median_3 = cv2.medianBlur(img, 3)
    median_7 = cv2.medianBlur(img, 7)
    
    # ==========================================
    # 4. 双边滤波 (Bilateral Filter)
    # ==========================================
    
    print("\n4️⃣ 双边滤波")
    print("  原理: 同时考虑空间距离和颜色差异")
    print("  特点: 保留边缘的同时平滑区域")
    
    # d: 邻域直径, sigmaColor: 颜色空间标准差, sigmaSpace: 坐标空间标准差
    bilateral = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    
    # ==========================================
    # 对比展示
    # ==========================================
    
    print("\n📊 对比不同滤波效果...")
    print("💡 按任意键退出")
    
    # 创建对比图
    row1 = np.hstack([
        add_label(img, "Original (Noisy)"),
        add_label(blur_7, "Box Blur"),
        add_label(gaussian_7, "Gaussian Blur"),
    ])
    
    row2 = np.hstack([
        add_label(median_7, "Median Filter"),
        add_label(bilateral, "Bilateral Filter"),
        add_label(gaussian_15, "Gaussian (Large)"),
    ])
    
    comparison = np.vstack([row1, row2])
    
    # 调整显示大小
    h, w = comparison.shape[:2]
    display = cv2.resize(comparison, (w // 2, h // 2))
    
    # macOS 优化
    cv2.namedWindow("Filter Comparison", cv2.WINDOW_NORMAL)
    cv2.imshow("Filter Comparison", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)  # macOS 需要额外的 waitKey
    
    print("\n✨ 总结:")
    print("  - 均值模糊: 简单快速，但会模糊边缘")
    print("  - 高斯模糊: 效果自然，是最常用的模糊方法")
    print("  - 中值滤波: 去除椒盐噪声的最佳选择")
    print("  - 双边滤波: 保边去噪，适合人像美颜等场景")


def create_noisy_image() -> np.ndarray:
    """创建带噪声的示例图像"""
    # 基础图像
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    
    # 添加一些形状
    cv2.rectangle(img, (50, 50), (150, 150), (100, 150, 200), -1)
    cv2.circle(img, (280, 100), 60, (200, 100, 100), -1)
    cv2.rectangle(img, (200, 180), (350, 260), (100, 200, 100), -1)
    
    # 添加高斯噪声
    noise = np.random.normal(0, 25, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # 添加一些椒盐噪声
    salt = np.random.random(img.shape[:2]) < 0.01
    pepper = np.random.random(img.shape[:2]) < 0.01
    img[salt] = 255
    img[pepper] = 0
    
    return img


def add_label(img: np.ndarray, label: str) -> np.ndarray:
    """给图像添加标签"""
    img = img.copy()
    cv2.putText(img, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return img


if __name__ == "__main__":
    main()

