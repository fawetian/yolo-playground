"""
边缘检测
=======

学习目标:
- Canny 边缘检测
- Sobel 算子
- Laplacian 算子
"""

import cv2
import numpy as np


def main():
    # 创建示例图像
    img = create_sample_image()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    print("=" * 50)
    print("🔲 边缘检测")
    print("=" * 50)
    
    # ==========================================
    # 1. Canny 边缘检测
    # ==========================================
    
    print("\n1️⃣ Canny 边缘检测")
    print("  最常用的边缘检测算法")
    print("  参数: threshold1 (低阈值), threshold2 (高阈值)")
    
    # 先进行高斯模糊减少噪声
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Canny 检测
    # 经验法则: threshold2 = 2~3 * threshold1
    canny_50_150 = cv2.Canny(blurred, 50, 150)
    canny_100_200 = cv2.Canny(blurred, 100, 200)
    canny_30_100 = cv2.Canny(blurred, 30, 100)  # 更敏感
    
    print("  阈值越低，检测到的边缘越多（可能包含噪声）")
    
    # ==========================================
    # 2. Sobel 算子
    # ==========================================
    
    print("\n2️⃣ Sobel 算子")
    print("  基于一阶导数，分别检测水平和垂直方向的边缘")
    
    # dx=1, dy=0: 检测垂直边缘（水平方向梯度）
    # dx=0, dy=1: 检测水平边缘（垂直方向梯度）
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # 转换为可显示的格式
    sobel_x = cv2.convertScaleAbs(sobel_x)
    sobel_y = cv2.convertScaleAbs(sobel_y)
    
    # 合并 x 和 y 方向的梯度
    sobel_combined = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)
    
    # ==========================================
    # 3. Laplacian 算子
    # ==========================================
    
    print("\n3️⃣ Laplacian 算子")
    print("  基于二阶导数，同时检测所有方向的边缘")
    
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian = cv2.convertScaleAbs(laplacian)
    
    # ==========================================
    # 4. Scharr 算子 (Sobel 的改进版)
    # ==========================================
    
    print("\n4️⃣ Scharr 算子")
    print("  比 Sobel 更精确的梯度计算")
    
    scharr_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
    scharr_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
    scharr_x = cv2.convertScaleAbs(scharr_x)
    scharr_y = cv2.convertScaleAbs(scharr_y)
    scharr_combined = cv2.addWeighted(scharr_x, 0.5, scharr_y, 0.5, 0)
    
    # ==========================================
    # 对比展示
    # ==========================================
    
    print("\n📊 边缘检测对比")
    print("💡 按任意键退出")
    
    # 转换为 3 通道以便拼接
    def to_color(img):
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    row1 = np.hstack([
        add_label(img, "Original"),
        add_label(to_color(canny_50_150), "Canny (50,150)"),
        add_label(to_color(canny_100_200), "Canny (100,200)"),
    ])
    
    row2 = np.hstack([
        add_label(to_color(sobel_x), "Sobel X"),
        add_label(to_color(sobel_y), "Sobel Y"),
        add_label(to_color(sobel_combined), "Sobel Combined"),
    ])
    
    row3 = np.hstack([
        add_label(to_color(laplacian), "Laplacian"),
        add_label(to_color(scharr_combined), "Scharr"),
        add_label(to_color(gray), "Grayscale"),
    ])
    
    comparison = np.vstack([row1, row2, row3])
    
    # 调整显示大小
    h, w = comparison.shape[:2]
    display = cv2.resize(comparison, (w * 2 // 3, h * 2 // 3))
    
    # macOS 优化
    cv2.namedWindow("Edge Detection Comparison", cv2.WINDOW_NORMAL)
    cv2.imshow("Edge Detection Comparison", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)  # macOS 需要额外的 waitKey
    
    print("\n✨ 总结:")
    print("  - Canny: 最常用，效果好，有噪声抑制")
    print("  - Sobel: 可分别检测水平/垂直边缘")
    print("  - Laplacian: 检测所有方向，对噪声敏感")
    print("  - Scharr: Sobel 的改进版，更精确")


def create_sample_image() -> np.ndarray:
    """创建有明显边缘特征的图像"""
    img = np.full((300, 400, 3), 200, dtype=np.uint8)
    
    # 各种形状
    cv2.rectangle(img, (50, 50), (150, 150), (50, 50, 50), -1)
    cv2.rectangle(img, (50, 50), (150, 150), (0, 0, 0), 2)
    
    cv2.circle(img, (280, 100), 50, (100, 100, 100), -1)
    cv2.circle(img, (280, 100), 50, (0, 0, 0), 2)
    
    pts = np.array([[200, 200], [300, 250], [250, 280], [180, 250]], np.int32)
    cv2.fillPoly(img, [pts], (150, 150, 150))
    cv2.polylines(img, [pts], True, (0, 0, 0), 2)
    
    # 渐变区域（测试边缘检测的敏感度）
    for x in range(320, 380):
        gray_value = int((x - 320) * 255 / 60)
        img[180:280, x] = [gray_value, gray_value, gray_value]
    
    return img


def add_label(img: np.ndarray, label: str) -> np.ndarray:
    """给图像添加标签"""
    img = img.copy()
    cv2.putText(img, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return img


if __name__ == "__main__":
    main()

