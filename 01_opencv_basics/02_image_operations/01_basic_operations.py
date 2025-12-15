"""
图像基本操作
==========

学习目标:
- 图像裁剪
- 图像缩放
- 图像旋转和翻转
- 图像拼接
"""

import cv2
import numpy as np
from pathlib import Path


def main():
    # 创建示例图像
    img = create_sample_image()
    h, w = img.shape[:2]
    print(f"原始图像尺寸: {w} x {h}")
    
    # ==========================================
    # 1. 图像裁剪 (ROI - Region of Interest)
    # ==========================================
    
    print("\n📐 1. 图像裁剪")
    
    # 使用 numpy 切片裁剪: img[y1:y2, x1:x2]
    roi = img[50:200, 100:300]
    print(f"  裁剪区域: (100, 50) 到 (300, 200)")
    print(f"  ROI 尺寸: {roi.shape[1]} x {roi.shape[0]}")
    
    # ==========================================
    # 2. 图像缩放
    # ==========================================
    
    print("\n🔍 2. 图像缩放")
    
    # 缩放到指定尺寸
    resized_fixed = cv2.resize(img, (200, 150))
    print(f"  固定尺寸: 200 x 150")
    
    # 按比例缩放
    scale = 0.5
    resized_scale = cv2.resize(img, None, fx=scale, fy=scale)
    print(f"  按比例 {scale}: {resized_scale.shape[1]} x {resized_scale.shape[0]}")
    
    # 不同插值方法
    # - INTER_NEAREST: 最近邻（最快）
    # - INTER_LINEAR: 双线性（默认，适合放大）
    # - INTER_AREA: 区域（适合缩小）
    # - INTER_CUBIC: 双三次（质量好，较慢）
    
    resized_quality = cv2.resize(img, (800, 600), interpolation=cv2.INTER_CUBIC)
    print(f"  高质量放大 (CUBIC): {resized_quality.shape[1]} x {resized_quality.shape[0]}")
    
    # ==========================================
    # 3. 图像翻转
    # ==========================================
    
    print("\n🔄 3. 图像翻转")
    
    # flipCode: 0=垂直, 1=水平, -1=同时
    flipped_h = cv2.flip(img, 1)   # 水平翻转
    flipped_v = cv2.flip(img, 0)   # 垂直翻转
    flipped_both = cv2.flip(img, -1)  # 同时翻转
    
    print("  flipCode=1: 水平翻转 (左右镜像)")
    print("  flipCode=0: 垂直翻转 (上下镜像)")
    print("  flipCode=-1: 同时翻转 (旋转180°)")
    
    # ==========================================
    # 4. 图像旋转
    # ==========================================
    
    print("\n🔃 4. 图像旋转")
    
    # 方式1: 使用预定义旋转代码（仅支持 90° 倍数）
    rotated_90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    rotated_180 = cv2.rotate(img, cv2.ROTATE_180)
    rotated_270 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    print("  90° 顺时针: ROTATE_90_CLOCKWISE")
    print("  180°: ROTATE_180")
    print("  90° 逆时针: ROTATE_90_COUNTERCLOCKWISE")
    
    # 方式2: 使用仿射变换（任意角度）
    angle = 45
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)
    rotated_45 = cv2.warpAffine(img, matrix, (w, h))
    
    print(f"  任意角度 ({angle}°): 使用 warpAffine")
    
    # ==========================================
    # 5. 图像拼接
    # ==========================================
    
    print("\n🧩 5. 图像拼接")
    
    # 水平拼接
    h_concat = np.hstack([img, flipped_h])
    # 或使用: cv2.hconcat([img, flipped_h])
    
    # 垂直拼接
    v_concat = np.vstack([img, flipped_v])
    # 或使用: cv2.vconcat([img, flipped_v])
    
    print(f"  水平拼接尺寸: {h_concat.shape[1]} x {h_concat.shape[0]}")
    print(f"  垂直拼接尺寸: {v_concat.shape[1]} x {v_concat.shape[0]}")
    
    # ==========================================
    # 显示结果
    # ==========================================
    
    print("\n💡 显示结果 (按任意键切换/退出)...")
    
    images = [
        ("Original", img),
        ("ROI", roi),
        ("Resized (scale=0.5)", resized_scale),
        ("Flipped Horizontal", flipped_h),
        ("Rotated 45°", rotated_45),
        ("Horizontal Concat", h_concat),
    ]
    
    for title, image in images:
        # macOS 优化: 使用 WINDOW_NORMAL 可以调整窗口大小
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.imshow(title, image)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)  # macOS 需要额外的 waitKey


def create_sample_image() -> np.ndarray:
    """创建带有明显特征的示例图像"""
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    
    # 背景渐变
    for y in range(400):
        img[y, :, 0] = int(y * 0.5)
    
    # 添加形状以便观察变换效果
    cv2.rectangle(img, (50, 50), (200, 150), (0, 255, 0), -1)
    cv2.circle(img, (400, 200), 80, (0, 0, 255), -1)
    cv2.line(img, (300, 300), (500, 350), (255, 255, 0), 5)
    
    # 添加文字标识方向
    cv2.putText(img, "TOP-LEFT", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, "BOTTOM-RIGHT", (380, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return img


if __name__ == "__main__":
    main()

