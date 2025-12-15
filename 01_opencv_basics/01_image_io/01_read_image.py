"""
01. 图像读取基础
===============

学习目标:
- 使用 cv2.imread() 读取图像
- 理解图像的 numpy 数组表示
- 了解不同的读取模式

知识点:
- OpenCV 默认使用 BGR 颜色空间（而非 RGB）
- 图像是一个 numpy 数组: (height, width, channels)
- 常用读取模式: IMREAD_COLOR, IMREAD_GRAYSCALE
"""

import cv2
import numpy as np
from pathlib import Path


def main():
    # ==========================================
    # 1. 基本图像读取
    # ==========================================
    
    # 创建示例图像（因为还没有数据集）
    sample_img = create_sample_image()
    sample_path = Path(__file__).parent / "sample.jpg"
    cv2.imwrite(str(sample_path), sample_img)
    print(f"✅ 创建示例图像: {sample_path}")
    
    # 读取彩色图像 (默认)
    img_color = cv2.imread(str(sample_path), cv2.IMREAD_COLOR)
    # 或简写: img_color = cv2.imread(str(sample_path))
    
    # 读取灰度图像
    img_gray = cv2.imread(str(sample_path), cv2.IMREAD_GRAYSCALE)
    
    # 读取包含 alpha 通道的图像
    img_unchanged = cv2.imread(str(sample_path), cv2.IMREAD_UNCHANGED)
    
    # ==========================================
    # 2. 检查图像属性
    # ==========================================
    
    print("\n" + "=" * 50)
    print("📷 彩色图像属性")
    print("=" * 50)
    print(f"  形状 (H, W, C): {img_color.shape}")
    print(f"  数据类型: {img_color.dtype}")
    print(f"  内存大小: {img_color.nbytes / 1024:.2f} KB")
    
    # 获取尺寸的便捷方式
    height, width, channels = img_color.shape
    print(f"  高度: {height} px")
    print(f"  宽度: {width} px")
    print(f"  通道数: {channels}")
    
    print("\n" + "=" * 50)
    print("📷 灰度图像属性")
    print("=" * 50)
    print(f"  形状 (H, W): {img_gray.shape}")
    print(f"  数据类型: {img_gray.dtype}")
    
    # ==========================================
    # 3. 访问像素值
    # ==========================================
    
    print("\n" + "=" * 50)
    print("🎨 像素值访问")
    print("=" * 50)
    
    # 获取指定位置的像素值 (注意: 先行后列, 即 [y, x])
    pixel_bgr = img_color[100, 100]
    print(f"  位置 (100, 100) 的 BGR 值: {pixel_bgr}")
    
    # 获取单个通道
    blue = img_color[100, 100, 0]
    green = img_color[100, 100, 1]
    red = img_color[100, 100, 2]
    print(f"  B={blue}, G={green}, R={red}")
    
    # 灰度图像的像素值
    gray_value = img_gray[100, 100]
    print(f"  灰度值: {gray_value}")
    
    # ==========================================
    # 4. 显示图像
    # ==========================================
    
    print("\n💡 按任意键关闭窗口...")
    
    # macOS 优化: 使用 WINDOW_NORMAL 可以调整窗口大小
    cv2.namedWindow("Color Image", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Grayscale Image", cv2.WINDOW_NORMAL)
    
    cv2.imshow("Color Image", img_color)
    cv2.imshow("Grayscale Image", img_gray)
    cv2.waitKey(0)  # 等待按键
    cv2.destroyAllWindows()
    cv2.waitKey(1)  # macOS 需要额外的 waitKey 来完全关闭窗口
    
    # ==========================================
    # 5. 错误处理
    # ==========================================
    
    # 读取不存在的文件
    non_exist = cv2.imread("not_exist.jpg")
    print(f"\n⚠️ 读取不存在的文件返回: {non_exist}")
    # OpenCV 不会报错，而是返回 None！记得检查！


def create_sample_image() -> np.ndarray:
    """创建一个示例图像用于测试"""
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    
    # 添加一些彩色区域
    img[50:150, 50:150] = [255, 0, 0]    # 蓝色方块
    img[100:200, 150:250] = [0, 255, 0]  # 绿色方块
    img[150:250, 250:350] = [0, 0, 255]  # 红色方块
    
    # 添加渐变背景
    for i in range(300):
        img[i, :, 1] = min(255, img[i, :, 1] + int(i * 0.3))
    
    return img


if __name__ == "__main__":
    main()

