"""
02. 图像保存
===========

学习目标:
- 使用 cv2.imwrite() 保存图像
- 了解不同图像格式的特点
- 掌握压缩参数设置
"""

import cv2
import numpy as np
from pathlib import Path


def main():
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    # 创建示例图像
    img = create_gradient_image()
    
    # ==========================================
    # 1. 基本保存
    # ==========================================
    
    # 保存为不同格式
    cv2.imwrite(str(output_dir / "image.jpg"), img)
    cv2.imwrite(str(output_dir / "image.png"), img)
    cv2.imwrite(str(output_dir / "image.bmp"), img)
    
    print("✅ 基本保存完成")
    
    # ==========================================
    # 2. JPEG 质量参数
    # ==========================================
    
    # JPEG 质量: 0-100 (越高质量越好，文件越大)
    for quality in [10, 50, 95]:
        path = output_dir / f"jpeg_quality_{quality}.jpg"
        cv2.imwrite(
            str(path), 
            img, 
            [cv2.IMWRITE_JPEG_QUALITY, quality]
        )
        size = path.stat().st_size / 1024
        print(f"  JPEG 质量 {quality}: {size:.1f} KB")
    
    # ==========================================
    # 3. PNG 压缩参数
    # ==========================================
    
    # PNG 压缩级别: 0-9 (越高压缩率越大，但更慢)
    for compression in [0, 5, 9]:
        path = output_dir / f"png_compression_{compression}.png"
        cv2.imwrite(
            str(path), 
            img, 
            [cv2.IMWRITE_PNG_COMPRESSION, compression]
        )
        size = path.stat().st_size / 1024
        print(f"  PNG 压缩 {compression}: {size:.1f} KB")
    
    # ==========================================
    # 4. 格式对比
    # ==========================================
    
    print("\n📊 文件大小对比:")
    print("-" * 40)
    
    formats = {
        "BMP (无压缩)": output_dir / "image.bmp",
        "PNG (无损)": output_dir / "image.png",
        "JPEG (有损)": output_dir / "image.jpg",
    }
    
    for name, path in formats.items():
        size = path.stat().st_size / 1024
        print(f"  {name}: {size:.1f} KB")
    
    print("\n💡 总结:")
    print("  - BMP: 无压缩，文件大，保存快")
    print("  - PNG: 无损压缩，适合截图/图标")
    print("  - JPEG: 有损压缩，适合照片")


def create_gradient_image() -> np.ndarray:
    """创建渐变图像用于测试压缩效果"""
    height, width = 400, 600
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 创建彩色渐变
    for y in range(height):
        for x in range(width):
            img[y, x] = [
                int(255 * x / width),           # B: 左右渐变
                int(255 * y / height),          # G: 上下渐变
                int(255 * (1 - x / width))      # R: 反向渐变
            ]
    
    # 添加一些细节（测试压缩质量）
    cv2.putText(
        img, "OpenCV Image Save Test", 
        (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 
        1.0, (255, 255, 255), 2
    )
    
    return img


if __name__ == "__main__":
    main()

