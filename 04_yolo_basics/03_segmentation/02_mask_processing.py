"""
掩码处理
=======

学习目标:
- 提取和处理分割掩码
- 掩码与原图尺寸对齐
- 应用掩码提取目标区域
"""

from pathlib import Path
import cv2
import numpy as np
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.model_loader import load_yolo_model
from utils.image_loader import get_sample_image


def main():
    print("=" * 60)
    print("🎭 掩码处理技术")
    print("=" * 60)
    
    # 加载分割模型
    model = load_yolo_model("yolo11n-seg.pt")
    
    # 从 datasets/images 加载测试图像
    test_image_path = get_sample_image("bus.jpg")
    print(f"\n📷 测试图像: {test_image_path}")
    print("🔍 执行实例分割...")
    
    results = model(str(test_image_path), verbose=False)
    result = results[0]
    
    if result.masks is None:
        print("⚠️ 未检测到可分割目标")
        return
    
    # 获取原始图像
    orig_img = result.orig_img.copy()
    h, w = orig_img.shape[:2]
    print(f"\n📐 原始图像尺寸: {w} x {h}")
    
    # ==========================================
    # 1. 掩码数据解析
    # ==========================================
    
    print("\n" + "=" * 60)
    print("📊 掩码数据解析")
    print("=" * 60)
    
    masks_data = result.masks.data.cpu().numpy()
    print(f"  掩码张量形状: {masks_data.shape}")
    print(f"  解释: ({masks_data.shape[0]} 个目标, "
          f"{masks_data.shape[1]}x{masks_data.shape[2]} 掩码尺寸)")
    
    # ==========================================
    # 2. 掩码尺寸对齐
    # ==========================================
    
    print("\n" + "=" * 60)
    print("📐 掩码尺寸对齐")
    print("=" * 60)
    
    # 掩码通常是低分辨率的，需要缩放到原图尺寸
    resized_masks = []
    for i, mask in enumerate(masks_data):
        # 缩放到原图尺寸
        mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        # 二值化
        mask_binary = (mask_resized > 0.5).astype(np.uint8)
        resized_masks.append(mask_binary)
        
        if i < 3:  # 只打印前3个
            cls_id = int(result.boxes.cls[i].item())
            cls_name = result.names[cls_id]
            pixel_count = np.sum(mask_binary)
            coverage = pixel_count / (w * h) * 100
            print(f"  目标 {i} ({cls_name}): {pixel_count:,} 像素 ({coverage:.1f}% 覆盖)")
    
    # ==========================================
    # 3. 提取单个目标
    # ==========================================
    
    print("\n" + "=" * 60)
    print("✂️ 提取单个目标")
    print("=" * 60)
    
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    # 提取第一个检测到的目标
    if len(resized_masks) > 0:
        mask = resized_masks[0]
        cls_id = int(result.boxes.cls[0].item())
        cls_name = result.names[cls_id]
        
        # 方法1: 使用 bitwise_and 提取
        extracted = cv2.bitwise_and(orig_img, orig_img, mask=mask)
        
        # 保存提取结果
        output_path = output_dir / f"extracted_{cls_name}.jpg"
        cv2.imwrite(str(output_path), extracted)
        print(f"  提取 {cls_name} 已保存: {output_path}")
        
        # 方法2: 创建透明背景 (RGBA)
        rgba = cv2.cvtColor(orig_img, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = mask * 255
        
        output_path_png = output_dir / f"extracted_{cls_name}_transparent.png"
        cv2.imwrite(str(output_path_png), rgba)
        print(f"  透明背景版本已保存: {output_path_png}")
    
    # ==========================================
    # 4. 合并所有掩码
    # ==========================================
    
    print("\n" + "=" * 60)
    print("🔗 合并掩码")
    print("=" * 60)
    
    # 创建彩色掩码可视化
    color_mask = np.zeros_like(orig_img)
    colors = [
        (255, 0, 0),    # 蓝
        (0, 255, 0),    # 绿
        (0, 0, 255),    # 红
        (255, 255, 0),  # 青
        (255, 0, 255),  # 紫
        (0, 255, 255),  # 黄
    ]
    
    for i, mask in enumerate(resized_masks):
        color = colors[i % len(colors)]
        # 将掩码区域着色
        color_mask[mask == 1] = color
    
    # 叠加到原图
    alpha = 0.5
    overlay = cv2.addWeighted(orig_img, 1, color_mask, alpha, 0)
    
    output_path = output_dir / "colored_masks.jpg"
    cv2.imwrite(str(output_path), overlay)
    print(f"  彩色掩码已保存: {output_path}")
    
    # ==========================================
    # 5. 掩码轮廓提取
    # ==========================================
    
    print("\n" + "=" * 60)
    print("📏 轮廓提取")
    print("=" * 60)
    
    contour_img = orig_img.copy()
    total_contours = 0
    
    for i, mask in enumerate(resized_masks):
        # 查找轮廓
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        total_contours += len(contours)
        
        # 绘制轮廓
        color = colors[i % len(colors)]
        cv2.drawContours(contour_img, contours, -1, color, 2)
    
    print(f"  共提取 {total_contours} 个轮廓")
    
    output_path = output_dir / "contours.jpg"
    cv2.imwrite(str(output_path), contour_img)
    print(f"  轮廓图已保存: {output_path}")
    
    print("\n✅ 掩码处理演示完成!")


if __name__ == "__main__":
    main()
