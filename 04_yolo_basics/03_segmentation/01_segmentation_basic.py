"""
实例分割基础
==========

学习目标:
- 理解实例分割与目标检测的区别
- 使用 YOLO 分割模型
- 访问和理解分割掩码数据
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
    print("🎭 实例分割基础")
    print("=" * 60)
    
    # 加载分割模型 (以 -seg 结尾)
    model = load_yolo_model("yolo11n-seg.pt")
    
    # ==========================================
    # 1. 加载测试图像
    # ==========================================
    
    # 从 datasets/images 加载，没有则自动下载
    test_image_path = get_sample_image("bus.jpg")
    print(f"\n📷 测试图像: {test_image_path}")
    
    # ==========================================
    # 2. 执行分割推理
    # ==========================================
    
    print("\n🔍 执行实例分割...")
    results = model(str(test_image_path), verbose=False)
    result = results[0]
    
    # ==========================================
    # 3. 理解分割结果
    # ==========================================
    
    print("\n📊 分割结果分析:")
    
    # 检测到的目标数量
    num_objects = len(result.boxes) if result.boxes is not None else 0
    print(f"  检测到 {num_objects} 个目标")
    
    # 访问边界框 (与检测相同)
    if result.boxes is not None and len(result.boxes) > 0:
        print("\n  📦 边界框信息:")
        for i, box in enumerate(result.boxes[:5]):  # 只显示前5个
            cls_id = int(box.cls.item())
            cls_name = result.names[cls_id]
            conf = box.conf.item()
            xyxy = box.xyxy[0].cpu().numpy()
            print(f"    目标 {i}: {cls_name} (置信度: {conf:.2%})")
            print(f"      位置: [{xyxy[0]:.0f}, {xyxy[1]:.0f}, {xyxy[2]:.0f}, {xyxy[3]:.0f}]")
        
        if num_objects > 5:
            print(f"    ... 还有 {num_objects - 5} 个目标")
    
    # 访问分割掩码 (分割特有)
    if result.masks is not None:
        masks = result.masks
        print("\n  🎭 掩码信息:")
        print(f"    掩码数量: {len(masks)}")
        print(f"    掩码形状: {masks.data.shape}")
        
        # 掩码数据详解
        masks_data = masks.data.cpu().numpy()
        print(f"    单个掩码尺寸: {masks_data[0].shape if len(masks_data) > 0 else 'N/A'}")
        print(f"    掩码值范围: [{masks_data.min():.2f}, {masks_data.max():.2f}]")
    else:
        print("\n  ⚠️ 未检测到可分割的目标")
    
    # ==========================================
    # 4. 自动可视化
    # ==========================================
    
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    # 使用 plot() 自动绘制分割结果
    annotated = result.plot()
    output_path = output_dir / "segmentation_result.jpg"
    cv2.imwrite(str(output_path), annotated)
    print(f"\n💾 分割结果已保存: {output_path}")
    
    # ==========================================
    # 5. 分割 vs 检测对比
    # ==========================================
    
    print("\n" + "=" * 60)
    print("📝 分割 vs 检测对比")
    print("=" * 60)
    print("""
    | 特性       | 目标检测      | 实例分割           |
    |-----------|--------------|-------------------|
    | 输出      | 边界框        | 边界框 + 掩码      |
    | 精度      | 矩形框        | 像素级轮廓         |
    | 速度      | 较快          | 较慢              |
    | 用途      | 定位目标      | 精确分割、背景移除  |
    | 模型后缀  | .pt           | -seg.pt           |
    """)
    
    print("✅ 实例分割基础演示完成!")


if __name__ == "__main__":
    main()
