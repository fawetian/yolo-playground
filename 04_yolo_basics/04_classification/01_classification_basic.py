"""
图像分类基础
==========

学习目标:
- 理解图像分类与目标检测的区别
- 使用 YOLO 分类模型
- 访问和理解分类结果
"""

from pathlib import Path
import cv2
import numpy as np
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.model_loader import load_yolo_model
from utils.image_loader import get_sample_image, get_all_sample_images


def main():
    print("=" * 60)
    print("🏷️ 图像分类基础")
    print("=" * 60)
    
    # 加载分类模型 (以 -cls 结尾)
    model = load_yolo_model("yolo11n-cls.pt")
    
    # ==========================================
    # 1. 分类 vs 检测概念
    # ==========================================
    
    print("\n📝 分类 vs 检测:")
    print("""
    | 任务     | 输出           | 问题                     |
    |---------|---------------|-------------------------|
    | 分类     | 整图类别       | "这张图是什么?"          |
    | 检测     | 多个目标位置    | "图中有什么? 在哪里?"    |
    """)
    
    # ==========================================
    # 2. 使用示例图像进行分类
    # ==========================================
    
    # 从 datasets/images 加载，没有则自动下载
    test_images = get_all_sample_images()
    
    print("=" * 60)
    print("🔍 执行图像分类")
    print("=" * 60)
    
    for img_path in test_images[:3]:  # 最多处理3张
        print(f"\n📷 图像: {img_path.name}")
        
        results = model(str(img_path), verbose=False)
        result = results[0]
        
        # ==========================================
        # 3. 访问分类结果
        # ==========================================
        
        probs = result.probs
        
        # Top-1 预测
        top1_idx = probs.top1
        top1_conf = probs.top1conf.item()
        top1_name = result.names[top1_idx]
        
        print(f"  🎯 预测结果: {top1_name}")
        print(f"     置信度: {top1_conf:.2%}")
        
        # Top-5 预测
        print(f"\n  📊 Top-5 预测:")
        top5_idx = probs.top5
        top5_conf = probs.top5conf.tolist()
        
        for i, (idx, conf) in enumerate(zip(top5_idx, top5_conf), 1):
            name = result.names[idx]
            bar = "█" * int(conf * 20)
            print(f"     {i}. {name:20s} {conf:6.2%} {bar}")
    
    # ==========================================
    # 4. 保存分类结果可视化
    # ==========================================
    
    print("\n" + "=" * 60)
    print("💾 保存分类结果")
    print("=" * 60)
    
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    # 使用 plot() 自动绘制分类结果
    if test_images:
        results = model(str(test_images[0]), verbose=False)
        annotated = results[0].plot()
        
        output_path = output_dir / "classification_result.jpg"
        cv2.imwrite(str(output_path), annotated)
        print(f"  结果已保存: {output_path}")
    
    # ==========================================
    # 5. 获取所有类别概率
    # ==========================================
    
    print("\n" + "=" * 60)
    print("📈 概率分布分析")
    print("=" * 60)
    
    if test_images:
        results = model(str(test_images[0]), verbose=False)
        all_probs = results[0].probs.data.cpu().numpy()
        print(f"  总类别数: {len(all_probs)}")
        print(f"  概率总和: {all_probs.sum():.4f} (应接近 1.0)")
        print(f"  最高概率: {all_probs.max():.4f}")
        print(f"  最低概率: {all_probs.min():.6f}")
        
        # 概率分布统计
        high_prob = (all_probs > 0.1).sum()
        medium_prob = ((all_probs > 0.01) & (all_probs <= 0.1)).sum()
        low_prob = (all_probs <= 0.01).sum()
        
        print(f"\n  概率分布:")
        print(f"    >10%: {high_prob} 个类别")
        print(f"    1-10%: {medium_prob} 个类别")
        print(f"    <1%: {low_prob} 个类别")
    
    print("\n✅ 图像分类基础演示完成!")


if __name__ == "__main__":
    main()
