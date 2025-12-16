"""
验证数据集
=========

学习目标:
- 检查图像和标签的一致性
- 可视化 Ground Truth (真实标签)
- 验证标注格式 (YOLO 格式)
"""

from pathlib import Path
import cv2
import numpy as np
import sys
import yaml
import random

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def main():
    print("=" * 60)
    print("🔍 验证数据集")
    print("=" * 60)
    
    # 加载前面创建的配置文件
    config_path = Path(__file__).parent / "coco8_local.yaml"
    if not config_path.exists():
        print("❌ 找不到配置文件，请先运行 01_create_sample_dataset.py")
        return
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    dataset_path = Path(config["path"])
    train_img_dir = dataset_path / config["train"]
    train_lbl_dir = dataset_path / "labels/train"
    class_names = config["names"]
    
    print(f"\n📂 数据集: {dataset_path}")
    print(f"📖 类别数: {len(class_names)}")
    
    # ==========================================
    # 1. 检查数据一致性
    # ==========================================
    
    print("\n1️⃣ 检查数据一致性...")
    
    img_files = sorted(list(train_img_dir.glob("*.jpg")))
    lbl_files = sorted(list(train_lbl_dir.glob("*.txt")))
    
    print(f"  图片文件: {len(img_files)}")
    print(f"  标签文件: {len(lbl_files)}")
    
    # 检查配对
    missing_labels = []
    for img_path in img_files:
        lbl_path = train_lbl_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            missing_labels.append(img_path.name)
    
    if missing_labels:
        print(f"  ⚠️ 警告: {len(missing_labels)} 张图片缺少标签")
    else:
        print("  ✅ 所有图片都有对应的标签文件")
    
    # ==========================================
    # 2. 验证标注格式并可视化
    # ==========================================
    
    print("\n2️⃣ 可视化标注 (随机抽取 2 张)...")
    
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    # 随机选择图片
    sample_imgs = random.sample(img_files, min(2, len(img_files)))
    
    for img_path in sample_imgs:
        lbl_path = train_lbl_dir / f"{img_path.stem}.txt"
        
        # 读取图片
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        
        print(f"\n  📄 文件: {img_path.name}")
        print(f"     尺寸: {w}x{h}")
        
        # 读取标签
        # YOLO 格式: <class_id> <x_center> <y_center> <width> <height> (归一化 0-1)
        if lbl_path.exists():
            with open(lbl_path, "r") as f:
                lines = f.readlines()
                
            print(f"     标签数: {len(lines)}")
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    # 反归一化
                    cx = float(parts[1]) * w
                    cy = float(parts[2]) * h
                    bw = float(parts[3]) * w
                    bh = float(parts[4]) * h
                    
                    # 计算左上角和右下角
                    x1 = int(cx - bw / 2)
                    y1 = int(cy - bh / 2)
                    x2 = int(cx + bw / 2)
                    y2 = int(cy + bh / 2)
                    
                    # 绘制矩形
                    color = (0, 255, 0)
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    
                    # 绘制类别名
                    cls_name = class_names.get(cls_id, str(cls_id))
                    cv2.putText(img, cls_name, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
                    print(f"       - {cls_name}: xyxy=[{x1},{y1},{x2},{y2}]")
        
        # 保存可视化
        out_path = output_dir / f"vis_{img_path.name}"
        cv2.imwrite(str(out_path), img)
        print(f"     💾 已保存可视化: {out_path}")
    
    print("\n✅ 数据集验证完成!")


if __name__ == "__main__":
    main()
