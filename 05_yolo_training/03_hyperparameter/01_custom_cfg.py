"""
超参数配置
=========

学习目标:
- 理解 YOLO 的关键超参数
- 自定义超参数字典
- 使用自定义配置进行训练
"""

from pathlib import Path
import sys
from ultralytics import YOLO

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.model_loader import load_yolo_model


def main():
    print("=" * 60)
    print("🎛️ 超参数配置")
    print("=" * 60)
    
    # ==========================================
    # 1. 默认超参数
    # ==========================================
    
    print("\n📝 关键超参数说明:")
    print("""
    优化器:
      lr0: 0.01        # 初始学习率 (SGD=0.01, Adam=0.001)
      lrf: 0.01        # 最终学习率 (lr0 * lrf)
      momentum: 0.937  # 动量
      weight_decay: 0.0005 # 权重衰减
    
    增强 (Augmentation):
      hsv_h: 0.015     # HSV-Hue 增强
      hsv_s: 0.7       # HSV-Saturation 增强
      hsv_v: 0.4       # HSV-Value 增强
      degrees: 0.0     # 旋转 (+/- deg)
      translate: 0.1   # 平移 (+/- fraction)
      scale: 0.5       # 缩放 (+/- gain)
      flipud: 0.0      # 上下翻转概率
      fliplr: 0.5      # 左右翻转概率
      mosaic: 1.0      # Mosaic 增强概率 (非常重要!)
      mixup: 0.0       # Mixup 增强概率
    """)
    
    # ==========================================
    # 2. 自定义参数训练
    # ==========================================
    
    print("\n🧪 使用自定义超参训练...")
    
    dataset_cfg = Path(__file__).parent.parent / "01_dataset_prep/coco8_local.yaml"
    if not dataset_cfg.exists():
        print("❌ 找不到数据集配置")
        return

    model = load_yolo_model("yolo11n.pt")
    
    # 定义自定义参数
    # 在 train() 中直接传递参数即可覆盖默认值
    results = model.train(
        data=str(dataset_cfg),
        epochs=3,
        imgsz=640,
        device="mps",
        project="runs/hyperparam",
        name="custom_lr_run",
        
        # 自定义超参
        lr0=0.001,       # 降低学习率
        optimizer="Adam",# 更换优化器
        mosaic=0.5,      # 减少 Mosaic 增强
        degrees=10.0,    # 增加旋转增强
        fliplr=0.5,      # 开启左右翻转
    )
    
    print(f"\n✅ 训练完成: {results.save_dir}")
    print("  已应用自定义超参数 (lr0=0.001, optimizer=Adam, ...)")


if __name__ == "__main__":
    main()
