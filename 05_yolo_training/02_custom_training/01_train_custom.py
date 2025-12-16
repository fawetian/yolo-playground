"""
自定义训练基础
============

学习目标:
- 使用 YOLO 进行自定义训练
- 配置训练参数 (epochs, batch, imgsz)
- 使用 Apple Silicon (MPS) 加速
"""

from pathlib import Path
import sys
from ultralytics import YOLO

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.model_loader import load_yolo_model


def main():
    print("=" * 60)
    print("🚀 自定义训练基础")
    print("=" * 60)
    
    # ==========================================
    # 1. 准备配置
    # ==========================================
    
    # 配置文件路径 (由 01_dataset_prep/01_create_sample_dataset.py 生成)
    dataset_cfg = Path(__file__).parent.parent / "01_dataset_prep/coco8_local.yaml"
    
    if not dataset_cfg.exists():
        print("❌ 找不到数据集配置，请先运行 01_dataset_prep/01_create_sample_dataset.py")
        return
    
    print(f"\n📂 数据集配置: {dataset_cfg}")
    
    # ==========================================
    # 2. 加载模型
    # ==========================================
    
    print("\n📦 加载预训练模型 (transfer learning)...")
    # 使用 nano 模型进行快速演示
    # 推荐: yolo11n.pt (nano), yolo11s.pt (small), yolo11m.pt (medium)
    model = load_yolo_model("yolo11n.pt")
    
    # ==========================================
    # 3. 开始训练
    # ==========================================
    
    print("\n🔄 开始训练...")
    print("  注意: 这只是演示，epoch 设置很少")
    
    # 训练参数详解: https://docs.ultralytics.com/modes/train/
    try:
        results = model.train(
            data=str(dataset_cfg),   # 数据集配置
            epochs=3,                # 训练轮数 (实际训练通常 100+)
            imgsz=640,               # 输入图像尺寸
            batch=8,                 # 批次大小 (根据显存调整)
            device="mps",            # Apple Silicon 使用 mps, Nvidia 使用 0, cpu 使用 cpu
            project="runs/train",    # 保存路径
            name="demo_run",         # 实验名称
            exist_ok=True,           # 覆盖已存在的实验
            plots=True,              # 生成训练曲线图
            save=True,               # 保存 checkpoint
        )
        
        print("\n✅ 训练完成!")
        print(f"  结果保存在: {results.save_dir}")
        
    except Exception as e:
        print(f"\n❌ 训练出错: {e}")
        print("  如果是内存不足，请尝试减小 batch 或 imgsz")
    
    # ==========================================
    # 4. 验证模型
    # ==========================================
    
    print("\n🔍 验证模型...")
    metrics = model.val()
    
    print(f"  mAP50: {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
