"""
自定义模型训练 (macOS 版)
=======================

学习目标:
- 使用自定义数据集训练 YOLO 模型
- 理解训练参数配置
- 监控训练过程

macOS 说明:
- Apple Silicon 使用 device='mps' 获得 GPU 加速
- 如果 MPS 内存不足，减小 batch 或使用 CPU
"""

from ultralytics import YOLO
from pathlib import Path
import torch


def get_device():
    """获取最佳可用设备"""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "0"
    else:
        return "cpu"


def main():
    print("=" * 60)
    print("🏋️ YOLO 自定义模型训练 (macOS)")
    print("=" * 60)
    
    # 检测设备
    device = get_device()
    device_names = {"mps": "Apple Silicon GPU (MPS)", "cpu": "CPU", "0": "NVIDIA GPU"}
    print(f"💻 使用设备: {device_names.get(device, device)}")
    
    # ==========================================
    # 1. 准备数据集配置文件
    # ==========================================
    
    # 创建示例 data.yaml (你需要根据实际数据集修改)
    data_yaml = Path(__file__).parent / "example_data.yaml"
    
    if not data_yaml.exists():
        create_example_data_yaml(data_yaml)
        print(f"✅ 创建示例配置: {data_yaml}")
        print("⚠️ 请根据你的数据集修改 data.yaml 后再运行训练!")
        return
    
    # ==========================================
    # 2. 加载预训练模型
    # ==========================================
    
    print("\n📦 加载预训练模型...")
    
    # 从预训练模型开始 (迁移学习)
    model = YOLO("yolo11n.pt")
    
    # 或者从头开始训练 (需要更多数据和时间)
    # model = YOLO("yolo11n.yaml")
    
    # ==========================================
    # 3. 训练参数说明
    # ==========================================
    
    print("\n⚙️ 训练参数配置:")
    
    # macOS MPS 建议: 
    # - batch 设置为 8-16 (MPS 内存有限)
    # - 如果遇到内存错误，减小 batch 或 imgsz
    
    train_args = {
        # 数据集配置
        "data": str(data_yaml),
        
        # 训练轮次
        "epochs": 100,        # 训练总轮次
        
        # 批次大小 (MPS 建议 8-16，内存不足时减小)
        "batch": 8 if device == "mps" else 16,
        
        # 图像尺寸
        "imgsz": 640,         # 输入图像尺寸
        
        # 学习率
        "lr0": 0.01,          # 初始学习率
        "lrf": 0.01,          # 最终学习率 (lr0 * lrf)
        
        # 优化器
        "optimizer": "auto",  # SGD, Adam, AdamW, auto
        
        # 设备 (macOS: mps, Intel Mac: cpu)
        "device": device,
        
        # 输出目录
        "project": "runs/train",
        "name": "custom_model",
        
        # 其他
        "patience": 50,       # 早停耐心值
        "save": True,         # 保存检查点
        "save_period": 10,    # 每 N 轮保存一次
        "verbose": True,      # 详细输出
    }
    
    for key, value in train_args.items():
        print(f"  {key}: {value}")
    
    # ==========================================
    # 4. 开始训练
    # ==========================================
    
    print("\n🚀 开始训练...")
    print("=" * 60)
    
    # 取消下面的注释开始训练
    # results = model.train(**train_args)
    
    # 训练完成后，最佳模型保存在:
    # runs/train/custom_model/weights/best.pt
    
    print("\n⏸️ 训练代码已准备好")
    print("   请准备好数据集后取消注释 model.train() 行")
    
    # ==========================================
    # 5. 训练后验证
    # ==========================================
    
    print("\n📊 训练完成后可以运行验证:")
    print("""
    # 加载训练好的模型
    model = YOLO("runs/train/custom_model/weights/best.pt")
    
    # 在验证集上评估
    metrics = model.val()
    
    print(f"mAP50: {metrics.box.map50:.3f}")
    print(f"mAP50-95: {metrics.box.map:.3f}")
    """)
    
    # ==========================================
    # 6. 恢复训练
    # ==========================================
    
    print("\n🔄 如果训练中断，可以恢复训练:")
    print("""
    # 从最后的检查点恢复
    model = YOLO("runs/train/custom_model/weights/last.pt")
    results = model.train(resume=True)
    """)


def create_example_data_yaml(path: Path):
    """创建示例 data.yaml 配置"""
    content = """# 数据集配置示例
# 请根据你的实际数据集路径修改

# 数据集根目录
path: /path/to/your/dataset

# 图像目录 (相对于 path)
train: train/images
val: val/images
test: test/images  # 可选

# 类别数量
nc: 2

# 类别名称
names:
  0: class_1
  1: class_2
"""
    path.write_text(content)


if __name__ == "__main__":
    main()

