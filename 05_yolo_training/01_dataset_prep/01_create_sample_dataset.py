"""
创建示例数据集
============

学习目标:
- 下载标准示例数据集 (coco8)
- 理解 YOLO 数据集目录结构
- 创建自定义数据集 YAML 配置文件
"""

from pathlib import Path
import sys
import yaml
from ultralytics.utils.downloads import download

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.model_loader import MODELS_DIR

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent

# 数据集根目录
DATASETS_DIR = PROJECT_ROOT / "datasets"
DATASETS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 60)
    print("📦 创建示例数据集")
    print("=" * 60)
    
    # ==========================================
    # 1. 下载 COCO8 数据集
    # ==========================================
    
    # coco8 是一个极小的数据集 (8张图)，用于快速测试
    dataset_name = "coco8"
    dataset_dir = DATASETS_DIR / dataset_name
    
    print(f"\n📥 准备数据集: {dataset_name}")
    
    if not dataset_dir.exists():
        print("  正在下载数据集...")
        # Ultralytics 会自动下载并解压
        # 我们手动指定 URL 以便控制下载位置
        url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8.zip"
        download(url, dir=DATASETS_DIR)
        print(f"✅ 数据集已保存到: {dataset_dir}")
    else:
        print(f"✅ 数据集已存在: {dataset_dir}")
    
    # ==========================================
    # 2. 检查目录结构
    # ==========================================
    
    print("\n📂 数据集结构:")
    
    print(f"{dataset_name}/")
    print("├── images/")
    print("│   ├── train/  (训练图片)")
    print("│   └── val/    (验证图片)")
    print("└── labels/")
    print("    ├── train/  (训练标签 .txt)")
    print("    └── val/    (验证标签 .txt)")
    
    # 验证文件数量
    train_imgs = len(list((dataset_dir / "images/train").glob("*.jpg")))
    val_imgs = len(list((dataset_dir / "images/val").glob("*.jpg")))
    print(f"\n统计:")
    print(f"  训练集: {train_imgs} 张图片")
    print(f"  验证集: {val_imgs} 张图片")
    
    # ==========================================
    # 3. 创建数据集配置文件 (YAML)
    # ==========================================
    
    print("\n📝 创建 dataset.yaml 配置文件")
    
    # 定义数据集配置
    # 注意: YOLO 需要绝对路径，或者相对于 datasets 目录的路径
    dataset_config = {
        "path": str(dataset_dir.absolute()),  # 数据集根目录
        "train": "images/train",              # 训练集 (相对于 path)
        "val": "images/val",                  # 验证集 (相对于 path)
        
        # 类别定义
        "names": {
            0: "person",
            1: "bicycle",
            2: "car",
            3: "motorcycle",
            4: "airplane",
            5: "bus",
            6: "train",
            7: "truck",
            8: "boat",
            9: "traffic light",
            10: "fire hydrant",
            # ... coco8 只包含部分类别，但为了兼容性通常保留标准 COCO 类别
        }
    }
    
    # 保存配置
    config_path = Path(__file__).parent / "coco8_local.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(dataset_config, f, sort_keys=False)
    
    print(f"✅ 配置文件已保存: {config_path}")
    print("\n内容预览:")
    with open(config_path, "r") as f:
        print(f.read())
        
    print("\n✅ 数据集准备完成!")
    print(f"你可以在训练脚本中使用此配置文件: path='{config_path}'")


if __name__ == "__main__":
    main()
