"""
训练过程可视化
============

学习目标:
- 解析训练日志 (results.csv)
- 绘制损失曲线和准确率曲线
- 理解训练指标
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def main():
    print("=" * 60)
    print("📊 训练过程可视化")
    print("=" * 60)
    
    # 查找训练结果文件
    # 假设运行了 01_train_custom.py
    results_dir = Path("runs/train/demo_run")
    csv_path = results_dir / "results.csv"
    
    print(f"\n📂 结果目录: {results_dir}")
    
    if not csv_path.exists():
        print(f"❌ 找不到 {csv_path}")
        print("  请先运行 01_train_custom.py")
        
        # 尝试查找其他 results.csv
        demos = list(Path("runs/train").glob("**/results.csv"))
        if demos:
            print(f"  发现其他结果: {demos[0]}")
            csv_path = demos[0]
        else:
            return
            
    # ==========================================
    # 1. 加载训练数据
    # ==========================================
    
    print(f"\n1️⃣ 加载数据: {csv_path}")
    
    # 读取 CSV (清除列名前后的空格)
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    
    print(f"  包含 {len(df)} 个 epoch 的数据")
    print(f"  列名: {list(df.columns)}")
    
    # ==========================================
    # 2. 绘制训练曲线
    # ==========================================
    
    print("\n2️⃣ 绘制曲线...")
    
    plt.figure(figsize=(12, 10))
    
    # 2.1 损失曲线
    plt.subplot(2, 2, 1)
    plt.plot(df["epoch"], df["train/box_loss"], label="Train Box Loss")
    plt.plot(df["epoch"], df["val/box_loss"], label="Val Box Loss")
    plt.title("Box Loss (Bounding Box)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.plot(df["epoch"], df["train/cls_loss"], label="Train Cls Loss")
    plt.plot(df["epoch"], df["val/cls_loss"], label="Val Cls Loss")
    plt.title("Class Loss (Classification)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2.2 准确率曲线 (mAP)
    plt.subplot(2, 2, 3)
    plt.plot(df["epoch"], df["metrics/mAP50(B)"], label="mAP@50")
    plt.plot(df["epoch"], df["metrics/mAP50-95(B)"], label="mAP@50-95")
    plt.title("Mean Average Precision (mAP)")
    plt.xlabel("Epoch")
    plt.ylabel("mAP")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2.3 学习率
    plt.subplot(2, 2, 4)
    if "lr/pg0" in df.columns:
        plt.plot(df["epoch"], df["lr/pg0"], label="Learning Rate")
    plt.title("Learning Rate")
    plt.xlabel("Epoch")
    plt.ylabel("LR")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    output_path = Path(__file__).parent / "training_metrics.png"
    plt.savefig(output_path)
    print(f"✅ 图表已保存: {output_path}")
    
    # ==========================================
    # 3. 分析最佳结果
    # ==========================================
    
    best_epoch = df.loc[df["metrics/mAP50-95(B)"].idxmax()]
    print(f"\n🏆 最佳结果 (Epoch {int(best_epoch['epoch'])}):")
    print(f"  mAP@50:    {best_epoch['metrics/mAP50(B)']:.4f}")
    print(f"  mAP@50-95: {best_epoch['metrics/mAP50-95(B)']:.4f}")


if __name__ == "__main__":
    main()
