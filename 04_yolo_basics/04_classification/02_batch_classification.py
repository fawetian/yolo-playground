"""
批量图像分类
==========

学习目标:
- 批量处理多张图像
- 分类结果统计与分析
- 结果导出
"""

from pathlib import Path
import cv2
import numpy as np
import sys
import json
from collections import Counter

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.model_loader import load_yolo_model
from utils.image_loader import get_all_sample_images, IMAGES_DIR


def main():
    print("=" * 60)
    print("📦 批量图像分类")
    print("=" * 60)
    
    # 加载分类模型
    model = load_yolo_model("yolo11n-cls.pt")
    
    # ==========================================
    # 1. 获取测试图像
    # ==========================================
    
    print("\n📷 获取测试图像...")
    
    # 从 datasets/images 获取所有图像
    image_paths = get_all_sample_images()
    print(f"  共 {len(image_paths)} 张图像")
    
    if len(image_paths) == 0:
        print("⚠️ 没有可用的测试图像")
        return
    
    # ==========================================
    # 2. 批量推理
    # ==========================================
    
    print("\n" + "=" * 60)
    print("🔍 批量分类")
    print("=" * 60)
    
    # 方法1: 一次性处理所有图像
    results = model([str(p) for p in image_paths], verbose=False)
    
    # 收集结果
    classification_results = []
    
    for result in results:
        img_path = Path(result.path)
        top1_idx = result.probs.top1
        top1_conf = result.probs.top1conf.item()
        top1_name = result.names[top1_idx]
        
        # Top-3
        top3_idx = result.probs.top5[:3]
        top3_names = [result.names[idx] for idx in top3_idx]
        top3_confs = result.probs.top5conf[:3].tolist()
        
        classification_results.append({
            "filename": img_path.name,
            "predicted_class": top1_name,
            "confidence": top1_conf,
            "top3": list(zip(top3_names, top3_confs))
        })
        
        print(f"  {img_path.name}: {top1_name} ({top1_conf:.2%})")
    
    # ==========================================
    # 3. 结果统计分析
    # ==========================================
    
    print("\n" + "=" * 60)
    print("📊 统计分析")
    print("=" * 60)
    
    # 类别分布
    class_counter = Counter([r["predicted_class"] for r in classification_results])
    
    print("\n  类别分布:")
    for cls, count in class_counter.most_common(10):
        bar = "█" * count
        print(f"    {cls:20s}: {count:3d} {bar}")
    
    # 置信度统计
    confidences = [r["confidence"] for r in classification_results]
    print(f"\n  置信度统计:")
    print(f"    平均置信度: {np.mean(confidences):.2%}")
    print(f"    最高置信度: {np.max(confidences):.2%}")
    print(f"    最低置信度: {np.min(confidences):.2%}")
    
    if len(confidences) > 1:
        print(f"    标准差: {np.std(confidences):.2%}")
    
    # 置信度分布
    high_conf = sum(1 for c in confidences if c > 0.8)
    medium_conf = sum(1 for c in confidences if 0.5 <= c <= 0.8)
    low_conf = sum(1 for c in confidences if c < 0.5)
    
    print(f"\n  置信度分布:")
    print(f"    高 (>80%): {high_conf} 张")
    print(f"    中 (50-80%): {medium_conf} 张")
    print(f"    低 (<50%): {low_conf} 张")
    
    # ==========================================
    # 4. 按置信度筛选
    # ==========================================
    
    print("\n" + "=" * 60)
    print("🔍 置信度筛选")
    print("=" * 60)
    
    threshold = 0.3
    high_conf_results = [r for r in classification_results if r["confidence"] > threshold]
    print(f"\n  置信度 > {threshold:.0%} 的结果: {len(high_conf_results)}/{len(classification_results)}")
    
    for r in high_conf_results[:5]:  # 只显示前5个
        print(f"    {r['filename']}: {r['predicted_class']} ({r['confidence']:.2%})")
    
    # ==========================================
    # 5. 结果导出
    # ==========================================
    
    print("\n" + "=" * 60)
    print("💾 导出结果")
    print("=" * 60)
    
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    # 导出 JSON
    json_results = []
    for r in classification_results:
        json_results.append({
            "filename": r["filename"],
            "predicted_class": r["predicted_class"],
            "confidence": round(r["confidence"], 4),
            "top3": [(name, round(conf, 4)) for name, conf in r["top3"]]
        })
    
    json_path = output_dir / "classification_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    print(f"  JSON 结果: {json_path}")
    
    # 导出 CSV
    csv_path = output_dir / "classification_results.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("filename,predicted_class,confidence\n")
        for r in classification_results:
            f.write(f"{r['filename']},{r['predicted_class']},{r['confidence']:.4f}\n")
    print(f"  CSV 结果: {csv_path}")
    
    # ==========================================
    # 6. 创建结果摘要图
    # ==========================================
    
    if len(classification_results) > 0:
        print("\n  创建结果摘要图...")
        create_summary_image(classification_results, output_dir, IMAGES_DIR)
    
    print("\n✅ 批量分类演示完成!")


def create_summary_image(results, output_dir: Path, images_dir: Path):
    """创建结果摘要马赛克图"""
    # 每行显示5张图
    cols = min(5, len(results))
    rows = (len(results) + cols - 1) // cols
    
    thumb_size = 120
    padding = 5
    text_height = 40
    
    total_width = cols * (thumb_size + padding) + padding
    total_height = rows * (thumb_size + text_height + padding) + padding
    
    summary = np.full((total_height, total_width, 3), 255, dtype=np.uint8)
    
    for i, result in enumerate(results):
        row = i // cols
        col = i % cols
        
        x = col * (thumb_size + padding) + padding
        y = row * (thumb_size + text_height + padding) + padding
        
        # 加载并缩放图像
        img_path = images_dir / result["filename"]
        if img_path.exists():
            img = cv2.imread(str(img_path))
            if img is not None:
                img_resized = cv2.resize(img, (thumb_size, thumb_size))
                summary[y:y+thumb_size, x:x+thumb_size] = img_resized
        
        # 添加文字
        text = f"{result['predicted_class'][:12]}"
        conf_text = f"{result['confidence']:.0%}"
        
        cv2.putText(summary, text, (x, y + thumb_size + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
        cv2.putText(summary, conf_text, (x, y + thumb_size + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 128, 0), 1)
    
    cv2.imwrite(str(output_dir / "batch_summary.jpg"), summary)
    print(f"  摘要图: {output_dir / 'batch_summary.jpg'}")


if __name__ == "__main__":
    main()
