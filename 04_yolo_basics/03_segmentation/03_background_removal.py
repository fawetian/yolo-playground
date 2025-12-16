"""
背景移除
=======

学习目标:
- 使用分割结果进行背景移除
- 创建透明背景图像
- 替换背景
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
    print("🎨 背景移除")
    print("=" * 60)
    
    # 加载分割模型
    model = load_yolo_model("yolo11n-seg.pt")
    
    # 使用包含人物的示例图像
    test_image_path = get_sample_image("zidane.jpg")
    print(f"\n📷 测试图像: {test_image_path}")
    print("🔍 执行实例分割...")
    
    results = model(str(test_image_path), verbose=False)
    result = results[0]
    
    if result.masks is None:
        print("⚠️ 未检测到可分割目标")
        return
    
    orig_img = result.orig_img.copy()
    h, w = orig_img.shape[:2]
    
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    # ==========================================
    # 1. 找到人物目标
    # ==========================================
    
    print("\n" + "=" * 60)
    print("👤 寻找人物目标")
    print("=" * 60)
    
    person_indices = []
    for i, cls in enumerate(result.boxes.cls):
        cls_id = int(cls.item())
        if result.names[cls_id] == "person":
            person_indices.append(i)
            conf = result.boxes.conf[i].item()
            print(f"  找到人物 #{i}, 置信度: {conf:.2%}")
    
    if not person_indices:
        print("  ⚠️ 未检测到人物，将使用第一个目标演示")
        person_indices = [0]
    
    # ==========================================
    # 2. 创建人物掩码
    # ==========================================
    
    print("\n" + "=" * 60)
    print("🎭 创建目标掩码")
    print("=" * 60)
    
    masks_data = result.masks.data.cpu().numpy()
    
    # 合并所有人物掩码
    combined_mask = np.zeros((h, w), dtype=np.uint8)
    for idx in person_indices:
        mask = masks_data[idx]
        mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        mask_binary = (mask_resized > 0.5).astype(np.uint8)
        combined_mask = np.maximum(combined_mask, mask_binary)
    
    print(f"  掩码覆盖像素: {np.sum(combined_mask):,}")
    print(f"  覆盖比例: {np.sum(combined_mask) / (w * h) * 100:.1f}%")
    
    # ==========================================
    # 3. 背景移除 - 透明背景
    # ==========================================
    
    print("\n" + "=" * 60)
    print("✨ 背景移除 - 透明背景")
    print("=" * 60)
    
    # 创建 RGBA 图像
    rgba = cv2.cvtColor(orig_img, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = combined_mask * 255
    
    output_path = output_dir / "person_transparent.png"
    cv2.imwrite(str(output_path), rgba)
    print(f"  透明背景图已保存: {output_path}")
    
    # ==========================================
    # 4. 背景替换 - 纯色背景
    # ==========================================
    
    print("\n" + "=" * 60)
    print("🎨 背景替换 - 纯色")
    print("=" * 60)
    
    backgrounds = [
        ("white", (255, 255, 255)),
        ("blue", (200, 100, 50)),
        ("green", (50, 200, 50)),
    ]
    
    for name, color in backgrounds:
        # 创建纯色背景
        bg = np.full_like(orig_img, color)
        
        # 合成
        mask_3ch = np.stack([combined_mask] * 3, axis=-1)
        result_img = np.where(mask_3ch == 1, orig_img, bg)
        
        output_path = output_dir / f"bg_{name}.jpg"
        cv2.imwrite(str(output_path), result_img)
        print(f"  {name} 背景已保存: {output_path}")
    
    # ==========================================
    # 5. 背景替换 - 渐变背景
    # ==========================================
    
    print("\n" + "=" * 60)
    print("🌈 背景替换 - 渐变")
    print("=" * 60)
    
    # 创建渐变背景
    gradient_bg = np.zeros_like(orig_img)
    for i in range(h):
        # 从紫色渐变到橙色
        ratio = i / h
        color = (
            int(150 * (1 - ratio) + 50 * ratio),   # B
            int(50 * (1 - ratio) + 150 * ratio),   # G  
            int(200 * (1 - ratio) + 255 * ratio),  # R
        )
        gradient_bg[i, :] = color
    
    # 合成
    mask_3ch = np.stack([combined_mask] * 3, axis=-1)
    result_img = np.where(mask_3ch == 1, orig_img, gradient_bg)
    
    output_path = output_dir / "bg_gradient.jpg"
    cv2.imwrite(str(output_path), result_img)
    print(f"  渐变背景已保存: {output_path}")
    
    # ==========================================
    # 6. 背景模糊效果
    # ==========================================
    
    print("\n" + "=" * 60)
    print("🔮 背景模糊效果")
    print("=" * 60)
    
    # 模糊整个图像
    blurred = cv2.GaussianBlur(orig_img, (51, 51), 0)
    
    # 前景保持清晰，背景模糊
    mask_3ch = np.stack([combined_mask] * 3, axis=-1)
    result_img = np.where(mask_3ch == 1, orig_img, blurred)
    
    output_path = output_dir / "bg_blurred.jpg"
    cv2.imwrite(str(output_path), result_img)
    print(f"  模糊背景已保存: {output_path}")
    
    # ==========================================
    # 7. 边缘羽化
    # ==========================================
    
    print("\n" + "=" * 60)
    print("🪶 边缘羽化")
    print("=" * 60)
    
    # 创建羽化掩码
    mask_float = combined_mask.astype(np.float32)
    mask_blurred = cv2.GaussianBlur(mask_float, (21, 21), 0)
    
    # 使用羽化掩码混合
    mask_3ch = np.stack([mask_blurred] * 3, axis=-1)
    white_bg = np.full_like(orig_img, 255)
    result_img = (orig_img * mask_3ch + white_bg * (1 - mask_3ch)).astype(np.uint8)
    
    output_path = output_dir / "feathered_edge.jpg"
    cv2.imwrite(str(output_path), result_img)
    print(f"  羽化边缘已保存: {output_path}")
    
    print("\n✅ 背景移除演示完成!")
    print(f"📁 所有结果保存在: {output_dir}")


if __name__ == "__main__":
    main()
