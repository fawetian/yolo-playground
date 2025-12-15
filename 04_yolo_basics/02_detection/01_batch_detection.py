"""
批量目标检测
==========

学习目标:
- 批量处理多张图像
- 理解推理参数配置
- 过滤检测结果
"""

from pathlib import Path
import cv2
import numpy as np
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from utils.model_loader import load_yolo_model


def main():
    print("=" * 60)
    print("📦 批量目标检测")
    print("=" * 60)
    
    # 加载模型 (优先从本地 models/yolo/ 目录加载)
    model = load_yolo_model("yolo11n.pt")
    
    # ==========================================
    # 1. 批量处理多张图像
    # ==========================================
    
    # 创建测试图像
    test_dir = Path(__file__).parent / "test_images"
    test_dir.mkdir(exist_ok=True)
    
    create_test_images(test_dir)
    
    # 批量推理
    image_paths = list(test_dir.glob("*.jpg"))
    print(f"\n📷 找到 {len(image_paths)} 张图像")
    
    # 一次性处理所有图像
    results = model(image_paths)
    
    # ==========================================
    # 2. 推理参数配置
    # ==========================================
    
    print("\n⚙️ 推理参数示例:")
    
    # 常用参数:
    results = model(
        image_paths[0],
        conf=0.5,          # 置信度阈值 (过滤低置信度检测)
        iou=0.45,          # NMS IoU 阈值
        classes=[0, 2, 5],  # 只检测特定类别 (person, car, bus)
        verbose=False       # 关闭日志输出
    )
    
    print("  conf=0.5    : 只保留置信度 > 50% 的检测")
    print("  iou=0.45    : NMS 重叠阈值")
    print("  classes=[0] : 只检测 person 类别")
    
    # ==========================================
    # 3. 结果过滤
    # ==========================================
    
    print("\n🔍 结果过滤示例:")
    
    # 重新检测 (不过滤)
    result = model(image_paths[0], verbose=False)[0]
    boxes = result.boxes
    
    # 按置信度过滤
    high_conf_mask = boxes.conf > 0.7
    high_conf_boxes = boxes[high_conf_mask]
    print(f"  置信度 > 70% 的检测: {len(high_conf_boxes)} 个")
    
    # 按类别过滤
    person_mask = boxes.cls == 0  # 0 = person
    person_boxes = boxes[person_mask]
    print(f"  person 类别检测: {len(person_boxes)} 个")
    
    # 按面积过滤
    if len(boxes) > 0:
        areas = (boxes.xyxy[:, 2] - boxes.xyxy[:, 0]) * (boxes.xyxy[:, 3] - boxes.xyxy[:, 1])
        large_mask = areas > 10000  # 面积大于 10000 像素
        large_boxes = boxes[large_mask]
        print(f"  面积 > 10000 的检测: {len(large_boxes)} 个")
    
    # ==========================================
    # 4. 保存批量结果
    # ==========================================
    
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n💾 保存结果到: {output_dir}")
    
    results = model(image_paths, verbose=False)
    for i, result in enumerate(results):
        annotated = result.plot()
        cv2.imwrite(str(output_dir / f"detected_{i}.jpg"), annotated)
    
    print(f"✅ 已保存 {len(results)} 张检测结果")


def create_test_images(output_dir: Path):
    """创建测试图像"""
    colors = [
        (50, 50, 200),   # 红色调
        (50, 200, 50),   # 绿色调
        (200, 50, 50),   # 蓝色调
    ]
    
    for i, color in enumerate(colors):
        img = np.full((480, 640, 3), color, dtype=np.uint8)
        
        # 添加一些形状
        cv2.rectangle(img, (100, 100), (300, 300), (255, 255, 255), 3)
        cv2.circle(img, (450, 240), 80, (255, 255, 255), 3)
        
        cv2.imwrite(str(output_dir / f"test_{i}.jpg"), img)


if __name__ == "__main__":
    main()

