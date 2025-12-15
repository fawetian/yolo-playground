"""
YOLO 快速入门
============

学习目标:
- 安装和导入 Ultralytics
- 了解 YOLO 的基本使用方式
- 运行第一个目标检测

前置要求:
- pip install ultralytics

macOS 说明:
- Apple Silicon (M1/M2/M3) 自动使用 MPS 加速
- Intel Mac 使用 CPU
"""

from pathlib import Path
import urllib.request
import torch
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.model_loader import load_yolo_model


def get_device():
    """获取最佳可用设备 (macOS 优化)"""
    if torch.backends.mps.is_available():
        return "mps"  # Apple Silicon GPU
    elif torch.cuda.is_available():
        return "0"    # NVIDIA GPU
    else:
        return "cpu"


def main():
    print("=" * 60)
    print("🚀 YOLO 快速入门 (macOS)")
    print("=" * 60)
    
    # 显示设备信息
    device = get_device()
    device_name = {
        "mps": "Apple Silicon GPU (MPS)",
        "cpu": "CPU",
        "0": "NVIDIA GPU"
    }.get(device, device)
    print(f"\n💻 使用设备: {device_name}")
    
    # ==========================================
    # 1. 加载预训练模型
    # ==========================================
    
    # YOLO11 模型系列 (按大小排序):
    # - yolo11n.pt: Nano (最快，精度较低)
    # - yolo11s.pt: Small
    # - yolo11m.pt: Medium
    # - yolo11l.pt: Large
    # - yolo11x.pt: Extra-Large (最慢，精度最高)
    
    print("\n📦 加载模型...")
    # 优先从本地 models/yolo/ 目录加载，如果没有则自动下载
    model = load_yolo_model("yolo11n.pt")
    print("✅ 模型加载成功!")
    
    # ==========================================
    # 2. 模型信息
    # ==========================================
    
    print("\n📋 模型信息:")
    print(f"  任务类型: {model.task}")
    print(f"  模型名称: {model.model_name if hasattr(model, 'model_name') else 'YOLO11n'}")
    
    # 查看模型可检测的类别
    print(f"\n  可检测类别数: {len(model.names)}")
    print(f"  前10个类别: {list(model.names.values())[:10]}")
    
    # ==========================================
    # 3. 准备测试图像
    # ==========================================
    
    # 使用 Ultralytics 官方示例图像
    test_url = "https://ultralytics.com/images/bus.jpg"
    test_image = Path(__file__).parent / "bus.jpg"
    
    if not test_image.exists():
        print(f"\n📥 下载测试图像...")
        urllib.request.urlretrieve(test_url, test_image)
        print(f"✅ 保存到: {test_image}")
    
    # ==========================================
    # 4. 运行推理
    # ==========================================
    
    print("\n🔍 运行目标检测...")
    # 使用最佳设备进行推理
    results = model(str(test_image), conf=0.90, device=device)
    
    # results 是一个列表，每个元素对应一张输入图像
    result = results[0]
    
    # ==========================================
    # 5. 解析结果
    # ==========================================
    
    print("\n📊 检测结果:")
    print("-" * 40)
    
    # 获取检测框
    boxes = result.boxes
    
    for i, box in enumerate(boxes):
        # 边界框坐标 (x1, y1, x2, y2)
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        
        # 置信度
        confidence = box.conf[0].item()
        
        # 类别
        class_id = int(box.cls[0].item())
        class_name = model.names[class_id]
        
        print(f"  [{i+1}] {class_name}")
        print(f"      置信度: {confidence:.2%}")
        print(f"      边界框: ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")
        print()
    
    # ==========================================
    # 6. 保存可视化结果
    # ==========================================
    
    output_path = Path(__file__).parent / "bus_detected.jpg"
    
    # 方式1: 使用 result.plot() 获取带标注的图像
    annotated = result.plot()
    
    import cv2
    cv2.imwrite(str(output_path), annotated)
    print(f"✅ 结果已保存: {output_path}")
    
    # 方式2: 直接保存到目录
    # result.save(save_dir="outputs/")
    
    # ==========================================
    # 7. 显示结果
    # ==========================================
    
    print("\n💡 按任意键关闭窗口...")
    
    # macOS 优化的窗口显示
    cv2.namedWindow("YOLO Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLO Detection", 800, 600)
    cv2.imshow("YOLO Detection", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)  # macOS 需要额外的 waitKey 来完全关闭窗口
    
    print("\n🎉 恭喜！你已完成第一个 YOLO 目标检测!")


if __name__ == "__main__":
    main()

