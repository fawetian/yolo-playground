"""
OpenCV 与 YOLO 基础集成
=====================

学习目标:
- 理解 OpenCV 图像读取与 YOLO 推理的结合
- 手动解析 YOLO 结果 (Results 对象)
- 使用 OpenCV 原生绘图函数绘制检测框和标签
"""

from pathlib import Path
import cv2
import numpy as np
import sys
import random

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.model_loader import load_yolo_model
from utils.image_loader import get_sample_image


def main():
    print("=" * 60)
    print("🔄 OpenCV + YOLO 基础集成")
    print("=" * 60)
    
    # 1. 加载模型
    model = load_yolo_model("yolo11n.pt")
    
    # 2. 读取图像 (使用 OpenCV)
    # 获取测试图像路径
    img_path = get_sample_image("bus.jpg")
    print(f"\n📷 读取图像: {img_path}")
    
    # cv2.imread 读取为 BGR 格式
    frame = cv2.imread(str(img_path))
    if frame is None:
        print("❌ 无法读取图像")
        return

    # 3. 执行推理
    print("🔍 执行推理...")
    # YOLOv8+ 可以直接接受 BGR numpy array
    results = model(frame, verbose=False)
    result = results[0]
    
    # 4. 手动绘制结果
    # 相比 result.plot()，手动绘制给我们更多控制权 (样式、颜色、逻辑)
    print("🎨 绘制检测结果...")
    
    annotated_frame = frame.copy()
    
    boxes = result.boxes
    print(f"  检测到 {len(boxes)} 个目标")
    
    for box in boxes:
        # 获取坐标 (xyxy)
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        
        # 获取类别和置信度
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        class_name = result.names[cls_id]
        
        # 仅绘制置信度 > 0.5 的目标
        if conf > 0.5:
            # 生成随机颜色 (基于类别ID)
            random.seed(cls_id)
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            
            # 1. 画矩形框
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # 2. 准备标签文字
            label = f"{class_name} {conf:.2f}"
            
            # 3. 计算文字背景尺寸
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            
            # 4. 画文字背景 (填充矩形)
            cv2.rectangle(annotated_frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
            
            # 5. 画白色文字
            text_color = (255, 255, 255)
            cv2.putText(annotated_frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
            
            print(f"    - {class_name}: {conf:.2%}")
            
    # 5. 保存结果
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "cv2_integration_result.jpg"
    
    cv2.imwrite(str(output_path), annotated_frame)
    print(f"\n💾 结果已保存: {output_path}")
    
    # 注意: 在服务器/无头环境中不要使用 cv2.imshow
    # cv2.imshow("Result", annotated_frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
