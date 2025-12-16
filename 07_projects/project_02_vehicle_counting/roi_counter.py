"""
项目 2: 车辆计数器
================

描述:
基于感兴趣区域 (ROI) 的车辆计数。
检测车辆 -> 跟踪中心点 -> 判断是否穿越计数线。
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
    print("🚗 车辆计数器 (ROI Loop Demo)")
    print("=" * 60)
    
    model = load_yolo_model("yolo11n.pt")
    
    # 使用静态图片模拟视频流 (循环处理同一张图并移动 ROI 线来演示)
    # 在真实项目中，这里应读取视频流 (cv2.VideoCapture)
    img_path = get_sample_image("bus.jpg")
    base_frame = cv2.imread(str(img_path))
    
    print(f"\n🎥 模拟视频流输入: {img_path.name}")
    
    # 定义车辆类别 ID (COCO 格式)
    # 2=car, 3=motorcycle, 5=bus, 7=truck
    VEHICLE_CLASSES = [2, 3, 5, 7]
    
    # 定义计数线 (屏幕中间水平线)
    h, w = base_frame.shape[:2]
    line_y = int(h * 0.6)  # 60% 高度处
    offset = 10  # 判定偏移量
    
    vehicle_count = 0
    
    # 模拟 3 帧的处理
    for i in range(3):
        print(f"\n--- Frame {i+1} ---")
        frame = base_frame.copy()
        
        # 绘制计数线
        cv2.line(frame, (0, line_y), (w, line_y), (0, 0, 255), 3)
        
        results = model(frame, verbose=False)
        result = results[0]
        
        # 1. 检测车辆
        detections = []
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            if cls_id in VEHICLE_CLASSES:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].item())
                
                # 计算中心点
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                
                detections.append((cx, cy, x1, y1, x2, y2, cls_id))
                
                # 绘制车辆框
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)
        
        print(f"  当前帧检测到 {len(detections)} 辆车")
        
        # 2. 计数逻辑 (简化版)
        for (cx, cy, _, _, _, _, cls_id) in detections:
            # 判断是否在计数线附近 (实际项目需要 Object Tracking ID 来避免重复计数)
            # 这里简单演示逻辑：如果在范围内则判定为"计数" (仅演示)
            if line_y - offset < cy < line_y + offset:
                # 假设这是被 Track 的对象首次经过
                vehicle_count += 1
                cv2.line(frame, (0, line_y), (w, line_y), (0, 255, 0), 3) # 变绿
                print(f"  ✨ 车辆穿越! 类型: {result.names[cls_id]}")
        
        # 显示计数
        cv2.putText(frame, f"Count: {vehicle_count}", (30, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                    
        # 保存演示帧
        output_dir = Path(__file__).parent / "outputs"
        output_dir.mkdir(exist_ok=True)
        cv2.imwrite(str(output_dir / f"count_frame_{i}.jpg"), frame)
    
    print(f"\n✅ 模拟结束")
    print(f"  总计数: {vehicle_count}")
    print(f"  演示帧保存在: {output_dir}")


if __name__ == "__main__":
    main()
