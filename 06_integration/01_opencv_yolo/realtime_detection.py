"""
OpenCV + YOLO 实时检测 (macOS 版)
==============================

学习目标:
- 结合 OpenCV 视频捕获和 YOLO 检测
- 实现实时目标检测
- 自定义可视化效果

macOS 说明:
- 首次运行会请求摄像头权限
- Apple Silicon 使用 MPS 加速
"""

import cv2
import numpy as np
from ultralytics import YOLO
import torch
import time


def get_device():
    """获取最佳可用设备 (macOS 优化)"""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "0"
    else:
        return "cpu"


def main():
    print("=" * 60)
    print("🎥 实时目标检测 (macOS)")
    print("=" * 60)
    
    # 检测设备
    device = get_device()
    device_names = {"mps": "Apple Silicon GPU", "cpu": "CPU", "0": "NVIDIA GPU"}
    print(f"💻 使用设备: {device_names.get(device, device)}")
    
    # 加载模型 (选择较小的模型以保证速度)
    model = YOLO("yolo11n.pt")
    
    # ==========================================
    # 1. 初始化视频捕获
    # ==========================================
    
    # 使用摄像头
    # macOS: 首次运行会请求摄像头权限，请点击"允许"
    print("\n📷 正在访问摄像头...")
    print("   如果弹出权限请求，请点击'允许'")
    
    cap = cv2.VideoCapture(0)
    
    # 如果没有摄像头，使用视频文件
    # cap = cv2.VideoCapture("path/to/video.mp4")
    
    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        print("💡 macOS 提示:")
        print("   1. 确保已授予摄像头权限")
        print("   2. 系统设置 → 隐私与安全性 → 摄像头 → 开启 Terminal/IDE")
        print("   3. 或修改代码使用视频文件")
        return
    
    # 设置分辨率 (降低分辨率可提高帧率)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("✅ 摄像头已打开")
    
    # ==========================================
    # 2. 主循环
    # ==========================================
    
    print("\n🎬 按 'q' 退出")
    print("   按 's' 截图")
    print("   按 'p' 暂停/继续")
    
    paused = False
    frame_count = 0
    fps_start_time = time.time()
    fps = 0
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("❌ 无法读取帧")
                break
            
            # 运行 YOLO 检测 (使用最佳设备)
            results = model(frame, verbose=False, device=device)
            result = results[0]
            
            # ==========================================
            # 3. 自定义可视化
            # ==========================================
            
            annotated_frame = custom_visualization(frame, result, model.names)
            
            # 计算 FPS
            frame_count += 1
            if frame_count % 30 == 0:
                fps = 30 / (time.time() - fps_start_time)
                fps_start_time = time.time()
            
            # 显示 FPS
            cv2.putText(
                annotated_frame, 
                f"FPS: {fps:.1f}", 
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2
            )
            
            # 显示检测数量
            num_detections = len(result.boxes)
            cv2.putText(
                annotated_frame, 
                f"Detections: {num_detections}", 
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2
            )
        
        # macOS: 使用 WINDOW_NORMAL 可以调整窗口大小
        cv2.namedWindow("YOLO Realtime Detection", cv2.WINDOW_NORMAL)
        cv2.imshow("YOLO Realtime Detection", annotated_frame)
        
        # ==========================================
        # 4. 键盘控制
        # ==========================================
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s'):
            # 截图
            screenshot_path = f"screenshot_{int(time.time())}.jpg"
            cv2.imwrite(screenshot_path, annotated_frame)
            print(f"📸 截图保存: {screenshot_path}")
        elif key == ord('p'):
            paused = not paused
            print("⏸️ 暂停" if paused else "▶️ 继续")
    
    # 清理
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)  # macOS 需要额外的 waitKey 来完全关闭窗口
    print("\n👋 检测结束")


def custom_visualization(frame: np.ndarray, result, class_names: dict) -> np.ndarray:
    """
    自定义可视化效果
    
    Args:
        frame: 原始帧
        result: YOLO 检测结果
        class_names: 类别名称字典
    
    Returns:
        带标注的帧
    """
    annotated = frame.copy()
    boxes = result.boxes
    
    # 为不同类别定义颜色
    colors = {
        0: (0, 255, 0),     # person - 绿色
        2: (255, 0, 0),     # car - 蓝色
        5: (0, 0, 255),     # bus - 红色
        7: (255, 255, 0),   # truck - 青色
    }
    default_color = (128, 128, 128)
    
    for box in boxes:
        # 获取信息
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = box.conf[0].item()
        cls_id = int(box.cls[0].item())
        cls_name = class_names[cls_id]
        
        # 选择颜色
        color = colors.get(cls_id, default_color)
        
        # 绘制边界框
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        
        # 绘制标签背景
        label = f"{cls_name} {conf:.0%}"
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
        )
        
        cv2.rectangle(
            annotated, 
            (x1, y1 - label_h - 10), 
            (x1 + label_w + 4, y1), 
            color, -1
        )
        
        # 绘制标签文字
        cv2.putText(
            annotated, label, 
            (x1 + 2, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, (255, 255, 255), 1
        )
        
        # 可选: 绘制中心点
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(annotated, (cx, cy), 4, color, -1)
    
    return annotated


if __name__ == "__main__":
    main()

