"""
实时摄像头检测
============

学习目标:
- 打开并读取摄像头视频流
- 实现实时推理循环
- 性能优化技巧 (跳帧、分辨率调整)
"""

from pathlib import Path
import cv2
import time
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.model_loader import load_yolo_model


def main():
    print("=" * 60)
    print("📹 实时摄像头检测")
    print("=" * 60)
    
    # 1. 加载模型
    # 使用 nano 模型以获得最快速度
    model = load_yolo_model("yolo11n.pt")
    
    # 2. 打开摄像头
    # mac通常是 0 或 1
    camera_id = 0
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"❌ 无法打开摄像头 (ID: {camera_id})")
        print("  请检查摄像头权限或连接")
        return
        
    print(f"✅ 摄像头已打开")
    print("  按 'q' 键退出...")
    print("  按 's' 键保存截图...")
    
    # 3. 性能参数
    prev_time = 0
    fps_history = []
    skip_frames = 2  # 每隔 N 帧处理一次 (优化性能)
    frame_count = 0
    
    # 存储上一帧的检测结果，用于跳帧时的平滑显示
    last_results = None
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ 无法读取视频帧")
                break
                
            frame_count += 1
            
            # 镜像翻转 (自拍模式)
            frame = cv2.flip(frame, 1)
            
            # --- 推理逻辑 ---
            # 仅在非跳过帧时进行推理
            if frame_count % (skip_frames + 1) == 0:
                results = model(frame, verbose=False)
                last_results = results[0]
            
            # --- 绘制逻辑 ---
            annotated_frame = frame.copy()
            
            if last_results:
                # 使用 YOLO 自带的 plot 绘制，或参考 01_cv2_yolo_basic.py 手动绘制
                annotated_frame = last_results.plot()
            
            # --- 计算 FPS ---
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
            prev_time = curr_time
            
            # 平滑 FPS 显示
            fps_history.append(fps)
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_fps = sum(fps_history) / len(fps_history)
            
            # 显示 FPS
            cv2.putText(annotated_frame, f"FPS: {avg_fps:.1f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if frame_count % (skip_frames + 1) != 0:
                cv2.putText(annotated_frame, "(Cached)", (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

            # --- 显示结果 ---
            cv2.imshow("YOLO Real-time Detection", annotated_frame)
            
            # --- 键盘控制 ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                timestamp = int(time.time())
                save_path = f"webcam_capture_{timestamp}.jpg"
                cv2.imwrite(save_path, annotated_frame)
                print(f"📸 截图已保存: {save_path}")

    except KeyboardInterrupt:
        print("\n🛑 用户中断")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n✅ 程序结束")


if __name__ == "__main__":
    main()
