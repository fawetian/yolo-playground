"""
视频文件分析
==========

学习目标:
- 使用生成器逐帧处理视频
- 保存处理后的视频文件
- 进度条显示处理进度
"""

from pathlib import Path
import cv2
import sys
import time

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.model_loader import load_yolo_model
from utils.image_loader import DATASETS_DIR

def main():
    print("=" * 60)
    print("🎬 视频文件分析")
    print("=" * 60)
    
    # 检查是否有测试视频，如果没有则警告提示
    # 这里可以使用 datasets/videos 目录，如果为空则需要用户提供
    video_dir = DATASETS_DIR / "videos"
    video_files = list(video_dir.glob("*.mp4"))
    
    if not video_files:
        print(f"⚠️ 在 {video_dir} 中未找到 .mp4 视频")
        print("  请放入一个测试视频 (例如 test.mp4) 后再运行")
        return

    # 选择第一个视频
    input_video_path = video_files[0]
    print(f"\n📂 输入视频: {input_video_path}")
    
    # 加载模型
    model = load_yolo_model("yolo11n.pt")
    
    # 视频处理
    process_video(model, input_video_path)


def process_video(model, video_path):
    cap = cv2.VideoCapture(str(video_path))
    
    # 获取视频属性
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"  分辨率: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  总帧数: {total_frames}")
    
    # 设置输出路径
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"processed_{video_path.name}"
    
    # 视频写入器 (.mp4 / H.264)
    # macOS 上 'avc1' 通常兼容性较好，如果失败可尝试 'mp4v'
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    print("\n🚀 开始处理 (请稍候)...")
    start_time = time.time()
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_idx += 1
        
        # 推理
        results = model(frame, verbose=False)
        annotated_frame = results[0].plot()
        
        # 写入结果
        out.write(annotated_frame)
        
        # 进度条
        if frame_idx % 10 == 0:
            percent = frame_idx / total_frames
            bar_length = 30
            filled = int(bar_length * percent)
            bar = "█" * filled + "-" * (bar_length - filled)
            print(f"\r  [{bar}] {percent:.1%} ({frame_idx}/{total_frames})", end="")
            
    cap.release()
    out.release()
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n\n✅ 处理完成!")
    print(f"  耗时: {duration:.2f} 秒")
    print(f"  平均 FPS: {frame_idx / duration:.1f}")
    print(f"  输出文件: {output_path}")


if __name__ == "__main__":
    main()
