"""
姿态估计基础
==========

学习目标:
- 理解人体姿态估计
- 使用 YOLO Pose 模型
- 访问和理解关键点数据
"""

from pathlib import Path
import cv2
import numpy as np
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.model_loader import load_yolo_model
from utils.image_loader import get_sample_image

# COCO 关键点定义
KEYPOINT_NAMES = [
    "nose",           # 0: 鼻子
    "left_eye",       # 1: 左眼
    "right_eye",      # 2: 右眼
    "left_ear",       # 3: 左耳
    "right_ear",      # 4: 右耳
    "left_shoulder",  # 5: 左肩
    "right_shoulder", # 6: 右肩
    "left_elbow",     # 7: 左肘
    "right_elbow",    # 8: 右肘
    "left_wrist",     # 9: 左腕
    "right_wrist",    # 10: 右腕
    "left_hip",       # 11: 左髋
    "right_hip",      # 12: 右髋
    "left_knee",      # 13: 左膝
    "right_knee",     # 14: 右膝
    "left_ankle",     # 15: 左踝
    "right_ankle",    # 16: 右踝
]


def main():
    print("=" * 60)
    print("🏃 姿态估计基础")
    print("=" * 60)
    
    # 加载姿态估计模型 (以 -pose 结尾)
    model = load_yolo_model("yolo11n-pose.pt")
    
    # ==========================================
    # 1. 姿态估计概念
    # ==========================================
    
    print("\n📝 姿态估计概念:")
    print("""
    姿态估计检测人体的 17 个关键点 (COCO 格式):
    
    头部: 鼻子、左右眼、左右耳
    上肢: 左右肩、左右肘、左右腕
    下肢: 左右髋、左右膝、左右踝
    
    每个关键点包含: (x, y, confidence)
    """)
    
    # ==========================================
    # 2. 使用示例图像进行姿态估计
    # ==========================================
    
    # 从 datasets/images 加载包含人物的图像
    test_image_path = get_sample_image("zidane.jpg")
    
    print("=" * 60)
    print("🔍 执行姿态估计")
    print("=" * 60)
    print(f"\n📷 测试图像: {test_image_path}")
    
    results = model(str(test_image_path), verbose=False)
    result = results[0]
    
    # ==========================================
    # 3. 理解姿态结果
    # ==========================================
    
    print("\n📊 姿态估计结果:")
    
    if result.keypoints is None:
        print("  ⚠️ 未检测到人物姿态")
        return
    
    keypoints = result.keypoints
    kpts_data = keypoints.data.cpu().numpy()
    
    print(f"  检测到 {len(kpts_data)} 个人物")
    print(f"  关键点数据形状: {kpts_data.shape}")
    print(f"  解释: ({kpts_data.shape[0]} 人, {kpts_data.shape[1]} 关键点, 3=[x,y,conf])")
    
    # ==========================================
    # 4. 访问关键点数据
    # ==========================================
    
    print("\n" + "=" * 60)
    print("🦴 关键点详情")
    print("=" * 60)
    
    for person_idx, person_kpts in enumerate(kpts_data):
        print(f"\n👤 人物 #{person_idx}:")
        
        # 统计可见关键点
        visible_count = 0
        high_conf_count = 0
        
        for kpt_idx, (x, y, conf) in enumerate(person_kpts):
            kpt_name = KEYPOINT_NAMES[kpt_idx]
            
            if conf > 0.5:
                visible_count += 1
                if conf > 0.8:
                    high_conf_count += 1
                    
                # 只打印高置信度的关键点
                if kpt_idx < 7:  # 只打印头部和肩膀
                    print(f"    {kpt_name:15s}: ({x:6.1f}, {y:6.1f}) conf={conf:.2f}")
        
        print(f"    ...")
        print(f"    可见关键点: {visible_count}/17")
        print(f"    高置信度 (>80%): {high_conf_count}/17")
    
    # ==========================================
    # 5. 边界框信息
    # ==========================================
    
    print("\n" + "=" * 60)
    print("📦 边界框信息")
    print("=" * 60)
    
    if result.boxes is not None:
        for i, box in enumerate(result.boxes):
            xyxy = box.xyxy[0].cpu().numpy()
            conf = box.conf.item()
            print(f"  人物 #{i}: 位置=[{xyxy[0]:.0f}, {xyxy[1]:.0f}, {xyxy[2]:.0f}, {xyxy[3]:.0f}], 置信度={conf:.2%}")
    
    # ==========================================
    # 6. 保存可视化结果
    # ==========================================
    
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    # 使用 plot() 自动绘制骨架
    annotated = result.plot()
    
    output_path = output_dir / "pose_result.jpg"
    cv2.imwrite(str(output_path), annotated)
    print(f"\n💾 姿态结果已保存: {output_path}")
    
    # ==========================================
    # 7. 关键点索引参考
    # ==========================================
    
    print("\n" + "=" * 60)
    print("📋 关键点索引参考")
    print("=" * 60)
    print("""
       0: 鼻子        1: 左眼       2: 右眼
       3: 左耳        4: 右耳       5: 左肩
       6: 右肩        7: 左肘       8: 右肘
       9: 左腕       10: 右腕      11: 左髋
      12: 右髋       13: 左膝      14: 右膝
      15: 左踝       16: 右踝
    """)
    
    print("✅ 姿态估计基础演示完成!")


if __name__ == "__main__":
    main()
