"""
骨架绘制
=======

学习目标:
- 理解人体骨架连接关系
- 手动绘制骨架
- 自定义骨架样式
"""

from pathlib import Path
import cv2
import numpy as np
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.model_loader import load_yolo_model
from utils.image_loader import get_sample_image

# COCO 关键点名称
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# 骨架连接定义 (关键点索引对)
SKELETON_CONNECTIONS = [
    # 头部
    (0, 1), (0, 2),      # 鼻子 - 眼睛
    (1, 3), (2, 4),      # 眼睛 - 耳朵
    
    # 躯干
    (5, 6),              # 左肩 - 右肩
    (5, 11), (6, 12),    # 肩膀 - 髋部
    (11, 12),            # 左髋 - 右髋
    
    # 左臂
    (5, 7), (7, 9),      # 肩 - 肘 - 腕
    
    # 右臂
    (6, 8), (8, 10),     # 肩 - 肘 - 腕
    
    # 左腿
    (11, 13), (13, 15),  # 髋 - 膝 - 踝
    
    # 右腿
    (12, 14), (14, 16),  # 髋 - 膝 - 踝
]

# 按身体部位分组的颜色
SKELETON_COLORS = {
    "head": (255, 200, 100),     # 浅蓝色 - 头部
    "torso": (100, 255, 100),    # 绿色 - 躯干
    "left_arm": (255, 100, 100), # 蓝色 - 左臂
    "right_arm": (100, 100, 255),# 红色 - 右臂
    "left_leg": (255, 255, 100), # 青色 - 左腿
    "right_leg": (100, 255, 255),# 黄色 - 右腿
}

# 每个连接对应的身体部位
CONNECTION_PARTS = [
    "head", "head", "head", "head",  # 头部连接
    "torso", "torso", "torso", "torso",  # 躯干连接
    "left_arm", "left_arm",  # 左臂
    "right_arm", "right_arm",  # 右臂
    "left_leg", "left_leg",  # 左腿
    "right_leg", "right_leg",  # 右腿
]


def main():
    print("=" * 60)
    print("🦴 骨架绘制")
    print("=" * 60)
    
    # 加载姿态估计模型
    model = load_yolo_model("yolo11n-pose.pt")
    
    # 从 datasets/images 加载测试图像
    test_image_path = get_sample_image("zidane.jpg")
    
    print(f"\n📷 测试图像: {test_image_path}")
    print("🔍 执行姿态估计...")
    
    results = model(str(test_image_path), verbose=False)
    result = results[0]
    
    if result.keypoints is None:
        print("⚠️ 未检测到人物姿态")
        return
    
    orig_img = result.orig_img.copy()
    kpts_data = result.keypoints.data.cpu().numpy()
    
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n检测到 {len(kpts_data)} 个人物")
    
    # ==========================================
    # 1. 基础骨架绘制
    # ==========================================
    
    print("\n" + "=" * 60)
    print("🎨 基础骨架绘制")
    print("=" * 60)
    
    basic_skeleton = orig_img.copy()
    
    for person_kpts in kpts_data:
        draw_basic_skeleton(basic_skeleton, person_kpts)
    
    output_path = output_dir / "skeleton_basic.jpg"
    cv2.imwrite(str(output_path), basic_skeleton)
    print(f"  已保存: {output_path}")
    
    # ==========================================
    # 2. 彩色骨架 (按身体部位着色)
    # ==========================================
    
    print("\n" + "=" * 60)
    print("🌈 彩色骨架 (按身体部位)")
    print("=" * 60)
    
    colored_skeleton = orig_img.copy()
    
    for person_kpts in kpts_data:
        draw_colored_skeleton(colored_skeleton, person_kpts)
    
    output_path = output_dir / "skeleton_colored.jpg"
    cv2.imwrite(str(output_path), colored_skeleton)
    print(f"  已保存: {output_path}")
    
    # ==========================================
    # 3. 仅关键点 (无连线)
    # ==========================================
    
    print("\n" + "=" * 60)
    print("⚫ 仅关键点")
    print("=" * 60)
    
    keypoints_only = orig_img.copy()
    
    for person_kpts in kpts_data:
        draw_keypoints_only(keypoints_only, person_kpts)
    
    output_path = output_dir / "keypoints_only.jpg"
    cv2.imwrite(str(output_path), keypoints_only)
    print(f"  已保存: {output_path}")
    
    # ==========================================
    # 4. 带标签的关键点
    # ==========================================
    
    print("\n" + "=" * 60)
    print("🏷️ 带标签的关键点")
    print("=" * 60)
    
    labeled_keypoints = orig_img.copy()
    
    # 只标注第一个人
    if len(kpts_data) > 0:
        draw_labeled_keypoints(labeled_keypoints, kpts_data[0])
    
    output_path = output_dir / "keypoints_labeled.jpg"
    cv2.imwrite(str(output_path), labeled_keypoints)
    print(f"  已保存: {output_path}")
    
    # ==========================================
    # 5. 置信度热图
    # ==========================================
    
    print("\n" + "=" * 60)
    print("🔥 置信度热图")
    print("=" * 60)
    
    confidence_img = orig_img.copy()
    
    for person_kpts in kpts_data:
        draw_confidence_keypoints(confidence_img, person_kpts)
    
    output_path = output_dir / "confidence_heatmap.jpg"
    cv2.imwrite(str(output_path), confidence_img)
    print(f"  已保存: {output_path}")
    
    # ==========================================
    # 6. 粗线条风格
    # ==========================================
    
    print("\n" + "=" * 60)
    print("🖌️ 粗线条风格")
    print("=" * 60)
    
    thick_skeleton = orig_img.copy()
    
    for person_kpts in kpts_data:
        draw_thick_skeleton(thick_skeleton, person_kpts)
    
    output_path = output_dir / "skeleton_thick.jpg"
    cv2.imwrite(str(output_path), thick_skeleton)
    print(f"  已保存: {output_path}")
    
    print("\n✅ 骨架绘制演示完成!")
    print(f"📁 所有结果保存在: {output_dir}")


def draw_basic_skeleton(img, kpts, color=(0, 255, 0), thickness=2, radius=4):
    """绘制基础骨架"""
    # 绘制连接线
    for (start_idx, end_idx) in SKELETON_CONNECTIONS:
        start_pt = kpts[start_idx]
        end_pt = kpts[end_idx]
        
        # 只绘制置信度高的连接
        if start_pt[2] > 0.5 and end_pt[2] > 0.5:
            pt1 = (int(start_pt[0]), int(start_pt[1]))
            pt2 = (int(end_pt[0]), int(end_pt[1]))
            cv2.line(img, pt1, pt2, color, thickness)
    
    # 绘制关键点
    for kpt in kpts:
        if kpt[2] > 0.5:
            pt = (int(kpt[0]), int(kpt[1]))
            cv2.circle(img, pt, radius, color, -1)


def draw_colored_skeleton(img, kpts, thickness=2, radius=5):
    """按身体部位着色的骨架"""
    # 绘制连接线
    for i, (start_idx, end_idx) in enumerate(SKELETON_CONNECTIONS):
        start_pt = kpts[start_idx]
        end_pt = kpts[end_idx]
        
        if start_pt[2] > 0.5 and end_pt[2] > 0.5:
            pt1 = (int(start_pt[0]), int(start_pt[1]))
            pt2 = (int(end_pt[0]), int(end_pt[1]))
            
            part = CONNECTION_PARTS[i]
            color = SKELETON_COLORS[part]
            cv2.line(img, pt1, pt2, color, thickness)
    
    # 绘制关键点
    for i, kpt in enumerate(kpts):
        if kpt[2] > 0.5:
            pt = (int(kpt[0]), int(kpt[1]))
            # 根据关键点位置选择颜色
            if i <= 4:
                color = SKELETON_COLORS["head"]
            elif i in [5, 7, 9]:
                color = SKELETON_COLORS["left_arm"]
            elif i in [6, 8, 10]:
                color = SKELETON_COLORS["right_arm"]
            elif i in [11, 13, 15]:
                color = SKELETON_COLORS["left_leg"]
            elif i in [12, 14, 16]:
                color = SKELETON_COLORS["right_leg"]
            else:
                color = SKELETON_COLORS["torso"]
            
            cv2.circle(img, pt, radius, color, -1)
            cv2.circle(img, pt, radius, (255, 255, 255), 1)


def draw_keypoints_only(img, kpts, radius=6):
    """仅绘制关键点"""
    colors = [
        (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
        (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85),
        (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
        (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255),
        (255, 0, 170)
    ]
    
    for i, kpt in enumerate(kpts):
        if kpt[2] > 0.5:
            pt = (int(kpt[0]), int(kpt[1]))
            cv2.circle(img, pt, radius, colors[i], -1)
            cv2.circle(img, pt, radius + 2, (255, 255, 255), 2)


def draw_labeled_keypoints(img, kpts, radius=4):
    """绘制带标签的关键点"""
    for i, kpt in enumerate(kpts):
        if kpt[2] > 0.5:
            pt = (int(kpt[0]), int(kpt[1]))
            
            # 绘制关键点
            cv2.circle(img, pt, radius, (0, 255, 0), -1)
            
            # 添加标签
            label = f"{i}"
            cv2.putText(img, label, (pt[0] + 5, pt[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
            cv2.putText(img, label, (pt[0] + 5, pt[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)


def draw_confidence_keypoints(img, kpts, max_radius=12):
    """根据置信度绘制不同大小的关键点"""
    for kpt in kpts:
        if kpt[2] > 0.3:
            pt = (int(kpt[0]), int(kpt[1]))
            conf = kpt[2]
            
            # 根据置信度计算半径和颜色
            radius = int(max_radius * conf)
            
            # 置信度低 -> 红色, 置信度高 -> 绿色
            red = int(255 * (1 - conf))
            green = int(255 * conf)
            color = (0, green, red)
            
            cv2.circle(img, pt, radius, color, -1)
            cv2.circle(img, pt, radius, (255, 255, 255), 1)


def draw_thick_skeleton(img, kpts, thickness=8, radius=10):
    """粗线条风格骨架"""
    # 先画阴影
    shadow_offset = 3
    for (start_idx, end_idx) in SKELETON_CONNECTIONS:
        start_pt = kpts[start_idx]
        end_pt = kpts[end_idx]
        
        if start_pt[2] > 0.5 and end_pt[2] > 0.5:
            pt1 = (int(start_pt[0]) + shadow_offset, int(start_pt[1]) + shadow_offset)
            pt2 = (int(end_pt[0]) + shadow_offset, int(end_pt[1]) + shadow_offset)
            cv2.line(img, pt1, pt2, (50, 50, 50), thickness + 2)
    
    # 再画骨架
    for (start_idx, end_idx) in SKELETON_CONNECTIONS:
        start_pt = kpts[start_idx]
        end_pt = kpts[end_idx]
        
        if start_pt[2] > 0.5 and end_pt[2] > 0.5:
            pt1 = (int(start_pt[0]), int(start_pt[1]))
            pt2 = (int(end_pt[0]), int(end_pt[1]))
            cv2.line(img, pt1, pt2, (0, 255, 255), thickness)
    
    # 绘制关键点
    for kpt in kpts:
        if kpt[2] > 0.5:
            pt = (int(kpt[0]), int(kpt[1]))
            cv2.circle(img, pt, radius + 2, (50, 50, 50), -1)
            cv2.circle(img, pt, radius, (0, 255, 255), -1)


if __name__ == "__main__":
    main()
