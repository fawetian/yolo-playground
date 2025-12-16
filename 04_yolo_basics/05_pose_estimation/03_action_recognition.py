"""
动作识别
=======

学习目标:
- 基于关键点进行简单动作识别
- 计算肢体角度
- 实现常见动作检测
"""

from pathlib import Path
import cv2
import numpy as np
import sys
import math

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.model_loader import load_yolo_model
from utils.image_loader import get_sample_image

# COCO 关键点索引
class KeypointIndex:
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16


def main():
    print("=" * 60)
    print("🎬 动作识别")
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
    # 1. 分析每个人的动作
    # ==========================================
    
    print("\n" + "=" * 60)
    print("🏃 动作分析")
    print("=" * 60)
    
    annotated_img = orig_img.copy()
    
    for person_idx, person_kpts in enumerate(kpts_data):
        print(f"\n👤 人物 #{person_idx}:")
        
        actions = analyze_actions(person_kpts)
        
        for action, detected in actions.items():
            status = "✅" if detected else "❌"
            print(f"    {status} {action}")
        
        # 在图上标注检测到的动作
        detected_actions = [a for a, d in actions.items() if d]
        if detected_actions and len(result.boxes) > person_idx:
            box = result.boxes[person_idx].xyxy[0].cpu().numpy()
            x1, y1 = int(box[0]), int(box[1])
            
            for i, action in enumerate(detected_actions[:3]):  # 最多显示3个
                cv2.putText(annotated_img, action, (x1, y1 - 10 - i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    output_path = output_dir / "action_detection.jpg"
    cv2.imwrite(str(output_path), annotated_img)
    print(f"\n💾 结果已保存: {output_path}")
    
    # ==========================================
    # 2. 角度计算演示
    # ==========================================
    
    print("\n" + "=" * 60)
    print("📐 关节角度计算")
    print("=" * 60)
    
    if len(kpts_data) > 0:
        person_kpts = kpts_data[0]
        
        # 计算各关节角度
        angles = calculate_joint_angles(person_kpts)
        
        for joint, angle in angles.items():
            if angle is not None:
                print(f"    {joint}: {angle:.1f}°")
            else:
                print(f"    {joint}: 无法计算 (关键点不可见)")
        
        # 创建角度可视化
        angle_img = orig_img.copy()
        draw_angles(angle_img, person_kpts, angles)
        
        output_path = output_dir / "joint_angles.jpg"
        cv2.imwrite(str(output_path), angle_img)
        print(f"\n💾 角度可视化已保存: {output_path}")
    
    # ==========================================
    # 3. 姿态对称性分析
    # ==========================================
    
    print("\n" + "=" * 60)
    print("⚖️ 姿态对称性分析")
    print("=" * 60)
    
    if len(kpts_data) > 0:
        person_kpts = kpts_data[0]
        symmetry = analyze_symmetry(person_kpts)
        
        for part, score in symmetry.items():
            if score is not None:
                bar = "█" * int(score * 10)
                print(f"    {part}: {score:.0%} {bar}")
            else:
                print(f"    {part}: 无法计算")
    
    # ==========================================
    # 4. 创建动作识别示例代码参考
    # ==========================================
    
    print("\n" + "=" * 60)
    print("📖 动作识别代码示例")
    print("=" * 60)
    print("""
    # 检测举手动作
    def is_hand_raised(kpts):
        left_wrist = kpts[9]   # 左腕
        left_shoulder = kpts[5]  # 左肩
        right_wrist = kpts[10]
        right_shoulder = kpts[6]
        
        left_raised = (left_wrist[2] > 0.5 and 
                       left_wrist[1] < left_shoulder[1])
        right_raised = (right_wrist[2] > 0.5 and 
                        right_wrist[1] < right_shoulder[1])
        
        return left_raised or right_raised
    
    # 检测蹲姿
    def is_squatting(kpts):
        hip = kpts[11]  # 左髋
        knee = kpts[13]  # 左膝
        ankle = kpts[15]  # 左踝
        
        if all(p[2] > 0.5 for p in [hip, knee, ankle]):
            # 计算膝关节角度
            angle = calculate_angle(hip, knee, ankle)
            return angle < 120  # 角度小于 120 度认为是蹲
        return False
    """)
    
    print("\n✅ 动作识别演示完成!")


def analyze_actions(kpts):
    """分析检测到的动作"""
    KI = KeypointIndex
    actions = {}
    
    # 1. 检测举手 (手腕高于肩膀)
    actions["举左手"] = is_keypoint_above(kpts, KI.LEFT_WRIST, KI.LEFT_SHOULDER)
    actions["举右手"] = is_keypoint_above(kpts, KI.RIGHT_WRIST, KI.RIGHT_SHOULDER)
    
    # 2. 检测双臂展开
    actions["双臂展开"] = are_arms_spread(kpts)
    
    # 3. 检测站立 (髋部高于膝盖)
    actions["站立"] = is_standing(kpts)
    
    # 4. 检测面向前方 (两眼可见且水平)
    actions["面向前方"] = is_facing_forward(kpts)
    
    # 5. 检测转头 (一只眼睛比另一只更可见)
    actions["转头"] = is_head_turned(kpts)
    
    return actions


def is_keypoint_above(kpts, upper_idx, lower_idx, conf_threshold=0.5):
    """检查一个关键点是否在另一个关键点上方"""
    upper = kpts[upper_idx]
    lower = kpts[lower_idx]
    
    if upper[2] > conf_threshold and lower[2] > conf_threshold:
        return upper[1] < lower[1]  # y 坐标更小表示更高
    return False


def are_arms_spread(kpts, conf_threshold=0.5):
    """检测双臂是否展开"""
    KI = KeypointIndex
    
    # 获取关键点
    left_shoulder = kpts[KI.LEFT_SHOULDER]
    right_shoulder = kpts[KI.RIGHT_SHOULDER]
    left_wrist = kpts[KI.LEFT_WRIST]
    right_wrist = kpts[KI.RIGHT_WRIST]
    
    # 检查可见性
    if not all(p[2] > conf_threshold for p in [left_shoulder, right_shoulder, 
                                                 left_wrist, right_wrist]):
        return False
    
    # 计算肩宽
    shoulder_width = abs(right_shoulder[0] - left_shoulder[0])
    
    # 计算手腕间距
    wrist_width = abs(right_wrist[0] - left_wrist[0])
    
    # 如果手腕间距大于肩宽的 1.5 倍，认为双臂展开
    return wrist_width > shoulder_width * 1.5


def is_standing(kpts, conf_threshold=0.5):
    """检测是否站立"""
    KI = KeypointIndex
    
    left_hip = kpts[KI.LEFT_HIP]
    left_knee = kpts[KI.LEFT_KNEE]
    right_hip = kpts[KI.RIGHT_HIP]
    right_knee = kpts[KI.RIGHT_KNEE]
    
    # 检查左侧
    left_standing = False
    if left_hip[2] > conf_threshold and left_knee[2] > conf_threshold:
        left_standing = left_hip[1] < left_knee[1]
    
    # 检查右侧
    right_standing = False
    if right_hip[2] > conf_threshold and right_knee[2] > conf_threshold:
        right_standing = right_hip[1] < right_knee[1]
    
    return left_standing or right_standing


def is_facing_forward(kpts, conf_threshold=0.5):
    """检测是否面向前方"""
    KI = KeypointIndex
    
    nose = kpts[KI.NOSE]
    left_eye = kpts[KI.LEFT_EYE]
    right_eye = kpts[KI.RIGHT_EYE]
    
    if not all(p[2] > conf_threshold for p in [nose, left_eye, right_eye]):
        return False
    
    # 两眼应该大致水平
    eye_height_diff = abs(left_eye[1] - right_eye[1])
    eye_width = abs(left_eye[0] - right_eye[0])
    
    if eye_width > 0:
        return eye_height_diff / eye_width < 0.3
    return False


def is_head_turned(kpts, conf_threshold=0.3):
    """检测是否转头"""
    KI = KeypointIndex
    
    left_eye = kpts[KI.LEFT_EYE]
    right_eye = kpts[KI.RIGHT_EYE]
    
    # 如果一只眼睛置信度明显高于另一只
    if left_eye[2] > conf_threshold or right_eye[2] > conf_threshold:
        conf_diff = abs(left_eye[2] - right_eye[2])
        return conf_diff > 0.3
    return False


def calculate_joint_angles(kpts):
    """计算关节角度"""
    KI = KeypointIndex
    angles = {}
    
    # 左肘角度 (肩-肘-腕)
    angles["左肘"] = calculate_angle_from_points(
        kpts[KI.LEFT_SHOULDER], kpts[KI.LEFT_ELBOW], kpts[KI.LEFT_WRIST]
    )
    
    # 右肘角度
    angles["右肘"] = calculate_angle_from_points(
        kpts[KI.RIGHT_SHOULDER], kpts[KI.RIGHT_ELBOW], kpts[KI.RIGHT_WRIST]
    )
    
    # 左膝角度 (髋-膝-踝)
    angles["左膝"] = calculate_angle_from_points(
        kpts[KI.LEFT_HIP], kpts[KI.LEFT_KNEE], kpts[KI.LEFT_ANKLE]
    )
    
    # 右膝角度
    angles["右膝"] = calculate_angle_from_points(
        kpts[KI.RIGHT_HIP], kpts[KI.RIGHT_KNEE], kpts[KI.RIGHT_ANKLE]
    )
    
    # 左肩角度 (肘-肩-髋)
    angles["左肩"] = calculate_angle_from_points(
        kpts[KI.LEFT_ELBOW], kpts[KI.LEFT_SHOULDER], kpts[KI.LEFT_HIP]
    )
    
    # 右肩角度
    angles["右肩"] = calculate_angle_from_points(
        kpts[KI.RIGHT_ELBOW], kpts[KI.RIGHT_SHOULDER], kpts[KI.RIGHT_HIP]
    )
    
    return angles


def calculate_angle_from_points(p1, p2, p3, conf_threshold=0.5):
    """计算三个点形成的角度 (p2 为顶点)"""
    if not all(p[2] > conf_threshold for p in [p1, p2, p3]):
        return None
    
    # 向量
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    
    # 计算角度
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    cos_angle = np.clip(cos_angle, -1, 1)
    angle = np.arccos(cos_angle)
    
    return np.degrees(angle)


def analyze_symmetry(kpts):
    """分析身体对称性"""
    KI = KeypointIndex
    symmetry = {}
    
    # 肩膀对称性
    symmetry["肩膀"] = calculate_symmetry_score(
        kpts[KI.LEFT_SHOULDER], kpts[KI.RIGHT_SHOULDER], kpts[KI.NOSE]
    )
    
    # 髋部对称性
    symmetry["髋部"] = calculate_symmetry_score(
        kpts[KI.LEFT_HIP], kpts[KI.RIGHT_HIP], kpts[KI.NOSE]
    )
    
    return symmetry


def calculate_symmetry_score(left, right, center, conf_threshold=0.5):
    """计算左右对称性分数"""
    if not all(p[2] > conf_threshold for p in [left, right, center]):
        return None
    
    # 计算左右到中心的距离
    left_dist = abs(left[0] - center[0])
    right_dist = abs(right[0] - center[0])
    
    # 对称性分数 (越接近 1 越对称)
    if max(left_dist, right_dist) > 0:
        return min(left_dist, right_dist) / max(left_dist, right_dist)
    return 1.0


def draw_angles(img, kpts, angles):
    """在图像上绘制角度"""
    KI = KeypointIndex
    
    # 关节位置映射
    joint_positions = {
        "左肘": KI.LEFT_ELBOW,
        "右肘": KI.RIGHT_ELBOW,
        "左膝": KI.LEFT_KNEE,
        "右膝": KI.RIGHT_KNEE,
        "左肩": KI.LEFT_SHOULDER,
        "右肩": KI.RIGHT_SHOULDER,
    }
    
    for joint, angle in angles.items():
        if angle is not None and joint in joint_positions:
            idx = joint_positions[joint]
            pt = kpts[idx]
            
            if pt[2] > 0.5:
                x, y = int(pt[0]), int(pt[1])
                
                # 绘制角度文字
                text = f"{angle:.0f}"
                cv2.putText(img, text, (x + 10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(img, text, (x + 10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)


if __name__ == "__main__":
    main()
