"""
项目 1: 人脸检测应用
==================

描述:
使用 YOLO 检测人脸，并进行增强展示（马赛克模糊、添加装饰等）。
虽然有专门的人脸模型，但这里我们使用通用模型的 'person' 类，
配合逻辑判断 (上半身/头部区域) 来模拟，或者尝试加载人脸专用模型。
"""

from pathlib import Path
import cv2
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.model_loader import load_yolo_model
from utils.image_loader import get_sample_image


def main():
    print("=" * 60)
    print("👤 人脸检测应用 (Demo)")
    print("=" * 60)
    
    # 尝试加载人脸模型，如果没有则使用普通 yolo 检测 'person'
    # 注意: yolo11n-face.pt 不是官方内置的标准权重名称，通常需要第三方转换
    # 这里我们演示如何使用 pose 模型更精准地定位头部 (利用鼻子、眼睛关键点)
    print("📦 加载 YOLO Pose 模型用于精准头部定位...")
    model = load_yolo_model("yolo11n-pose.pt")
    
    # 加载测试图 (齐达内图包含人脸)
    img_path = get_sample_image("zidane.jpg")
    frame = cv2.imread(str(img_path))
    
    print(f"\n📷 处理: {img_path.name}")
    
    # 推理
    results = model(frame, verbose=False)
    result = results[0]
    
    # 复制图像用于不同的效果
    mosaic_frame = frame.copy()
    decoration_frame = frame.copy()
    
    # 关键点索引: 0=鼻子, 1=左眼, 2=右眼, 3=左耳, 4=右耳
    if result.keypoints is not None:
        kpts_data = result.keypoints.data.cpu().numpy()
        
        print(f"  检测到 {len(kpts_data)} 个人物")
        
        for i, kpts in enumerate(kpts_data):
            # 获取头部关键点 (0-4)
            head_kpts = kpts[:5]
            
            # 过滤掉置信度低的点
            valid_points = [p for p in head_kpts if p[2] > 0.5]
            
            if len(valid_points) >= 2:
                # 计算头部边界框
                xs = [p[0] for p in valid_points]
                ys = [p[1] for p in valid_points]
                
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                
                # 扩大边界框以覆盖整个头部
                w = x_max - x_min
                h = y_max - y_min
                pad_x = w * 0.5
                pad_y = h * 0.8
                
                x1 = int(max(0, x_min - pad_x))
                y1 = int(max(0, y_min - pad_y))
                x2 = int(min(frame.shape[1], x_max + pad_x))
                y2 = int(min(frame.shape[0], y_max + pad_y * 0.5))
                
                print(f"    人物 {i}: 头部位置 [{x1}, {y1}, {x2}, {y2}]")
                
                # --- 效果 1: 隐私保护 (马赛克) ---
                apply_mosaic(mosaic_frame, x1, y1, x2, y2, block_size=15)
                
                # --- 效果 2: 添加虚拟墨镜 ---
                # 使用眼睛坐标 (idx 1, 2)
                left_eye = kpts[1]
                right_eye = kpts[2]
                if left_eye[2] > 0.5 and right_eye[2] > 0.5:
                    add_sunglasses(decoration_frame, left_eye, right_eye)
                
                # 绘制头部框
                cv2.rectangle(decoration_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(decoration_frame, "Face", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 保存结果
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    cv2.imwrite(str(output_dir / "face_mosaic.jpg"), mosaic_frame)
    cv2.imwrite(str(output_dir / "face_decoration.jpg"), decoration_frame)
    
    print(f"\n✅ 结果已保存:")
    print(f"  隐私保护: {output_dir / 'face_mosaic.jpg'}")
    print(f"  趣味效果: {output_dir / 'face_decoration.jpg'}")


def apply_mosaic(img, x1, y1, x2, y2, block_size=10):
    """区域马赛克效果"""
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0: return

    # 缩小
    small = cv2.resize(img[y1:y2, x1:x2], (w // block_size, h // block_size), interpolation=cv2.INTER_LINEAR)
    # 放大回原尺寸
    mosaic = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    
    img[y1:y2, x1:x2] = mosaic


def add_sunglasses(img, left_eye, right_eye):
    """在两眼之间绘制墨镜"""
    # 计算中心点和角度
    eye_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
    # 简单的黑色矩形模拟墨镜
    width = abs(right_eye[0] - left_eye[0]) * 2.2
    height = width * 0.4
    
    x1 = int(eye_center[0] - width / 2)
    y1 = int(eye_center[1] - height / 2)
    x2 = int(eye_center[0] + width / 2)
    y2 = int(eye_center[1] + height / 2)
    
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), -1)
    
    # 镜框连接
    line_thickness = max(1, int(height * 0.1))
    cv2.line(img, (int(x1), int(eye_center[1])), (int(x2), int(eye_center[1])), (50, 50, 50), line_thickness)


if __name__ == "__main__":
    main()
