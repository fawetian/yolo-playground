"""
项目 4: 异常检测 (基于规则)
=========================

描述:
简单的基于规则的异常检测。
场景: 
1. 检测这一区域是否出现了不该出现的人 (闯入检测)。
2. 检测某人是否未佩戴特定装备 (这里用"是否携带背包"模拟，假设 backpack 为安全装备)。
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
    print("🚨 异常检测系统 (规则演示)")
    print("=" * 60)
    
    model = load_yolo_model("yolo11n.pt")
    
    # 模拟输入
    img_path = get_sample_image("bus.jpg")
    frame = cv2.imread(str(img_path))
    h, w = frame.shape[:2]
    
    print(f"\n📷 场景: 公交车站 ({w}x{h})")
    
    # ==========================
    # 规则 1: 禁区检测
    # ==========================
    # 定义左侧 20% 区域为"禁止行人区"
    restricted_x = int(w * 0.2)
    
    # 绘制禁区 (半透明红色)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (restricted_x, h), (0, 0, 255), -1)
    frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
    cv2.putText(frame, "RESTRICTED AREA", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # ==========================
    # 推理
    # ==========================
    results = model(frame, verbose=False)
    result = results[0]
    
    alerts = []
    
    for box in result.boxes:
        cls_id = int(box.cls[0].item())
        name = result.names[cls_id]
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        
        # 1. 检查禁区闯入 (只针对 Person)
        if name == "person":
            # 计算人中心点
            cx = (x1 + x2) // 2
            
            if cx < restricted_x:
                alert_msg = f"🚨 警报: 人员闯入禁区! (ID: {cls_id})"
                alerts.append(alert_msg)
                
                # 画红框
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(frame, "INTRUDER", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                # 正常区域，画绿框
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 显示警报
    print("\n📝 检测报告:")
    if alerts:
        for alert in alerts:
            print(f"  {alert}")
    else:
        print("  ✅ 区域安全，无违规")
        
    # 保存结果
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    out_path = output_dir / "anomaly_result.jpg"
    cv2.imwrite(str(out_path), frame)
    print(f"\n💾 结果图: {out_path}")


if __name__ == "__main__":
    main()
