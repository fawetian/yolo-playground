"""
项目 3: OCR 文字识别流程
=====================

描述:
YOLO (检测文本区域 / 或其它目标) -> OpenCV (预处理) -> Tesseract (OCR 识别)。
注意: 需要系统安装 tesseract。如果没有安装，脚本会优雅降级提示。
"""

from pathlib import Path
import cv2
import sys
import shutil

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.model_loader import load_yolo_model
from utils.image_loader import get_sample_image

# 尝试导入 pytesseract
try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False
    print("⚠️ 未安装 pytesseract 库 (pip install pytesseract)")


def main():
    print("=" * 60)
    print("📝 OCR 识别流程")
    print("=" * 60)
    
    # 检查系统 Tesseract
    if HAS_TESSERACT:
        if shutil.which("tesseract") is None:
            print("❌ 系统未找到 'tesseract' 可执行文件")
            print("  macOS 安装: brew install tesseract")
            tesseract_available = False
        else:
            tesseract_available = True
    else:
        tesseract_available = False

    # 1. 场景: 识别公交车上的文字 (模拟车牌/广告牌识别)
    img_path = get_sample_image("bus.jpg")
    frame = cv2.imread(str(img_path))
    print(f"\n📷 输入图像: {img_path.name}")
    
    # 2. YOLO 检测目标 (比如检测公交车 'bus')
    model = load_yolo_model("yolo11n.pt")
    results = model(frame, verbose=False)
    
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    # 3. 提取目标并进行 OCR
    count = 0
    for box in results[0].boxes:
        cls_id = int(box.cls[0].item())
        name = results[0].names[cls_id]
        
        if name == "bus":
            count += 1
            # 提取 ROI
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            roi = frame[y1:y2, x1:x2]
            
            print(f"\n🚌 检测到公交车 #{count}，正在尝试 OCR...")
            
            # 预处理: 转灰度 -> 阈值化 -> 降噪
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            # Otsu 阈值
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 保存预处理图
            cv2.imwrite(str(output_dir / f"roi_bus_{count}_binary.jpg"), binary)
            
            if tesseract_available:
                try:
                    # OCR 识别
                    # --psm 6 表示假设单一文本块，普通英文
                    text = pytesseract.image_to_string(binary, config='--psm 6')
                    stripped_text = text.strip()
                    
                    if stripped_text:
                        print(f"  📄 识别结果: \"{stripped_text}\"")
                    else:
                        print("  (OCR 未识别出清晰文字)")
                except Exception as e:
                    print(f"  OCR 出错: {e}")
            else:
                print("  ⏭️  跳过 OCR (未安装 tesseract)")
                print("  已保存 ROI 图像供查看")
    
    print("\n✅ OCR 流程演示完成")
    if not tesseract_available:
        print("💡 提示: 安装 Tesseract 以开启实际文字识别功能")


if __name__ == "__main__":
    main()
