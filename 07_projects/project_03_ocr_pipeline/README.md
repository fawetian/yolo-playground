# 项目 03 - OCR 流水线 📝

## 项目概述

构建一个文字检测与识别系统：
- 文字区域检测
- 文字识别 (OCR)
- 结构化输出

## 难度等级

⭐⭐⭐ 进阶级

## 预计时间

5 天

## 技术栈

- OpenCV: 图像预处理
- YOLO/PaddleOCR: 文字检测
- PaddleOCR/EasyOCR: 文字识别

## 功能需求

### 基础功能
- [ ] 检测图像中的文字区域
- [ ] 识别文字内容
- [ ] 输出识别结果

### 进阶功能
- [ ] 表格识别
- [ ] 证件信息提取
- [ ] 批量处理

## 安装额外依赖

```bash
conda activate yolo

# PaddleOCR (推荐)
pip install paddlepaddle paddleocr

# 或 EasyOCR
pip install easyocr
```

## 实现思路

### 方案 A: 使用 PaddleOCR (推荐)
```python
from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=True, lang='ch')
result = ocr.ocr(img_path)

for line in result[0]:
    bbox = line[0]      # 文字框坐标
    text = line[1][0]   # 识别文字
    conf = line[1][1]   # 置信度
```

### 方案 B: YOLO 检测 + EasyOCR 识别
```python
import easyocr
from ultralytics import YOLO

# 1. 检测文字区域
detector = YOLO("yolo-text.pt")  # 文字检测模型
boxes = detector(img)

# 2. 裁剪并识别
reader = easyocr.Reader(['ch_sim', 'en'])
for box in boxes:
    roi = crop(img, box)
    text = reader.readtext(roi)
```

### 图像预处理
```python
def preprocess(img):
    # 灰度
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 二值化
    _, binary = cv2.threshold(gray, 0, 255, 
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 去噪
    denoised = cv2.fastNlMeansDenoising(binary)
    
    return denoised
```

## 目录结构

```
project_03_ocr_pipeline/
├── README.md
├── main.py
├── src/
│   ├── detector.py      # 文字检测
│   ├── recognizer.py    # 文字识别
│   ├── preprocessor.py  # 图像预处理
│   └── postprocessor.py # 结果后处理
├── test_images/         # 测试图片
└── outputs/             # 输出结果
```

## 运行方式

```bash
conda activate yolo
python main.py --image test_images/document.jpg
```

## 扩展想法

- 身份证/银行卡识别
- 发票信息提取
- 手写文字识别

