# YOLO 数据集格式指南 📁

## 1. 目录结构

YOLO 训练需要按照特定格式组织数据集：

```
dataset/
├── train/
│   ├── images/
│   │   ├── img001.jpg
│   │   ├── img002.jpg
│   │   └── ...
│   └── labels/
│       ├── img001.txt
│       ├── img002.txt
│       └── ...
├── val/
│   ├── images/
│   └── labels/
├── test/  (可选)
│   ├── images/
│   └── labels/
└── data.yaml
```

## 2. 标注格式

每个 `.txt` 标注文件包含对应图像的所有目标，每行一个目标：

```
<class_id> <x_center> <y_center> <width> <height>
```

### 参数说明

| 参数 | 说明 | 范围 |
|-----|------|------|
| class_id | 类别索引 (从0开始) | 0, 1, 2, ... |
| x_center | 边界框中心 x 坐标 (归一化) | 0.0 ~ 1.0 |
| y_center | 边界框中心 y 坐标 (归一化) | 0.0 ~ 1.0 |
| width | 边界框宽度 (归一化) | 0.0 ~ 1.0 |
| height | 边界框高度 (归一化) | 0.0 ~ 1.0 |

### 示例

假设图像尺寸为 640x480，目标边界框为 (100, 150) 到 (300, 350)：

```python
# 计算归一化坐标
img_w, img_h = 640, 480
x1, y1, x2, y2 = 100, 150, 300, 350

x_center = (x1 + x2) / 2 / img_w  # = 0.3125
y_center = (y1 + y2) / 2 / img_h  # = 0.5208
width = (x2 - x1) / img_w         # = 0.3125
height = (y2 - y1) / img_h        # = 0.4167
```

标注文件内容：
```
0 0.3125 0.5208 0.3125 0.4167
```

## 3. data.yaml 配置文件

```yaml
# 数据集路径
path: /path/to/dataset  # 数据集根目录
train: train/images     # 训练图像目录 (相对于 path)
val: val/images         # 验证图像目录
test: test/images       # 测试图像目录 (可选)

# 类别数量
nc: 3

# 类别名称
names:
  0: cat
  1: dog
  2: bird
```

## 4. 常用标注工具

### LabelImg (推荐新手)
```bash
pip install labelimg
labelimg
```

### CVAT (在线工具)
- 网址: https://www.cvat.ai/
- 支持团队协作

### Label Studio
```bash
pip install label-studio
label-studio
```

### Roboflow (强大但收费)
- 网址: https://roboflow.com/
- 支持自动标注和数据增强

## 5. 转换脚本示例

### COCO 格式转 YOLO 格式

```python
import json
from pathlib import Path

def coco_to_yolo(coco_json, output_dir, image_dir):
    """将 COCO 格式标注转换为 YOLO 格式"""
    
    with open(coco_json) as f:
        coco = json.load(f)
    
    # 创建类别映射
    categories = {cat['id']: cat['name'] for cat in coco['categories']}
    
    # 图像信息映射
    images = {img['id']: img for img in coco['images']}
    
    # 按图像分组标注
    img_annotations = {}
    for ann in coco['annotations']:
        img_id = ann['image_id']
        if img_id not in img_annotations:
            img_annotations[img_id] = []
        img_annotations[img_id].append(ann)
    
    # 转换并保存
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for img_id, anns in img_annotations.items():
        img_info = images[img_id]
        img_w, img_h = img_info['width'], img_info['height']
        
        # 生成 YOLO 格式标注
        lines = []
        for ann in anns:
            cat_id = ann['category_id']
            x, y, w, h = ann['bbox']  # COCO: [x, y, width, height]
            
            # 转换为 YOLO 格式 (中心点 + 归一化)
            x_center = (x + w / 2) / img_w
            y_center = (y + h / 2) / img_h
            w_norm = w / img_w
            h_norm = h / img_h
            
            lines.append(f"{cat_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
        
        # 保存
        label_file = output_dir / f"{Path(img_info['file_name']).stem}.txt"
        with open(label_file, 'w') as f:
            f.write('\n'.join(lines))
    
    print(f"✅ 转换完成: {len(img_annotations)} 个标注文件")
```

## 6. 数据集验证

```python
from ultralytics.data.utils import check_det_dataset

# 验证数据集配置
data_dict = check_det_dataset("data.yaml")
print(f"训练样本: {len(data_dict['train'])} 张")
print(f"验证样本: {len(data_dict['val'])} 张")
```

## 7. 常见问题

### Q: 标注文件为空怎么办？
A: 如果图像中没有目标，对应的 `.txt` 文件应该是空的或不存在。

### Q: 一张图像有多个目标怎么标注？
A: 每个目标一行，例如：
```
0 0.5 0.5 0.2 0.3
1 0.2 0.3 0.1 0.2
0 0.8 0.7 0.15 0.25
```

### Q: 图像和标注文件名必须匹配吗？
A: 是的，必须只有扩展名不同，例如 `img001.jpg` 对应 `img001.txt`。

---

准备好数据集后，就可以开始训练了！参见 `02_custom_training/` 目录。

