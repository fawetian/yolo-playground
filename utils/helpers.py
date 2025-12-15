"""
通用辅助函数 (macOS 优化版)
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Union, Tuple, Optional
import platform


def get_device():
    """
    获取最佳可用计算设备
    
    Returns:
        str: 'mps' (Apple Silicon), '0' (NVIDIA GPU), 或 'cpu'
    """
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "0"
    except ImportError:
        pass
    return "cpu"


def is_apple_silicon() -> bool:
    """检查是否为 Apple Silicon Mac"""
    return platform.processor() == 'arm'


def load_image(path: Union[str, Path], color_mode: str = "bgr") -> np.ndarray:
    """
    加载图像
    
    Args:
        path: 图像路径
        color_mode: 颜色模式 ('bgr', 'rgb', 'gray')
    
    Returns:
        图像数组
    """
    path = str(path)
    
    if color_mode == "gray":
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if color_mode == "rgb":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if img is None:
        raise FileNotFoundError(f"无法加载图像: {path}")
    
    return img


def save_image(img: np.ndarray, path: Union[str, Path], create_dirs: bool = True) -> bool:
    """
    保存图像
    
    Args:
        img: 图像数组
        path: 保存路径
        create_dirs: 是否自动创建目录
    
    Returns:
        是否保存成功
    """
    path = Path(path)
    
    if create_dirs:
        path.parent.mkdir(parents=True, exist_ok=True)
    
    return cv2.imwrite(str(path), img)


def resize_image(
    img: np.ndarray,
    width: Optional[int] = None,
    height: Optional[int] = None,
    keep_ratio: bool = True
) -> np.ndarray:
    """
    调整图像大小
    
    Args:
        img: 图像数组
        width: 目标宽度
        height: 目标高度
        keep_ratio: 是否保持宽高比
    
    Returns:
        调整后的图像
    """
    h, w = img.shape[:2]
    
    if width is None and height is None:
        return img
    
    if keep_ratio:
        if width is not None and height is not None:
            scale = min(width / w, height / h)
        elif width is not None:
            scale = width / w
        else:
            scale = height / h
        
        new_w = int(w * scale)
        new_h = int(h * scale)
    else:
        new_w = width if width else w
        new_h = height if height else h
    
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def show_image(
    img: np.ndarray,
    title: str = "Image",
    wait_key: int = 0,
    destroy: bool = True,
    width: Optional[int] = None,
    height: Optional[int] = None
) -> int:
    """
    显示图像窗口 (macOS 优化)
    
    Args:
        img: 图像数组
        title: 窗口标题
        wait_key: 等待按键时间 (0 = 无限等待)
        destroy: 是否关闭窗口
        width: 窗口宽度 (可选，用于 Retina 屏幕)
        height: 窗口高度 (可选)
    
    Returns:
        按下的键值
    """
    # macOS: 使用 WINDOW_NORMAL 以便调整窗口大小
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    
    # 如果指定了尺寸，调整窗口大小 (对 Retina 屏幕有用)
    if width and height:
        cv2.resizeWindow(title, width, height)
    
    cv2.imshow(title, img)
    key = cv2.waitKey(wait_key) & 0xFF
    
    if destroy:
        cv2.destroyWindow(title)
        cv2.waitKey(1)  # macOS 需要额外的 waitKey 来完全关闭窗口
    
    return key


def show_images_grid(
    images: list,
    titles: Optional[list] = None,
    cols: int = 3,
    figsize: Tuple[int, int] = (15, 10)
):
    """
    网格显示多张图像（使用 matplotlib）
    
    Args:
        images: 图像列表
        titles: 标题列表
        cols: 每行列数
        figsize: 图像大小
    """
    import matplotlib.pyplot as plt
    
    n = len(images)
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.array(axes).flatten()
    
    for i, img in enumerate(images):
        # BGR to RGB for matplotlib
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axes[i].imshow(img)
        else:
            axes[i].imshow(img, cmap='gray')
        
        if titles and i < len(titles):
            axes[i].set_title(titles[i])
        axes[i].axis('off')
    
    # 隐藏多余的子图
    for i in range(n, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def draw_bbox(
    img: np.ndarray,
    bbox: Tuple[int, int, int, int],
    label: str = "",
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    在图像上绘制边界框
    
    Args:
        img: 图像数组
        bbox: 边界框 (x1, y1, x2, y2)
        label: 标签文本
        color: 颜色 (BGR)
        thickness: 线宽
    
    Returns:
        绘制后的图像
    """
    img = img.copy()
    x1, y1, x2, y2 = [int(v) for v in bbox]
    
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 1
        
        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
        
        cv2.rectangle(img, (x1, y1 - text_h - 10), (x1 + text_w + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 5), font, font_scale, (255, 255, 255), font_thickness)
    
    return img


def get_image_info(img: np.ndarray) -> dict:
    """
    获取图像信息
    
    Args:
        img: 图像数组
    
    Returns:
        图像信息字典
    """
    info = {
        "shape": img.shape,
        "dtype": str(img.dtype),
        "size_bytes": img.nbytes,
    }
    
    if len(img.shape) == 2:
        info["height"], info["width"] = img.shape
        info["channels"] = 1
        info["color_mode"] = "grayscale"
    else:
        info["height"], info["width"], info["channels"] = img.shape
        info["color_mode"] = "color"
    
    info["min_value"] = int(img.min())
    info["max_value"] = int(img.max())
    info["mean_value"] = float(img.mean())
    
    return info


def print_image_info(img: np.ndarray, name: str = "Image"):
    """打印图像信息"""
    info = get_image_info(img)
    print(f"\n{'='*40}")
    print(f"📷 {name}")
    print(f"{'='*40}")
    print(f"  尺寸: {info['width']} x {info['height']}")
    print(f"  通道: {info['channels']}")
    print(f"  类型: {info['dtype']}")
    print(f"  模式: {info['color_mode']}")
    print(f"  内存: {info['size_bytes'] / 1024:.2f} KB")
    print(f"  像素范围: [{info['min_value']}, {info['max_value']}]")
    print(f"  平均值: {info['mean_value']:.2f}")
    print(f"{'='*40}\n")

