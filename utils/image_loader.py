"""
图像加载工具
统一管理测试图像的加载
优先从本地 datasets/images/ 目录加载，如果没有则从网络下载
"""

from pathlib import Path
import urllib.request
import os

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 数据集目录
DATASETS_DIR = PROJECT_ROOT / "datasets"
IMAGES_DIR = DATASETS_DIR / "images"
VIDEOS_DIR = DATASETS_DIR / "videos"

# 确保目录存在
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

# 预定义的示例图像 URL
SAMPLE_IMAGES = {
    "bus.jpg": "https://ultralytics.com/images/bus.jpg",
    "zidane.jpg": "https://ultralytics.com/images/zidane.jpg",
}


def get_sample_image(name: str = "bus.jpg", download_if_missing: bool = True) -> Path:
    """
    获取示例图像路径，如果本地不存在则下载
    
    Args:
        name: 图像名称，如 "bus.jpg", "zidane.jpg"
        download_if_missing: 如果本地不存在，是否自动下载
    
    Returns:
        图像的本地路径
    
    Examples:
        >>> img_path = get_sample_image("bus.jpg")
        >>> img_path = get_sample_image("zidane.jpg")
    """
    local_path = IMAGES_DIR / name
    
    # 如果本地存在，直接返回
    if local_path.exists():
        return local_path
    
    # 检查是否有预定义的 URL
    if name in SAMPLE_IMAGES and download_if_missing:
        url = SAMPLE_IMAGES[name]
        print(f"📥 下载示例图像: {name}")
        print(f"   URL: {url}")
        print(f"   保存到: {local_path}")
        
        try:
            urllib.request.urlretrieve(url, local_path)
            print(f"✅ 下载完成: {name}")
            return local_path
        except Exception as e:
            print(f"❌ 下载失败: {e}")
            raise
    
    # 本地不存在且无法下载
    if not local_path.exists():
        raise FileNotFoundError(
            f"图像 {name} 不存在于 {IMAGES_DIR}，"
            f"且没有预定义的下载 URL"
        )
    
    return local_path


def list_sample_images() -> list:
    """
    列出所有可用的示例图像
    
    Returns:
        本地图像文件列表
    """
    extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    images = []
    for ext in extensions:
        images.extend(IMAGES_DIR.glob(f"*{ext}"))
    return sorted(images)


def get_all_sample_images(download_all: bool = True) -> list:
    """
    获取所有示例图像，如果需要则下载
    
    Args:
        download_all: 是否下载所有预定义的示例图像
    
    Returns:
        所有图像路径列表
    """
    if download_all:
        for name in SAMPLE_IMAGES:
            try:
                get_sample_image(name)
            except Exception:
                pass
    
    return list_sample_images()


if __name__ == "__main__":
    print("=" * 60)
    print("🖼️ 图像加载工具测试")
    print("=" * 60)
    
    print(f"\n📁 图像目录: {IMAGES_DIR}")
    
    print("\n📋 本地已有图像:")
    local_images = list_sample_images()
    if local_images:
        for img in local_images:
            size_kb = img.stat().st_size / 1024
            print(f"  ✅ {img.name} ({size_kb:.1f} KB)")
    else:
        print("  (暂无)")
    
    print("\n📥 可下载的示例图像:")
    for name, url in SAMPLE_IMAGES.items():
        local_path = IMAGES_DIR / name
        status = "✅ 已下载" if local_path.exists() else "⬇️ 待下载"
        print(f"  {status} {name}")
    
    print("\n💡 使用示例:")
    print("  from utils.image_loader import get_sample_image")
    print("  img_path = get_sample_image('bus.jpg')")
