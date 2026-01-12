import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import os
import time
import logging
from datetime import datetime
import tifffile # 用于读取TIFF文件

from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union

# --- 配置日志 ---
def setup_logging(output_dir):
    """设置日志记录"""
    log_file = os.path.join(output_dir, f"combined_processing_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# --- 配置类 ---
class CombinedConfig:
    def __init__(self):
        # === 文件路径 (用于加载原始TIFF) ===
        self.HE_local_path = '/media/raid/tanzhiyao/qilu_hospital/202105341-2.tiff'  # 修改为.tiff
        self.Fluoro_Nuclei_local_path = '/media/raid/med/qilu_hospital/20250922-李连杰老师-正式实验结果-20251105/202105341-Merge.tiff' # 修改为.tiff

        # === TIFF通道选择 ===
        self.fluoro_channel_index = 0 # 选择荧光图像的哪个通道 (0-based index)

        # === 物理像素大小 (微米/像素) - 保留但不用于缩放 ===
        self.HE_pixel_size_microns = 0.325 # H&E图像每个像素代表的物理尺寸 (微米) - 示例值
        self.Fluoro_pixel_size_microns = 0.345 # 荧光图像每个像素代表的物理尺寸 (微米) - 示例值

        # === 旋转角度 ===
        self.rotation_angle = 120 # 旋转角度：是否旋转荧光染色，正值为逆时针旋转

        # === 输出主目录 ===
        self.main_output_dir = '/media/raid/tanzhiyao/qilu_hospital_align/202105341-2'

        # === 子目录和中间文件路径 ===
        self.patch_processing_output_dir = os.path.join(self.main_output_dir, 'patch_processing_output')
        self.registration_output_dir = os.path.join(self.main_output_dir, 'registration_output')
        # 配准脚本将从这里加载特征图像
        self.img1_path_for_registration = os.path.join(self.patch_processing_output_dir, "debug_he_white_raw_small.png")
        self.img2_path_for_registration = os.path.join(self.patch_processing_output_dir, "debug_fluoro_black_raw_small.png")

        # === 参数：胆管特征提取参数 (来自脚本2) ===
        self.he_white_threshold = 240  # H&E图像中白色区域阈值 (0-255)
        # 荧光图像阈值类型变为 uint16
        self.fluoro_black_threshold = 50 # 荧光图像中黑色区域阈值 (0-65535 for uint16)
        # 形态学操作核大小 (虽然不用于mask，但保留参数结构)
        self.morphology_kernel_size = 21  # 从5增大到21
        # 最小轮廓面积 (虽然不用于mask，但保留参数结构)
        self.min_contour_area = 10000  # 从100增大到10000
        # 可视化参数 - 大图像可能不适合直接显示，建议保存
        self.save_debug_images = True
        # 对于大图像，可能需要先缩小再可视化
        self.visualization_downsample_factor = 0.1 # 可视化时缩小到原图的10%

        # === 图像尺寸处理策略 (用于配准) (来自脚本1) ===
        # 注意：这里的尺寸调整将应用于 patch_processing 输出的黑白图像
        self.resize_strategy = 'resize_to_smaller' # 'resize_to_smaller', 'resize_to_larger', 'no_resize'
        self.interpolation_method = cv2.INTER_LINEAR
        # ECC配准参数
        self.motion_types = [
            cv2.MOTION_TRANSLATION,
            cv2.MOTION_EUCLIDEAN,
            cv2.MOTION_AFFINE,
            # cv2.MOTION_HOMOGRAPHY # Homography需要更多特征点，对于简单图像可能不稳定
        ]
        self.motion_names = ['TRANSLATION', 'EUCLIDEAN', 'AFFINE', 'HOMOGRAPHY']
        # 停止条件
        self.max_iterations = 5000
        self.epsilon = 1e-10 # 更严格的收敛条件
        # 可视化参数
        self.registration_save_debug_images = True
        self.display_plots = False

config = CombinedConfig()

# --- 脚本2中的函数 (patch_processing) ---

# def roration(img, angle):
#     """旋转图像"""
#     rows, cols = img.shape[:2]
#     M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
#     return cv2.warpAffine(img, M, (cols, rows))

def rotation(img, angle, background='min'):
    """
    智能旋转：保留完整图像，背景填充为白色或原图最大值
    只处理单通道图像 (H, W)
    
    Args:
        img: 输入图像，单通道 (H, W)
        angle: 旋转角度（度），正值为逆时针
        background: 背景填充方式
            - 'white': 使用白色填充（根据数据类型自动选择最大值）
            - 'max': 使用原图中的最大值填充
            - 'mean': 使用原图的平均值填充
            - 'min': 使用原图的最小值填充
    
    Returns:
        旋转后的图像，保持原始数据类型
    """
    # 确保输入是单通道图像
    if len(img.shape) != 2:
        raise ValueError(f"只支持单通道图像 (H, W)，但输入的形状是 {img.shape}")
    
    h, w = img.shape
    
    # 计算旋转矩阵和新尺寸（确保完整保留图像）
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # 计算新图像的尺寸（确保完整包含旋转后的图像）
    cos_val = np.abs(M[0, 0])
    sin_val = np.abs(M[0, 1])
    new_w = int((h * sin_val) + (w * cos_val))
    new_h = int((h * cos_val) + (w * sin_val))
    
    # 调整旋转矩阵的平移部分，使图像居中
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
    
    # 第一步：旋转图像（使用黑色作为临时背景）
    rotated = cv2.warpAffine(img, M, (new_w, new_h), borderValue=0)
    
    # 第二步：创建掩码，标记哪些像素来自原始图像
    # 创建一个全白的掩码，旋转它，这样就能知道哪些区域是原始图像
    source_mask = np.ones((h, w), dtype=np.uint8) * 255
    rotated_mask = cv2.warpAffine(source_mask, M, (new_w, new_h), borderValue=0)
    
    # 第三步：确定背景填充值
    if background == 'white':
        # 根据数据类型自动选择白色值
        if img.dtype == np.uint8:
            bg_value = 255
        elif img.dtype == np.uint16:
            bg_value = 65535
        elif img.dtype == np.uint32:
            bg_value = 4294967295
        elif np.issubdtype(img.dtype, np.integer):
            bg_value = np.iinfo(img.dtype).max
        elif np.issubdtype(img.dtype, np.floating):
            bg_value = 1.0
        else:
            bg_value = np.iinfo(np.uint16).max  # 默认使用uint16的最大值
            
    elif background == 'max':
        # 使用原图中的最大值
        bg_value = np.max(img)
        
    elif background == 'mean':
        # 使用原图的平均值
        bg_value = np.mean(img)
        
    elif background == 'min':
        # 使用原图的最小值
        bg_value = np.min(img)
        
    else:
        raise ValueError(f"未知的背景选项: {background}。请使用 'white', 'max', 'mean' 或 'min'")
    
    # 第四步：创建背景并填充
    result = np.full((new_h, new_w), bg_value, dtype=img.dtype)
    
    # 将旋转后的图像复制到结果中（只复制原始图像区域）
    result[rotated_mask > 0] = rotated[rotated_mask > 0]
    
    return result

def visualize_multiple(
    images: List[np.ndarray],
    output_path: Union[str, Path],
    titles: List[str] = None,
    figsize: Tuple[int, int] = (20, 10),) -> bool:
    """
    Visualize and save multiple images in a grid.

    Args:
        images: List of images.
        output_path: Path where the figure will be saved.
        titles: Optional list of titles for each subplot.
        figsize: Figure size (width, height) in inches.

    Returns:
        True if saved successfully, False otherwise.
    """
    try:
        n_images = len(images)
        cols = min(4, n_images)
        rows = (n_images + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if rows == 1:
            axes = [axes] if cols == 1 else axes
        else:
            axes = axes.flatten()

        for i, image in enumerate(images):
            if i >= len(axes):
                break
            if len(image.shape) == 3:
                axes[i].imshow(image)
            else:
                axes[i].imshow(image, cmap="gray")

            if titles and i < len(titles):
                axes[i].set_title(titles[i])
            axes[i].axis("off")

        # Hide unused subplots
        for j in range(n_images, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        plt.savefig(output_path, dpi=60, bbox_inches="tight")
        plt.close()
        return True
    except Exception as e:
        logger.error(f"Failed to save multi-image visualization to {output_path}: {e}")
        plt.close()
        return False

def demo_rotation():
    """
    演示旋转函数
    """
    if_path = '/media/raid/med/qilu_hospital/20250922-李连杰老师-正式实验结果-20251105/202105341-Merge.tiff'
    img_fluoro_tiff = tifffile.imread(if_path)
    img_fluoro_gray = img_fluoro_tiff[0, :, :]
    h_gray, w_gray = img_fluoro_gray.shape


    rotation_angle = 120
    img_fluoro_rotated = rotation(img_fluoro_gray, rotation_angle)
    h_rotated, w_rotated = img_fluoro_rotated.shape

    img_gray_small = cv2.resize(img_fluoro_gray, (int(h_gray*0.1), int(w_gray*0.1)))
    img_rotated_small = cv2.resize(img_fluoro_rotated, (int(h_rotated*0.1), int(w_rotated*0.1)))

    visualize_multiple([img_gray_small, img_rotated_small], 'demo_rotation.png', titles=['original', 'rotated'])


def load_and_preprocess_tiff(config, logger=None):
    """加载TIFF图像，处理通道和物理尺寸"""
    if logger:
        logger.info("开始加载和预处理TIFF图像...")
    # 1. 加载 H&E TIFF - 假设总是 (H, W, 3)
    if logger:
        logger.info(f"加载 H&E TIFF 图像: {config.HE_local_path}")
    try:
        img_he_tiff = tifffile.imread(config.HE_local_path)
        if logger:
            logger.info(f"H&E TIFF 原始形状: {img_he_tiff.shape}")
        # 强制假设 H&E 图像形状为 (H, W, 3)
        if len(img_he_tiff.shape) == 3 and img_he_tiff.shape[2] == 3:
            img_he_rgb = img_he_tiff
        else:
            logger.error(f"H&E 图像形状 {img_he_tiff.shape} 不符合预期 (H, W, 3)")
            return None, None, None, None
        # 确保数据类型为 uint8
        if img_he_rgb.dtype != np.uint8:
            logger.info(f"H&E 图像数据类型为 {img_he_rgb.dtype}, 将转换为 uint8.")
            img_he_rgb = np.clip(img_he_rgb, 0, 255).astype(np.uint8)
    except Exception as e:
        logger.error(f"加载 H&E TIFF 图像失败: {e}")
        return None, None, None, None
    # 2. 加载 荧光 TIFF - 假设总是 (7, H, W)
    if logger:
        logger.info(f"加载 荧光 TIFF 图像: {config.Fluoro_Nuclei_local_path}")
    try:
        img_fluoro_tiff = tifffile.imread(config.Fluoro_Nuclei_local_path)
        if logger:
            logger.info(f"荧光 TIFF 原始形状: {img_fluoro_tiff.shape}")
        # 强制假设 荧光图像形状为 (7, H, W)
        if len(img_fluoro_tiff.shape) == 3 and img_fluoro_tiff.shape[0] == 7:
            # 提取指定通道 (0, H, W) -> (H, W)
            img_fluoro_gray = img_fluoro_tiff[config.fluoro_channel_index, :, :]
        else:
            logger.error(f"荧光图像形状 {img_fluoro_tiff.shape} 不符合预期 (7, H, W)")
            return None, None, None, None
        # 确保数据类型为 uint16
        if img_fluoro_gray.dtype != np.uint16:
            logger.info(f"荧光图像数据类型为 {img_fluoro_gray.dtype}, 将转换为 uint16.")
            img_fluoro_gray = img_fluoro_gray.astype(np.uint16)

        # 旋转荧光图像
        if config.rotation_angle != 0:
            logger.info(f"旋转荧光图像 {config.rotation_angle} 度")
            img_fluoro_gray = rotation(img_fluoro_gray, config.rotation_angle)
            logger.info(f"旋转后荧光图像形状: {img_fluoro_gray.shape}")
        else:
            logger.info(f"不旋转荧光图像")
    except Exception as e:
        logger.error(f"加载 荧光 TIFF 图像失败: {e}")
        return None, None, None, None
    if logger:
        logger.info("TIFF图像加载和预处理完成。")
    return img_he_rgb, img_fluoro_gray, config.HE_pixel_size_microns, config.Fluoro_pixel_size_microns

def match_images_by_pixel_size(img_he_rgb, img_fluoro_gray, he_pixel_size, fluoro_pixel_size, target_pixel_size, logger=None):
    """
    调整两张图像到相同像素尺寸。
    本次修改：不进行缩放，直接以 H&E 图像为目标尺寸进行裁剪或填充。
    """
    if logger:
        logger.info("开始匹配图像尺寸 (仅填充，不缩放)...")
        logger.info(f"  H&E 像素大小: {he_pixel_size} 微米/像素")
        logger.info(f"  荧光 像素大小: {fluoro_pixel_size} 微米/像素")
        logger.info(f"  目标像素大小: {target_pixel_size} 微米/像素 (此参数本次不用于缩放)")
    he_h, he_w = img_he_rgb.shape[:2]
    fluoro_h, fluoro_w = img_fluoro_gray.shape[:2]
    # 1. 确定目标尺寸 (以 H&E 图像尺寸为准)
    target_h, target_w = he_h, he_w
    if logger:
        logger.info(f"  目标匹配尺寸: {target_h} x {target_w}")
    # 2. 裁剪或填充 H&E 图像到目标尺寸
    if he_h > target_h or he_w > target_w:
        img_he_final = img_he_rgb[:target_h, :target_w]
    elif he_h < target_h or he_w < target_w:
        pad_h = max(0, target_h - he_h)
        pad_w = max(0, target_w - he_w)
        # 假设填充黑色/0
        if len(img_he_rgb.shape) == 3:
            img_he_final = np.pad(img_he_rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)
        else:
            img_he_final = np.pad(img_he_rgb, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
        # 确保精确尺寸
        img_he_final = img_he_final[:target_h, :target_w]
    else:
        img_he_final = img_he_rgb
    # 3. 裁剪或填充 荧光图像到目标尺寸
    if fluoro_h > target_h or fluoro_w > target_w:
        img_fluoro_final = img_fluoro_gray[:target_h, :target_w]
    elif fluoro_h < target_h or fluoro_w < target_w:
        pad_h = max(0, target_h - fluoro_h)
        pad_w = max(0, target_w - fluoro_w)
        img_fluoro_final = np.pad(img_fluoro_gray, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
        # 确保精确尺寸
        img_fluoro_final = img_fluoro_final[:target_h, :target_w]
    else:
        img_fluoro_final = img_fluoro_gray
    if logger:
        logger.info(f"  最终 H&E 尺寸: {img_he_final.shape}")
        logger.info(f"  最终 荧光 尺寸: {img_fluoro_final.shape}")
        logger.info("图像尺寸匹配完成 (仅填充)。")
    # 由于不缩放，transform_info 信息简化
    transform_info = {
        'method': 'pad_to_match_he_size',
        'target_pixel_size': target_pixel_size, # 保留供参考
        'he_scale_factor': 1.0, # 无缩放
        'fluoro_scale_factor': 1.0, # 无缩放
        'final_size': (target_h, target_w)
    }
    return img_he_final, img_fluoro_final, transform_info

def extract_bile_duct_features_he(img_rgb, white_threshold=200, kernel_size=5, min_area=100,
                                  save_debug=False, output_dir=None, logger=None):
    """
    从H&E图像中提取胆管特征（白色区域）
    Parameters:
    - img_rgb: H&E RGB图像 (uint8)
    - white_threshold: 白色区域阈值 (0-255)
    - kernel_size: 形态学操作核大小
    - min_area: 最小轮廓面积
    Returns:
    - bile_duct_mask: 胆管特征掩码
    - processed_image: 处理后的特征图像 (与输入类型一致)
    """
    if logger:
        logger.info("开始从H&E图像提取胆管特征（白色区域）...")
    # 转换为灰度图
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    # 提取白色区域（胆管腔）
    _, white_mask = cv2.threshold(img_gray, white_threshold, 255, cv2.THRESH_BINARY)
    # 本次修改：移除形态学操作和轮廓过滤，只保留阈值后的原始黑白图
    filtered_mask = white_mask
    if logger:
        logger.info(f"H&E胆管特征提取完成:")
        logger.info(f"  白色阈值: {white_threshold}")
        logger.info(f"  特征区域像素: {np.sum(filtered_mask > 0)}")
    # 保存调试图像
    if save_debug and output_dir:
        # 对于大图像，先缩小再保存
        downsample_factor = config.visualization_downsample_factor
        if downsample_factor < 1.0:
            dsize = (int(img_gray.shape[1] * downsample_factor), int(img_gray.shape[0] * downsample_factor))
            img_gray_small = cv2.resize(img_gray, dsize, interpolation=cv2.INTER_AREA)
            white_mask_small = cv2.resize(white_mask, dsize, interpolation=cv2.INTER_NEAREST)
            filtered_mask_small = cv2.resize(filtered_mask, dsize, interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(os.path.join(output_dir, "debug_he_gray_small.png"), img_gray_small)
            cv2.imwrite(os.path.join(output_dir, "debug_he_white_raw_small.png"), white_mask_small)
        else:
            cv2.imwrite(os.path.join(output_dir, "debug_he_gray.png"), img_gray)
            cv2.imwrite(os.path.join(output_dir, "debug_he_white_raw.png"), white_mask)
        if logger:
            logger.info("H&E胆管特征调试图像已保存 (可能为缩小版)")
    return filtered_mask, white_mask

def extract_bile_duct_features_fluoro(img_gray_uint16, black_threshold_uint16=5000, kernel_size=5, min_area=100,
                                     save_debug=False, output_dir=None, logger=None):
    """
    从荧光图像中提取胆管特征（黑色区域）
    Parameters:
    - img_gray_uint16: 荧光灰度图像 (uint16)
    - black_threshold_uint16: 黑色区域阈值 (0-65535)
    - kernel_size: 形态学操作核大小
    - min_area: 最小轮廓面积
    Returns:
    - bile_duct_mask: 胆管特征掩码 (uint8)
    - processed_image: 处理后的特征图像 (uint8)
    """
    if logger:
        logger.info("开始从荧光图像提取胆管特征（黑色区域）...")
        logger.info(f"  输入图像数据类型: {img_gray_uint16.dtype}")
        logger.info(f"  黑色阈值 (uint16): {black_threshold_uint16}")
    # 提取黑色区域（胆管腔）- 直接使用 uint16 阈值，小于阈值认为是黑色
    _, black_mask = cv2.threshold(img_gray_uint16, black_threshold_uint16, 65535, cv2.THRESH_BINARY_INV)
    # 转换为 uint8 掩码以便后续处理 (0 或 255)
    black_mask_uint8 = (black_mask > 0).astype(np.uint8) * 255
    # 本次修改：移除形态学操作和轮廓过滤，只保留阈值后的原始黑白图
    filtered_mask = black_mask_uint8
    if logger:
        logger.info(f"荧光胆管特征提取完成:")
        logger.info(f"  黑色阈值 (uint16): {black_threshold_uint16}")
        logger.info(f"  特征区域像素: {np.sum(filtered_mask > 0)}")
    # 为了与H&E掩码一致，返回 uint8 类型的掩码
    processed_image = filtered_mask # 或者是 black_mask_uint8，根据需要
    # 保存调试图像
    if save_debug and output_dir:
         # 对于大图像，先缩小再保存
        downsample_factor = config.visualization_downsample_factor
        if downsample_factor < 1.0:
            dsize = (int(img_gray_uint16.shape[1] * downsample_factor), int(img_gray_uint16.shape[0] * downsample_factor))
            # 将uint16图像转换为uint8用于可视化 (简单缩放)
            img_gray_uint8_vis = cv2.convertScaleAbs(img_gray_uint16, alpha=(255.0/65535.0))
            img_gray_small = cv2.resize(img_gray_uint8_vis, dsize, interpolation=cv2.INTER_AREA)
            black_mask_small = cv2.resize(black_mask_uint8, dsize, interpolation=cv2.INTER_NEAREST)
            filtered_mask_small = cv2.resize(filtered_mask, dsize, interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(os.path.join(output_dir, "debug_fluoro_gray_small.png"), img_gray_small)
            cv2.imwrite(os.path.join(output_dir, "debug_fluoro_black_raw_small.png"), black_mask_small)
        else:
             # 将uint16图像转换为uint8用于可视化
            img_gray_uint8_vis = cv2.convertScaleAbs(img_gray_uint16, alpha=(255.0/65535.0))
            cv2.imwrite(os.path.join(output_dir, "debug_fluoro_gray.png"), img_gray_uint8_vis)
            cv2.imwrite(os.path.join(output_dir, "debug_fluoro_black_raw.png"), black_mask_uint8)
        if logger:
            logger.info("荧光胆管特征调试图像已保存 (可能为缩小版)")
    return filtered_mask, black_mask_uint8

# --- 脚本1中的函数 (registration) ---

def resize_images_to_match(img1, img2, strategy='resize_to_smaller', interpolation=cv2.INTER_LINEAR, logger=None):
    """将两个图像调整为相同尺寸"""
    if len(img1.shape) == 3:
        h1, w1, _ = img1.shape
    else:
        h1, w1 = img1.shape
    if len(img2.shape) == 3:
        h2, w2, _ = img2.shape
    else:
        h2, w2 = img2.shape
    if logger:
        logger.info(f"图像尺寸处理:")
        logger.info(f"  图像1尺寸: {img1.shape}")
        logger.info(f"  图像2尺寸: {img2.shape}")
        logger.info(f"  处理策略: {strategy}")
    # 如果尺寸已经相同，直接返回
    if h1 == h2 and w1 == w2:
        if logger:
            logger.info("  图像尺寸已匹配，无需处理")
        return img1.copy(), img2.copy(), {'method': 'no_change', 'scale_x': 1.0, 'scale_y': 1.0}
    transform_info = {'method': strategy}
    if strategy == 'resize_to_smaller':
        target_h = min(h1, h2)
        target_w = min(w1, w2)
    elif strategy == 'resize_to_larger':
        target_h = max(h1, h2)
        target_w = max(w1, w2)
    elif strategy == 'no_resize':
        # 选择一个作为目标尺寸，这里选择第一个
        target_h, target_w = h1, w1
        # 调整第二个图像
        resized_img2 = cv2.resize(img2, (target_w, target_h), interpolation=interpolation)
        return img1.copy(), resized_img2, {'method': strategy, 'target_size': (target_h, target_w), 'scale2_x': target_w / w2, 'scale2_y': target_h / h2}
    else: # 默认调整到较小
        target_h = min(h1, h2)
        target_w = min(w1, w2)
    resized_img1 = cv2.resize(img1, (target_w, target_h), interpolation=interpolation)
    resized_img2 = cv2.resize(img2, (target_w, target_h), interpolation=interpolation)
    transform_info.update({
        'target_size': (target_h, target_w),
        'scale1_x': target_w / w1 if w1 > 0 else 1.0,
        'scale1_y': target_h / h1 if h1 > 0 else 1.0,
        'scale2_x': target_w / w2 if w2 > 0 else 1.0,
        'scale2_y': target_h / h2 if h2 > 0 else 1.0
    })
    if logger:
        logger.info(f"  处理后尺寸: {resized_img1.shape}")
        logger.info(f"  变换信息: {transform_info}")
    return resized_img1, resized_img2, transform_info

def compute_image_statistics(img, name="Image", logger=None):
    """计算图像统计信息"""
    stats = {
        'mean': np.mean(img),
        'std': np.std(img),
        'min': np.min(img),
        'max': np.max(img),
        'shape': img.shape,
        'dtype': img.dtype
    }
    if logger:
        logger.info(f"{name} 统计信息:")
        logger.info(f"  形状: {stats['shape']}, 类型: {stats['dtype']}")
        logger.info(f"  均值: {stats['mean']:.3f}, 标准差: {stats['std']:.3f}")
        logger.info(f"  范围: [{stats['min']:.3f}, {stats['max']:.3f}]")
    return stats

def enhanced_register_images_ecc(img_source, img_dest, motion_type, config, logger=None):
    """增强的ECC配准函数"""
    motion_name = config.motion_names[config.motion_types.index(motion_type)]
    if logger:
        logger.info(f"开始 {motion_name} 配准...")
    # 设置停止条件
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, config.max_iterations, config.epsilon)
    # 初始化变换矩阵
    if motion_type == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)
    start_time = time.time()
    try:
        # 确保输入是 float32
        if img_source.dtype != np.float32:
            img_source_f = img_source.astype(np.float32)
        else:
            img_source_f = img_source
        if img_dest.dtype != np.float32:
            img_dest_f = img_dest.astype(np.float32)
        else:
            img_dest_f = img_dest
        # 执行ECC配准
        cc, warp_matrix = cv2.findTransformECC(
            img_source_f, img_dest_f, warp_matrix, motion_type, criteria, None, 1 # Gauss-Newton step
        )
        elapsed_time = time.time() - start_time
        if logger:
            logger.info(f"{motion_name} 配准完成:")
            logger.info(f"  相关系数: {cc:.6f}")
            logger.info(f"  耗时: {elapsed_time:.2f}秒")
            logger.info(f"  变换矩阵:\n{warp_matrix}")
        # 确保返回3x3矩阵
        if motion_type != cv2.MOTION_HOMOGRAPHY:
            M = np.eye(3, 3, dtype=np.float32)
            M[:2, :] = warp_matrix
            return M, cc, elapsed_time
        else:
            return warp_matrix, cc, elapsed_time
    except cv2.error as e:
        elapsed_time = time.time() - start_time
        if logger:
            logger.error(f"{motion_name} 配准失败: {e}")
            logger.info(f"失败耗时: {elapsed_time:.2f}秒")
        return None, 0.0, elapsed_time
    except Exception as e: # 捕获其他可能的异常
        elapsed_time = time.time() - start_time
        if logger:
            logger.error(f"{motion_name} 配准过程中发生未预期错误: {e}")
            logger.info(f"失败耗时: {elapsed_time:.2f}秒")
        return None, 0.0, elapsed_time

def try_multiple_motion_types(img_source, img_dest, config, logger=None):
    """尝试多种运动模型"""
    if logger:
        logger.info("尝试多种运动模型进行配准...")
    results = []
    for motion_type, motion_name in zip(config.motion_types, config.motion_names):
        result = enhanced_register_images_ecc(img_source, img_dest, motion_type, config, logger)
        if result[0] is not None:  # 配准成功
            results.append({
                'motion_type': motion_type,
                'motion_name': motion_name,
                'matrix': result[0],
                'cc': result[1],
                'time': result[2]
            })
        else:
            results.append({
                'motion_type': motion_type,
                'motion_name': motion_name,
                'matrix': None,
                'cc': 0.0,
                'time': result[2]
            })
    # 选择最佳结果 (基于相关系数)
    successful_results = [r for r in results if r['matrix'] is not None]
    if successful_results:
        best_result = max(successful_results, key=lambda x: x['cc'])
        if logger:
            logger.info(f"最佳配准结果: {best_result['motion_name']}, CC={best_result['cc']:.6f}")
        return best_result, results
    else:
        if logger:
            logger.error("所有运动模型配准均失败")
        return None, results

def compute_registration_metrics(img_source, img_dest, warp_matrix, logger=None):
    """计算配准评估指标"""
    h, w = img_dest.shape[:2]
    img_warped = cv2.warpPerspective(img_source, warp_matrix, (w, h))
    # 结构相似性指数 (确保输入是 uint8)
    img_dest_uint8 = img_dest.astype(np.uint8)
    img_warped_uint8 = img_warped.astype(np.uint8)
    data_range = img_dest_uint8.max() - img_dest_uint8.min()
    if data_range == 0: data_range = 255 # 防止除零
    ssim_score = ssim(img_dest_uint8, img_warped_uint8, data_range=data_range)
    # 均方误差
    mse = np.mean((img_dest.astype(float) - img_warped.astype(float)) ** 2)
    # 峰值信噪比
    if mse > 0:
        psnr = 10 * np.log10((255.0 ** 2) / mse)
    else:
        psnr = float('inf')
    metrics = {
        'ssim': ssim_score,
        'mse': mse,
        'psnr': psnr
    }
    if logger:
        logger.info(f"配准评估指标:")
        logger.info(f"  SSIM: {ssim_score:.4f}")
        logger.info(f"  MSE: {mse:.2f}")
        logger.info(f"  PSNR: {psnr:.2f} dB")
    return metrics, img_warped

def create_visualization(img1_orig, img2_orig, img1_processed, img2_processed,
                         best_result, all_results, metrics, output_dir, logger=None):
    """创建配准的可视化结果"""
    if logger:
        logger.info("创建配准可视化结果...")
    # 确保所有图像尺寸匹配 (使用处理后的尺寸)
    target_h, target_w = img1_processed.shape[:2]
    # 调整原始图像尺寸以匹配处理后的图像
    img1_resized = cv2.resize(img1_orig, (target_w, target_h))
    img2_resized = cv2.resize(img2_orig, (target_w, target_h))
    if logger:
        logger.info(f"可视化图像尺寸调整:")
        logger.info(f"  目标尺寸: {target_h} x {target_w}")
        logger.info(f"  原始图像1: {img1_orig.shape} -> {img1_resized.shape}")
        logger.info(f"  原始图像2: {img2_orig.shape} -> {img2_resized.shape}")
    # 1. 图像和处理后图像 (现在处理后就是原图)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    # 第一行：原始图像和处理后图像
    axes[0, 0].imshow(cv2.cvtColor(img1_resized, cv2.COLOR_BGR2RGB) if len(img1_resized.shape)==3 else img1_resized, cmap='gray' if len(img1_resized.shape)!=3 else None)
    axes[0, 0].set_title('图像1 (调整尺寸)')
    axes[0, 0].axis('off')
    axes[0, 1].imshow(cv2.cvtColor(img2_resized, cv2.COLOR_BGR2RGB) if len(img2_resized.shape)==3 else img2_resized, cmap='gray' if len(img2_resized.shape)!=3 else None)
    axes[0, 1].set_title('图像2 (调整尺寸)')
    axes[0, 1].axis('off')
    axes[0, 2].imshow(cv2.cvtColor(img1_processed, cv2.COLOR_BGR2RGB) if len(img1_processed.shape)==3 else img1_processed, cmap='gray' if len(img1_processed.shape)!=3 else None)
    axes[0, 2].set_title('图像1 (用于配准)')
    axes[0, 2].axis('off')
    # 第二行：配准结果
    if best_result and best_result['matrix'] is not None:
        h, w = img2_processed.shape[:2]
        # 对处理后的图像1进行配准变换
        img1_aligned = cv2.warpPerspective(img1_processed, best_result['matrix'], (w, h))
        axes[1, 0].imshow(cv2.cvtColor(img1_aligned, cv2.COLOR_BGR2RGB) if len(img1_aligned.shape)==3 else img1_aligned, cmap='gray' if len(img1_aligned.shape)!=3 else None)
        axes[1, 0].set_title(f'配准后图像1 ({best_result["motion_name"]})')
        axes[1, 0].axis('off')
        axes[1, 1].imshow(cv2.cvtColor(img2_processed, cv2.COLOR_BGR2RGB) if len(img2_processed.shape)==3 else img2_processed, cmap='gray' if len(img2_processed.shape)!=3 else None)
        axes[1, 1].set_title('图像2 (参考)')
        axes[1, 1].axis('off')
        # 叠加效果
        if len(img1_aligned.shape) == 3 or len(img2_processed.shape) == 3:
             # 如果任一图像是彩色的，则转换为彩色进行叠加
             img1_aligned_vis = cv2.cvtColor(img1_aligned, cv2.COLOR_BGR2RGB) if len(img1_aligned.shape)==3 else cv2.cvtColor(img1_aligned, cv2.COLOR_GRAY2RGB)
             img2_processed_vis = cv2.cvtColor(img2_processed, cv2.COLOR_BGR2RGB) if len(img2_processed.shape)==3 else cv2.cvtColor(img2_processed, cv2.COLOR_GRAY2RGB)
        else:
             img1_aligned_vis = img1_aligned
             img2_processed_vis = img2_processed
        overlay = cv2.addWeighted(img1_aligned_vis, 0.5, img2_processed_vis, 0.5, 0)
        axes[1, 2].imshow(overlay)
        axes[1, 2].set_title('配准叠加效果')
        axes[1, 2].axis('off')
        # 添加指标文本
        if metrics:
            metrics_text = f"CC: {best_result['cc']:.4f}\nSSIM: {metrics['ssim']:.4f}\nMSE: {metrics['mse']:.2f}\nPSNR: {metrics['psnr']:.2f} dB"
            fig.text(0.02, 0.02, metrics_text, fontsize=10,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))
    else:
        # 如果没有配准结果，显示失败信息
        axes[1, 0].text(0.5, 0.5, '配准失败', ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].axis('off')
        axes[1, 1].axis('off')
        axes[1, 2].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'registration_visualization.png'),
                dpi=300, bbox_inches='tight')
    if config.display_plots:
        plt.show()
    plt.close()
    if logger:
        logger.info("配准可视化已保存")
    # 2. 配准结果对比图
    plt.figure(figsize=(10, 6))
    motion_names = [r['motion_name'] for r in all_results]
    cc_scores = [r['cc'] for r in all_results]
    colors = ['green' if r['matrix'] is not None else 'red' for r in all_results]
    bars = plt.bar(motion_names, cc_scores, color=colors, alpha=0.7)
    plt.ylabel('相关系数 (CC)')
    plt.title('配准：不同运动模型结果对比')
    plt.xticks(rotation=45)
    # 在柱状图上添加数值
    for bar, cc in zip(bars, cc_scores):
        if cc > 0:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f'{cc:.3f}', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'registration_comparison.png'),
                dpi=300, bbox_inches='tight')
    if config.display_plots:
        plt.show()
    plt.close()
    if logger:
        logger.info("配准结果对比图已保存")

def save_registration_report(all_results, best_result, metrics, config, output_dir,
                           transform_info=None, logger=None):
    """保存详细的配准报告"""
    report_path = os.path.join(output_dir, 'registration_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("图像配准详细报告\n")
        f.write("=" * 50 + "\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("配准原理:\n")
        f.write("  - 直接使用ECC算法对输入图像进行配准\n")
        f.write("  - 未进行任何预处理（如直方图均衡化、高斯模糊）\n")
        f.write("配置参数:\n")
        f.write(f"  图像1路径: {config.img1_path_for_registration}\n") # 修改为实际使用的路径
        f.write(f"  图像2路径: {config.img2_path_for_registration}\n") # 修改为实际使用的路径
        f.write(f"  尺寸处理策略: {config.resize_strategy}\n")
        f.write(f"  最大迭代次数: {config.max_iterations}\n")
        f.write(f"  收敛阈值 (epsilon): {config.epsilon}\n")
        if transform_info:
            f.write("图像尺寸变换信息:\n")
            f.write(f"  变换方法: {transform_info['method']}\n")
            if 'target_size' in transform_info:
                f.write(f"  目标尺寸: {transform_info['target_size']}\n")
            f.write(f"  详细信息: {transform_info}\n")
        f.write("所有运动模型结果:\n")
        f.write("-" * 30 + "\n")
        for result in all_results:
            f.write(f"运动模型: {result['motion_name']}\n")
            f.write(f"  相关系数: {result['cc']:.6f}\n")
            f.write(f"  耗时: {result['time']:.2f}秒\n")
            if result['matrix'] is not None:
                f.write(f"  变换矩阵:\n")
                for row in result['matrix']:
                    f.write(f"    [{', '.join([f'{x:10.6f}' for x in row])}]\n")
            f.write("\n")
        if best_result:
            f.write(f"最佳配准结果: {best_result['motion_name']}\n")
            f.write("-" * 30 + "\n")
            f.write(f"相关系数: {best_result['cc']:.6f}\n")
            if metrics:
                f.write("\n配准评估指标:\n")
                f.write(f"  结构相似性指数 (SSIM): {metrics['ssim']:.4f}\n")
                f.write(f"  均方误差 (MSE): {metrics['mse']:.2f}\n")
                f.write(f"  峰值信噪比 (PSNR): {metrics['psnr']:.2f} dB\n")
    if logger:
        logger.info(f"配准报告已保存至: {report_path}")

# --- 主程序 ---
def main():
    # 创建主输出目录和日志
    os.makedirs(config.main_output_dir, exist_ok=True)
    logger = setup_logging(config.main_output_dir)
    logger.info("开始合并的图像处理和配准程序")
    logger.info(f"主输出目录: {config.main_output_dir}")
    logger.info("处理流程：加载TIFF -> 匹配尺寸(仅填充) -> 提取黑白特征 -> ECC配准")

    try:
        # --- 阶段1: Patch Processing (来自脚本2) ---
        logger.info("=" * 60)
        logger.info("阶段 1: Patch Processing (特征提取)")
        os.makedirs(config.patch_processing_output_dir, exist_ok=True)

        # 1. 加载和预处理TIFF图像
        logger.info("-" * 40)
        logger.info("步骤 1.1: 加载和预处理TIFF图像")
        img_he_rgb, img_fluoro_gray, he_pixel_size, fluoro_pixel_size = load_and_preprocess_tiff(config, logger)
        if img_he_rgb is None or img_fluoro_gray is None:
            logger.error("图像加载失败，程序退出。")
            return

        # 2. 匹配图像尺寸 (仅填充，不缩放)
        logger.info("-" * 40)
        logger.info("步骤 1.2: 匹配图像尺寸 (仅填充)")
        # 目标像素大小设为H&E的像素大小 (虽然本次不用于缩放，但保留参数)
        target_pixel_size = he_pixel_size
        img_he_matched, img_fluoro_matched, transform_info_patch = match_images_by_pixel_size(
            img_he_rgb, img_fluoro_gray, he_pixel_size, fluoro_pixel_size, target_pixel_size, logger
        )
        if img_he_matched is None or img_fluoro_matched is None:
            logger.error("图像尺寸匹配失败，程序退出。")
            return

        # 3. 胆管特征提取 (简化版，只生成黑白图)
        logger.info("-" * 40)
        logger.info("步骤 1.3: 胆管特征提取 (简化版)")
        # 从H&E图像提取白色区域（胆管）
        he_bile_mask, he_white_raw = extract_bile_duct_features_he(
            img_he_matched, config.he_white_threshold, config.morphology_kernel_size,
            config.min_contour_area, config.save_debug_images, config.patch_processing_output_dir, logger
        )
        # 从荧光图像提取黑色区域（胆管）- 使用uint16阈值
        fluoro_bile_mask, fluoro_black_raw = extract_bile_duct_features_fluoro(
            img_fluoro_matched, config.fluoro_black_threshold, config.morphology_kernel_size,
            config.min_contour_area, config.save_debug_images, config.patch_processing_output_dir, logger
        )

        # 4. 保存特定的调试图像 (最终目标)
        logger.info("-" * 40)
        logger.info("步骤 1.4: 保存目标调试图像")
        if config.save_debug_images:
            # 保存缩小版的黑白图像
            downsample_factor = config.visualization_downsample_factor
            if downsample_factor < 1.0:
                dsize_he = (int(he_white_raw.shape[1] * downsample_factor), int(he_white_raw.shape[0] * downsample_factor))
                dsize_fluoro = (int(fluoro_black_raw.shape[1] * downsample_factor), int(fluoro_black_raw.shape[0] * downsample_factor))
                he_white_raw_small = cv2.resize(he_white_raw, dsize_he, interpolation=cv2.INTER_NEAREST)
                fluoro_black_raw_small = cv2.resize(fluoro_black_raw, dsize_fluoro, interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(os.path.join(config.patch_processing_output_dir, "debug_fluoro_black_raw_small.png"), fluoro_black_raw_small)
                cv2.imwrite(os.path.join(config.patch_processing_output_dir, "debug_he_white_raw_small.png"), he_white_raw_small)
                logger.info("目标调试图像已保存: debug_fluoro_black_raw_small.png, debug_he_white_raw_small.png")
            else:
                cv2.imwrite(os.path.join(config.patch_processing_output_dir, "debug_fluoro_black_raw.png"), fluoro_black_raw)
                cv2.imwrite(os.path.join(config.patch_processing_output_dir, "debug_he_white_raw.png"), he_white_raw)
                logger.info("目标调试图像已保存: debug_fluoro_black_raw.png, debug_he_white_raw.png")
        else:
           logger.info("未启用保存调试图像。")
        logger.info("阶段 1 完成!")

        # --- 阶段2: Image Registration (来自脚本1) ---
        logger.info("=" * 60)
        logger.info("阶段 2: Image Registration (ECC配准)")
        os.makedirs(config.registration_output_dir, exist_ok=True)
        # 重新设置日志记录器，指向配准输出目录
        registration_logger = setup_logging(config.registration_output_dir) # 可以选择是否覆盖主日志
        registration_logger.info("开始图像配准程序 (使用特征图像)")

        # 1. 加载图像 (从 patch processing 的输出加载)
        registration_logger.info("-" * 40)
        registration_logger.info("步骤 2.1: 加载特征图像")
        registration_logger.info(f"加载 图像1 (特征): {config.img1_path_for_registration}")
        img1_bgr = cv2.imread(config.img1_path_for_registration, cv2.IMREAD_GRAYSCALE) # 特征图像是灰度图
        if img1_bgr is None:
            registration_logger.error(f"无法加载 图像1: {config.img1_path_for_registration}")
            return
        registration_logger.info(f"加载 图像2 (特征): {config.img2_path_for_registration}")
        img2_bgr = cv2.imread(config.img2_path_for_registration, cv2.IMREAD_GRAYSCALE)
        if img2_bgr is None:
            registration_logger.error(f"无法加载 图像2: {config.img2_path_for_registration}")
            return
        compute_image_statistics(img1_bgr, "图像1 (特征原始)", registration_logger)
        compute_image_statistics(img2_bgr, "图像2 (特征原始)", registration_logger)

        # 2. 尺寸匹配
        registration_logger.info("-" * 40)
        registration_logger.info("步骤 2.2: 图像尺寸匹配")
        img1_resized, img2_resized, transform_info_reg = resize_images_to_match(
            img1_bgr, img2_bgr, config.resize_strategy,
            config.interpolation_method, registration_logger
        )

        # 3. 图像预处理 (已删除，直接使用调整尺寸后的图像)
        registration_logger.info("-" * 40)
        registration_logger.info("步骤 2.3: 图像预处理 (已跳过)")
        img1_processed = img1_resized # 直接使用调整尺寸后的图像
        img2_processed = img2_resized # 直接使用调整尺寸后的图像
        compute_image_statistics(img1_processed, "图像1 (用于配准)", registration_logger)
        compute_image_statistics(img2_processed, "图像2 (用于配准)", registration_logger)
        # 保存处理后的图像 (现在是调整尺寸后的图像)
        if config.registration_save_debug_images:
            cv2.imwrite(os.path.join(config.registration_output_dir, "img1_for_registration.png"), img1_processed)
            cv2.imwrite(os.path.join(config.registration_output_dir, "img2_for_registration.png"), img2_processed)
            registration_logger.info("用于配准的图像已保存")

        # 4. 执行配准 (使用处理后的图像)
        registration_logger.info("-" * 40)
        registration_logger.info("步骤 2.4: 执行图像配准")
        # 转换为 float32 用于 ECC
        img1_float = img1_processed.astype(np.float32)
        img2_float = img2_processed.astype(np.float32)
        best_result, all_results = try_multiple_motion_types(img1_float, img2_float, config, registration_logger)
        if best_result is None:
            registration_logger.error("所有配准方法均失败")
            # 即使配准失败，也尝试创建可视化
            registration_logger.info("尝试创建失败情况的可视化...")

        # 5. 计算配准指标（如果有最佳结果）
        metrics = None
        img_warped = None
        if best_result is not None:
            registration_logger.info("-" * 40)
            registration_logger.info("步骤 2.5: 计算配准评估指标")
            metrics, img_warped = compute_registration_metrics(
                img1_processed, img2_processed, best_result['matrix'], registration_logger # 使用 uint8 图像计算指标
            )

        # 6. 保存结果
        registration_logger.info("-" * 40)
        registration_logger.info("步骤 2.6: 保存配准结果")
        if best_result is not None:
            # 保存变换矩阵
            matrix_path = os.path.join(config.registration_output_dir, "transformation_matrix.txt")
            np.savetxt(matrix_path, best_result['matrix'], fmt='%.8f')
            registration_logger.info(f"配准变换矩阵已保存: {matrix_path}")
            # 保存配准后的图像
            if img_warped is not None:
                cv2.imwrite(os.path.join(config.registration_output_dir, "img1_warped.png"), img_warped)
                registration_logger.info("配准后的图像1已保存")

        # 保存配准报告
        save_registration_report(all_results, best_result, metrics, config, config.registration_output_dir,
                                transform_info_reg, registration_logger)

        # 7. 创建可视化
        registration_logger.info("-" * 40)
        registration_logger.info("步骤 2.7: 创建配准可视化结果")
        create_visualization(
            cv2.cvtColor(img1_bgr, cv2.COLOR_GRAY2BGR), cv2.cvtColor(img2_bgr, cv2.COLOR_GRAY2BGR), # 为了可视化，转为BGR
            cv2.cvtColor(img1_processed, cv2.COLOR_GRAY2BGR), cv2.cvtColor(img2_processed, cv2.COLOR_GRAY2BGR),
            best_result, all_results, metrics,
            config.registration_output_dir, registration_logger
        )

        # 8. 输出使用说明
        registration_logger.info("-" * 40)
        if best_result is not None:
            registration_logger.info("图像配准完成！使用说明:")
            registration_logger.info(f"最佳配准方法: {best_result['motion_name']}")
            registration_logger.info(f"相关系数: {best_result['cc']:.6f}")
            registration_logger.info("请查看输出目录中的可视化结果和详细报告")
        else:
            registration_logger.info("图像配准失败，但已生成分析报告")
            registration_logger.info("请检查:")
            registration_logger.info("1. 输入图像路径是否正确")
            registration_logger.info("2. 图像内容是否有足够的纹理或特征用于配准")
            registration_logger.info("3. 尝试不同的运动模型 (在 config.motion_types 中添加/移除)")

        logger.info("=" * 60)
        logger.info("合并处理流程完成！")
        logger.info(f"Patch Processing 输出目录: {config.patch_processing_output_dir}")
        logger.info(f"Registration 输出目录: {config.registration_output_dir}")

    except Exception as e:
        logger.error(f"程序执行过程中发生错误: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
    # demo_rotation()