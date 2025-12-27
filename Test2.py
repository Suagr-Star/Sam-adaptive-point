import os
import json
import random
import csv
import time
import numpy as np
import cv2
import torch
from pycocotools.coco import COCO
from skimage import metrics
from segment_anything import sam_model_registry, SamPredictor

# ====================== 1. 配置参数（新增：复杂图阈值+单张图片结果路径） ======================
class Config:
    # COCO数据集路径
    COCO_RAW_ROOT = "E:\\SAM_Model\\datasets\\COCO"
    COCO_IMG_PATH = os.path.join(COCO_RAW_ROOT, "val2017")
    COCO_ANN_PATH = os.path.join(COCO_RAW_ROOT, "annotations", "instances_val2017.json")
    
    # SAM权重路径（ViT-B）
    SAM_CHECKPOINT_PATH = "E:\\SAM_Model\\weights\\sam_vit_b_01ec64.pth"
    MODEL_TYPE = "vit_b"
    
    # 结果保存路径（已有样本路径）
    RESULT_ROOT = "E:\\SAM_Model\\results"
    EXP1_ROOT = os.path.join(RESULT_ROOT, "coco_exp1")
    EXP1_IMG_PATH = os.path.join(EXP1_ROOT, "images", "val2017")
    EXP1_ANN_PATH = os.path.join(EXP1_ROOT, "annotations", "instances_val2017_1000.json")
    SAMPLE_LIST_PATH = os.path.join(EXP1_ROOT, "sample_list.txt")
    RESULT_CSV_PATH = os.path.join(EXP1_ROOT, "exp1_results_adaptive.csv")  # 汇总结果
    LOG_PATH = os.path.join(EXP1_ROOT, "exp1_log_adaptive.txt")
    
    # 新增：每张图片详细结果保存路径（四项检测+复杂图标签）
    PER_IMAGE_RESULT_ROOT = os.path.join(RESULT_ROOT, "per_image_details")
    PER_IMAGE_CSV_PATH = os.path.join(PER_IMAGE_RESULT_ROOT, "image_details_with_detection.csv")
    
    # 实验参数
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    RANDOM_SEED = 42
    SAMPLE_NUM = 1000
    IMG_TARGET_SIZE = 1024
    MASK_THRESHOLD = 0.5
    MIN_INSTANCE_AREA = 100
    # 自适应选点参数
    SIZE_QUANTILE = 0.25  # 尺寸特征25分位数（小实例阈值）
    SHAPE_QUANTILE = 0.75  # 形状特征75分位数（不规则阈值）
    # 新增：复杂图判断阈值（难例占比超过该值则为复杂图）
    IMG_COMPLEX_THRESHOLD = 0.5

# ====================== 2. 工具函数（新增：四项检测+复杂图判断+结果保存） ======================
def create_dirs():
    # 原有目录
    os.makedirs(Config.EXP1_IMG_PATH, exist_ok=True)
    os.makedirs(os.path.dirname(Config.EXP1_ANN_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(Config.RESULT_CSV_PATH), exist_ok=True)
    # 新增：单张图片结果目录
    os.makedirs(Config.PER_IMAGE_RESULT_ROOT, exist_ok=True)

def init_per_image_csv():
    """初始化单张图片结果CSV，创建表头（四项检测+复杂图+三种mIoU）"""
    if not os.path.exists(Config.PER_IMAGE_CSV_PATH):
        with open(Config.PER_IMAGE_CSV_PATH, "w", encoding="utf-8", newline="") as f:
            fieldnames = [
                "img_id", "img_name", "img_width", "img_height",
                # 四项检测指标
                "total_valid_instances",  # 有效实例数（过滤小实例后）
                "hard_instance_num",      # 难例数
                "hard_instance_ratio",    # 难例占比
                "is_complex_image",       # 是否为复杂图（True/False）
                "complex_label",          # 复杂图标签（"复杂图片"/"简单图片"）
                # 三种提示方式mIoU
                "Point-only_mIoU",
                "Box-only_mIoU",
                "Point+Box(Adaptive)_mIoU"
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
    print(f"单张图片四项检测结果将保存至：{Config.PER_IMAGE_CSV_PATH}（追加模式，不覆盖）")

def judge_img_complexity(ann_list, img_info, size_thresh, shape_thresh):
    """
    四项检测+复杂图判断核心函数
    返回：total_valid（有效实例数）、hard_num（难例数）、hard_ratio（难例占比）、is_complex（是否复杂图）、complex_label（标签）
    """
    img_w, img_h = img_info["width"], img_info["height"]
    total_valid = 0  # 有效实例数（过滤小实例）
    hard_instance_num = 0  # 难例数
    
    for ann in ann_list:
        # 过滤小面积实例（无效实例）
        if ann["area"] < Config.MIN_INSTANCE_AREA:
            continue
        total_valid += 1
        
        # 提取实例四项特征（尺寸、形状、遮挡、分类）
        size_feat = get_size_feature(ann, img_w, img_h)
        mask = coco_exp1.annToMask(ann)
        shape_feat = get_shape_feature(mask)
        occlusion_feat = get_occlusion_feature(ann)
        instance_type = classify_instance(size_feat, shape_feat, occlusion_feat, size_thresh, shape_thresh)
        
        # 统计难例数
        if instance_type == "hard":
            hard_instance_num += 1
    
    # 计算难例占比
    hard_ratio = hard_instance_num / total_valid if total_valid > 0 else 0.0
    # 判断是否为复杂图
    is_complex = hard_ratio >= Config.IMG_COMPLEX_THRESHOLD
    complex_label = "复杂图片" if is_complex else "简单图片"
    
    return total_valid, hard_instance_num, round(hard_ratio, 4), is_complex, complex_label

def append_per_image_result(img_data):
    """追加单张图片结果到CSV（不覆盖已有数据）"""
    with open(Config.PER_IMAGE_CSV_PATH, "a", encoding="utf-8", newline="") as f:
        fieldnames = [
            "img_id", "img_name", "img_width", "img_height",
            "total_valid_instances", "hard_instance_num", "hard_instance_ratio",
            "is_complex_image", "complex_label",
            "Point-only_mIoU", "Box-only_mIoU", "Point+Box(Adaptive)_mIoU"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(img_data)

# ====================== 3. 样本筛选（原有逻辑，无修改） ======================
def filter_coco_samples():
    random.seed(Config.RANDOM_SEED)
    np.random.seed(Config.RANDOM_SEED)
    
    # 判断是否已有样本，有则直接加载
    if os.path.exists(Config.SAMPLE_LIST_PATH) and os.path.exists(Config.EXP1_ANN_PATH):
        print(f"检测到已有样本列表和标注，直接加载...")
        # 加载样本列表
        with open(Config.SAMPLE_LIST_PATH, "r", encoding="utf-8") as f:
            sample_lines = f.readlines()
        selected_img_ids = [int(line.split(",")[0]) for line in sample_lines]
        # 加载原始COCO
        coco = COCO(Config.COCO_ANN_PATH)
        print(f"已加载 {len(selected_img_ids)} 张已有样本")
        return selected_img_ids, coco
    
    # 无样本则重新生成（原有逻辑）
    print(f"正在加载原始COCO标注：{Config.COCO_ANN_PATH}")
    coco = COCO(Config.COCO_ANN_PATH)
    img_ids = coco.getImgIds()
    print(f"原始COCO val2017共有 {len(img_ids)} 张图片")
    
    selected_img_ids = random.sample(img_ids, Config.SAMPLE_NUM)
    print(f"已抽取 {len(selected_img_ids)} 张图片，ID: {selected_img_ids[:5]}...")
    
    # 复制图片
    sample_list = []
    for idx, img_id in enumerate(selected_img_ids):
        img_info = coco.loadImgs(img_id)[0]
        img_name = img_info["file_name"]
        img_original_path = os.path.join(Config.COCO_IMG_PATH, img_name)
        img_target_path = os.path.join(Config.EXP1_IMG_PATH, img_name)
        
        if not os.path.exists(img_target_path):
            import shutil
            shutil.copy(img_original_path, img_target_path)
        
        sample_list.append(f"{img_id},{img_name},{img_target_path}")
        
        if (idx + 1) % 100 == 0:
            print(f"图片复制进度：{idx+1}/{len(selected_img_ids)}")
    
    # 保存样本列表
    with open(Config.SAMPLE_LIST_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(sample_list))
    print(f"样本列表已保存至 {Config.SAMPLE_LIST_PATH}")
    
    # 生成专属标注
    print("正在生成专属标注文件...")
    selected_anns = coco.loadAnns(coco.getAnnIds(imgIds=selected_img_ids))
    exp1_anno = {
        "info": coco.dataset["info"],
        "licenses": coco.dataset["licenses"],
        "categories": coco.dataset["categories"],
        "images": [coco.loadImgs(img_id)[0] for img_id in selected_img_ids],
        "annotations": selected_anns
    }
    
    with open(Config.EXP1_ANN_PATH, "w", encoding="utf-8") as f:
        json.dump(exp1_anno, f)
    print(f"专属标注文件已保存至 {Config.EXP1_ANN_PATH}")
    
    return selected_img_ids, coco

# ====================== 4. 实例复杂度特征计算 + 自适应选点函数（原有逻辑，无修改） ======================
def get_mask_centroid(mask):
    """计算掩码质心（像素坐标）"""
    y_coords, x_coords = np.where(mask == 1)
    if len(x_coords) == 0 or len(y_coords) == 0:
        return None
    centroid_x = int(np.mean(x_coords))
    centroid_y = int(np.mean(y_coords))
    return (centroid_x, centroid_y)

def get_size_feature(ann, img_w, img_h):
    """计算尺寸特征：实例面积/图像面积"""
    return ann['area'] / (img_w * img_h)

def get_shape_feature(mask):
    """计算形状特征：(周长)^2 / 面积"""
    mask_uint8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    max_contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(max_contour, closed=True)
    area = cv2.contourArea(max_contour)
    if area == 0:
        return 0.0
    return (perimeter ** 2) / area

def get_occlusion_feature(ann):
    """获取遮挡特征：iscrowd字段"""
    return ann.get('iscrowd', 0)

def calculate_feature_thresholds(coco_exp1):
    """离线计算特征阈值（基于分位数）"""
    print("正在计算实例复杂度特征阈值...")
    all_anns = coco_exp1.loadAnns(coco_exp1.getAnnIds())
    size_features = []
    shape_features = []
    
    for ann in all_anns:
        img_info = coco_exp1.loadImgs(ann['image_id'])[0]
        img_w, img_h = img_info['width'], img_info['height']
        # 尺寸特征
        size_feat = get_size_feature(ann, img_w, img_h)
        size_features.append(size_feat)
        # 形状特征
        mask = coco_exp1.annToMask(ann)
        shape_feat = get_shape_feature(mask)
        shape_features.append(shape_feat)
    
    # 计算分位数阈值
    size_thresh = np.quantile(size_features, Config.SIZE_QUANTILE)
    shape_thresh = np.quantile(shape_features, Config.SHAPE_QUANTILE)
    print(f"特征阈值计算完成：")
    print(f"  尺寸特征阈值（小实例）：{size_thresh:.6f}")
    print(f"  形状特征阈值（不规则）：{shape_thresh:.2f}")
    return size_thresh, shape_thresh

def classify_instance(size_feat, shape_feat, occlusion_feat, size_thresh, shape_thresh):
    """规则分类：易例/难例"""
    if (size_feat < size_thresh) or (shape_feat > shape_thresh) or (occlusion_feat == 1):
        return "hard"  # 难例
    else:
        return "easy"  # 易例

def get_core_region_points(mask):
    """易例：获取核心内点（远离边缘）"""
    mask_uint8 = (mask * 255).astype(np.uint8)
    # 距离变换：计算每个前景像素到边缘的距离
    dist_transform = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, 5)
    if dist_transform.max() == 0:
        centroid = get_mask_centroid(mask)
        return np.array([centroid], dtype=np.float32) if centroid else None
    # 筛选核心区域（距离>50%最大距离）
    core_threshold = dist_transform.max() * 0.5
    core_mask = (dist_transform > core_threshold).astype(np.uint8)
    # 计算核心区域质心
    core_centroid = get_mask_centroid(core_mask)
    if not core_centroid:
        core_centroid = get_mask_centroid(mask)
    return np.array([core_centroid], dtype=np.float32) if core_centroid else None

def get_multi_core_points(mask, max_points=2):
    """难例：获取多核心点（基于连通域）"""
    mask_uint8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        centroid = get_mask_centroid(mask)
        return np.array([centroid] * min(1, max_points), dtype=np.float32) if centroid else None
    # 按面积排序，取前max_points个连通域
    contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)[:max_points]
    multi_centroids = []
    for cnt in contours_sorted:
        # 计算单个连通域的质心
        M = cv2.moments(cnt)
        if M['m00'] == 0:
            continue
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        multi_centroids.append((cx, cy))
    # 不足max_points则补充整体质心
    if len(multi_centroids) < max_points:
        global_centroid = get_mask_centroid(mask)
        if global_centroid:
            multi_centroids.append(global_centroid)
    # 去重并限制数量
    multi_centroids = list(set(multi_centroids))[:max_points]
    return np.array(multi_centroids, dtype=np.float32) if multi_centroids else None

def get_high_response_points(predictor, mask, img_w, img_h, top_k=2):
    """难例：获取高响应点（基于SAM特征图）"""
    # 获取SAM的图像嵌入特征
    img_embedding = predictor.get_image_embedding()
    # 统一转换为numpy数组（兼容GPU/CPU）
    if torch.is_tensor(img_embedding):
        img_embedding = img_embedding.cpu().numpy()
    embedding_h, embedding_w = img_embedding.shape[2], img_embedding.shape[3]
    # 缩放掩码到特征图尺寸
    mask_emb_size = cv2.resize(mask.astype(np.uint8), (embedding_w, embedding_h), interpolation=cv2.INTER_NEAREST)
    # 提取掩码区域的特征响应（取通道均值）
    feat_mean = np.mean(img_embedding[0], axis=0)
    mask_feat = feat_mean * mask_emb_size
    # 找到响应最高的top_k个像素
    flat_feat = mask_feat.flatten()
    top_indices = np.argsort(flat_feat)[-top_k:]
    # 转换为特征图坐标
    y_emb = (top_indices // embedding_w).astype(np.int32)
    x_emb = (top_indices % embedding_w).astype(np.int32)
    # 映射回原始图像坐标
    scale_x = img_w / embedding_w
    scale_y = img_h / embedding_h
    x_ori = (x_emb * scale_x).astype(np.int32)
    y_ori = (y_emb * scale_y).astype(np.int32)
    # 组合坐标
    high_response_points = [(x, y) for x, y in zip(x_ori, y_ori)]
    return np.array(high_response_points, dtype=np.float32) if high_response_points else None

def generate_negative_point(img_w, img_h, bbox):
    """生成负点（像素坐标，在bbox外）"""
    x1, y1, x2, y2 = bbox
    for _ in range(10):
        px = random.randint(0, img_w-1)
        py = random.randint(0, img_h-1)
        if not (x1 <= px <= x2 and y1 <= py <= y2):
            return (px, py)
    return (0, 0)

def build_prompts(img_info, ann, coco, predictor, size_thresh, shape_thresh):
    """
    自适应选点版本：返回像素坐标提示
    参数：
        predictor: SAM预测器（用于高响应点计算）
        size_thresh/shape_thresh: 特征阈值
    返回：point_prompt, box_prompt, mixed_prompt（自适应点）
    """
    img_w, img_h = img_info["width"], img_info["height"]
    bbox = ann["bbox"]
    mask = coco.annToMask(ann)
    instance_area = ann["area"]
    
    if instance_area < Config.MIN_INSTANCE_AREA:
        return None, None, None
    
    # 1. 框提示：xyxy像素坐标
    x, y, w, h = bbox
    box_xyxy = np.array([[x, y, x + w, y + h]], dtype=np.float32)
    
    # 2. 原始点提示（Point-only用，保持不变）
    centroid = get_mask_centroid(mask)
    if centroid is None:
        return None, None, None
    pos_x, pos_y = centroid
    neg_x, neg_y = generate_negative_point(img_w, img_h, box_xyxy[0])
    points = np.array([[pos_x, pos_y], [neg_x, neg_y]], dtype=np.float32)
    point_labels = np.array([1, 0], dtype=np.int32)
    
    # 3. 自适应混合提示（Point+Box用）
    # 提取实例复杂度特征
    size_feat = get_size_feature(ann, img_w, img_h)
    shape_feat = get_shape_feature(mask)
    occlusion_feat = get_occlusion_feature(ann)
    # 实例分类
    instance_type = classify_instance(size_feat, shape_feat, occlusion_feat, size_thresh, shape_thresh)
    
    if instance_type == "easy":
        # 易例：1个核心内点
        adaptive_points = get_core_region_points(mask)
    else:
        # 难例：多核心点 + 高响应点（合并去重）
        multi_core_pts = get_multi_core_points(mask, max_points=2)
        high_response_pts = get_high_response_points(predictor, mask, img_w, img_h, top_k=2)
        # 合并点
        adaptive_points_list = []
        if multi_core_pts is not None:
            adaptive_points_list.extend(multi_core_pts.tolist())
        if high_response_pts is not None:
            adaptive_points_list.extend(high_response_pts.tolist())
        # 去重并限制为2个点
        adaptive_points_list = list(set(tuple(pt) for pt in adaptive_points_list))[:2]
        adaptive_points = np.array(adaptive_points_list, dtype=np.float32) if adaptive_points_list else None
    
    # 兜底：自适应点失效时用原始质心
    if adaptive_points is None or len(adaptive_points) == 0:
        adaptive_points = np.array([[pos_x, pos_y]], dtype=np.float32)
    adaptive_labels = np.array([1] * len(adaptive_points), dtype=np.int32)
    
    # 封装返回
    point_prompt = (points, point_labels)
    box_prompt = (box_xyxy,)
    mixed_prompt = (adaptive_points, adaptive_labels, box_xyxy)
    
    return point_prompt, box_prompt, mixed_prompt

# ====================== 5. IoU计算函数（无需修改） ======================
def calculate_iou(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 0.0
    return intersection / union

def scale_mask(mask, target_h, target_w):
    """将SAM输出的掩码缩放至图像原始尺寸"""
    mask = mask.astype(np.uint8) * 255
    scaled_mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    return (scaled_mask > Config.MASK_THRESHOLD * 255).astype(np.uint8)

# ====================== 6. 核心实验流程（修正张量/numpy转换，新增四项检测+复杂图判断） ======================
def run_exp1():
    create_dirs()
    init_per_image_csv()  # 初始化单张图片结果CSV
    config = Config()
    global coco_exp1  # 全局变量，用于复杂图判断
    
    # 打印配置
    print("="*50)
    print("当前实验配置信息（新增四项检测+复杂图判断，已修正数据类型报错）")
    print("="*50)
    print(f"COCO数据集路径：{config.COCO_RAW_ROOT}")
    print(f"SAM权重路径：{config.SAM_CHECKPOINT_PATH}")
    print(f"SAM模型类型：{config.MODEL_TYPE}")
    print(f"实验结果保存路径：{config.EXP1_ROOT}")
    print(f"单张图片检测结果路径：{config.PER_IMAGE_CSV_PATH}")
    print(f"运行设备：{config.DEVICE}")
    print(f"抽取样本数：{config.SAMPLE_NUM}")
    print(f"复杂图判断阈值：{config.IMG_COMPLEX_THRESHOLD}（难例占比≥该值为复杂图）")
    print("="*50 + "\n")
    
    # 筛选/加载样本
    selected_img_ids, coco_original = filter_coco_samples()
    
    # 加载专属标注
    print(f"\n正在加载专属标注文件：{config.EXP1_ANN_PATH}")
    coco_exp1 = COCO(config.EXP1_ANN_PATH)
    print("专属标注加载完成！")
    
    # 计算实例复杂度特征阈值
    size_thresh, shape_thresh = calculate_feature_thresholds(coco_exp1)
    
    # 加载SAM模型
    print(f"\n正在加载SAM模型：{config.MODEL_TYPE}")
    print(f"权重路径：{config.SAM_CHECKPOINT_PATH}")
    sam = sam_model_registry[config.MODEL_TYPE](checkpoint=config.SAM_CHECKPOINT_PATH)
    sam.to(device=config.DEVICE)
    sam.eval()
    predictor = SamPredictor(sam)
    print("SAM模型加载完成，已进入评估模式\n")
    
    # 初始化结果记录
    prompt_types = ["Point-only", "Box-only", "Point+Box(Adaptive)"]
    result_dict = {pt: {"img_miou_list": [], "instance_iou_list": []} for pt in prompt_types}
    log_content = []
    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    log_content.append(f"实验1：SAM不同提示方式性能对比（新增四项检测+复杂图判断，已修正数据类型报错）")
    log_content.append(f"配置：样本数={config.SAMPLE_NUM}, 模型={config.MODEL_TYPE}, 输入尺寸={config.IMG_TARGET_SIZE}")
    log_content.append(f"COCO路径：{config.COCO_RAW_ROOT}")
    log_content.append(f"SAM权重：{config.SAM_CHECKPOINT_PATH}")
    log_content.append(f"实例复杂度阈值：尺寸={size_thresh:.6f}, 形状={shape_thresh:.2f}")
    log_content.append(f"复杂图判断阈值：{config.IMG_COMPLEX_THRESHOLD}")
    log_content.append(f"开始时间：{start_time}")
    print("\n".join(log_content))
    print("\n即将开始逐图推理+四项检测...")
    
    # 逐张推理
    total_img = len(selected_img_ids)
    # 用于统计复杂图数量
    complex_img_count = 0
    simple_img_count = 0
    for idx, img_id in enumerate(selected_img_ids):
        img_info = coco_exp1.loadImgs(img_id)[0]
        img_name = img_info["file_name"]
        img_path = os.path.join(config.EXP1_IMG_PATH, img_name)
        img_w, img_h = img_info["width"], img_info["height"]
        ann_ids = coco_exp1.getAnnIds(imgIds=img_id)
        anns = coco_exp1.loadAnns(ann_ids)
        
        # 读取图像
        image = cv2.imread(img_path)
        if image is None:
            print(f"警告：无法读取图片 {img_path}，已跳过")
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image_rgb)  # SAM自动处理图像尺寸
        
        # 初始化单张图片IoU
        img_instance_iou = {pt: [] for pt in prompt_types}
        
        # 逐实例推理
        for ann in anns:
            # 构建自适应提示
            point_prompt, box_prompt, mixed_prompt = build_prompts(
                img_info, ann, coco_exp1, predictor, size_thresh, shape_thresh
            )
            if point_prompt is None:
                continue
            gt_mask = coco_exp1.annToMask(ann)
            
            # （1）Point-only 推理（修正张量/numpy转换）
            points, point_labels = point_prompt
            with torch.no_grad():
                if config.DEVICE == "cuda":
                    with torch.cuda.amp.autocast():
                        masks, scores, _ = predictor.predict(
                            point_coords=points,
                            point_labels=point_labels,
                            multimask_output=True
                        )
                else:
                    masks, scores, _ = predictor.predict(
                        point_coords=points,
                        point_labels=point_labels,
                        multimask_output=True
                    )
            best_mask_idx = np.argmax(scores)
            # 统一处理：判断是否为张量，再转换为numpy
            pred_mask_point = masks[best_mask_idx]
            if torch.is_tensor(pred_mask_point):
                pred_mask_point = pred_mask_point.cpu().numpy()
            pred_mask_point_scaled = scale_mask(pred_mask_point, img_h, img_w)
            iou_point = calculate_iou(pred_mask_point_scaled, gt_mask)
            img_instance_iou["Point-only"].append(iou_point)
            result_dict["Point-only"]["instance_iou_list"].append(iou_point)
            
            # （2）Box-only 推理（修正张量/numpy转换）
            box_xyxy = box_prompt[0]
            with torch.no_grad():
                if config.DEVICE == "cuda":
                    with torch.cuda.amp.autocast():
                        masks, scores, _ = predictor.predict(
                            box=box_xyxy,
                            multimask_output=True
                        )
                else:
                    masks, scores, _ = predictor.predict(
                        box=box_xyxy,
                        multimask_output=True
                    )
            best_mask_idx = np.argmax(scores)
            pred_mask_box = masks[best_mask_idx]
            if torch.is_tensor(pred_mask_box):
                pred_mask_box = pred_mask_box.cpu().numpy()
            pred_mask_box_scaled = scale_mask(pred_mask_box, img_h, img_w)
            iou_box = calculate_iou(pred_mask_box_scaled, gt_mask)
            img_instance_iou["Box-only"].append(iou_box)
            result_dict["Box-only"]["instance_iou_list"].append(iou_box)
            
            # （3）Point+Box(Adaptive) 推理（修正张量/numpy转换）
            adaptive_points, adaptive_labels, mixed_box = mixed_prompt
            with torch.no_grad():
                if config.DEVICE == "cuda":
                    with torch.cuda.amp.autocast():
                        masks, scores, _ = predictor.predict(
                            point_coords=adaptive_points,
                            point_labels=adaptive_labels,
                            box=mixed_box,
                            multimask_output=True
                        )
                else:
                    masks, scores, _ = predictor.predict(
                        point_coords=adaptive_points,
                        point_labels=adaptive_labels,
                        box=mixed_box,
                        multimask_output=True
                    )
            best_mask_idx = np.argmax(scores)
            pred_mask_mixed = masks[best_mask_idx]
            if torch.is_tensor(pred_mask_mixed):
                pred_mask_mixed = pred_mask_mixed.cpu().numpy()
            pred_mask_mixed_scaled = scale_mask(pred_mask_mixed, img_h, img_w)
            iou_mixed = calculate_iou(pred_mask_mixed_scaled, gt_mask)
            img_instance_iou["Point+Box(Adaptive)"].append(iou_mixed)
            result_dict["Point+Box(Adaptive)"]["instance_iou_list"].append(iou_mixed)
        
        # 计算单张图片mIoU
        current_img_miou = {}
        for pt in prompt_types:
            if len(img_instance_iou[pt]) > 0:
                img_miou = np.mean(img_instance_iou[pt])
                result_dict[pt]["img_miou_list"].append(img_miou)
                current_img_miou[pt] = round(img_miou, 4)
            else:
                current_img_miou[pt] = 0.0
        
        # 新增：四项检测+复杂图判断
        total_valid, hard_num, hard_ratio, is_complex, complex_label = judge_img_complexity(
            anns, img_info, size_thresh, shape_thresh
        )
        
        # 统计复杂图/简单图数量
        if is_complex:
            complex_img_count += 1
        else:
            simple_img_count += 1
        
        # 整理单张图片结果数据
        img_detail_data = {
            "img_id": img_id,
            "img_name": img_name,
            "img_width": img_w,
            "img_height": img_h,
            # 四项检测指标
            "total_valid_instances": total_valid,
            "hard_instance_num": hard_num,
            "hard_instance_ratio": hard_ratio,
            "is_complex_image": is_complex,
            "complex_label": complex_label,
            # 三种提示方式mIoU
            "Point-only_mIoU": current_img_miou["Point-only"],
            "Box-only_mIoU": current_img_miou["Box-only"],
            "Point+Box(Adaptive)_mIoU": current_img_miou["Point+Box(Adaptive)"]
        }
        
        # 保存单张图片结果（追加模式，不覆盖）
        append_per_image_result(img_detail_data)
        
        # 打印进度（新增四项检测+复杂图信息）
        if (idx + 1) % 10 == 0 or (idx + 1) == total_img:
            print(f"推理进度：{idx+1}/{total_img} | 图片：{img_name}")
            print(f"  四项检测：有效实例={total_valid} | 难例数={hard_num} | 难例占比={hard_ratio}")
            print(f"  图片类型：{complex_label}（是否复杂图：{is_complex}）")
            print(f"  Point-only mIoU：{current_img_miou['Point-only']:.4f}")
            print(f"  Box-only mIoU：{current_img_miou['Box-only']:.4f}")
            print(f"  Point+Box(Adaptive) mIoU：{current_img_miou['Point+Box(Adaptive)']:.4f}")
            print("-"*50)
        
        # 清空显存（仅GPU有效）
        if config.DEVICE == "cuda":
            torch.cuda.empty_cache()
    
    # 计算最终汇总结果（原有逻辑，无修改）
    final_results = []
    end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    for pt in prompt_types:
        instance_iou_list = result_dict[pt]["instance_iou_list"]
        img_miou_list = result_dict[pt]["img_miou_list"]
        final_instance_miou = np.mean(instance_iou_list) if instance_iou_list else 0.0
        final_img_miou = np.mean(img_miou_list) if img_miou_list else 0.0
        
        final_results.append({
            "提示方式": pt,
            "实例数量": len(instance_iou_list),
            "图像数量": len(img_miou_list),
            "实例级mIoU": final_instance_miou,
            "图像级mIoU": final_img_miou
        })
        
        log_content.append(f"\n{pt} 结果：")
        log_content.append(f"  - 有效实例数：{len(instance_iou_list)}")
        log_content.append(f"  - 有效图像数：{len(img_miou_list)}")
        log_content.append(f"  - 实例级mIoU：{final_instance_miou:.4f}")
        log_content.append(f"  - 图像级mIoU：{final_img_miou:.4f}")
    
    # 保存汇总结果（原有逻辑）
    with open(config.RESULT_CSV_PATH, "w", encoding="utf-8", newline="") as f:
        fieldnames = ["提示方式", "实例数量", "图像数量", "实例级mIoU", "图像级mIoU"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final_results)
    print(f"\n核心对比汇总表格已保存至：{config.RESULT_CSV_PATH}")
    
    # 保存详细日志（新增复杂图统计）
    log_content.append(f"\n复杂图统计：")
    log_content.append(f"  - 总有效图片数：{complex_img_count + simple_img_count}")
    log_content.append(f"  - 复杂图片数：{complex_img_count}（占比：{complex_img_count/(complex_img_count+simple_img_count):.4f}）")
    log_content.append(f"  - 简单图片数：{simple_img_count}（占比：{simple_img_count/(complex_img_count+simple_img_count):.4f}）")
    log_content.append(f"\n结束时间：{end_time}")
    
    with open(config.LOG_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(log_content))
    print(f"详细实验日志（含复杂图统计）已保存至：{config.LOG_PATH}")
    print(f"单张图片四项检测结果已保存至：{config.PER_IMAGE_CSV_PATH}")
    
    # 打印最终结果
    print("\n" + "="*80)
    print("实验1 最终结果对比（新增四项检测+复杂图判断，已修正数据类型报错）")
    print("="*80)
    for res in final_results:
        print(f"{res['提示方式']:20s} | 实例级mIoU：{res['实例级mIoU']:.4f} | 图像级mIoU：{res['图像级mIoU']:.4f}")
    print("="*80)
    # 对比自适应与原始Box-only
    adaptive_instance_miou = final_results[2]["实例级mIoU"]
    box_only_instance_miou = final_results[1]["实例级mIoU"]
    if adaptive_instance_miou > box_only_instance_miou:
        print(f"实验结论：Point+Box(Adaptive) 性能超越Box-only，提升 {adaptive_instance_miou - box_only_instance_miou:.4f}")
    else:
        print(f"实验结论：Point+Box(Adaptive) 性能略低于Box-only，差异 {box_only_instance_miou - adaptive_instance_miou:.4f}")
    # 打印复杂图统计
    total_valid_img = complex_img_count + simple_img_count
    print(f"\n复杂图统计：共{complex_img_count}张复杂图片（占比{complex_img_count/total_valid_img:.2%}），{simple_img_count}张简单图片")

# ====================== 运行实验 ======================
if __name__ == "__main__":
    try:
        from segment_anything import sam_model_registry
    except ImportError:
        print("错误：未找到segment-anything库，请先克隆官方仓库并添加到Python环境！")
        print("克隆命令：git clone https://github.com/facebookresearch/segment-anything.git")
        exit(1)
    
    run_exp1()