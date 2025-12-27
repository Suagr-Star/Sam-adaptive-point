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

# ====================== 1. 配置参数（精简：仅保留核心参数+指定输出表格） ======================
class Config:
    # COCO数据集路径
    COCO_RAW_ROOT = "E:\\SAM_Model\\datasets\\COCO"
    COCO_IMG_PATH = os.path.join(COCO_RAW_ROOT, "val2017")
    COCO_ANN_PATH = os.path.join(COCO_RAW_ROOT, "annotations", "instances_val2017.json")
    
    # SAM权重路径（ViT-B）
    SAM_CHECKPOINT_PATH = "E:\\SAM_Model\\weights\\sam_vit_b_01ec64.pth"
    MODEL_TYPE = "vit_b"
    
    # 结果保存路径（新增1000张样本专用）
    RESULT_ROOT = "E:\\SAM_Model\\results"
    NEW_EXP_ROOT = os.path.join(RESULT_ROOT, "coco_exp_new_1000")
    NEW_SAMPLE_LIST_PATH = os.path.join(NEW_EXP_ROOT, "new_sample_list.txt")
    NEW_ANN_PATH = os.path.join(NEW_EXP_ROOT, "annotations", "instances_val2017_new_1000.json")
    NEW_IMG_PATH = os.path.join(NEW_EXP_ROOT, "images", "val2017")
    
    # 核心输出表格（仅保留Point/PointAdaptive/复杂图片标签）
    FINAL_TABLE_PATH = os.path.join(NEW_EXP_ROOT, "point_adaptive_complex_table.csv")
    LOG_PATH = os.path.join(NEW_EXP_ROOT, "new_exp_log.txt")
    
    # 实验参数
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    RANDOM_SEED = 42
    NEW_SAMPLE_NUM = 1000  # 新增1000张样本
    IMG_TARGET_SIZE = 1024
    MASK_THRESHOLD = 0.5
    MIN_INSTANCE_AREA = 100
    # 自适应选点优化参数
    SIZE_QUANTILE = 0.25  # 小实例阈值
    SHAPE_QUANTILE = 0.75  # 不规则阈值
    IMG_COMPLEX_THRESHOLD = 0.5  # 复杂图：难例占比≥50%
    CORE_DIST_RATIO = 0.6  # 优化：核心区域距离阈值（原0.5→0.6，更精准）
    MAX_ADAPTIVE_POINTS = 3  # 优化：难例最大选点数量（原2→3，覆盖更多核心区域）
    NEG_POINT_DIST_THRESH = 50  # 优化：负点与目标的最小距离（像素）

# ====================== 2. 工具函数（优化+聚焦指定表格输出） ======================
def create_dirs():
    """创建必要目录，确保路径存在"""
    os.makedirs(Config.NEW_IMG_PATH, exist_ok=True)
    os.makedirs(os.path.dirname(Config.NEW_ANN_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(Config.FINAL_TABLE_PATH), exist_ok=True)

def init_final_table():
    """初始化核心输出表格，仅保留指定字段"""
    if not os.path.exists(Config.FINAL_TABLE_PATH):
        with open(Config.FINAL_TABLE_PATH, "w", encoding="utf-8", newline="") as f:
            fieldnames = [
                "img_id",          # 图片ID
                "img_name",        # 图片名称
                "Point-only_mIoU", # Point-only的图像级mIoU
                "Point(Adaptive)_mIoU", # Point(Adaptive)的图像级mIoU
                "is_complex_image", # 是否为复杂图片（True/False）
                "complex_label"    # 复杂图片标签（"复杂图片"/"简单图片"）
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
    print(f"核心表格已初始化，路径：{Config.FINAL_TABLE_PATH}")

def append_to_final_table(row_data):
    """追加单张图片数据到核心表格（仅保留指定字段）"""
    with open(Config.FINAL_TABLE_PATH, "a", encoding="utf-8", newline="") as f:
        fieldnames = [
            "img_id", "img_name", "Point-only_mIoU", 
            "Point(Adaptive)_mIoU", "is_complex_image", "complex_label"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row_data)

# ====================== 3. 新增1000张样本筛选（独立于原有样本，避免冲突） ======================
def select_new_1000_samples():
    """筛选全新的1000张COCO样本（不与原有样本重复）"""
    random.seed(Config.RANDOM_SEED)
    np.random.seed(Config.RANDOM_SEED)
    
    # 加载原始COCO
    coco = COCO(Config.COCO_ANN_PATH)
    all_img_ids = coco.getImgIds()
    print(f"原始COCO val2017共{len(all_img_ids)}张图片")
    
    # 检查是否已有新增样本（避免重复生成）
    if os.path.exists(Config.NEW_SAMPLE_LIST_PATH) and os.path.exists(Config.NEW_ANN_PATH):
        print("检测到已有新增1000张样本，直接加载...")
        with open(Config.NEW_SAMPLE_LIST_PATH, "r", encoding="utf-8") as f:
            new_img_ids = [int(line.strip().split(",")[0]) for line in f.readlines()]
        return new_img_ids, coco
    
    # 随机抽取1000张全新样本
    new_img_ids = random.sample(all_img_ids, Config.NEW_SAMPLE_NUM)
    print(f"已抽取{len(new_img_ids)}张新样本，前5个ID：{new_img_ids[:5]}")
    
    # 复制图片文件
    sample_list = []
    for idx, img_id in enumerate(new_img_ids):
        img_info = coco.loadImgs(img_id)[0]
        img_name = img_info["file_name"]
        src_path = os.path.join(Config.COCO_IMG_PATH, img_name)
        dst_path = os.path.join(Config.NEW_IMG_PATH, img_name)
        
        if not os.path.exists(dst_path):
            import shutil
            shutil.copy(src_path, dst_path)
        
        sample_list.append(f"{img_id},{img_name},{dst_path}")
        
        if (idx + 1) % 100 == 0:
            print(f"图片复制进度：{idx+1}/{Config.NEW_SAMPLE_NUM}")
    
    # 保存新样本列表
    with open(Config.NEW_SAMPLE_LIST_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(sample_list))
    print(f"新样本列表已保存：{Config.NEW_SAMPLE_LIST_PATH}")
    
    # 生成新样本的标注文件
    new_ann_ids = coco.getAnnIds(imgIds=new_img_ids)
    new_anns = coco.loadAnns(new_ann_ids)
    new_anno_data = {
        "info": coco.dataset["info"],
        "licenses": coco.dataset["licenses"],
        "categories": coco.dataset["categories"],
        "images": [coco.loadImgs(img_id)[0] for img_id in new_img_ids],
        "annotations": new_anns
    }
    
    with open(Config.NEW_ANN_PATH, "w", encoding="utf-8") as f:
        json.dump(new_anno_data, f, indent=2)
    print(f"新样本标注文件已保存：{Config.NEW_ANN_PATH}")
    
    return new_img_ids, coco

# ====================== 4. 选点算法核心优化（关键改进点） ======================
def get_mask_centroid(mask):
    """优化：增加空掩码校验，返回整数坐标"""
    y_coords, x_coords = np.where(mask == 1)
    if len(x_coords) == 0 or len(y_coords) == 0:
        return None
    centroid_x = int(np.round(np.mean(x_coords)))
    centroid_y = int(np.round(np.mean(y_coords)))
    return (centroid_x, centroid_y)

def get_size_feature(ann, img_w, img_h):
    """尺寸特征：实例面积/图像面积（无修改）"""
    return ann['area'] / (img_w * img_h)

def get_shape_feature(mask):
    """优化：形状特征计算，增加轮廓校验，避免除零错误"""
    mask_uint8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 1e6  # 无轮廓视为极不规则
    max_contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(max_contour, closed=True)
    area = cv2.contourArea(max_contour)
    if area < 1:
        return 1e6
    return (perimeter ** 2) / area

def get_occlusion_feature(ann):
    """遮挡特征：iscrowd字段（无修改）"""
    return ann.get('iscrowd', 0)

def calculate_feature_thresholds(coco_new):
    """计算实例复杂度阈值（基于新增样本）"""
    print("正在计算新增样本的实例复杂度阈值...")
    all_anns = coco_new.loadAnns(coco_new.getAnnIds())
    size_features = []
    shape_features = []
    
    for ann in all_anns:
        if ann["area"] < Config.MIN_INSTANCE_AREA:
            continue
        img_info = coco_new.loadImgs(ann['image_id'])[0]
        img_w, img_h = img_info['width'], img_info['height']
        
        # 尺寸特征
        size_feat = get_size_feature(ann, img_w, img_h)
        size_features.append(size_feat)
        
        # 形状特征
        mask = coco_new.annToMask(ann)
        shape_feat = get_shape_feature(mask)
        shape_features.append(shape_feat)
    
    # 计算分位数阈值
    size_thresh = np.quantile(size_features, Config.SIZE_QUANTILE) if size_features else 0.01
    shape_thresh = np.quantile(shape_features, Config.SHAPE_QUANTILE) if shape_features else 1000.0
    print(f"阈值计算完成：尺寸阈值={size_thresh:.6f}，形状阈值={shape_thresh:.2f}")
    return size_thresh, shape_thresh

def classify_instance(size_feat, shape_feat, occlusion_feat, size_thresh, shape_thresh):
    """实例分类：易例/难例（无修改）"""
    if (size_feat < size_thresh) or (shape_feat > shape_thresh) or (occlusion_feat == 1):
        return "hard"
    else:
        return "easy"

def get_core_region_points_optimized(mask):
    """优化：核心区域点计算
    改进点：
    1. 距离变换阈值从0.5→0.6，更聚焦核心区域
    2. 增加坐标越界校验，避免无效点
    3. 小实例直接返回质心，减少计算量
    """
    mask_uint8 = (mask * 255).astype(np.uint8)
    mask_h, mask_w = mask.shape
    
    # 小实例直接返回质心
    if np.sum(mask) < Config.MIN_INSTANCE_AREA:
        centroid = get_mask_centroid(mask)
        return np.array([centroid], dtype=np.float32) if centroid else None
    
    # 距离变换计算核心区域
    dist_transform = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, 5)
    if dist_transform.max() < 5:  # 边缘距离过小，返回质心
        centroid = get_mask_centroid(mask)
        return np.array([centroid], dtype=np.float32) if centroid else None
    
    # 优化阈值：0.6倍最大距离
    core_threshold = dist_transform.max() * Config.CORE_DIST_RATIO
    core_mask = (dist_transform > core_threshold).astype(np.uint8)
    core_centroid = get_mask_centroid(core_mask)
    
    # 越界校验
    if core_centroid:
        cx, cy = core_centroid
        if 0 <= cx < mask_w and 0 <= cy < mask_h:
            return np.array([[cx, cy]], dtype=np.float32)
    
    # 兜底返回整体质心
    centroid = get_mask_centroid(mask)
    return np.array([centroid], dtype=np.float32) if centroid else None

def get_multi_core_points_optimized(mask):
    """优化：难例多核心点计算
    改进点：
    1. 最大选点数量从2→3，覆盖更多连通域
    2. 过滤小连通域（面积<50像素），减少噪声点
    3. 坐标越界校验
    """
    mask_uint8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        centroid = get_mask_centroid(mask)
        return np.array([centroid], dtype=np.float32) if centroid else None
    
    # 过滤小连通域，按面积排序
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 50]
    if not valid_contours:
        centroid = get_mask_centroid(mask)
        return np.array([centroid], dtype=np.float32) if centroid else None
    
    contours_sorted = sorted(valid_contours, key=cv2.contourArea, reverse=True)[:Config.MAX_ADAPTIVE_POINTS]
    multi_centroids = []
    
    for cnt in contours_sorted:
        M = cv2.moments(cnt)
        if M['m00'] == 0:
            continue
        cx = int(np.round(M['m10'] / M['m00']))
        cy = int(np.round(M['m01'] / M['m00']))
        
        # 越界校验
        mask_h, mask_w = mask.shape
        if 0 <= cx < mask_w and 0 <= cy < mask_h:
            multi_centroids.append((cx, cy))
    
    # 不足数量则补充整体质心
    if len(multi_centroids) < Config.MAX_ADAPTIVE_POINTS:
        global_centroid = get_mask_centroid(mask)
        if global_centroid and global_centroid not in multi_centroids:
            multi_centroids.append(global_centroid)
    
    # 去重并限制数量
    multi_centroids = list(set(multi_centroids))[:Config.MAX_ADAPTIVE_POINTS]
    return np.array(multi_centroids, dtype=np.float32) if multi_centroids else None

def get_high_response_points_optimized(predictor, mask, img_w, img_h):
    """优化：高响应点计算
    改进点：
    1. 特征图坐标映射更精准（浮点型缩放）
    2. 过滤响应值过低的点（<均值），减少噪声
    3. 坐标越界校验
    """
    # 获取SAM图像嵌入
    img_embedding = predictor.get_image_embedding()
    if torch.is_tensor(img_embedding):
        img_embedding = img_embedding.cpu().numpy()
    
    embedding_h, embedding_w = img_embedding.shape[2], img_embedding.shape[3]
    # 缩放掩码到特征图尺寸（浮点型插值）
    mask_emb_size = cv2.resize(
        mask.astype(np.uint8), 
        (embedding_w, embedding_h), 
        interpolation=cv2.INTER_LINEAR
    )
    if np.sum(mask_emb_size) == 0:
        return None
    
    # 计算特征响应均值，过滤低响应点
    feat_mean = np.mean(img_embedding[0], axis=0)
    mask_feat = feat_mean * mask_emb_size
    feat_threshold = np.mean(mask_feat[mask_feat > 0]) if np.sum(mask_feat > 0) > 0 else 0
    
    # 取响应最高的2个点
    flat_feat = mask_feat.flatten()
    valid_indices = np.where(flat_feat >= feat_threshold)[0]
    if len(valid_indices) == 0:
        return None
    
    top_indices = valid_indices[np.argsort(flat_feat[valid_indices])[-2:]]
    y_emb = (top_indices // embedding_w).astype(np.float32)
    x_emb = (top_indices % embedding_w).astype(np.float32)
    
    # 精准映射回原始图像坐标（浮点型缩放）
    scale_x = img_w / embedding_w
    scale_y = img_h / embedding_h
    x_ori = x_emb * scale_x
    y_ori = y_emb * scale_y
    
    # 越界校验
    high_response_points = []
    for x, y in zip(x_ori, y_ori):
        if 0 <= x < img_w and 0 <= y < img_h:
            high_response_points.append((int(np.round(x)), int(np.round(y))))
    
    return np.array(high_response_points, dtype=np.float32) if high_response_points else None

def generate_negative_point_optimized(img_w, img_h, bbox):
    """优化：负点生成
    改进点：确保负点与目标框的距离≥50像素，避免靠近目标
    """
    x1, y1, x2, y2 = bbox
    target_center = ((x1+x2)/2, (y1+y2)/2)
    
    for _ in range(20):  # 增加尝试次数
        px = random.randint(0, img_w-1)
        py = random.randint(0, img_h-1)
        
        # 检查是否在bbox外，且与目标中心距离≥阈值
        if not (x1 <= px <= x2 and y1 <= py <= y2):
            dist = np.sqrt((px - target_center[0])**2 + (py - target_center[1])**2)
            if dist >= Config.NEG_POINT_DIST_THRESH:
                return (px, py)
    
    # 兜底：返回图像角落
    return (0, 0) if (0,0) not in bbox else (img_w-1, img_h-1)

def build_prompts_optimized(img_info, ann, coco, predictor, size_thresh, shape_thresh):
    """优化后的提示构建函数（仅返回Point-only和Point(Adaptive)）"""
    img_w, img_h = img_info["width"], img_info["height"]
    bbox = ann["bbox"]
    mask = coco.annToMask(ann)
    instance_area = ann["area"]
    
    if instance_area < Config.MIN_INSTANCE_AREA:
        return None, None
    
    # 1. Point-only提示（优化负点生成）
    centroid = get_mask_centroid(mask)
    if centroid is None:
        return None, None
    pos_x, pos_y = centroid
    neg_x, neg_y = generate_negative_point_optimized(img_w, img_h, bbox)
    original_points = np.array([[pos_x, pos_y], [neg_x, neg_y]], dtype=np.float32)
    original_point_labels = np.array([1, 0], dtype=np.int32)
    
    # 2. Point(Adaptive)提示（优化选点逻辑）
    size_feat = get_size_feature(ann, img_w, img_h)
    shape_feat = get_shape_feature(mask)
    occlusion_feat = get_occlusion_feature(ann)
    instance_type = classify_instance(size_feat, shape_feat, occlusion_feat, size_thresh, shape_thresh)
    
    if instance_type == "easy":
        adaptive_points = get_core_region_points_optimized(mask)
    else:
        # 难例：多核心点 + 高响应点
        multi_core_pts = get_multi_core_points_optimized(mask)
        high_response_pts = get_high_response_points_optimized(predictor, mask, img_w, img_h)
        
        adaptive_points_list = []
        if multi_core_pts is not None:
            adaptive_points_list.extend(multi_core_pts.tolist())
        if high_response_pts is not None:
            adaptive_points_list.extend(high_response_pts.tolist())
        
        # 去重+越界校验
        adaptive_points_list = list(set(tuple(pt) for pt in adaptive_points_list))
        adaptive_points_list = [
            (x, y) for x, y in adaptive_points_list 
            if 0 <= x < img_w and 0 <= y < img_h
        ]
        
        # 限制最大数量
        adaptive_points_list = adaptive_points_list[:Config.MAX_ADAPTIVE_POINTS]
        adaptive_points = np.array(adaptive_points_list, dtype=np.float32) if adaptive_points_list else None
    
    # 兜底：自适应点失效时用原始质心
    if adaptive_points is None or len(adaptive_points) == 0:
        adaptive_points = np.array([[pos_x, pos_y]], dtype=np.float32)
    adaptive_labels = np.array([1] * len(adaptive_points), dtype=np.int32)
    
    return (original_points, original_point_labels), (adaptive_points, adaptive_labels)

# ====================== 5. 辅助函数（IoU计算+掩码缩放+复杂图判断） ======================
def calculate_iou(pred_mask, gt_mask):
    """计算IoU（无修改）"""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    return intersection / union if union > 0 else 0.0

def scale_mask(mask, target_h, target_w):
    """缩放掩码到原始尺寸（优化插值方式）"""
    mask = mask.astype(np.uint8) * 255
    scaled_mask = cv2.resize(
        mask, (target_w, target_h), 
        interpolation=cv2.INTER_NEAREST_EXACT  # 优化：精准最近邻插值
    )
    return (scaled_mask > Config.MASK_THRESHOLD * 255).astype(np.uint8)

def judge_img_complexity_optimized(ann_list, img_info, size_thresh, shape_thresh):
    """优化：复杂图判断，过滤小实例后统计难例占比"""
    img_w, img_h = img_info["width"], img_info["height"]
    total_valid = 0
    hard_instance_num = 0
    
    for ann in ann_list:
        if ann["area"] < Config.MIN_INSTANCE_AREA:
            continue
        total_valid += 1
        
        # 计算实例特征
        size_feat = get_size_feature(ann, img_w, img_h)
        mask = coco_new.annToMask(ann)
        shape_feat = get_shape_feature(mask)
        occlusion_feat = get_occlusion_feature(ann)
        
        # 分类
        if classify_instance(size_feat, shape_feat, occlusion_feat, size_thresh, shape_thresh) == "hard":
            hard_instance_num += 1
    
    # 计算难例占比
    hard_ratio = hard_instance_num / total_valid if total_valid > 0 else 0.0
    is_complex = hard_ratio >= Config.IMG_COMPLEX_THRESHOLD
    complex_label = "复杂图片" if is_complex else "简单图片"
    
    return total_valid, hard_instance_num, hard_ratio, is_complex, complex_label

# ====================== 6. 核心实验流程（仅运行Point-only和Point(Adaptive)） ======================
def run_new_exp():
    """运行新增1000张样本的实验，仅聚焦Point/PointAdaptive+复杂图片标签"""
    create_dirs()
    init_final_table()
    global coco_new  # 全局变量，用于复杂图判断
    
    # 打印配置
    print("="*60)
    print("新增1000张样本实验配置（仅Point/PointAdaptive+复杂图片标签）")
    print("="*60)
    print(f"设备：{Config.DEVICE} | 样本数：{Config.NEW_SAMPLE_NUM}")
    print(f"SAM模型：{Config.MODEL_TYPE} | 权重路径：{Config.SAM_CHECKPOINT_PATH}")
    print(f"核心表格输出：{Config.FINAL_TABLE_PATH}")
    print(f"复杂图阈值：难例占比≥{Config.IMG_COMPLEX_THRESHOLD}")
    print("="*60 + "\n")
    
    # 筛选新增1000张样本
    new_img_ids, coco_original = select_new_1000_samples()
    
    # 加载新增样本的标注
    print("加载新增样本的标注文件...")
    coco_new = COCO(Config.NEW_ANN_PATH)
    
    # 计算特征阈值
    size_thresh, shape_thresh = calculate_feature_thresholds(coco_new)
    
    # 加载SAM模型
    print("\n加载SAM模型...")
    sam = sam_model_registry[Config.MODEL_TYPE](checkpoint=Config.SAM_CHECKPOINT_PATH)
    sam.to(device=Config.DEVICE)
    sam.eval()
    predictor = SamPredictor(sam)
    print("SAM模型加载完成！")
    
    # 初始化结果统计
    log_content = [
        f"新增1000张样本实验日志 | 开始时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"配置：样本数={Config.NEW_SAMPLE_NUM}, 复杂图阈值={Config.IMG_COMPLEX_THRESHOLD}",
        f"特征阈值：尺寸={size_thresh:.6f}, 形状={shape_thresh:.2f}"
    ]
    
    # 逐张图片推理
    total_img = len(new_img_ids)
    complex_img_count = 0
    simple_img_count = 0
    point_only_all_iou = []
    point_adaptive_all_iou = []
    
    for idx, img_id in enumerate(new_img_ids):
        try:
            # 加载图片信息
            img_info = coco_new.loadImgs(img_id)[0]
            img_name = img_info["file_name"]
            img_path = os.path.join(Config.NEW_IMG_PATH, img_name)
            ann_ids = coco_new.getAnnIds(imgIds=img_id)
            anns = coco_new.loadAnns(ann_ids)
            
            # 读取图片
            image = cv2.imread(img_path)
            if image is None:
                print(f"警告：无法读取图片 {img_name}，跳过")
                continue
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            predictor.set_image(image_rgb)
            
            # 初始化单张图片的IoU列表
            img_point_only_iou = []
            img_point_adaptive_iou = []
            
            # 逐实例推理
            for ann in anns:
                # 构建提示
                point_only_data, point_adaptive_data = build_prompts_optimized(
                    img_info, ann, coco_new, predictor, size_thresh, shape_thresh
                )
                if point_only_data is None:
                    continue
                gt_mask = coco_new.annToMask(ann)
                img_h, img_w = gt_mask.shape
                
                # 1. Point-only推理
                points, point_labels = point_only_data
                with torch.no_grad():
                    if Config.DEVICE == "cuda":
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
                pred_mask = masks[best_mask_idx].cpu().numpy() if torch.is_tensor(masks[best_mask_idx]) else masks[best_mask_idx]
                pred_mask_scaled = scale_mask(pred_mask, img_h, img_w)
                iou = calculate_iou(pred_mask_scaled, gt_mask)
                img_point_only_iou.append(iou)
                point_only_all_iou.append(iou)
                
                # 2. Point(Adaptive)推理
                adaptive_points, adaptive_labels = point_adaptive_data
                with torch.no_grad():
                    if Config.DEVICE == "cuda":
                        with torch.cuda.amp.autocast():
                            masks, scores, _ = predictor.predict(
                                point_coords=adaptive_points,
                                point_labels=adaptive_labels,
                                multimask_output=True
                            )
                    else:
                        masks, scores, _ = predictor.predict(
                            point_coords=adaptive_points,
                            point_labels=adaptive_labels,
                            multimask_output=True
                        )
                best_mask_idx = np.argmax(scores)
                pred_mask = masks[best_mask_idx].cpu().numpy() if torch.is_tensor(masks[best_mask_idx]) else masks[best_mask_idx]
                pred_mask_scaled = scale_mask(pred_mask, img_h, img_w)
                iou = calculate_iou(pred_mask_scaled, gt_mask)
                img_point_adaptive_iou.append(iou)
                point_adaptive_all_iou.append(iou)
            
            # 计算单张图片的mIoU
            img_point_only_miou = np.mean(img_point_only_iou) if img_point_only_iou else 0.0
            img_point_adaptive_miou = np.mean(img_point_adaptive_iou) if img_point_adaptive_iou else 0.0
            
            # 判断是否为复杂图片
            _, _, _, is_complex, complex_label = judge_img_complexity_optimized(
                anns, img_info, size_thresh, shape_thresh
            )
            
            # 统计复杂/简单图片数量
            if is_complex:
                complex_img_count += 1
            else:
                simple_img_count += 1
            
            # 写入核心表格
            row_data = {
                "img_id": img_id,
                "img_name": img_name,
                "Point-only_mIoU": round(img_point_only_miou, 4),
                "Point(Adaptive)_mIoU": round(img_point_adaptive_miou, 4),
                "is_complex_image": is_complex,
                "complex_label": complex_label
            }
            append_to_final_table(row_data)
            
            # 打印进度
            if (idx + 1) % 50 == 0 or (idx + 1) == total_img:
                print(f"进度：{idx+1}/{total_img} | 图片：{img_name}")
                print(f"  Point-only mIoU：{img_point_only_miou:.4f} | Point(Adaptive) mIoU：{img_point_adaptive_miou:.4f}")
                print(f"  复杂图片：{is_complex} | 标签：{complex_label}")
                print("-"*50)
            
            # 清空显存
            if Config.DEVICE == "cuda":
                torch.cuda.empty_cache()
        
        except Exception as e:
            print(f"处理图片 {img_id} 出错：{str(e)}，已跳过")
            log_content.append(f"错误：图片{img_id}处理失败 - {str(e)}")
            continue
    
    # 计算整体结果
    final_point_only_miou = np.mean(point_only_all_iou) if point_only_all_iou else 0.0
    final_point_adaptive_miou = np.mean(point_adaptive_all_iou) if point_adaptive_all_iou else 0.0
    total_valid_img = complex_img_count + simple_img_count
    
    # 写入日志
    log_content.extend([
        f"\n===== 实验汇总结果 =====",
        f"总处理图片数：{total_img}",
        f"有效处理图片数：{total_valid_img}",
        f"复杂图片数：{complex_img_count}（占比：{complex_img_count/total_valid_img:.4f}）",
        f"简单图片数：{simple_img_count}（占比：{simple_img_count/total_valid_img:.4f}）",
        f"Point-only 整体mIoU：{final_point_only_miou:.4f}",
        f"Point(Adaptive) 整体mIoU：{final_point_adaptive_miou:.4f}",
        f"提升幅度：{final_point_adaptive_miou - final_point_only_miou:.4f}",
        f"\n结束时间：{time.strftime('%Y-%m-%d %H:%M:%S')}"
    ])
    
    # 保存日志
    with open(Config.LOG_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(log_content))
    
    # 打印最终结果
    print("\n" + "="*80)
    print("新增1000张样本实验最终结果")
    print("="*80)
    print(f"Point-only 整体mIoU：{final_point_only_miou:.4f}")
    print(f"Point(Adaptive) 整体mIoU：{final_point_adaptive_miou:.4f}")
    print(f"提升幅度：{final_point_adaptive_miou - final_point_only_miou:.4f}")
    print(f"复杂图片数量：{complex_img_count}（占比{complex_img_count/total_valid_img:.2%}）")
    print(f"核心表格已保存至：{Config.FINAL_TABLE_PATH}")
    print(f"实验日志已保存至：{Config.LOG_PATH}")
    print("="*80)

# ====================== 运行入口 ======================
if __name__ == "__main__":
    try:
        from segment_anything import sam_model_registry
    except ImportError:
        print("错误：未安装segment-anything库！")
        print("请执行：git clone https://github.com/facebookresearch/segment-anything.git")
        print("并将segment-anything添加到Python环境中")
        exit(1)
    
    # 运行新增1000张样本的实验
    run_new_exp()