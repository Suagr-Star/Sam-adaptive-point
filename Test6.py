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

# ====================== 1. 配置参数 ======================
class Config:
    COCO_RAW_ROOT = "E:\\SAM_Model\\datasets\\COCO"
    COCO_IMG_PATH = os.path.join(COCO_RAW_ROOT, "val2017")
    COCO_ANN_PATH = os.path.join(COCO_RAW_ROOT, "annotations", "instances_val2017.json")

    SAM_CHECKPOINT_PATH = "E:\\SAM_Model\\weights\\sam_vit_b_01ec64.pth"
    MODEL_TYPE = "vit_b"

    RESULT_ROOT = "E:\\SAM_Model\\results"
    NEW_EXP_ROOT = os.path.join(RESULT_ROOT, "coco_exp_new_1000")
    NEW_SAMPLE_LIST_PATH = os.path.join(NEW_EXP_ROOT, "new_sample_list.txt")
    NEW_ANN_PATH = os.path.join(NEW_EXP_ROOT, "annotations", "instances_val2017_new_1000.json")
    NEW_IMG_PATH = os.path.join(NEW_EXP_ROOT, "images", "val2017")

    FINAL_TABLE_PATH = os.path.join(NEW_EXP_ROOT, "point_adaptive_complex_table.csv")
    LOG_PATH = os.path.join(NEW_EXP_ROOT, "new_exp_log.txt")

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    RANDOM_SEED = 42
    NEW_SAMPLE_NUM = 1000
    IMG_TARGET_SIZE = 1024
    MASK_THRESHOLD = 0.5
    MIN_INSTANCE_AREA = 100

    SIZE_QUANTILE = 0.25
    SHAPE_QUANTILE = 0.75
    IMG_COMPLEX_THRESHOLD = 0.5
    CORE_DIST_RATIO = 0.6
    MAX_ADAPTIVE_POINTS = 3

    # ========= 负点策略（优化版）=========
    # Adaptive 负点门控：仅 hard 实例加负点（强烈建议 True）
    ADAPTIVE_NEG_ONLY_HARD = True

    # ring负点：mask 外侧环带参数（像素半径）
    NEG_RING_OUTER = 25    # 外扩半径（越大 ring 越宽）
    NEG_RING_INNER = 5     # 内缩空洞（避免太贴近边界造成误伤）
    NEG_RING_TRIES = 30    # ring采样尝试次数

    # bbox外安全边界（避免负点落入 bbox 内部附近）
    NEG_BBOX_MARGIN = 3

    # 兜底策略：如果 ring 失败，则退化为 bbox 外随机采样
    NEG_FALLBACK_TRIES = 40
    NEG_POINT_DIST_THRESH = 50  # 兜底时与目标中心最小距离

# ====================== 2. 工具函数 ======================
def create_dirs():
    os.makedirs(Config.NEW_IMG_PATH, exist_ok=True)
    os.makedirs(os.path.dirname(Config.NEW_ANN_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(Config.FINAL_TABLE_PATH), exist_ok=True)

def init_final_table():
    if not os.path.exists(Config.FINAL_TABLE_PATH):
        with open(Config.FINAL_TABLE_PATH, "w", encoding="utf-8", newline="") as f:
            fieldnames = [
                "img_id", "img_name",
                "Point-only_mIoU",
                "Point(Adaptive)_mIoU",
                "is_complex_image", "complex_label"
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
    print(f"核心表格已初始化，路径：{Config.FINAL_TABLE_PATH}")

def append_to_final_table(row_data):
    with open(Config.FINAL_TABLE_PATH, "a", encoding="utf-8", newline="") as f:
        fieldnames = [
            "img_id", "img_name",
            "Point-only_mIoU",
            "Point(Adaptive)_mIoU",
            "is_complex_image", "complex_label"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row_data)

# ====================== 3. 新增1000张样本筛选 ======================
def select_new_1000_samples():
    random.seed(Config.RANDOM_SEED)
    np.random.seed(Config.RANDOM_SEED)

    coco = COCO(Config.COCO_ANN_PATH)
    all_img_ids = coco.getImgIds()
    print(f"原始COCO val2017共{len(all_img_ids)}张图片")

    if os.path.exists(Config.NEW_SAMPLE_LIST_PATH) and os.path.exists(Config.NEW_ANN_PATH):
        print("检测到已有新增1000张样本，直接加载...")
        with open(Config.NEW_SAMPLE_LIST_PATH, "r", encoding="utf-8") as f:
            new_img_ids = [int(line.strip().split(",")[0]) for line in f.readlines()]
        return new_img_ids, coco

    new_img_ids = random.sample(all_img_ids, Config.NEW_SAMPLE_NUM)
    print(f"已抽取{len(new_img_ids)}张新样本，前5个ID：{new_img_ids[:5]}")

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

    with open(Config.NEW_SAMPLE_LIST_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(sample_list))
    print(f"新样本列表已保存：{Config.NEW_SAMPLE_LIST_PATH}")

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

# ====================== 4. 特征与选点 ======================
def get_mask_centroid(mask):
    y_coords, x_coords = np.where(mask == 1)
    if len(x_coords) == 0 or len(y_coords) == 0:
        return None
    return (int(np.round(np.mean(x_coords))), int(np.round(np.mean(y_coords))))

def get_size_feature(ann, img_w, img_h):
    return ann['area'] / (img_w * img_h)

def get_shape_feature(mask):
    mask_uint8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 1e6
    max_contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(max_contour, closed=True)
    area = cv2.contourArea(max_contour)
    if area < 1:
        return 1e6
    return (perimeter ** 2) / area

def get_occlusion_feature(ann):
    return ann.get('iscrowd', 0)

def calculate_feature_thresholds(coco_new):
    print("正在计算新增样本的实例复杂度阈值...")
    all_anns = coco_new.loadAnns(coco_new.getAnnIds())
    size_features, shape_features = [], []

    for ann in all_anns:
        if ann["area"] < Config.MIN_INSTANCE_AREA:
            continue
        img_info = coco_new.loadImgs(ann['image_id'])[0]
        img_w, img_h = img_info['width'], img_info['height']

        size_features.append(get_size_feature(ann, img_w, img_h))
        mask = coco_new.annToMask(ann)
        shape_features.append(get_shape_feature(mask))

    size_thresh = np.quantile(size_features, Config.SIZE_QUANTILE) if size_features else 0.01
    shape_thresh = np.quantile(shape_features, Config.SHAPE_QUANTILE) if shape_features else 1000.0
    print(f"阈值计算完成：尺寸阈值={size_thresh:.6f}，形状阈值={shape_thresh:.2f}")
    return size_thresh, shape_thresh

def classify_instance(size_feat, shape_feat, occlusion_feat, size_thresh, shape_thresh):
    return "hard" if (size_feat < size_thresh) or (shape_feat > shape_thresh) or (occlusion_feat == 1) else "easy"

def get_core_region_points_optimized(mask):
    mask_uint8 = (mask * 255).astype(np.uint8)
    mask_h, mask_w = mask.shape

    if np.sum(mask) < Config.MIN_INSTANCE_AREA:
        c = get_mask_centroid(mask)
        return np.array([c], dtype=np.float32) if c else None

    dist_transform = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, 5)
    if dist_transform.max() < 5:
        c = get_mask_centroid(mask)
        return np.array([c], dtype=np.float32) if c else None

    core_threshold = dist_transform.max() * Config.CORE_DIST_RATIO
    core_mask = (dist_transform > core_threshold).astype(np.uint8)
    core_centroid = get_mask_centroid(core_mask)

    if core_centroid:
        cx, cy = core_centroid
        if 0 <= cx < mask_w and 0 <= cy < mask_h:
            return np.array([[cx, cy]], dtype=np.float32)

    c = get_mask_centroid(mask)
    return np.array([c], dtype=np.float32) if c else None

def get_multi_core_points_optimized(mask):
    mask_uint8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        c = get_mask_centroid(mask)
        return np.array([c], dtype=np.float32) if c else None

    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 50]
    if not valid_contours:
        c = get_mask_centroid(mask)
        return np.array([c], dtype=np.float32) if c else None

    contours_sorted = sorted(valid_contours, key=cv2.contourArea, reverse=True)[:Config.MAX_ADAPTIVE_POINTS]
    mask_h, mask_w = mask.shape
    multi_centroids = []

    for cnt in contours_sorted:
        M = cv2.moments(cnt)
        if M['m00'] == 0:
            continue
        cx = int(np.round(M['m10'] / M['m00']))
        cy = int(np.round(M['m01'] / M['m00']))
        if 0 <= cx < mask_w and 0 <= cy < mask_h:
            multi_centroids.append((cx, cy))

    if len(multi_centroids) < Config.MAX_ADAPTIVE_POINTS:
        g = get_mask_centroid(mask)
        if g and g not in multi_centroids:
            multi_centroids.append(g)

    multi_centroids = list(set(multi_centroids))[:Config.MAX_ADAPTIVE_POINTS]
    return np.array(multi_centroids, dtype=np.float32) if multi_centroids else None

def get_high_response_points_optimized(predictor, mask, img_w, img_h):
    img_embedding = predictor.get_image_embedding()
    if torch.is_tensor(img_embedding):
        img_embedding = img_embedding.cpu().numpy()

    embedding_h, embedding_w = img_embedding.shape[2], img_embedding.shape[3]
    mask_emb_size = cv2.resize(mask.astype(np.uint8), (embedding_w, embedding_h), interpolation=cv2.INTER_LINEAR)
    if np.sum(mask_emb_size) == 0:
        return None

    feat_mean = np.mean(img_embedding[0], axis=0)
    mask_feat = feat_mean * mask_emb_size
    feat_threshold = np.mean(mask_feat[mask_feat > 0]) if np.sum(mask_feat > 0) > 0 else 0

    flat_feat = mask_feat.flatten()
    valid_indices = np.where(flat_feat >= feat_threshold)[0]
    if len(valid_indices) == 0:
        return None

    top_indices = valid_indices[np.argsort(flat_feat[valid_indices])[-2:]]
    y_emb = (top_indices // embedding_w).astype(np.float32)
    x_emb = (top_indices % embedding_w).astype(np.float32)

    scale_x = img_w / embedding_w
    scale_y = img_h / embedding_h
    x_ori = x_emb * scale_x
    y_ori = y_emb * scale_y

    pts = []
    for x, y in zip(x_ori, y_ori):
        if 0 <= x < img_w and 0 <= y < img_h:
            pts.append((int(np.round(x)), int(np.round(y))))
    return np.array(pts, dtype=np.float32) if pts else None

# ====================== 负点：优化为 ring 负点 + 兜底 ======================
def _bbox_to_xyxy(bbox):
    x1, y1, w, h = bbox
    x2, y2 = x1 + w, y1 + h
    return x1, y1, x2, y2

def generate_negative_point_ring(mask, img_w, img_h, bbox):
    """
    优化负点：从 GT mask 外侧环带 ring 采样，提供紧贴边界的背景约束。
    ring = dilate(mask, outer) - dilate(mask, inner)
    同时避免落入 bbox (含margin) 内。
    """
    x1, y1, x2, y2 = _bbox_to_xyxy(bbox)
    margin = Config.NEG_BBOX_MARGIN
    bx1 = max(0, int(x1 - margin))
    by1 = max(0, int(y1 - margin))
    bx2 = min(img_w - 1, int(x2 + margin))
    by2 = min(img_h - 1, int(y2 + margin))

    mask_u8 = (mask.astype(np.uint8) * 255)

    outer_k = 2 * Config.NEG_RING_OUTER + 1
    inner_k = 2 * Config.NEG_RING_INNER + 1
    outer_k = max(3, outer_k | 1)
    inner_k = max(3, inner_k | 1)

    outer = cv2.dilate(mask_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (outer_k, outer_k)))
    inner = cv2.dilate(mask_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (inner_k, inner_k)))

    ring = ((outer > 0) & (inner == 0)).astype(np.uint8)
    ys, xs = np.where(ring == 1)
    if len(xs) == 0:
        return None

    for _ in range(Config.NEG_RING_TRIES):
        idx = random.randint(0, len(xs) - 1)
        px, py = int(xs[idx]), int(ys[idx])

        # 避免落入 bbox 内（含margin）
        if bx1 <= px <= bx2 and by1 <= py <= by2:
            continue
        if 0 <= px < img_w and 0 <= py < img_h:
            return (px, py)

    return None

def generate_negative_point_fallback(img_w, img_h, bbox):
    """
    兜底负点：bbox外随机 + 与目标中心保持距离（你原方案的稳健版）
    """
    x1, y1, x2, y2 = _bbox_to_xyxy(bbox)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

    for _ in range(Config.NEG_FALLBACK_TRIES):
        px = random.randint(0, img_w - 1)
        py = random.randint(0, img_h - 1)
        if x1 <= px <= x2 and y1 <= py <= y2:
            continue
        dist = np.sqrt((px - cx) ** 2 + (py - cy) ** 2)
        if dist >= Config.NEG_POINT_DIST_THRESH:
            return (px, py)

    # 最终兜底：角点
    return (0, 0)

def generate_negative_point(mask, img_w, img_h, bbox):
    """
    总入口：优先 ring 负点，失败则 fallback。
    """
    p = generate_negative_point_ring(mask, img_w, img_h, bbox)
    if p is not None:
        return p
    return generate_negative_point_fallback(img_w, img_h, bbox)

# ====================== 提示构建（门控 + ring负点） ======================
def build_prompts_optimized(img_info, ann, coco, predictor, size_thresh, shape_thresh):
    img_w, img_h = img_info["width"], img_info["height"]
    bbox = ann["bbox"]
    mask = coco.annToMask(ann)
    instance_area = ann["area"]

    if instance_area < Config.MIN_INSTANCE_AREA:
        return None, None

    centroid = get_mask_centroid(mask)
    if centroid is None:
        return None, None
    pos_x, pos_y = centroid

    # Point-only：保持 1正+1负（用优化后的负点生成，更稳定）
    neg_x, neg_y = generate_negative_point(mask, img_w, img_h, bbox)
    original_points = np.array([[pos_x, pos_y], [neg_x, neg_y]], dtype=np.float32)
    original_point_labels = np.array([1, 0], dtype=np.int32)

    # Adaptive：先算实例类型
    size_feat = get_size_feature(ann, img_w, img_h)
    shape_feat = get_shape_feature(mask)
    occlusion_feat = get_occlusion_feature(ann)
    instance_type = classify_instance(size_feat, shape_feat, occlusion_feat, size_thresh, shape_thresh)

    # 自适应正点
    if instance_type == "easy":
        adaptive_pos_points = get_core_region_points_optimized(mask)
    else:
        multi_core_pts = get_multi_core_points_optimized(mask)
        high_response_pts = get_high_response_points_optimized(predictor, mask, img_w, img_h)

        adaptive_points_list = []
        if multi_core_pts is not None:
            adaptive_points_list.extend(multi_core_pts.tolist())
        if high_response_pts is not None:
            adaptive_points_list.extend(high_response_pts.tolist())

        adaptive_points_list = list(set(tuple(pt) for pt in adaptive_points_list))
        adaptive_points_list = [(x, y) for x, y in adaptive_points_list if 0 <= x < img_w and 0 <= y < img_h]
        adaptive_points_list = adaptive_points_list[:Config.MAX_ADAPTIVE_POINTS]
        adaptive_pos_points = np.array(adaptive_points_list, dtype=np.float32) if adaptive_points_list else None

    if adaptive_pos_points is None or len(adaptive_pos_points) == 0:
        adaptive_pos_points = np.array([[pos_x, pos_y]], dtype=np.float32)

    # ======= 关键：门控负点（只对 hard 启用）=======
    add_neg = True
    if Config.ADAPTIVE_NEG_ONLY_HARD and instance_type == "easy":
        add_neg = False

    if add_neg:
        npx, npy = generate_negative_point(mask, img_w, img_h, bbox)
        adaptive_points = np.vstack([adaptive_pos_points, np.array([[npx, npy]], dtype=np.float32)])
        adaptive_labels = np.array([1] * len(adaptive_pos_points) + [0], dtype=np.int32)
    else:
        adaptive_points = adaptive_pos_points
        adaptive_labels = np.array([1] * len(adaptive_pos_points), dtype=np.int32)

    return (original_points, original_point_labels), (adaptive_points, adaptive_labels)

# ====================== 5. 评估与复杂图判断 ======================
def calculate_iou(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    return intersection / union if union > 0 else 0.0

def scale_mask(mask, target_h, target_w):
    mask = mask.astype(np.uint8) * 255
    scaled_mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST_EXACT)
    return (scaled_mask > Config.MASK_THRESHOLD * 255).astype(np.uint8)

def judge_img_complexity_optimized(ann_list, img_info, size_thresh, shape_thresh):
    img_w, img_h = img_info["width"], img_info["height"]
    total_valid = 0
    hard_instance_num = 0

    for ann in ann_list:
        if ann["area"] < Config.MIN_INSTANCE_AREA:
            continue
        total_valid += 1

        size_feat = get_size_feature(ann, img_w, img_h)
        mask = coco_new.annToMask(ann)
        shape_feat = get_shape_feature(mask)
        occlusion_feat = get_occlusion_feature(ann)

        if classify_instance(size_feat, shape_feat, occlusion_feat, size_thresh, shape_thresh) == "hard":
            hard_instance_num += 1

    hard_ratio = hard_instance_num / total_valid if total_valid > 0 else 0.0
    is_complex = hard_ratio >= Config.IMG_COMPLEX_THRESHOLD
    complex_label = "复杂图片" if is_complex else "简单图片"
    return total_valid, hard_instance_num, hard_ratio, is_complex, complex_label

# ====================== 6. 核心实验流程 ======================
def run_new_exp():
    create_dirs()
    init_final_table()
    global coco_new

    print("=" * 60)
    print("新增1000张样本实验配置（Point/PointAdaptive + 复杂图片标签）")
    print("=" * 60)
    print(f"设备：{Config.DEVICE} | 样本数：{Config.NEW_SAMPLE_NUM}")
    print(f"SAM模型：{Config.MODEL_TYPE} | 权重：{Config.SAM_CHECKPOINT_PATH}")
    print(f"复杂图阈值：难例占比≥{Config.IMG_COMPLEX_THRESHOLD}")
    print(f"Adaptive负点门控(仅hard)：{Config.ADAPTIVE_NEG_ONLY_HARD}")
    print(f"Ring负点：outer={Config.NEG_RING_OUTER}, inner={Config.NEG_RING_INNER}")
    print("=" * 60 + "\n")

    new_img_ids, _ = select_new_1000_samples()

    print("加载新增样本标注...")
    coco_new = COCO(Config.NEW_ANN_PATH)

    size_thresh, shape_thresh = calculate_feature_thresholds(coco_new)

    print("\n加载SAM模型...")
    sam = sam_model_registry[Config.MODEL_TYPE](checkpoint=Config.SAM_CHECKPOINT_PATH)
    sam.to(device=Config.DEVICE)
    sam.eval()
    predictor = SamPredictor(sam)
    print("SAM模型加载完成！")

    log_content = [
        f"新增1000张样本实验日志 | 开始：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"配置：样本数={Config.NEW_SAMPLE_NUM}, 复杂图阈值={Config.IMG_COMPLEX_THRESHOLD}",
        f"阈值：尺寸={size_thresh:.6f}, 形状={shape_thresh:.2f}",
        f"Adaptive负点门控(仅hard)={Config.ADAPTIVE_NEG_ONLY_HARD}",
        f"Ring负点 outer={Config.NEG_RING_OUTER}, inner={Config.NEG_RING_INNER}"
    ]

    total_img = len(new_img_ids)
    complex_img_count, simple_img_count = 0, 0
    point_only_all_iou, point_adaptive_all_iou = [], []

    for idx, img_id in enumerate(new_img_ids):
        try:
            img_info = coco_new.loadImgs(img_id)[0]
            img_name = img_info["file_name"]
            img_path = os.path.join(Config.NEW_IMG_PATH, img_name)
            ann_ids = coco_new.getAnnIds(imgIds=img_id)
            anns = coco_new.loadAnns(ann_ids)

            image = cv2.imread(img_path)
            if image is None:
                print(f"警告：无法读取 {img_name}，跳过")
                continue
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            predictor.set_image(image_rgb)

            img_point_only_iou, img_point_adaptive_iou = [], []

            for ann in anns:
                point_only_data, point_adaptive_data = build_prompts_optimized(
                    img_info, ann, coco_new, predictor, size_thresh, shape_thresh
                )
                if point_only_data is None:
                    continue

                gt_mask = coco_new.annToMask(ann)
                h, w = gt_mask.shape

                # Point-only
                points, labels = point_only_data
                with torch.no_grad():
                    if Config.DEVICE == "cuda":
                        with torch.cuda.amp.autocast():
                            masks, scores, _ = predictor.predict(
                                point_coords=points, point_labels=labels, multimask_output=True
                            )
                    else:
                        masks, scores, _ = predictor.predict(
                            point_coords=points, point_labels=labels, multimask_output=True
                        )
                best_idx = np.argmax(scores)
                pred_mask = masks[best_idx].cpu().numpy() if torch.is_tensor(masks[best_idx]) else masks[best_idx]
                pred_mask_scaled = scale_mask(pred_mask, h, w)
                iou = calculate_iou(pred_mask_scaled, gt_mask)
                img_point_only_iou.append(iou)
                point_only_all_iou.append(iou)

                # Point(Adaptive)
                a_points, a_labels = point_adaptive_data
                with torch.no_grad():
                    if Config.DEVICE == "cuda":
                        with torch.cuda.amp.autocast():
                            masks, scores, _ = predictor.predict(
                                point_coords=a_points, point_labels=a_labels, multimask_output=True
                            )
                    else:
                        masks, scores, _ = predictor.predict(
                            point_coords=a_points, point_labels=a_labels, multimask_output=True
                        )
                best_idx = np.argmax(scores)
                pred_mask = masks[best_idx].cpu().numpy() if torch.is_tensor(masks[best_idx]) else masks[best_idx]
                pred_mask_scaled = scale_mask(pred_mask, h, w)
                iou = calculate_iou(pred_mask_scaled, gt_mask)
                img_point_adaptive_iou.append(iou)
                point_adaptive_all_iou.append(iou)

            miou_p = np.mean(img_point_only_iou) if img_point_only_iou else 0.0
            miou_a = np.mean(img_point_adaptive_iou) if img_point_adaptive_iou else 0.0

            _, _, _, is_complex, complex_label = judge_img_complexity_optimized(anns, img_info, size_thresh, shape_thresh)
            if is_complex:
                complex_img_count += 1
            else:
                simple_img_count += 1

            append_to_final_table({
                "img_id": img_id,
                "img_name": img_name,
                "Point-only_mIoU": round(miou_p, 4),
                "Point(Adaptive)_mIoU": round(miou_a, 4),
                "is_complex_image": is_complex,
                "complex_label": complex_label
            })

            if (idx + 1) % 50 == 0 or (idx + 1) == total_img:
                print(f"进度：{idx+1}/{total_img} | {img_name}")
                print(f"  Point-only mIoU：{miou_p:.4f} | Adaptive mIoU：{miou_a:.4f} | {complex_label}")
                print("-" * 50)

            if Config.DEVICE == "cuda":
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"处理图片 {img_id} 出错：{str(e)}，跳过")
            log_content.append(f"错误：img_id={img_id} - {str(e)}")
            continue

    final_p = np.mean(point_only_all_iou) if point_only_all_iou else 0.0
    final_a = np.mean(point_adaptive_all_iou) if point_adaptive_all_iou else 0.0
    total_valid = complex_img_count + simple_img_count

    log_content.extend([
        "\n===== 实验汇总结果 =====",
        f"总处理图片数：{total_img}",
        f"有效处理图片数：{total_valid}",
        f"复杂图片数：{complex_img_count}（占比：{complex_img_count / total_valid:.4f}）",
        f"简单图片数：{simple_img_count}（占比：{simple_img_count / total_valid:.4f}）",
        f"Point-only 整体mIoU：{final_p:.4f}",
        f"Point(Adaptive) 整体mIoU：{final_a:.4f}",
        f"提升幅度：{final_a - final_p:.4f}",
        f"结束：{time.strftime('%Y-%m-%d %H:%M:%S')}"
    ])

    with open(Config.LOG_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(log_content))

    print("\n" + "=" * 80)
    print("新增1000张样本实验最终结果")
    print("=" * 80)
    print(f"Point-only 整体mIoU：{final_p:.4f}")
    print(f"Point(Adaptive) 整体mIoU：{final_a:.4f}")
    print(f"提升幅度：{final_a - final_p:.4f}")
    print(f"复杂图片：{complex_img_count}（{complex_img_count/total_valid:.2%}）")
    print(f"结果表：{Config.FINAL_TABLE_PATH}")
    print(f"日志：{Config.LOG_PATH}")
    print("=" * 80)

if __name__ == "__main__":
    try:
        from segment_anything import sam_model_registry
    except ImportError:
        print("错误：未安装segment-anything库！")
        exit(1)
    run_new_exp()
