import os
import json
import random
import csv
import time
import numpy as np
import cv2
import torch
from pycocotools.coco import COCO
from segment_anything import sam_model_registry, SamPredictor

# ====================== 1. 配置参数（强制CPU运行） ======================
class Config:
    # COCO数据集路径
    COCO_RAW_ROOT = "E:\\SAM_Model\\datasets\\COCO"
    COCO_IMG_PATH = os.path.join(COCO_RAW_ROOT, "val2017")
    COCO_ANN_PATH = os.path.join(COCO_RAW_ROOT, "annotations", "instances_val2017.json")
    
    # SAM权重路径（ViT-B）
    SAM_CHECKPOINT_PATH = "E:\\SAM_Model\\weights\\sam_vit_b_01ec64.pth"
    MODEL_TYPE = "vit_b"
    
    # 原有结果保存路径
    RESULT_ROOT = "E:\\SAM_Model\\results"
    EXP1_ROOT = os.path.join(RESULT_ROOT, "coco_exp1")
    EXP1_IMG_PATH = os.path.join(EXP1_ROOT, "images", "val2017")
    EXP1_ANN_PATH = os.path.join(EXP1_ROOT, "annotations", "instances_val2017_1000.json")
    SAMPLE_LIST_PATH = os.path.join(EXP1_ROOT, "sample_list.txt")
    RESULT_CSV_PATH = os.path.join(EXP1_ROOT, "exp1_results_adaptive.csv")
    LOG_PATH = os.path.join(EXP1_ROOT, "exp1_log_adaptive.txt")
    
    # 每张图片独立结果保存目录（避免覆盖）
    PER_IMAGE_RESULT_ROOT = os.path.join(RESULT_ROOT, "per_image_results")
    PER_IMAGE_CSV_PATH = os.path.join(PER_IMAGE_RESULT_ROOT, "per_image_miou_results.csv")
    
    # 实验参数（强制CPU运行）
    DEVICE = "cpu"  # 直接指定为CPU，避免设备兼容问题
    RANDOM_SEED = 42
    SAMPLE_NUM = 1000
    IMG_TARGET_SIZE = 1024
    MASK_THRESHOLD = 0.5
    MIN_INSTANCE_AREA = 100
    # 自适应选点参数
    SIZE_QUANTILE = 0.25
    SHAPE_QUANTILE = 0.75
    IMG_COMPLEX_THRESHOLD = 0.5

# ====================== 2. 工具函数 ======================
def create_dirs():
    os.makedirs(Config.EXP1_IMG_PATH, exist_ok=True)
    os.makedirs(os.path.dirname(Config.EXP1_ANN_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(Config.RESULT_CSV_PATH), exist_ok=True)
    os.makedirs(Config.PER_IMAGE_RESULT_ROOT, exist_ok=True)

def init_per_image_csv():
    if not os.path.exists(Config.PER_IMAGE_CSV_PATH):
        with open(Config.PER_IMAGE_CSV_PATH, "w", encoding="utf-8", newline="") as f:
            fieldnames = [
                "img_id", "img_name", "img_width", "img_height",
                "total_valid_instances", "hard_instance_num", "hard_instance_ratio",
                "img_complexity",
                "Point-only_mIoU", "Point-only_improved_mIoU",
                "Box-only_mIoU", "Box+Point_improved_mIoU"
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
    print(f"每张图片结果将保存至：{Config.PER_IMAGE_CSV_PATH}（追加模式，不覆盖）")

def judge_img_complexity(ann_list, img_info, size_thresh, shape_thresh):
    img_w, img_h = img_info["width"], img_info["height"]
    total_valid = 0
    hard_instance_num = 0
    
    for ann in ann_list:
        if ann["area"] < Config.MIN_INSTANCE_AREA:
            continue
        total_valid += 1
        
        size_feat = get_size_feature(ann, img_w, img_h)
        mask = coco_exp1.annToMask(ann)
        shape_feat = get_shape_feature(mask)
        occlusion_feat = get_occlusion_feature(ann)
        instance_type = classify_instance(size_feat, shape_feat, occlusion_feat, size_thresh, shape_thresh)
        
        if instance_type == "hard":
            hard_instance_num += 1
    
    hard_ratio = hard_instance_num / total_valid if total_valid > 0 else 0.0
    img_complexity = "复杂图片" if hard_ratio >= Config.IMG_COMPLEX_THRESHOLD else "简单图片"
    return total_valid, hard_instance_num, hard_ratio, img_complexity

def append_per_image_result(img_data):
    with open(Config.PER_IMAGE_CSV_PATH, "a", encoding="utf-8", newline="") as f:
        fieldnames = [
            "img_id", "img_name", "img_width", "img_height",
            "total_valid_instances", "hard_instance_num", "hard_instance_ratio",
            "img_complexity",
            "Point-only_mIoU", "Point-only_improved_mIoU",
            "Box-only_mIoU", "Box+Point_improved_mIoU"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(img_data)

# ====================== 3. 样本筛选 ======================
def filter_coco_samples():
    random.seed(Config.RANDOM_SEED)
    np.random.seed(Config.RANDOM_SEED)
    
    if os.path.exists(Config.SAMPLE_LIST_PATH) and os.path.exists(Config.EXP1_ANN_PATH):
        print(f"检测到已有样本列表和标注，直接加载...")
        with open(Config.SAMPLE_LIST_PATH, "r", encoding="utf-8") as f:
            sample_lines = f.readlines()
        selected_img_ids = [int(line.split(",")[0]) for line in sample_lines]
        coco = COCO(Config.COCO_ANN_PATH)
        print(f"已加载 {len(selected_img_ids)} 张已有样本")
        return selected_img_ids, coco
    
    print(f"正在加载原始COCO标注：{Config.COCO_ANN_PATH}")
    coco = COCO(Config.COCO_ANN_PATH)
    img_ids = coco.getImgIds()
    print(f"原始COCO val2017共有 {len(img_ids)} 张图片")
    
    selected_img_ids = random.sample(img_ids, Config.SAMPLE_NUM)
    print(f"已抽取 {len(selected_img_ids)} 张图片，ID: {selected_img_ids[:5]}...")
    
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
    
    with open(Config.SAMPLE_LIST_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(sample_list))
    print(f"样本列表已保存至 {Config.SAMPLE_LIST_PATH}")
    
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

# ====================== 4. 实例复杂度与选点函数 ======================
def get_mask_centroid(mask):
    y_coords, x_coords = np.where(mask == 1)
    if len(x_coords) == 0 or len(y_coords) == 0:
        return None
    centroid_x = int(np.mean(x_coords))
    centroid_y = int(np.mean(y_coords))
    return (centroid_x, centroid_y)

def get_size_feature(ann, img_w, img_h):
    return ann['area'] / (img_w * img_h)

def get_shape_feature(mask):
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
    return ann.get('iscrowd', 0)

def calculate_feature_thresholds(coco_exp1):
    print("正在计算实例复杂度特征阈值...")
    all_anns = coco_exp1.loadAnns(coco_exp1.getAnnIds())
    size_features = []
    shape_features = []
    
    for ann in all_anns:
        img_info = coco_exp1.loadImgs(ann['image_id'])[0]
        img_w, img_h = img_info['width'], img_info['height']
        size_feat = get_size_feature(ann, img_w, img_h)
        size_features.append(size_feat)
        
        mask = coco_exp1.annToMask(ann)
        shape_feat = get_shape_feature(mask)
        shape_features.append(shape_feat)
    
    size_thresh = np.quantile(size_features, Config.SIZE_QUANTILE)
    shape_thresh = np.quantile(shape_features, Config.SHAPE_QUANTILE)
    print(f"特征阈值计算完成：")
    print(f"  尺寸特征阈值：{size_thresh:.6f}")
    print(f"  形状特征阈值：{shape_thresh:.2f}")
    return size_thresh, shape_thresh

def classify_instance(size_feat, shape_feat, occlusion_feat, size_thresh, shape_thresh):
    if (size_feat < size_thresh) or (shape_feat > shape_thresh) or (occlusion_feat == 1):
        return "hard"
    else:
        return "easy"

def get_core_region_points(mask):
    mask_uint8 = (mask * 255).astype(np.uint8)
    dist_transform = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, 5)
    if dist_transform.max() == 0:
        centroid = get_mask_centroid(mask)
        return np.array([centroid], dtype=np.float32) if centroid else None
    core_threshold = dist_transform.max() * 0.5
    core_mask = (dist_transform > core_threshold).astype(np.uint8)
    core_centroid = get_mask_centroid(core_mask)
    if not core_centroid:
        core_centroid = get_mask_centroid(mask)
    return np.array([core_centroid], dtype=np.float32) if core_centroid else None

def get_multi_core_points(mask, max_points=2):
    mask_uint8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        centroid = get_mask_centroid(mask)
        return np.array([centroid] * min(1, max_points), dtype=np.float32) if centroid else None
    contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)[:max_points]
    multi_centroids = []
    for cnt in contours_sorted:
        M = cv2.moments(cnt)
        if M['m00'] == 0:
            continue
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        multi_centroids.append((cx, cy))
    if len(multi_centroids) < max_points:
        global_centroid = get_mask_centroid(mask)
        if global_centroid:
            multi_centroids.append(global_centroid)
    multi_centroids = list(set(multi_centroids))[:max_points]
    return np.array(multi_centroids, dtype=np.float32) if multi_centroids else None

def get_high_response_points(predictor, mask, img_w, img_h, top_k=2):
    img_embedding = predictor.get_image_embedding().numpy()  # CPU下直接转numpy
    embedding_h, embedding_w = img_embedding.shape[2], img_embedding.shape[3]
    mask_emb_size = cv2.resize(mask.astype(np.uint8), (embedding_w, embedding_h), interpolation=cv2.INTER_NEAREST)
    feat_mean = np.mean(img_embedding[0], axis=0)
    mask_feat = feat_mean * mask_emb_size
    flat_feat = mask_feat.flatten()
    top_indices = np.argsort(flat_feat)[-top_k:]
    y_emb = (top_indices // embedding_w).astype(np.int32)
    x_emb = (top_indices % embedding_w).astype(np.int32)
    scale_x = img_w / embedding_w
    scale_y = img_h / embedding_h
    x_ori = (x_emb * scale_x).astype(np.int32)
    y_ori = (y_emb * scale_y).astype(np.int32)
    high_response_points = [(x, y) for x, y in zip(x_ori, y_ori)]
    return np.array(high_response_points, dtype=np.float32) if high_response_points else None

def generate_negative_point(img_w, img_h, bbox):
    x1, y1, x2, y2 = bbox
    for _ in range(10):
        px = random.randint(0, img_w-1)
        py = random.randint(0, img_h-1)
        if not (x1 <= px <= x2 and y1 <= py <= y2):
            return (px, py)
    return (0, 0)

def build_prompts(img_info, ann, coco, predictor, size_thresh, shape_thresh):
    img_w, img_h = img_info["width"], img_info["height"]
    bbox = ann["bbox"]
    mask = coco.annToMask(ann)
    instance_area = ann["area"]
    
    if instance_area < Config.MIN_INSTANCE_AREA:
        return None, None, None, None
    
    x, y, w, h = bbox
    box_xyxy = np.array([[x, y, x + w, y + h]], dtype=np.float32)
    
    centroid = get_mask_centroid(mask)
    if centroid is None:
        return None, None, None, None
    pos_x, pos_y = centroid
    neg_x, neg_y = generate_negative_point(img_w, img_h, box_xyxy[0])
    original_point_prompt = (np.array([[pos_x, pos_y], [neg_x, neg_y]], dtype=np.float32), np.array([1, 0], dtype=np.int32))
    
    size_feat = get_size_feature(ann, img_w, img_h)
    shape_feat = get_shape_feature(mask)
    occlusion_feat = get_occlusion_feature(ann)
    instance_type = classify_instance(size_feat, shape_feat, occlusion_feat, size_thresh, shape_thresh)
    
    if instance_type == "easy":
        adaptive_points = get_core_region_points(mask)
    else:
        multi_core_pts = get_multi_core_points(mask, max_points=2)
        high_response_pts = get_high_response_points(predictor, mask, img_w, img_h, top_k=2)
        adaptive_points_list = []
        if multi_core_pts is not None:
            adaptive_points_list.extend(multi_core_pts.tolist())
        if high_response_pts is not None:
            adaptive_points_list.extend(high_response_pts.tolist())
        adaptive_points_list = list(set(tuple(pt) for pt in adaptive_points_list))[:2]
        adaptive_points = np.array(adaptive_points_list, dtype=np.float32) if adaptive_points_list else None
    
    if adaptive_points is None or len(adaptive_points) == 0:
        adaptive_points = np.array([[pos_x, pos_y]], dtype=np.float32)
    adaptive_labels = np.array([1] * len(adaptive_points), dtype=np.int32)
    
    improved_point_prompt = (adaptive_points, adaptive_labels)
    mixed_prompt = (adaptive_points, adaptive_labels, box_xyxy)
    
    return original_point_prompt, improved_point_prompt, (box_xyxy,), mixed_prompt

# ====================== 5. IoU计算函数 ======================
def calculate_iou(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 0.0
    return intersection / union

def scale_mask(mask, target_h, target_w):
    mask = mask.astype(np.uint8) * 255
    scaled_mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    return (scaled_mask > Config.MASK_THRESHOLD * 255).astype(np.uint8)

# ====================== 6. 核心实验流程（纯CPU） ======================
def run_exp1():
    create_dirs()
    init_per_image_csv()
    config = Config()
    
    print("="*50)
    print("当前实验配置信息（纯CPU运行）")
    print("="*50)
    print(f"COCO数据集路径：{config.COCO_RAW_ROOT}")
    print(f"SAM权重路径：{config.SAM_CHECKPOINT_PATH}")
    print(f"运行设备：{config.DEVICE}（已强制CPU，无兼容问题）")
    print(f"抽取样本数：{config.SAMPLE_NUM}")
    print("="*50 + "\n")
    
    selected_img_ids, coco_original = filter_coco_samples()
    
    print(f"\n正在加载专属标注文件：{config.EXP1_ANN_PATH}")
    global coco_exp1
    coco_exp1 = COCO(config.EXP1_ANN_PATH)
    print("专属标注加载完成！")
    
    size_thresh, shape_thresh = calculate_feature_thresholds(coco_exp1)
    
    print(f"\n正在加载SAM模型：{config.MODEL_TYPE}")
    print(f"权重路径：{config.SAM_CHECKPOINT_PATH}")
    sam = sam_model_registry[config.MODEL_TYPE](checkpoint=config.SAM_CHECKPOINT_PATH)
    sam.to(device=config.DEVICE)  # 模型移到CPU
    sam.eval()
    predictor = SamPredictor(sam)
    print("SAM模型加载完成，已进入评估模式\n")
    
    prompt_types = ["Point-only", "Point-only_improved", "Box-only", "Box+Point_improved"]
    result_dict = {pt: {"img_miou_list": [], "instance_iou_list": []} for pt in prompt_types}
    log_content = []
    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    log_content.append(f"实验1：SAM不同提示方式性能对比（纯CPU运行）")
    log_content.append(f"配置：样本数={config.SAMPLE_NUM}, 模型={config.MODEL_TYPE}")
    log_content.append(f"实例复杂度阈值：尺寸={size_thresh:.6f}, 形状={shape_thresh:.2f}")
    log_content.append(f"开始时间：{start_time}")
    print("\n".join(log_content))
    print("\n即将开始逐图推理...")
    
    total_img = len(selected_img_ids)
    for idx, img_id in enumerate(selected_img_ids):
        img_info = coco_exp1.loadImgs(img_id)[0]
        img_name = img_info["file_name"]
        img_path = os.path.join(config.EXP1_IMG_PATH, img_name)
        img_w, img_h = img_info["width"], img_info["height"]
        ann_ids = coco_exp1.getAnnIds(imgIds=img_id)
        anns = coco_exp1.loadAnns(ann_ids)
        
        image = cv2.imread(img_path)
        if image is None:
            print(f"警告：无法读取图片 {img_path}，已跳过")
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image_rgb)
        
        img_instance_iou = {pt: [] for pt in prompt_types}
        
        for ann in anns:
            original_point_prompt, improved_point_prompt, box_prompt, mixed_prompt = build_prompts(
                img_info, ann, coco_exp1, predictor, size_thresh, shape_thresh
            )
            if original_point_prompt is None:
                continue
            gt_mask = coco_exp1.annToMask(ann)
            
            # （1）原始Point-only推理（纯CPU）
            points, point_labels = original_point_prompt
            with torch.no_grad():
                masks, scores, _ = predictor.predict(
                    point_coords=points,
                    point_labels=point_labels,
                    multimask_output=True
                )
            best_mask_idx = np.argmax(scores)
            pred_mask_point = masks[best_mask_idx].numpy()  # CPU张量直接转numpy
            pred_mask_point_scaled = scale_mask(pred_mask_point, img_h, img_w)
            iou_point = calculate_iou(pred_mask_point_scaled, gt_mask)
            img_instance_iou["Point-only"].append(iou_point)
            result_dict["Point-only"]["instance_iou_list"].append(iou_point)
            
            # （2）Point-only改进版推理
            improved_points, improved_labels = improved_point_prompt
            with torch.no_grad():
                masks, scores, _ = predictor.predict(
                    point_coords=improved_points,
                    point_labels=improved_labels,
                    multimask_output=True
                )
            best_mask_idx = np.argmax(scores)
            pred_mask_point_improved = masks[best_mask_idx].numpy()
            pred_mask_point_improved_scaled = scale_mask(pred_mask_point_improved, img_h, img_w)
            iou_point_improved = calculate_iou(pred_mask_point_improved_scaled, gt_mask)
            img_instance_iou["Point-only_improved"].append(iou_point_improved)
            result_dict["Point-only_improved"]["instance_iou_list"].append(iou_point_improved)
            
            # （3）原始Box-only推理
            box_xyxy = box_prompt[0]
            with torch.no_grad():
                masks, scores, _ = predictor.predict(
                    box=box_xyxy,
                    multimask_output=True
                )
            best_mask_idx = np.argmax(scores)
            pred_mask_box = masks[best_mask_idx].numpy()
            pred_mask_box_scaled = scale_mask(pred_mask_box, img_h, img_w)
            iou_box = calculate_iou(pred_mask_box_scaled, gt_mask)
            img_instance_iou["Box-only"].append(iou_box)
            result_dict["Box-only"]["instance_iou_list"].append(iou_box)
            
            # （4）Box+Point改进版推理
            adaptive_points, adaptive_labels, mixed_box = mixed_prompt
            with torch.no_grad():
                masks, scores, _ = predictor.predict(
                    point_coords=adaptive_points,
                    point_labels=adaptive_labels,
                    box=mixed_box,
                    multimask_output=True
                )
            best_mask_idx = np.argmax(scores)
            pred_mask_mixed = masks[best_mask_idx].numpy()
            pred_mask_mixed_scaled = scale_mask(pred_mask_mixed, img_h, img_w)
            iou_mixed = calculate_iou(pred_mask_mixed_scaled, gt_mask)
            img_instance_iou["Box+Point_improved"].append(iou_mixed)
            result_dict["Box+Point_improved"]["instance_iou_list"].append(iou_mixed)
        
        # 计算单张图片mIoU
        current_img_miou = {}
        for pt in prompt_types:
            if len(img_instance_iou[pt]) > 0:
                img_miou = np.mean(img_instance_iou[pt])
                result_dict[pt]["img_miou_list"].append(img_miou)
                current_img_miou[pt] = img_miou
            else:
                current_img_miou[pt] = 0.0
        
        # 判断图片复杂度
        total_valid, hard_num, hard_ratio, img_complexity = judge_img_complexity(
            anns, img_info, size_thresh, shape_thresh
        )
        
        # 保存单张图片结果
        img_result_data = {
            "img_id": img_id,
            "img_name": img_name,
            "img_width": img_w,
            "img_height": img_h,
            "total_valid_instances": total_valid,
            "hard_instance_num": hard_num,
            "hard_instance_ratio": round(hard_ratio, 4),
            "img_complexity": img_complexity,
            "Point-only_mIoU": round(current_img_miou["Point-only"], 4),
            "Point-only_improved_mIoU": round(current_img_miou["Point-only_improved"], 4),
            "Box-only_mIoU": round(current_img_miou["Box-only"], 4),
            "Box+Point_improved_mIoU": round(current_img_miou["Box+Point_improved"], 4)
        }
        append_per_image_result(img_result_data)
        
        # 打印进度
        if (idx + 1) % 10 == 0 or (idx + 1) == total_img:
            print(f"推理进度：{idx+1}/{total_img} | 图片：{img_name}")
            print(f"  图片复杂度：{img_complexity}（难例占比：{hard_ratio:.4f}）")
            print(f"  Point-only mIoU：{current_img_miou['Point-only']:.4f}")
            print(f"  Point-only（改进版）mIoU：{current_img_miou['Point-only_improved']:.4f}")
            print(f"  Box-only mIoU：{current_img_miou['Box-only']:.4f}")
            print(f"  Box+Point（改进版）mIoU：{current_img_miou['Box+Point_improved']:.4f}")
            print("-"*50)
    
    # 计算最终结果
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
    
    # 保存汇总结果
    with open(config.RESULT_CSV_PATH, "w", encoding="utf-8", newline="") as f:
        fieldnames = ["提示方式", "实例数量", "图像数量", "实例级mIoU", "图像级mIoU"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final_results)
    print(f"\n核心汇总表格已保存至：{config.RESULT_CSV_PATH}")
    
    # 保存日志
    with open(config.LOG_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(log_content))
        f.write(f"\n结束时间：{end_time}")
    print(f"详细实验日志已保存至：{config.LOG_PATH}")
    print(f"每张图片独立结果已保存至：{config.PER_IMAGE_CSV_PATH}")
    
    # 打印最终结果
    print("\n" + "="*80)
    print("实验1 最终结果对比（纯CPU运行）")
    print("="*80)
    for res in final_results:
        print(f"{res['提示方式']:20s} | 实例级mIoU：{res['实例级mIoU']:.4f} | 图像级mIoU：{res['图像级mIoU']:.4f}")
    print("="*80)
    adaptive_instance_miou = final_results[3]["实例级mIoU"]
    box_only_instance_miou = final_results[2]["实例级mIoU"]
    if adaptive_instance_miou > box_only_instance_miou:
        print(f"实验结论：Box+Point（改进版）性能超越Box-only，提升 {adaptive_instance_miou - box_only_instance_miou:.4f}")
    else:
        print(f"实验结论：Box+Point（改进版）性能略低于Box-only，差异 {box_only_instance_miou - adaptive_instance_miou:.4f}")

# ====================== 运行实验 ======================
if __name__ == "__main__":
    try:
        from segment_anything import sam_model_registry
    except ImportError:
        print("错误：未找到segment-anything库，请先克隆官方仓库并添加到Python环境！")
        print("克隆命令：git clone https://github.com/facebookresearch/segment-anything.git")
        exit(1)
    
    print("已强制设置为纯CPU运行，将避免所有设备兼容错误")
    run_exp1()