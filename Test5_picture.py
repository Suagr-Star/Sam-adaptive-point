import os
import json
import random
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from segment_anything import sam_model_registry, SamPredictor

# ====================== 1. 配置参数（关联已有1000张样本） ======================
class Config:
    # 关联已有实验数据（关键：读取之前1000张样本的表格）
    EXIST_TABLE_PATH = "E:\\SAM_Model\\results\\coco_exp_new_1000\\point_adaptive_complex_table.csv"
    EXIST_EXP_ROOT = "E:\\SAM_Model\\results\\coco_exp_new_1000"
    EXIST_IMG_PATH = os.path.join(EXIST_EXP_ROOT, "images", "val2017")
    EXIST_ANN_PATH = os.path.join(EXIST_EXP_ROOT, "annotations", "instances_val2017_new_1000.json")
    
    # SAM模型路径（与之前一致，保证结果统一）
    SAM_CHECKPOINT_PATH = "E:\\SAM_Model\\weights\\sam_vit_b_01ec64.pth"
    MODEL_TYPE = "vit_b"
    
    # 对比图输出配置
    VIS_ROOT = "E:\\SAM_Model\\results\\coco_exp_new_1000\\visualization_100"
    VIS_IMG_PATH = os.path.join(VIS_ROOT, "comparison_imgs")  # 对比图保存路径
    SELECTED_100_LIST = os.path.join(VIS_ROOT, "selected_100_samples.txt")  # 筛选的100张样本列表
    
    # 筛选参数
    SELECT_NUM = 100  # 筛选100张
    SIMPLE_RATIO = 0.5  # 简单图片占比50%（50张），复杂图片50%（50张）
    # 对比图参数
    FIG_SIZE = (16, 4)  # 4个子图（原图+GT+Point+Adaptive）
    FONT_SIZE = 11
    DPI = 300  # 高清，适配论文印刷
    MASK_THRESHOLD = 0.5
    MIN_INSTANCE_AREA = 100
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    RANDOM_SEED = 42

# ====================== 2. 核心工具函数 ======================
def create_dirs():
    os.makedirs(Config.VIS_IMG_PATH, exist_ok=True)
    print(f"100张对比图将保存至：{Config.VIS_IMG_PATH}")

def load_exist_data():
    """加载已有1000张样本的表格数据，返回DataFrame"""
    if not os.path.exists(Config.EXIST_TABLE_PATH):
        raise FileNotFoundError(f"未找到已有样本表格：{Config.EXIST_TABLE_PATH}")
    df = pd.read_csv(Config.EXIST_TABLE_PATH)
    print(f"已加载{len(df)}张已有样本数据")
    # 过滤无效样本（mIoU=0且两种方式均失效的）
    df = df[(df["Point-only_mIoU"] > 0) | (df["Point(Adaptive)_mIoU"] > 0)]
    print(f"过滤无效样本后剩余：{len(df)}张")
    return df

def select_100_samples(df):
    """按规则筛选100张样本：场景均衡+提升幅度全覆盖"""
    random.seed(Config.RANDOM_SEED)
    np.random.seed(Config.RANDOM_SEED)
    
    # 按简单/复杂分类
    df_simple = df[df["is_complex_image"] == False].reset_index(drop=True)
    df_complex = df[df["is_complex_image"] == True].reset_index(drop=True)
    
    # 计算需筛选的简单/复杂图片数量
    simple_num = int(Config.SELECT_NUM * Config.SIMPLE_RATIO)
    complex_num = Config.SELECT_NUM - simple_num
    print(f"筛选规则：简单图片{simple_num}张 + 复杂图片{complex_num}张")
    
    # 按提升幅度分层筛选（确保覆盖不同效果）
    def select_by_improvement(sub_df, select_num, label):
        # 计算提升幅度
        sub_df["improvement"] = sub_df["Point(Adaptive)_mIoU"] - sub_df["Point-only_mIoU"]
        # 分3层：大幅提升（>0.1）、小幅提升（0~0.1）、轻微下降（<0）
        df_large = sub_df[sub_df["improvement"] > 0.1].reset_index(drop=True)
        df_small = sub_df[(sub_df["improvement"] >= 0) & (sub_df["improvement"] <= 0.1)].reset_index(drop=True)
        df_drop = sub_df[sub_df["improvement"] < 0].reset_index(drop=True)
        
        # 按比例分配名额（确保每层都有样本，体现全面性）
        layer_ratio = [0.4, 0.4, 0.2]  # 大幅:小幅:下降
        num_large = max(1, int(select_num * layer_ratio[0]))
        num_small = max(1, int(select_num * layer_ratio[1]))
        num_drop = max(1, select_num - num_large - num_small)
        
        # 采样（不足则从其他层补充）
        selected = []
        if len(df_large) >= num_large:
            selected.extend(df_large.sample(num_large, random_state=Config.RANDOM_SEED).to_dict("records"))
        else:
            selected.extend(df_large.to_dict("records"))
            num_small += num_large - len(df_large)
        
        if len(df_small) >= num_small:
            selected.extend(df_small.sample(num_small, random_state=Config.RANDOM_SEED).to_dict("records"))
        else:
            selected.extend(df_small.to_dict("records"))
            num_drop += num_small - len(df_small)
        
        if len(df_drop) >= num_drop:
            selected.extend(df_drop.sample(num_drop, random_state=Config.RANDOM_SEED).to_dict("records"))
        else:
            selected.extend(df_drop.to_dict("records"))
        
        # 确保数量达标
        if len(selected) > select_num:
            selected = selected[:select_num]
        print(f"{label}筛选完成：{len(selected)}张（大幅提升{num_large}张+小幅提升{num_small}张+轻微下降{num_drop}张）")
        return selected
    
    # 分别筛选简单/复杂图片
    selected_simple = select_by_improvement(df_simple, simple_num, "简单图片")
    selected_complex = select_by_improvement(df_complex, complex_num, "复杂图片")
    
    # 合并并打乱顺序
    selected_all = selected_simple + selected_complex
    random.shuffle(selected_all)
    
    # 保存筛选列表
    with open(Config.SELECTED_100_LIST, "w", encoding="utf-8") as f:
        for item in selected_all:
            f.write(f"{item['img_id']},{item['img_name']},{item['complex_label']},{item['improvement']:.4f}\n")
    print(f"100张筛选样本列表已保存至：{Config.SELECTED_100_LIST}")
    return selected_all

def draw_comparison_img(img_rgb, gt_mask, point_pred_mask, adaptive_pred_mask, miou_dict, img_name, complex_label):
    """生成对比图：原图 + GT掩码 + Point-only预测 + Point(Adaptive)预测"""
    fig, axes = plt.subplots(1, 4, figsize=Config.FIG_SIZE)
    fig.suptitle(f"Sample: {img_name} | {complex_label} | Improvement: {miou_dict['improvement']:.4f}", fontsize=Config.FONT_SIZE+1)
    
    # 1. 原图
    axes[0].imshow(img_rgb)
    axes[0].set_title("Original Image", fontsize=Config.FONT_SIZE)
    axes[0].axis("off")
    
    # 2. GT掩码（白色背景+前景显原图纹理）
    gt_vis = np.ones_like(img_rgb) * 255
    gt_vis[gt_mask == 1] = img_rgb[gt_mask == 1]
    axes[1].imshow(gt_vis)
    axes[1].set_title("GT Mask", fontsize=Config.FONT_SIZE)
    axes[1].axis("off")
    
    # 3. Point-only预测
    point_vis = np.ones_like(img_rgb) * 255
    point_vis[point_pred_mask == 1] = img_rgb[point_pred_mask == 1]
    axes[2].imshow(point_vis)
    axes[2].set_title(f"Point-only\nmIoU: {miou_dict['point_only']:.4f}", fontsize=Config.FONT_SIZE)
    axes[2].axis("off")
    
    # 4. Point(Adaptive)预测
    adaptive_vis = np.ones_like(img_rgb) * 255
    adaptive_vis[adaptive_pred_mask == 1] = img_rgb[adaptive_pred_mask == 1]
    axes[3].imshow(adaptive_vis)
    axes[3].set_title(f"Point(Adaptive)\nmIoU: {miou_dict['point_adaptive']:.4f}", fontsize=Config.FONT_SIZE)
    axes[3].axis("off")
    
    # 调整间距，无白边保存
    plt.tight_layout()
    save_path = os.path.join(Config.VIS_IMG_PATH, f"{img_name.split('.')[0]}_comparison.png")
    plt.savefig(save_path, dpi=Config.DPI, bbox_inches="tight", pad_inches=0.05)
    plt.close()
    return save_path

# ====================== 3. 复用优化后的选点与推理函数 ======================
def get_mask_centroid(mask):
    y_coords, x_coords = np.where(mask == 1)
    if len(x_coords) == 0 or len(y_coords) == 0:
        return None
    centroid_x = int(np.round(np.mean(x_coords)))
    centroid_y = int(np.round(np.mean(y_coords)))
    return (centroid_x, centroid_y)

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

def classify_instance(size_feat, shape_feat, occlusion_feat, size_thresh, shape_thresh):
    if (size_feat < size_thresh) or (shape_feat > shape_thresh) or (occlusion_feat == 1):
        return "hard"
    else:
        return "easy"

def get_core_region_points_optimized(mask):
    mask_uint8 = (mask * 255).astype(np.uint8)
    mask_h, mask_w = mask.shape
    if np.sum(mask) < Config.MIN_INSTANCE_AREA:
        centroid = get_mask_centroid(mask)
        return np.array([centroid], dtype=np.float32) if centroid else None
    dist_transform = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, 5)
    if dist_transform.max() < 5:
        centroid = get_mask_centroid(mask)
        return np.array([centroid], dtype=np.float32) if centroid else None
    core_threshold = dist_transform.max() * 0.6
    core_mask = (dist_transform > core_threshold).astype(np.uint8)
    core_centroid = get_mask_centroid(core_mask)
    if core_centroid:
        cx, cy = core_centroid
        if 0 <= cx < mask_w and 0 <= cy < mask_h:
            return np.array([[cx, cy]], dtype=np.float32)
    centroid = get_mask_centroid(mask)
    return np.array([centroid], dtype=np.float32) if centroid else None

def get_multi_core_points_optimized(mask):
    mask_uint8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        centroid = get_mask_centroid(mask)
        return np.array([centroid], dtype=np.float32) if centroid else None
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 50]
    if not valid_contours:
        centroid = get_mask_centroid(mask)
        return np.array([centroid], dtype=np.float32) if centroid else None
    contours_sorted = sorted(valid_contours, key=cv2.contourArea, reverse=True)[:3]
    multi_centroids = []
    for cnt in contours_sorted:
        M = cv2.moments(cnt)
        if M['m00'] == 0:
            continue
        cx = int(np.round(M['m10'] / M['m00']))
        cy = int(np.round(M['m01'] / M['m00']))
        mask_h, mask_w = mask.shape
        if 0 <= cx < mask_w and 0 <= cy < mask_h:
            multi_centroids.append((cx, cy))
    if len(multi_centroids) < 3:
        global_centroid = get_mask_centroid(mask)
        if global_centroid and global_centroid not in multi_centroids:
            multi_centroids.append(global_centroid)
    multi_centroids = list(set(multi_centroids))[:3]
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
    high_response_points = []
    for x, y in zip(x_ori, y_ori):
        if 0 <= x < img_w and 0 <= y < img_h:
            high_response_points.append((int(np.round(x)), int(np.round(y))))
    return np.array(high_response_points, dtype=np.float32) if high_response_points else None

def generate_negative_point_optimized(img_w, img_h, bbox):
    x1, y1, x2, y2 = bbox
    target_center = ((x1+x2)/2, (y1+y2)/2)
    for _ in range(20):
        px = random.randint(0, img_w-1)
        py = random.randint(0, img_h-1)
        if not (x1 <= px <= x2 and y1 <= py <= y2):
            dist = np.sqrt((px - target_center[0])**2 + (py - target_center[1])**2)
            if dist >= 50:
                return (px, py)
    return (0, 0) if (0,0) not in bbox else (img_w-1, img_h-1)

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
    neg_x, neg_y = generate_negative_point_optimized(img_w, img_h, bbox)
    original_points = np.array([[pos_x, pos_y], [neg_x, neg_y]], dtype=np.float32)
    original_point_labels = np.array([1, 0], dtype=np.int32)
    size_feat = get_size_feature(ann, img_w, img_h)
    shape_feat = get_shape_feature(mask)
    occlusion_feat = get_occlusion_feature(ann)
    instance_type = classify_instance(size_feat, shape_feat, occlusion_feat, size_thresh, shape_thresh)
    if instance_type == "easy":
        adaptive_points = get_core_region_points_optimized(mask)
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
        adaptive_points_list = adaptive_points_list[:3]
        adaptive_points = np.array(adaptive_points_list, dtype=np.float32) if adaptive_points_list else None
    if adaptive_points is None or len(adaptive_points) == 0:
        adaptive_points = np.array([[pos_x, pos_y]], dtype=np.float32)
    adaptive_labels = np.array([1] * len(adaptive_points), dtype=np.int32)
    return (original_points, original_point_labels), (adaptive_points, adaptive_labels)

def calculate_iou(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    return intersection / union if union > 0 else 0.0

def scale_mask(mask, target_h, target_w):
    mask = mask.astype(np.uint8) * 255
    scaled_mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST_EXACT)
    return (scaled_mask > Config.MASK_THRESHOLD * 255).astype(np.uint8)

def calculate_feature_thresholds(coco):
    all_anns = coco.loadAnns(coco.getAnnIds())
    size_features = []
    shape_features = []
    for ann in all_anns:
        if ann["area"] < Config.MIN_INSTANCE_AREA:
            continue
        img_info = coco.loadImgs(ann['image_id'])[0]
        img_w, img_h = img_info['width'], img_info['height']
        size_feat = get_size_feature(ann, img_w, img_h)
        size_features.append(size_feat)
        mask = coco.annToMask(ann)
        shape_feat = get_shape_feature(mask)
        shape_features.append(shape_feat)
    size_thresh = np.quantile(size_features, 0.25) if size_features else 0.01
    shape_thresh = np.quantile(shape_features, 0.75) if shape_features else 1000.0
    return size_thresh, shape_thresh

# ====================== 4. 核心流程：筛选+生成对比图 ======================
def run_100_visualization():
    create_dirs()
    # 1. 加载已有数据并筛选100张
    df_exist = load_exist_data()
    selected_100 = select_100_samples(df_exist)
    
    # 2. 加载COCO标注和SAM模型
    print("\n加载COCO标注文件...")
    coco = COCO(Config.EXIST_ANN_PATH)
    size_thresh, shape_thresh = calculate_feature_thresholds(coco)
    print(f"特征阈值：尺寸={size_thresh:.6f}，形状={shape_thresh:.2f}")
    
    print("\n加载SAM模型...")
    sam = sam_model_registry[Config.MODEL_TYPE](checkpoint=Config.SAM_CHECKPOINT_PATH)
    sam.to(device=Config.DEVICE)
    sam.eval()
    predictor = SamPredictor(sam)
    
    # 3. 逐张生成对比图
    for idx, item in enumerate(selected_100):
        img_id = item["img_id"]
        img_name = item["img_name"]
        complex_label = item["complex_label"]
        point_only_miou = item["Point-only_mIoU"]
        adaptive_miou = item["Point(Adaptive)_mIoU"]
        improvement = item["improvement"]
        
        try:
            # 加载图片和标注
            img_info = coco.loadImgs(img_id)[0]
            img_path = os.path.join(Config.EXIST_IMG_PATH, img_name)
            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)
            
            # 读取图片
            image = cv2.imread(img_path)
            if image is None:
                print(f"警告：无法读取图片 {img_name}，跳过")
                continue
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            predictor.set_image(img_rgb)
            
            # 选择面积最大的有效实例（避免多实例混乱）
            valid_anns = [ann for ann in anns if ann["area"] >= Config.MIN_INSTANCE_AREA]
            if len(valid_anns) == 0:
                print(f"警告：图片 {img_name} 无有效实例，跳过")
                continue
            target_ann = max(valid_anns, key=lambda x: x["area"])
            gt_mask = coco.annToMask(target_ann)
            img_h, img_w = gt_mask.shape
            
            # 构建提示并推理
            point_only_data, adaptive_data = build_prompts_optimized(
                img_info, target_ann, coco, predictor, size_thresh, shape_thresh
            )
            if point_only_data is None:
                print(f"警告：图片 {img_name} 提示构建失败，跳过")
                continue
            
            # Point-only推理
            points, labels = point_only_data
            with torch.no_grad():
                if Config.DEVICE == "cuda":
                    with torch.cuda.amp.autocast():
                        masks, scores, _ = predictor.predict(point_coords=points, point_labels=labels, multimask_output=True)
                else:
                    masks, scores, _ = predictor.predict(point_coords=points, point_labels=labels, multimask_output=True)
            best_idx = np.argmax(scores)
            point_pred_mask = masks[best_idx].cpu().numpy() if torch.is_tensor(masks[best_idx]) else masks[best_idx]
            point_pred_mask_scaled = scale_mask(point_pred_mask, img_h, img_w)
            
            # Point(Adaptive)推理
            adaptive_points, adaptive_labels = adaptive_data
            with torch.no_grad():
                if Config.DEVICE == "cuda":
                    with torch.cuda.amp.autocast():
                        masks, scores, _ = predictor.predict(point_coords=adaptive_points, point_labels=adaptive_labels, multimask_output=True)
                else:
                    masks, scores, _ = predictor.predict(point_coords=adaptive_points, point_labels=adaptive_labels, multimask_output=True)
            best_idx = np.argmax(scores)
            adaptive_pred_mask = masks[best_idx].cpu().numpy() if torch.is_tensor(masks[best_idx]) else masks[best_idx]
            adaptive_pred_mask_scaled = scale_mask(adaptive_pred_mask, img_h, img_w)
            
            # 生成对比图
            miou_dict = {
                "point_only": point_only_miou,
                "point_adaptive": adaptive_miou,
                "improvement": improvement
            }
            save_path = draw_comparison_img(
                img_rgb, gt_mask, point_pred_mask_scaled, adaptive_pred_mask_scaled,
                miou_dict, img_name, complex_label
            )
            
            # 打印进度
            if (idx + 1) % 10 == 0 or (idx + 1) == Config.SELECT_NUM:
                print(f"进度：{idx+1}/{Config.SELECT_NUM} | 已生成：{save_path}")
            
            # 清空显存
            if Config.DEVICE == "cuda":
                torch.cuda.empty_cache()
        
        except Exception as e:
            print(f"处理图片 {img_name} 出错：{str(e)}，跳过")
            continue
    
    print(f"\n100张对比图生成完成！保存路径：{Config.VIS_IMG_PATH}")
    print(f"筛选样本列表：{Config.SELECTED_100_LIST}")

# ====================== 运行入口 ======================
if __name__ == "__main__":
    try:
        from segment_anything import sam_model_registry
    except ImportError:
        print("错误：未安装segment-anything库！")
        print("请执行：git clone https://github.com/facebookresearch/segment-anything.git")
        exit(1)
    
    run_100_visualization()