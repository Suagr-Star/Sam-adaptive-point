import os
import json
import random
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from segment_anything import sam_model_registry, SamPredictor

# ====================== 1. 配置参数（适配论文配图需求） ======================
class Config:
    # 基础路径（与原实验保持一致，确保数据兼容）
    COCO_RAW_ROOT = "E:\\SAM_Model\\datasets\\COCO"
    COCO_IMG_PATH = os.path.join(COCO_RAW_ROOT, "val2017")
    COCO_ANN_PATH = os.path.join(COCO_RAW_ROOT, "annotations", "instances_val2017.json")
    SAM_CHECKPOINT_PATH = "E:\\SAM_Model\\weights\\sam_vit_b_01ec64.pth"
    MODEL_TYPE = "vit_b"
    
    # 可视化专属配置（核心修改）
    VIS_SAMPLE_NUM = 100  # 仅选100张图生成配图
    VIS_ROOT = "E:\\SAM_Model\\paper_figures"  # 论文配图保存根目录
    VIS_IMG_SAVE_PATH = os.path.join(VIS_ROOT, "segmentation_comparisons")  # 对比图保存路径
    VIS_SAMPLE_LIST_PATH = os.path.join(VIS_ROOT, "vis_sample_list.txt")  # 100张样本列表（避免重复）
    
    # 实验参数（与原代码一致，保证结果可比）
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    RANDOM_SEED = 42
    MASK_THRESHOLD = 0.5
    MIN_INSTANCE_AREA = 100
    SIZE_QUANTILE = 0.25  # 实例复杂度分类阈值（复用原逻辑）
    SHAPE_QUANTILE = 0.75
    # 可视化参数
    FIG_SIZE = (15, 10)  # 对比图尺寸（宽x高），适配论文排版
    FONT_SIZE = 10       # 图中文字大小
    CMAP = "viridis"     # 掩码配色（避免黑白，论文更清晰）

# ====================== 2. 工具函数（可视化核心） ======================
def create_vis_dirs():
    """创建可视化结果保存目录"""
    os.makedirs(Config.VIS_IMG_SAVE_PATH, exist_ok=True)
    print(f"论文配图将保存至：{Config.VIS_IMG_SAVE_PATH}")

def filter_100_coco_samples():
    """筛选100张COCO样本（确保有实例，避免空图），保存样本列表"""
    random.seed(Config.RANDOM_SEED)
    np.random.seed(Config.RANDOM_SEED)
    
    # 若已存在100张样本列表，直接加载
    if os.path.exists(Config.VIS_SAMPLE_LIST_PATH):
        with open(Config.VIS_SAMPLE_LIST_PATH, "r", encoding="utf-8") as f:
            selected_img_ids = [int(line.strip()) for line in f.readlines()]
        print(f"已加载100张可视化样本，ID前5个：{selected_img_ids[:5]}...")
        return selected_img_ids
    
    # 重新筛选（确保每张图至少有1个有效实例）
    coco = COCO(Config.COCO_ANN_PATH)
    all_img_ids = coco.getImgIds()
    selected_img_ids = []
    
    print(f"正在筛选100张含有效实例的COCO样本...")
    for img_id in all_img_ids:
        # 检查该图是否有有效实例（面积≥MIN_INSTANCE_AREA）
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        valid_anns = [ann for ann in anns if ann["area"] >= Config.MIN_INSTANCE_AREA]
        
        if len(valid_anns) > 0:
            selected_img_ids.append(img_id)
            # 满100张停止
            if len(selected_img_ids) == Config.VIS_SAMPLE_NUM:
                break
    
    # 保存100张样本列表
    with open(Config.VIS_SAMPLE_LIST_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(map(str, selected_img_ids)))
    print(f"100张样本筛选完成，保存至：{Config.VIS_SAMPLE_LIST_PATH}")
    return selected_img_ids

def draw_segmentation_comparison(img, gt_mask, pred_masks_dict, miou_dict, img_name):
    """
    绘制分割结果对比图：1行7列（原图 + GT掩码 + 5种提示预测掩码）
    img: 原始图像（RGB）
    gt_mask: Ground Truth掩码（单通道，0=背景，1=前景）
    pred_masks_dict: 预测掩码字典，key=提示方式，value=预测掩码（单通道）
    miou_dict: mIoU字典，key=提示方式，value=mIoU值（保留3位小数）
    img_name: 图像文件名（用于保存）
    """
    # 初始化画布（1行7列，尺寸15x10）
    fig, axes = plt.subplots(1, 7, figsize=Config.FIG_SIZE)
    fig.suptitle(f"COCO Sample: {img_name} | Segmentation Comparison", fontsize=Config.FONT_SIZE + 2)
    
    # 1. 绘制原图
    axes[0].imshow(img)
    axes[0].set_title("Original Image", fontsize=Config.FONT_SIZE)
    axes[0].axis("off")  # 隐藏坐标轴（论文图更简洁）
    
    # 2. 绘制GT掩码（叠加半透明到白色背景）
    gt_vis = np.ones_like(img) * 255  # 白色背景
    gt_vis[gt_mask == 1] = img[gt_mask == 1]  # 前景区域显示原图纹理
    axes[1].imshow(gt_vis)
    axes[1].set_title("GT Mask", fontsize=Config.FONT_SIZE)
    axes[1].axis("off")
    
    # 3. 绘制5种提示方式的预测结果（按固定顺序）
    prompt_order = ["Point-only", "Point(Adaptive)", "Box-only", "Box-Point(Original)", "Box-Point(Adaptive)"]
    for idx, prompt in enumerate(prompt_order):
        pred_mask = pred_masks_dict[prompt]
        miou = miou_dict[prompt]
        
        # 预测掩码可视化（同GT风格：前景显原图，背景白色）
        pred_vis = np.ones_like(img) * 255
        pred_vis[pred_mask == 1] = img[pred_mask == 1]
        
        # 绘制并标注mIoU
        axes[2 + idx].imshow(pred_vis)
        axes[2 + idx].set_title(f"{prompt}\nmIoU: {miou:.3f}", fontsize=Config.FONT_SIZE)
        axes[2 + idx].axis("off")
    
    # 调整子图间距（避免重叠）
    plt.tight_layout()
    
    # 保存图片（PNG格式，高清，无白边）
    save_path = os.path.join(Config.VIS_IMG_SAVE_PATH, f"{img_name}_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")  # dpi=300适合论文印刷
    plt.close(fig)  # 关闭画布，释放内存
    print(f"对比图保存完成：{save_path}")

# ====================== 3. 复用原实验核心函数（选点+推理+IoU计算） ======================
def get_mask_centroid(mask):
    """计算掩码质心（复用原逻辑）"""
    y_coords, x_coords = np.where(mask == 1)
    if len(x_coords) == 0 or len(y_coords) == 0:
        return None
    centroid_x = int(np.mean(x_coords))
    centroid_y = int(np.mean(y_coords))
    return (centroid_x, centroid_y)

def get_size_feature(ann, img_w, img_h):
    """尺寸特征（复用原逻辑）"""
    return ann['area'] / (img_w * img_h)

def get_shape_feature(mask):
    """形状特征（复用原逻辑）"""
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
    """遮挡特征（复用原逻辑）"""
    return ann.get('iscrowd', 0)

def classify_instance(size_feat, shape_feat, occlusion_feat, size_thresh, shape_thresh):
    """易例/难例分类（复用原逻辑）"""
    if (size_feat < size_thresh) or (shape_feat > shape_thresh) or (occlusion_feat == 1):
        return "hard"
    else:
        return "easy"

def get_core_region_points(mask):
    """易例选点（复用原逻辑）"""
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
    """难例选点（复用原逻辑）"""
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
    """难例高响应点（复用原逻辑）"""
    img_embedding = predictor.get_image_embedding()
    if torch.is_tensor(img_embedding):
        img_embedding = img_embedding.cpu().numpy()
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
    """生成负点（复用原逻辑）"""
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]
    for _ in range(10):
        px = random.randint(0, img_w-1)
        py = random.randint(0, img_h-1)
        if not (x1 <= px <= x2 and y1 <= py <= y2):
            return (px, py)
    return (0, 0)

def build_prompts(img_info, ann, coco, predictor, size_thresh, shape_thresh):
    """构建5种提示方式的输入（复用原逻辑）"""
    img_w, img_h = img_info["width"], img_info["height"]
    bbox = ann["bbox"]
    mask = coco.annToMask(ann)
    instance_area = ann["area"]
    
    if instance_area < Config.MIN_INSTANCE_AREA:
        return None
    
    # 1. Point-only（原始正负点）
    centroid = get_mask_centroid(mask)
    if centroid is None:
        return None
    pos_x, pos_y = centroid
    neg_x, neg_y = generate_negative_point(img_w, img_h, bbox)
    point_only = (np.array([[pos_x, pos_y], [neg_x, neg_y]], dtype=np.float32), np.array([1, 0], dtype=np.int32))
    
    # 2. Point(Adaptive)（自适应点）
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
    point_adaptive = (adaptive_points, np.array([1] * len(adaptive_points), dtype=np.int32))
    
    # 3. Box-only（仅框）
    x, y, w, h = bbox
    box_only = (np.array([[x, y, x + w, y + h]], dtype=np.float32),)
    
    # 4. Box-Point(Original)（框+原始质心）
    box_point_original = (np.array([[pos_x, pos_y]], dtype=np.float32), np.array([1], dtype=np.int32), np.array([[x, y, x + w, y + h]], dtype=np.float32))
    
    # 5. Box-Point(Adaptive)（框+自适应点）
    box_point_adaptive = (adaptive_points, np.array([1] * len(adaptive_points), dtype=np.int32), np.array([[x, y, x + w, y + h]], dtype=np.float32))
    
    return {
        "Point-only": point_only,
        "Point(Adaptive)": point_adaptive,
        "Box-only": box_only,
        "Box-Point(Original)": box_point_original,
        "Box-Point(Adaptive)": box_point_adaptive
    }, mask  # 返回提示字典+GT掩码

def calculate_iou(pred_mask, gt_mask):
    """计算mIoU（复用原逻辑）"""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 0.0
    return intersection / union

def scale_mask(mask, target_h, target_w):
    """缩放掩码到原图尺寸（复用原逻辑）"""
    mask = mask.astype(np.uint8) * 255
    scaled_mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    return (scaled_mask > Config.MASK_THRESHOLD * 255).astype(np.uint8)

# ====================== 4. 核心可视化流程（100张图推理+绘图） ======================
def run_vis_pipeline():
    create_vis_dirs()
    config = Config()
    
    # 1. 筛选100张COCO样本
    selected_img_ids = filter_100_coco_samples()
    
    # 2. 加载COCO标注与SAM模型
    coco = COCO(config.COCO_ANN_PATH)
    print(f"\n加载SAM模型：{config.MODEL_TYPE}")
    sam = sam_model_registry[config.MODEL_TYPE](checkpoint=config.SAM_CHECKPOINT_PATH)
    sam.to(device=config.DEVICE)
    sam.eval()
    predictor = SamPredictor(sam)
    print("SAM模型加载完成，开始推理与可视化...")
    
    # 3. 计算实例复杂度阈值（复用原逻辑，确保分类标准一致）
    print(f"\n计算COCO实例复杂度阈值...")
    all_anns = coco.loadAnns(coco.getAnnIds())
    size_features = []
    shape_features = []
    for ann in all_anns[:10000]:  # 用1万实例计算阈值（平衡速度与准确性）
        img_info = coco.loadImgs(ann['image_id'])[0]
        img_w, img_h = img_info['width'], img_info['height']
        size_features.append(get_size_feature(ann, img_w, img_h))
        mask = coco.annToMask(ann)
        shape_features.append(get_shape_feature(mask))
    size_thresh = np.quantile(size_features, config.SIZE_QUANTILE)
    shape_thresh = np.quantile(shape_features, config.SHAPE_QUANTILE)
    print(f"阈值计算完成：尺寸={size_thresh:.6f}，形状={shape_thresh:.2f}")
    
    # 4. 逐张推理+可视化（100张）
    for idx, img_id in enumerate(selected_img_ids):
        img_info = coco.loadImgs(img_id)[0]
        img_name = img_info["file_name"]
        img_path = os.path.join(config.COCO_IMG_PATH, img_name)
        img_w, img_h = img_info["width"], img_info["height"]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        # 读取图像（转RGB）
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"警告：无法读取图像 {img_path}，跳过")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        predictor.set_image(img_rgb)  # SAM预处理图像
        
        # 选择1个有效实例进行可视化（避免单图多实例混乱，选面积最大的实例）
        valid_anns = [ann for ann in anns if ann["area"] >= config.MIN_INSTANCE_AREA]
        if len(valid_anns) == 0:
            print(f"警告：图像 {img_name} 无有效实例，跳过")
            continue
        target_ann = max(valid_anns, key=lambda x: x["area"])  # 选面积最大的实例
        
        # 构建提示+获取GT掩码
        prompts_dict, gt_mask = build_prompts(img_info, target_ann, coco, predictor, size_thresh, shape_thresh)
        if prompts_dict is None:
            print(f"警告：图像 {img_name} 提示构建失败，跳过")
            continue
        
        # 推理5种提示方式，保存预测掩码与mIoU
        pred_masks_dict = {}
        miou_dict = {}
        for prompt_name, prompt_data in prompts_dict.items():
            with torch.no_grad():
                if config.DEVICE == "cuda":
                    with torch.cuda.amp.autocast():
                        # 不同提示方式的SAM推理参数适配
                        if prompt_name == "Point-only":
                            points, labels = prompt_data
                            masks, scores, _ = predictor.predict(point_coords=points, point_labels=labels, multimask_output=True)
                        elif prompt_name == "Point(Adaptive)":
                            points, labels = prompt_data
                            masks, scores, _ = predictor.predict(point_coords=points, point_labels=labels, multimask_output=True)
                        elif prompt_name == "Box-only":
                            box, = prompt_data
                            masks, scores, _ = predictor.predict(box=box, multimask_output=True)
                        elif prompt_name in ["Box-Point(Original)", "Box-Point(Adaptive)"]:
                            points, labels, box = prompt_data
                            masks, scores, _ = predictor.predict(point_coords=points, point_labels=labels, box=box, multimask_output=True)
                        else:
                            continue
                else:
                    # CPU推理（参数与GPU一致）
                    if prompt_name == "Point-only":
                        points, labels = prompt_data
                        masks, scores, _ = predictor.predict(point_coords=points, point_labels=labels, multimask_output=True)
                    elif prompt_name == "Point(Adaptive)":
                        points, labels = prompt_data
                        masks, scores, _ = predictor.predict(point_coords=points, point_labels=labels, multimask_output=True)
                    elif prompt_name == "Box-only":
                        box, = prompt_data
                        masks, scores, _ = predictor.predict(box=box, multimask_output=True)
                    elif prompt_name in ["Box-Point(Original)", "Box-Point(Adaptive)"]:
                        points, labels, box = prompt_data
                        masks, scores, _ = predictor.predict(point_coords=points, point_labels=labels, box=box, multimask_output=True)
                    else:
                        continue
            
            # 取score最高的掩码，缩放至原图尺寸
            best_mask_idx = np.argmax(scores)
            pred_mask = masks[best_mask_idx].cpu().numpy() if torch.is_tensor(masks[best_mask_idx]) else masks[best_mask_idx]
            pred_mask_scaled = scale_mask(pred_mask, img_h, img_w)
            
            # 计算mIoU
            miou = calculate_iou(pred_mask_scaled, gt_mask)
            
            # 保存结果
            pred_masks_dict[prompt_name] = pred_mask_scaled
            miou_dict[prompt_name] = miou
        
        # 绘制并保存对比图
        draw_segmentation_comparison(img_rgb, gt_mask, pred_masks_dict, miou_dict, img_name.split(".")[0])
        
        # 进度打印
        print(f"可视化进度：{idx+1}/{config.VIS_SAMPLE_NUM} | 完成图像：{img_name}")
        
        # 每10张清空一次显存（避免GPU内存溢出）
        if config.DEVICE == "cuda" and (idx + 1) % 10 == 0:
            torch.cuda.empty_cache()
    
    print(f"\n100张样本可视化完成！所有对比图已保存至：{config.VIS_IMG_SAVE_PATH}")

# ====================== 运行可视化流程 ======================
if __name__ == "__main__":
    try:
        from segment_anything import sam_model_registry
    except ImportError:
        print("错误：未找到segment-anything库，请先克隆官方仓库！")
        print("克隆命令：git clone https://github.com/facebookresearch/segment-anything.git")
        exit(1)
    
    run_vis_pipeline()