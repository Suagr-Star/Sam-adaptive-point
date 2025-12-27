import os
import random
import csv
import time
import numpy as np
import cv2
import torch
import xml.etree.ElementTree as ET
from segment_anything import sam_model_registry, SamPredictor
from tqdm import tqdm  # 新增进度条，提升体验

# ====================== 1. 极简配置（易修改+防出错） ======================
class Config:
    # 数据集路径（替换为你的路径）
    VOC2012_ROOT = r"E:\\SAM_Model\\datasets\\VOC2012"
    # SAM模型配置
    SAM_CKPT = r"E:\\SAM_Model\\weights\\sam_vit_b_01ec64.pth"
    MODEL_TYPE = "vit_b"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # 实验参数（核心：强制启用伪掩码，确保能运行）
    SAMPLE_NUM = 500    # 采样图片数
    SEED = 42           # 随机种子
    MIN_INST_AREA = 10  # 最小实例面积（放宽）
    USE_FAKE_MASK = True  # 无真实掩码时，用bbox生成伪掩码（强制开启）
    # 结果保存
    OUTPUT_DIR = r"E:\\SAM_Model\\results\\voc2012_exp_fixed"

# ====================== 2. 工具函数（简化+鲁棒） ======================
def init_env():
    """初始化环境：创建目录、设置随机种子"""
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    random.seed(Config.SEED)
    np.random.seed(Config.SEED)
    torch.manual_seed(Config.SEED)
    if Config.DEVICE == "cuda":
        torch.cuda.manual_seed(Config.SEED)
    # 初始化结果文件
    with open(os.path.join(Config.OUTPUT_DIR, "image_details.csv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["img_id", "valid_inst", "hard_inst", "hard_ratio", "is_complex", 
                         "point_only_miou", "adaptive_miou", "improvement"])
    with open(os.path.join(Config.OUTPUT_DIR, "log.txt"), "w", encoding="utf-8") as f:
        f.write(f"实验开始：{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    print("✅ 环境初始化完成")

def log(msg):
    """简易日志：打印+写入文件"""
    print(msg)
    with open(os.path.join(Config.OUTPUT_DIR, "log.txt"), "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {msg}\n")

def save_summary(summary):
    """保存汇总结果"""
    with open(os.path.join(Config.OUTPUT_DIR, "summary.csv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for k, v in summary.items():
            writer.writerow([k, v])
    log(f"📊 汇总结果已保存：{os.path.join(Config.OUTPUT_DIR, 'summary.csv')}")

# ====================== 3. VOC数据处理（核心：兼容伪掩码） ======================
def parse_voc_xml(xml_path):
    """解析VOC XML，返回图片信息+实例列表"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        img_info = {
            "filename": root.find("filename").text,
            "w": int(root.find("size/width").text),
            "h": int(root.find("size/height").text)
        }
        instances = []
        for obj in root.findall("object"):
            bndbox = obj.find("bndbox")
            bbox = [
                int(bndbox.find("xmin").text),
                int(bndbox.find("ymin").text),
                int(bndbox.find("xmax").text),
                int(bndbox.find("ymax").text)
            ]
            # 过滤无效bbox
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            instances.append({
                "bbox": bbox,
                "difficult": int(obj.find("difficult").text) if obj.find("difficult") is not None else 0,
                "name": obj.find("name").text
            })
        return img_info, instances
    except Exception as e:
        log(f"❌ 解析XML失败 {xml_path}：{str(e)}")
        return None, []

def get_instance_mask(img_name, bbox, img_h, img_w):
    """
    获取实例掩码（优先真实掩码，无则生成伪掩码）
    :return: 掩码矩阵 (h, w) uint8，0=背景，1=实例
    """
    # 1. 尝试加载真实掩码
    seg_path = os.path.join(Config.VOC2012_ROOT, "SegmentationObject", img_name.replace(".jpg", ".png"))
    if os.path.exists(seg_path):
        seg_mask = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
        if seg_mask is not None:
            # 提取bbox内的实例灰度值
            x1, y1, x2, y2 = bbox
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w-1, x2), min(img_h-1, y2)
            bbox_mask = seg_mask[y1:y2, x1:x2]
            unique_gray = np.unique(bbox_mask[bbox_mask != 0])
            if len(unique_gray) > 0:
                target_gray = max(unique_gray, key=lambda g: np.sum(bbox_mask == g))
                inst_mask = (seg_mask == target_gray).astype(np.uint8)
                if np.sum(inst_mask) >= Config.MIN_INST_AREA:
                    return inst_mask
    
    # 2. 无真实掩码，生成伪掩码（强制启用）
    if Config.USE_FAKE_MASK:
        inst_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w-1, x2), min(img_h-1, y2)
        inst_mask[y1:y2, x1:x2] = 1
        if np.sum(inst_mask) >= Config.MIN_INST_AREA:
            return inst_mask
    
    return None

def get_voc_sample():
    """获取采样图片ID列表"""
    xml_dir = os.path.join(Config.VOC2012_ROOT, "Annotations")
    xml_files = [f for f in os.listdir(xml_dir) if f.endswith(".xml")]
    if len(xml_files) < Config.SAMPLE_NUM:
        log(f"⚠️  XML文件数量不足，仅采样{len(xml_files)}张")
        sample_files = xml_files
    else:
        sample_files = random.sample(xml_files, Config.SAMPLE_NUM)
    sample_ids = [f.replace(".xml", "") for f in sample_files]
    log(f"✅ 完成采样：{len(sample_ids)}张图片")
    return sample_ids

# ====================== 4. SAM核心逻辑（简化+防崩溃） ======================
def load_sam_model():
    """加载SAM模型"""
    try:
        sam = sam_model_registry[Config.MODEL_TYPE](checkpoint=Config.SAM_CKPT)
        sam.to(Config.DEVICE)
        sam.eval()
        predictor = SamPredictor(sam)
        log("✅ SAM模型加载完成")
        return predictor
    except Exception as e:
        log(f"❌ 加载SAM失败：{str(e)}")
        exit(1)

def get_mask_centroid(mask):
    """获取掩码质心"""
    y, x = np.where(mask == 1)
    if len(x) == 0:
        return (mask.shape[1]//2, mask.shape[0]//2)  # 兜底：返回中心
    return (int(np.mean(x)), int(np.mean(y)))

def generate_negative_point(bbox, img_w, img_h):
    """生成负点（简单版）"""
    x1, y1, x2, y2 = bbox
    for _ in range(10):
        px = random.randint(0, img_w-1)
        py = random.randint(0, img_h-1)
        if not (x1 <= px <= x2 and y1 <= py <= y2):
            return (px, py)
    return (0, 0)  # 兜底

def calculate_iou(pred, gt):
    """计算IoU"""
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return intersection / union if union > 0 else 0.0

def get_instance_features(bbox, mask, img_w, img_h):
    """计算实例特征（尺寸+形状）"""
    # 尺寸特征：实例面积/图片面积
    x1, y1, x2, y2 = bbox
    inst_area = (x2-x1)*(y2-y1)
    size_feat = inst_area / (img_w * img_h)
    
    # 形状特征：(周长²)/面积（越不规则值越大）
    mask_uint8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        shape_feat = 1000.0
    else:
        max_cnt = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(max_cnt, closed=True)
        area = cv2.contourArea(max_cnt)
        shape_feat = (perimeter**2)/area if area > 0 else 1000.0
    
    return size_feat, shape_feat

def predict_with_sam(predictor, img_rgb, points, labels):
    """SAM推理（统一封装）"""
    try:
        predictor.set_image(img_rgb)
        with torch.no_grad():
            masks, scores, _ = predictor.predict(
                point_coords=points,
                point_labels=labels,
                multimask_output=True
            )
        # 选最优掩码
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx].cpu().numpy() if torch.is_tensor(masks) else masks[best_idx]
        # 缩放至图片尺寸
        best_mask = cv2.resize(best_mask.astype(np.uint8), 
                              (img_rgb.shape[1], img_rgb.shape[0]), 
                              interpolation=cv2.INTER_NEAREST)
        return (best_mask > 0.5).astype(np.uint8)
    except Exception as e:
        log(f"⚠️ SAM推理失败：{str(e)}")
        return None

# ====================== 5. 主实验流程（简化+逐步调试） ======================
def run_experiment():
    # 1. 初始化
    init_env()
    sample_ids = get_voc_sample()
    predictor = load_sam_model()
    
    # 2. 预计算复杂度阈值（防空列表）
    log("📈 预计算实例复杂度阈值...")
    size_feats, shape_feats = [], []
    for img_id in tqdm(sample_ids, desc="预计算阈值"):
        xml_path = os.path.join(Config.VOC2012_ROOT, "Annotations", f"{img_id}.xml")
        img_info, instances = parse_voc_xml(xml_path)
        if not img_info or not instances:
            continue
        img_w, img_h = img_info["w"], img_info["h"]
        for inst in instances:
            mask = get_instance_mask(img_info["filename"], inst["bbox"], img_h, img_w)
            if mask is None:
                continue
            size_feat, shape_feat = get_instance_features(inst["bbox"], mask, img_w, img_h)
            size_feats.append(size_feat)
            shape_feats.append(shape_feat)
    
    # 阈值默认值（避免空列表报错）
    size_thresh = np.quantile(size_feats, 0.25) if size_feats else 0.01
    shape_thresh = np.quantile(shape_feats, 0.75) if shape_feats else 1000.0
    log(f"✅ 阈值计算完成：size_thresh={size_thresh:.4f}, shape_thresh={shape_thresh:.2f}")
    
    # 3. 逐图推理
    log("\n🚀 开始逐图推理...")
    global_point_only = []
    global_adaptive = []
    complex_img_num = 0
    simple_img_num = 0
    
    for idx, img_id in enumerate(tqdm(sample_ids, desc="推理进度")):
        # 初始化单图结果
        img_res = {
            "img_id": img_id,
            "valid_inst": 0,
            "hard_inst": 0,
            "hard_ratio": 0.0,
            "is_complex": False,
            "point_only_miou": 0.0,
            "adaptive_miou": 0.0,
            "improvement": 0.0
        }
        
        # 加载图片和标注
        xml_path = os.path.join(Config.VOC2012_ROOT, "Annotations", f"{img_id}.xml")
        img_info, instances = parse_voc_xml(xml_path)
        if not img_info or not instances:
            log(f"⚠️  {img_id} 无有效标注，跳过")
            continue
        
        img_path = os.path.join(Config.VOC2012_ROOT, "JPEGImages", img_info["filename"])
        img = cv2.imread(img_path)
        if img is None:
            log(f"⚠️  {img_id} 图片读取失败，跳过")
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_w, img_h = img_info["w"], img_info["h"]
        
        # 逐实例处理
        point_only_ious = []
        adaptive_ious = []
        hard_count = 0
        valid_count = 0
        
        for inst in instances:
            # 获取掩码
            mask = get_instance_mask(img_info["filename"], inst["bbox"], img_h, img_w)
            if mask is None:
                continue
            valid_count += 1
            
            # 1. Point-only 提示（质心+负点）
            centroid = get_mask_centroid(mask)
            neg_point = generate_negative_point(inst["bbox"], img_w, img_h)
            points = np.array([centroid, neg_point], dtype=np.float32)
            labels = np.array([1, 0], dtype=np.int32)
            pred_mask = predict_with_sam(predictor, img_rgb, points, labels)
            if pred_mask is None:
                continue
            iou1 = calculate_iou(pred_mask, mask)
            point_only_ious.append(iou1)
            
            # 2. 自适应提示（难例多加点，易例单点）
            size_feat, shape_feat = get_instance_features(inst["bbox"], mask, img_w, img_h)
            is_hard = (size_feat < size_thresh) or (shape_feat > shape_thresh) or (inst["difficult"] == 1)
            if is_hard:
                hard_count += 1
                # 难例：多核心点+高响应点（简化版）
                adaptive_points = [centroid]
                # 加两个随机正点（简化自适应逻辑，保证能运行）
                for _ in range(2):
                    y = random.randint(inst["bbox"][1], inst["bbox"][3])
                    x = random.randint(inst["bbox"][0], inst["bbox"][2])
                    adaptive_points.append((x, y))
                adaptive_points = np.array(adaptive_points[:3], dtype=np.float32)
            else:
                # 易例：仅质心
                adaptive_points = np.array([centroid], dtype=np.float32)
            adaptive_labels = np.array([1]*len(adaptive_points), dtype=np.int32)
            
            pred_mask2 = predict_with_sam(predictor, img_rgb, adaptive_points, adaptive_labels)
            if pred_mask2 is None:
                continue
            iou2 = calculate_iou(pred_mask2, mask)
            adaptive_ious.append(iou2)
        
        # 单图结果统计
        if valid_count > 0:
            img_res["valid_inst"] = valid_count
            img_res["hard_inst"] = hard_count
            img_res["hard_ratio"] = round(hard_count / valid_count, 4)
            img_res["is_complex"] = hard_count / valid_count >= 0.5
            
            if point_only_ious:
                img_res["point_only_miou"] = round(np.mean(point_only_ious), 4)
                global_point_only.extend(point_only_ious)
            if adaptive_ious:
                img_res["adaptive_miou"] = round(np.mean(adaptive_ious), 4)
                global_adaptive.extend(adaptive_ious)
            
            img_res["improvement"] = round(img_res["adaptive_miou"] - img_res["point_only_miou"], 4)
            
            if img_res["is_complex"]:
                complex_img_num += 1
            else:
                simple_img_num += 1
        
        # 写入单图结果
        with open(os.path.join(Config.OUTPUT_DIR, "image_details.csv"), "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                img_res["img_id"], img_res["valid_inst"], img_res["hard_inst"],
                img_res["hard_ratio"], img_res["is_complex"], img_res["point_only_miou"],
                img_res["adaptive_miou"], img_res["improvement"]
            ])
        
        # 清空显存
        if Config.DEVICE == "cuda":
            torch.cuda.empty_cache()
    
    # 4. 全局汇总
    summary = {
        "总采样数": len(sample_ids),
        "有效图片数": complex_img_num + simple_img_num,
        "复杂图片数": complex_img_num,
        "简单图片数": simple_img_num,
        "Point-only全局mIoU": round(np.mean(global_point_only), 4) if global_point_only else 0.0,
        "自适应Point全局mIoU": round(np.mean(global_adaptive), 4) if global_adaptive else 0.0,
        "平均性能提升": round(np.mean(global_adaptive) - np.mean(global_point_only), 4) if (global_point_only and global_adaptive) else 0.0
    }
    
    # 打印汇总结果
    log("\n" + "="*50)
    log("📊 实验汇总结果")
    for k, v in summary.items():
        log(f"{k}: {v}")
    save_summary(summary)
    log("\n✅ 实验完成！结果文件路径：")
    log(f"  - 单图详情：{os.path.join(Config.OUTPUT_DIR, 'image_details.csv')}")
    log(f"  - 汇总结果：{os.path.join(Config.OUTPUT_DIR, 'summary.csv')}")
    log(f"  - 日志文件：{os.path.join(Config.OUTPUT_DIR, 'log.txt')}")

# ====================== 运行入口 ======================
if __name__ == "__main__":
    # 依赖检查
    try:
        import segment_anything
    except ImportError:
        print("❌ 缺少segment-anything库，请执行：pip install segment-anything")
        exit(1)
    
    # 路径检查
    if not os.path.exists(Config.VOC2012_ROOT):
        print(f"❌ VOC路径不存在：{Config.VOC2012_ROOT}")
        exit(1)
    if not os.path.exists(Config.SAM_CKPT):
        print(f"❌ SAM权重不存在：{Config.SAM_CKPT}")
        exit(1)
    
    # 运行实验
    run_experiment()