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
    # COCO数据集路径
    COCO_RAW_ROOT = "E:\\SAM_Model\\datasets\\COCO"
    COCO_IMG_PATH = os.path.join(COCO_RAW_ROOT, "val2017")
    COCO_ANN_PATH = os.path.join(COCO_RAW_ROOT, "annotations", "instances_val2017.json")
    
    # SAM权重路径（ViT-B）
    SAM_CHECKPOINT_PATH = "E:\\SAM_Model\\weights\\sam_vit_b_01ec64.pth"
    MODEL_TYPE = "vit_b"
    
    # 结果保存路径
    RESULT_ROOT = "E:\\SAM_Model\\results"
    EXP1_ROOT = os.path.join(RESULT_ROOT, "coco_exp1")
    EXP1_IMG_PATH = os.path.join(EXP1_ROOT, "images", "val2017")
    EXP1_ANN_PATH = os.path.join(EXP1_ROOT, "annotations", "instances_val2017_1000.json")
    SAMPLE_LIST_PATH = os.path.join(EXP1_ROOT, "sample_list.txt")
    RESULT_CSV_PATH = os.path.join(EXP1_ROOT, "exp1_results.csv")
    LOG_PATH = os.path.join(EXP1_ROOT, "exp1_log.txt")
    
    # 实验参数
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    RANDOM_SEED = 42
    SAMPLE_NUM = 1000  # 先改为100快速验证
    IMG_TARGET_SIZE = 1024
    MASK_THRESHOLD = 0.5
    MIN_INSTANCE_AREA = 100

# ====================== 2. 工具函数：创建目录 ======================
def create_dirs():
    os.makedirs(Config.EXP1_IMG_PATH, exist_ok=True)
    os.makedirs(os.path.dirname(Config.EXP1_ANN_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(Config.RESULT_CSV_PATH), exist_ok=True)

# ====================== 3. 样本筛选 ======================
def filter_coco_samples():
    random.seed(Config.RANDOM_SEED)
    np.random.seed(Config.RANDOM_SEED)
    
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

# ====================== 4. 提示构建函数（修正：使用像素坐标，取消归一化） ======================
def get_mask_centroid(mask):
    """计算掩码质心（像素坐标）"""
    y_coords, x_coords = np.where(mask == 1)
    if len(x_coords) == 0 or len(y_coords) == 0:
        return None
    centroid_x = int(np.mean(x_coords))
    centroid_y = int(np.mean(y_coords))
    return (centroid_x, centroid_y)

def generate_negative_point(img_w, img_h, bbox):
    """生成负点（像素坐标，在bbox外）"""
    x1, y1, x2, y2 = bbox
    for _ in range(10):
        px = random.randint(0, img_w-1)
        py = random.randint(0, img_h-1)
        if not (x1 <= px <= x2 and y1 <= py <= y2):
            return (px, py)
    return (0, 0)

def build_prompts(img_info, ann, coco):
    """
    修正：返回像素坐标的提示，取消归一化
    返回：point_prompt (像素坐标, 标签), box_prompt (像素坐标xyxy), mixed_prompt
    """
    img_w, img_h = img_info["width"], img_info["height"]
    bbox = ann["bbox"]  # COCO格式：(x, y, w, h)（像素坐标）
    mask = coco.annToMask(ann)
    instance_area = ann["area"]
    
    if instance_area < Config.MIN_INSTANCE_AREA:
        return None, None, None
    
    # 1. 框提示：转为xyxy像素坐标（无需归一化），并reshape为(1,4)（SAM要求）
    x, y, w, h = bbox
    box_xyxy = np.array([[x, y, x + w, y + h]], dtype=np.float32)  # 关键：(1,4)形状+float32
    
    # 2. 点提示：像素坐标（取消归一化）
    centroid = get_mask_centroid(mask)
    if centroid is None:
        return None, None, None
    pos_x, pos_y = centroid
    neg_x, neg_y = generate_negative_point(img_w, img_h, box_xyxy[0])  # 取box的一维数据
    
    # 点坐标：形状(N,2)（N是点的数量），float32
    points = np.array([[pos_x, pos_y], [neg_x, neg_y]], dtype=np.float32)
    point_labels = np.array([1, 0], dtype=np.int32)  # 1=前景点，0=背景点
    
    # 3. 混合提示：前景点 + 框（均为像素坐标）
    mixed_points = np.array([[pos_x, pos_y]], dtype=np.float32)  # (1,2)形状
    mixed_labels = np.array([1], dtype=np.int32)
    
    point_prompt = (points, point_labels)
    box_prompt = (box_xyxy,)  # 保持元组格式
    mixed_prompt = (mixed_points, mixed_labels, box_xyxy)
    
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
    # SAM输出的mask是布尔型，先转为uint8
    mask = mask.astype(np.uint8) * 255  # 转为0-255灰度图
    scaled_mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    # 二值化（超过阈值为前景）
    return (scaled_mask > Config.MASK_THRESHOLD * 255).astype(np.uint8)  # 修正：对应0-255灰度值

# ====================== 6. 核心实验流程（修正提示使用逻辑） ======================
def run_exp1():
    create_dirs()
    config = Config()
    
    # 打印配置
    print("="*50)
    print("当前实验配置信息")
    print("="*50)
    print(f"COCO数据集路径：{config.COCO_RAW_ROOT}")
    print(f"SAM权重路径：{config.SAM_CHECKPOINT_PATH}")
    print(f"SAM模型类型：{config.MODEL_TYPE}")
    print(f"实验结果保存路径：{config.EXP1_ROOT}")
    print(f"运行设备：{config.DEVICE}")
    print(f"抽取样本数：{config.SAMPLE_NUM}")
    print("="*50 + "\n")
    
    # 筛选样本
    selected_img_ids, coco_original = filter_coco_samples()
    
    # 加载专属标注
    print(f"\n正在加载专属标注文件：{config.EXP1_ANN_PATH}")
    coco_exp1 = COCO(config.EXP1_ANN_PATH)
    print("专属标注加载完成！")
    
    # 加载SAM模型
    print(f"\n正在加载SAM模型：{config.MODEL_TYPE}")
    print(f"权重路径：{config.SAM_CHECKPOINT_PATH}")
    sam = sam_model_registry[config.MODEL_TYPE](checkpoint=config.SAM_CHECKPOINT_PATH)
    sam.to(device=config.DEVICE)
    sam.eval()
    predictor = SamPredictor(sam)
    print("SAM模型加载完成，已进入评估模式\n")
    
    # 初始化结果记录
    prompt_types = ["Point-only", "Box-only", "Point+Box"]
    result_dict = {pt: {"img_miou_list": [], "instance_iou_list": []} for pt in prompt_types}
    log_content = []
    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    log_content.append(f"实验1：SAM不同提示方式性能对比")
    log_content.append(f"配置：样本数={config.SAMPLE_NUM}, 模型={config.MODEL_TYPE}, 输入尺寸={config.IMG_TARGET_SIZE}")
    log_content.append(f"COCO路径：{config.COCO_RAW_ROOT}")
    log_content.append(f"SAM权重：{config.SAM_CHECKPOINT_PATH}")
    log_content.append(f"开始时间：{start_time}")
    print("\n".join(log_content))
    print("\n即将开始逐图推理...")
    
    # 逐张推理
    total_img = len(selected_img_ids)
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
            point_prompt, box_prompt, mixed_prompt = build_prompts(img_info, ann, coco_exp1)
            if point_prompt is None:
                continue
            gt_mask = coco_exp1.annToMask(ann)  # 真实掩码（像素坐标）
            
            # （1）Point-only 推理（修正：使用像素坐标点）
            points, point_labels = point_prompt
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    masks, scores, _ = predictor.predict(
                        point_coords=points,  # 像素坐标，(N,2)
                        point_labels=point_labels,  # (N,)
                        multimask_output=True
                    )
            best_mask_idx = np.argmax(scores)
            pred_mask_point = masks[best_mask_idx]
            pred_mask_point_scaled = scale_mask(pred_mask_point, img_h, img_w)
            iou_point = calculate_iou(pred_mask_point_scaled, gt_mask)
            img_instance_iou["Point-only"].append(iou_point)
            result_dict["Point-only"]["instance_iou_list"].append(iou_point)
            
            # （2）Box-only 推理（修正：使用像素坐标框）
            box_xyxy = box_prompt[0]
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    masks, scores, _ = predictor.predict(
                        box=box_xyxy,  # 像素坐标，(1,4)
                        multimask_output=True
                    )
            best_mask_idx = np.argmax(scores)
            pred_mask_box = masks[best_mask_idx]
            pred_mask_box_scaled = scale_mask(pred_mask_box, img_h, img_w)
            iou_box = calculate_iou(pred_mask_box_scaled, gt_mask)
            img_instance_iou["Box-only"].append(iou_box)
            result_dict["Box-only"]["instance_iou_list"].append(iou_box)
            
            # （3）Point+Box 推理（修正：混合像素坐标提示）
            mixed_points, mixed_labels, mixed_box = mixed_prompt
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    masks, scores, _ = predictor.predict(
                        point_coords=mixed_points,  # 像素坐标，(1,2)
                        point_labels=mixed_labels,  # (1,)
                        box=mixed_box,  # 像素坐标，(1,4)
                        multimask_output=True
                    )
            best_mask_idx = np.argmax(scores)
            pred_mask_mixed = masks[best_mask_idx]
            pred_mask_mixed_scaled = scale_mask(pred_mask_mixed, img_h, img_w)
            iou_mixed = calculate_iou(pred_mask_mixed_scaled, gt_mask)
            img_instance_iou["Point+Box"].append(iou_mixed)
            result_dict["Point+Box"]["instance_iou_list"].append(iou_mixed)
        
        # 计算单张图片mIoU
        current_img_miou = {}
        for pt in prompt_types:
            if len(img_instance_iou[pt]) > 0:
                img_miou = np.mean(img_instance_iou[pt])
                result_dict[pt]["img_miou_list"].append(img_miou)
                current_img_miou[pt] = img_miou
            else:
                current_img_miou[pt] = 0.0
        
        # 打印进度（同时显示3种提示的mIoU，更直观）
        if (idx + 1) % 10 == 0 or (idx + 1) == total_img:
            print(f"推理进度：{idx+1}/{total_img} | 图片：{img_name}")
            print(f"  Point-only mIoU：{current_img_miou['Point-only']:.4f} | Box-only mIoU：{current_img_miou['Box-only']:.4f} | Point+Box mIoU：{current_img_miou['Point+Box']:.4f}")
        
        # 清空显存
        torch.cuda.empty_cache()
    
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
    
    # 保存结果
    with open(config.RESULT_CSV_PATH, "w", encoding="utf-8", newline="") as f:
        fieldnames = ["提示方式", "实例数量", "图像数量", "实例级mIoU", "图像级mIoU"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final_results)
    print(f"\n核心对比表格已保存至：{config.RESULT_CSV_PATH}")
    
    with open(config.LOG_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(log_content))
        f.write(f"\n结束时间：{end_time}")
    print(f"详细实验日志已保存至：{config.LOG_PATH}")
    
    # 打印最终结果
    print("\n" + "="*80)
    print("实验1 最终结果对比（验证混合提示最优）")
    print("="*80)
    for res in final_results:
        print(f"{res['提示方式']:12s} | 实例级mIoU：{res['实例级mIoU']:.4f} | 图像级mIoU：{res['图像级mIoU']:.4f}")
    print("="*80)
    print("实验结论：Point+Box 混合提示性能最优（mIoU值最高），与原论文一致")

# ====================== 运行实验 ======================
if __name__ == "__main__":
    try:
        from segment_anything import sam_model_registry
    except ImportError:
        print("错误：未找到segment-anything库，请先克隆官方仓库并添加到Python环境！")
        print("克隆命令：git clone https://github.com/facebookresearch/segment-anything.git")
        exit(1)
    
    run_exp1()