import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch
import os  # 新增：导入os模块，用于创建目录
from segment_anything import sam_model_registry, SamPredictor

# -------------------------- 仅修改这3处路径 --------------------------
SAM_WEIGHT = "E:\\SAM_Model\\weights\\sam_vit_h_4b8939.pth"  # 权重路径
TEST_IMAGE = "E:\\SAM_Model\\images\\000000046804.jpg"  # 测试图路径
SAVE_DIR = "E:\\SAM_Model\\results\\000000046804\\"  # 保存目录
# ---------------------------------------------------------------------

# 新增：自动创建保存目录（不存在则创建）
os.makedirs(SAVE_DIR, exist_ok=True)

# 新增：异常处理 - 模型加载
try:
    sam = sam_model_registry["vit_h"](checkpoint=SAM_WEIGHT)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam.to(device=device)
    predictor = SamPredictor(sam)
    print(f"模型加载成功，使用设备：{device}")
except Exception as e:
    print(f"模型加载失败：{e}")
    exit()  # 模型加载失败则终止程序

# 新增：异常处理 - 图片加载与预处理
try:
    image = cv2.imread(TEST_IMAGE)
    if image is None:
        raise FileNotFoundError("测试图片不存在或损坏")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image_rgb.shape  # 获取图片宽高，用于验证坐标
    predictor.set_image(image_rgb)
    print("图片加载并预处理成功")
except Exception as e:
    print(f"图片处理失败：{e}")
    exit()

# 定义可视化与保存函数
def save_demo(img, mask, prompt_type, prompt_data):
    plt.figure(figsize=(10, 8))
    plt.imshow(img)
    plt.imshow(mask[0], alpha=0.5, cmap='jet')
    # 绘制提示（点/框）
    if prompt_type == "point":
        plt.scatter(prompt_data[:, 0], prompt_data[:, 1], c='red', marker='*', s=200, edgecolor='white')
    elif prompt_type == "box":
        x1, y1, x2, y2 = prompt_data[0]
        ax = plt.gca()
        rect = Rectangle((x1, y1), x2 - x1, y2 - y1, color='blue', linewidth=3, fill=False)
        ax.add_patch(rect)
    plt.axis('off')
    plt.title(f"SAM {prompt_type} prompt segmentation", fontsize=14)
    plt.savefig(f"{SAVE_DIR}{prompt_type}_demo.png", bbox_inches='tight', dpi=150)
    plt.close()

# 1. 点提示分割（选2个前景点：物体中心+边缘）
point_coords = np.array([[400,110], [273, 191]])  # 目标坐标
point_labels = np.array([1, 1])

# 新增：验证点坐标有效性
if (point_coords[:, 0] >= w).any() or (point_coords[:, 0] < 0).any() or \
   (point_coords[:, 1] >= h).any() or (point_coords[:, 1] < 0).any():
    print("警告：点坐标超出图片尺寸范围，请检查修改")
else:
    # 保存logits，开启多掩码输出（可选筛选最优）
    point_masks, point_scores, point_logits = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True  # 开启多掩码，方便筛选最优
    )
    # 选择得分最高的掩码（提升分割效果）
    best_point_idx = np.argmax(point_scores)
    point_masks = point_masks[best_point_idx:best_point_idx+1]  # 保持维度为1×H×W
    point_logits = point_logits[best_point_idx:best_point_idx+1]  # 对应最优掩码的logits
    save_demo(image_rgb, point_masks, "point", point_coords)
    print("点提示分割结果已保存")

# 2. 框提示分割（框选物体：[x1, y1, x2, y2]）
box_coords = np.array([[180,49,506,383]])

# 新增：验证框坐标有效性
x1, y1, x2, y2 = box_coords[0]
if x1 < 0 or y1 < 0 or x2 >= w or y2 >= h or x1 >= x2 or y1 >= y2:
    print("警告：框坐标超出图片尺寸范围或格式错误，请检查修改")
else:
    box_masks, _, _ = predictor.predict(box=box_coords, multimask_output=False)
    save_demo(image_rgb, box_masks, "box", box_coords)
    print("框提示分割结果已保存")

# 3. 掩码提示分割（用点提示的logits作为初始输入）
if 'point_logits' in locals():  # 确保点提示正常运行后再执行
    mask_masks, mask_scores, _ = predictor.predict(
        mask_input=point_logits,
        multimask_output=True  # 开启多掩码筛选
    )
    # 选择得分最高的优化掩码
    best_mask_idx = np.argmax(mask_scores)
    mask_masks = mask_masks[best_mask_idx:best_mask_idx+1]
    
    # 定义初始掩码（点提示的最优掩码）
    init_mask = point_masks[0]
    # 保存对比图
    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    plt.imshow(image_rgb)
    plt.imshow(init_mask, alpha=0.5, cmap='jet')
    plt.title("Initial Mask (Point Prompt)", fontsize=12)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(image_rgb)
    plt.imshow(mask_masks[0], alpha=0.5, cmap='jet')
    plt.title("Optimized Mask (Mask Prompt)", fontsize=12)
    plt.axis('off')
    plt.savefig(f"{SAVE_DIR}mask_optimize_demo.png", bbox_inches='tight', dpi=150)
    plt.close()
    print("掩码提示对比图已保存")

print(f"所有分割结果已保存到：{SAVE_DIR}")  