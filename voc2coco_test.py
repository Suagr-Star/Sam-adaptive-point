import json

# 你的COCO标注保存路径
coco_path = "E:\\SAM_Model\\datasets\\VOC2012\\annotation\\voc2012_val_coco.json"

# 读取标注文件
with open(coco_path, "r", encoding="utf-8") as f:
    coco_data = json.load(f)

# 验证核心结构和数据规模
print("=== VOC2012转COCO验证结果 ===")
# 核心数组完整性（仅表达式放大括号内）

# 图片总数（说明文字在大括号外）
print(f"2. 图片总数：{len(coco_data['images'])} 张（应≈1449张，对应VOC2012 val集）")
# 标注总数
print(f"3. 标注总数：{len(coco_data['annotations'])} 个")
# 类别总数
print(f"4. 类别总数：{len(coco_data['categories'])} 类（应=20类，匹配VOC2012官方类别）")
# 示例图片信息
print(f"5. 示例图片信息：{coco_data['images'][0]}")
# 示例标注信息
print(f"6. 示例标注信息：{coco_data['annotations'][0]}")