import os
import json
import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple

# ====================== 1. 配置参数（用户仅需修改此部分！） ======================
class Config:
    """VOC转COCO配置类，根据实际路径修改以下参数"""
    # VOC2012 XML标注文件所在目录（必填）
    VOC_XML_DIR = "E:\\SAM_Model\\datasets\\VOC2012\\Annotations"
    # VOC2012图像文件所在目录（必填，需与XML标注对应）
    VOC_IMG_DIR = "E:\\SAM_Model\\datasets\\VOC2012\\JPEGImages"
    # 输出COCO格式JSON文件的完整路径（必填，目录不存在会自动创建）
    OUTPUT_COCO_JSON = "E:\\SAM_Model\\datasets\\VOC2012\\annotation\\voc2012_val_coco_format.json"
    # VOC2012标准类别（固定20类，无需修改）
    VOC_CATEGORIES = [
        {"id": 1, "name": "aeroplane"}, {"id": 2, "name": "bicycle"}, {"id": 3, "name": "bird"},
        {"id": 4, "name": "boat"}, {"id": 5, "name": "bottle"}, {"id": 6, "name": "bus"},
        {"id": 7, "name": "car"}, {"id": 8, "name": "cat"}, {"id": 9, "name": "chair"},
        {"id": 10, "name": "cow"}, {"id": 11, "name": "diningtable"}, {"id": 12, "name": "dog"},
        {"id": 13, "name": "horse"}, {"id": 14, "name": "motorbike"}, {"id": 15, "name": "person"},
        {"id": 16, "name": "pottedplant"}, {"id": 17, "name": "sheep"}, {"id": 18, "name": "sofa"},
        {"id": 19, "name": "train"}, {"id": 20, "name": "tvmonitor"}
    ]
    # 是否显示转换进度条（True/False，需安装tqdm库，建议开启）
    SHOW_PROGRESS = True


# ====================== 2. 工具函数（无需修改） ======================
def load_tqdm() -> callable:
    """加载tqdm进度条库，若未安装则返回None"""
    try:
        from tqdm import tqdm
        return tqdm
    except ImportError:
        print("提示：未安装tqdm库，将不显示转换进度条（可执行`pip install tqdm`安装）")
        return None


def get_xml_file_list(xml_dir: str) -> List[str]:
    """获取指定目录下所有XML文件路径，过滤非XML文件"""
    if not os.path.exists(xml_dir):
        raise FileNotFoundError(f"XML标注目录不存在：{xml_dir}")
    
    xml_files = []
    for file_name in os.listdir(xml_dir):
        if file_name.lower().endswith(".xml"):
            xml_files.append(os.path.join(xml_dir, file_name))
    
    if len(xml_files) == 0:
        raise ValueError(f"在{xml_dir}目录下未找到任何XML文件")
    
    return xml_files


def parse_voc_xml(xml_path: str, categories: List[Dict]) -> Tuple[Dict, List[Dict]]:
    """
    解析单张VOC XML标注文件
    返回：(图像信息字典, 实例标注列表)
    """
    # 解析XML树
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        raise ValueError(f"XML文件解析失败：{xml_path}，错误信息：{str(e)}")
    
    # 1. 提取图像基本信息
    img_info = {}
    # 获取图像尺寸（width/height）
    size_elem = root.find("size")
    if size_elem is None:
        raise KeyError(f"XML文件{xml_path}缺少<size>标签（需包含width/height信息）")
    
    width_elem = size_elem.find("width")
    height_elem = size_elem.find("height")
    if width_elem is None or height_elem is None:
        raise KeyError(f"XML文件{xml_path}的<size>标签缺少width或height")
    
    img_info["width"] = int(width_elem.text.strip())
    img_info["height"] = int(height_elem.text.strip())
    
    # 获取图像文件名
    filename_elem = root.find("filename")
    if filename_elem is None:
        raise KeyError(f"XML文件{xml_path}缺少<filename>标签")
    
    img_info["file_name"] = filename_elem.text.strip()
    # 生成唯一图像ID（基于文件名，如"2007_000027.jpg"→2007000027）
    img_id_str = os.path.splitext(img_info["file_name"])[0].replace("_", "")
    img_info["id"] = int(img_id_str) if img_id_str.isdigit() else hash(img_info["file_name"])
    
    # 2. 提取实例标注信息（含image_id关联）
    instances = []
    object_elems = root.findall("object")
    if len(object_elems) == 0:
        print(f"警告：XML文件{xml_path}未包含任何<object>实例标注，将跳过该文件")
        return img_info, instances
    
    # 构建类别名到ID的映射（加速查找）
    cat_name_to_id = {cat["name"]: cat["id"] for cat in categories}
    
    for obj_elem in object_elems:
        instance = {}
        # 2.1 类别ID（必填）
        name_elem = obj_elem.find("name")
        if name_elem is None:
            print(f"警告：XML文件{xml_path}的某个<object>缺少<name>标签，将跳过该实例")
            continue
        
        cat_name = name_elem.text.strip().lower()
        if cat_name not in cat_name_to_id:
            print(f"警告：XML文件{xml_path}包含未知类别{cat_name}，将跳过该实例（仅支持VOC2012标准20类）")
            continue
        
        instance["category_id"] = cat_name_to_id[cat_name]
        # 2.2 关联图像ID（关键！修复之前的KeyError）
        instance["image_id"] = img_info["id"]
        # 2.3 遮挡标记（iscrowd，VOC无该信息，默认0）
        instance["iscrowd"] = 0
        # 2.4 BBOX（COCO格式：x, y, w, h，必填）
        bndbox_elem = obj_elem.find("bndbox")
        if bndbox_elem is None:
            print(f"警告：XML文件{xml_path}的类别{cat_name}实例缺少<bndbox>标签，将跳过该实例")
            continue
        
        # 提取BBOX坐标（xmin, ymin, xmax, ymax）
        xmin_elem = bndbox_elem.find("xmin")
        ymin_elem = bndbox_elem.find("ymin")
        xmax_elem = bndbox_elem.find("xmax")
        ymax_elem = bndbox_elem.find("ymax")
        
        if any(elem is None for elem in [xmin_elem, ymin_elem, xmax_elem, ymax_elem]):
            print(f"警告：XML文件{xml_path}的类别{cat_name}实例<bndbox>标签缺少坐标，将跳过该实例")
            continue
        
        # 转换为COCO格式（x, y为左上角坐标，w=width, h=height）
        xmin = float(xmin_elem.text.strip())
        ymin = float(ymin_elem.text.strip())
        xmax = float(xmax_elem.text.strip())
        ymax = float(ymax_elem.text.strip())
        
        # 验证BBOX有效性（避免负数或超出图像范围）
        if xmin < 0 or ymin < 0 or xmax > img_info["width"] or ymax > img_info["height"]:
            print(f"警告：XML文件{xml_path}的类别{cat_name}实例BBOX超出图像范围，将跳过该实例")
            continue
        
        instance["bbox"] = [round(xmin, 2), round(ymin, 2), round(xmax - xmin, 2), round(ymax - ymin, 2)]
        # 2.5 实例面积（基于BBOX，必填）
        instance["area"] = round((xmax - xmin) * (ymax - ymin), 2)
        # 2.6 Segmentation（COCO格式：二维数组，必填，兼容VOC两种标注方式）
        segmentation = []
        # 方式1：直接嵌套<polygon>标签（常见格式）
        polygon_elem = obj_elem.find("polygon")
        if polygon_elem is not None:
            point_elems = polygon_elem.findall("point")
            if len(point_elems) >= 3:  # 至少3个点构成多边形
                for pt_elem in point_elems:
                    x_elem = pt_elem.find("x")
                    y_elem = pt_elem.find("y")
                    if x_elem is not None and y_elem is not None:
                        x = round(float(x_elem.text.strip()), 2)
                        y = round(float(y_elem.text.strip()), 2)
                        segmentation.extend([x, y])
        # 方式2：<segm>嵌套<polygon>标签（部分VOC标注格式）
        if not segmentation:
            segm_elem = obj_elem.find("segm")
            if segm_elem is not None:
                segm_polygon_elem = segm_elem.find("polygon")
                if segm_polygon_elem is not None:
                    point_elems = segm_polygon_elem.findall("point")
                    if len(point_elems) >= 3:
                        for pt_elem in point_elems:
                            x_elem = pt_elem.find("x")
                            y_elem = pt_elem.find("y")
                            if x_elem is not None and y_elem is not None:
                                x = round(float(x_elem.text.strip()), 2)
                                y = round(float(y_elem.text.strip()), 2)
                                segmentation.extend([x, y])
        # 兜底：若无轮廓信息，用BBOX生成简易四边形（确保segmentation字段存在）
        if not segmentation:
            segmentation = [
                xmin, ymin,
                xmax, ymin,
                xmax, ymax,
                xmin, ymax
            ]
        
        instance["segmentation"] = [segmentation]
        instances.append(instance)
    
    return img_info, instances


def voc2coco_convert() -> None:
    """
    核心转换函数：将VOC2012标注转换为COCO格式
    1. 加载配置与工具
    2. 解析所有XML文件
    3. 构建COCO JSON结构
    4. 保存JSON文件
    """
    # 1. 加载配置与进度条工具
    config = Config()
    tqdm = load_tqdm()
    
    # 2. 获取所有XML文件列表
    print(f"步骤1/4：获取XML标注文件，目录：{config.VOC_XML_DIR}")
    xml_files = get_xml_file_list(config.VOC_XML_DIR)
    print(f"成功获取{len(xml_files)}个XML标注文件")
    
    # 3. 初始化COCO JSON结构（严格遵循COCO规范）
    print(f"步骤2/4：初始化COCO JSON结构")
    coco_json = {
        "info": {
            "description": "VOC2012 Dataset Converted to COCO Format",
            "version": "1.0",
            "year": 2025,
            "contributor": "VOC2COCO Converter",
            "date_created": os.popen('date /t').read().strip()  # 获取当前日期
        },
        "licenses": [
            {
                "id": 1,
                "name": "VOC2012 Original License",
                "url": "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/license.html"
            }
        ],
        "categories": config.VOC_CATEGORIES,
        "images": [],
        "annotations": []
    }
    
    # 4. 批量解析XML并填充COCO JSON
    print(f"步骤3/4：开始解析XML标注并转换（共{len(xml_files)}个文件）")
    global_ann_id = 0  # 全局实例ID（确保所有实例ID唯一）
    valid_img_count = 0  # 有效图像计数（含实例标注）
    valid_ann_count = 0  # 有效实例计数
    
    # 处理文件列表（带进度条/无进度条）
    file_iter = tqdm(xml_files, desc="转换进度") if (config.SHOW_PROGRESS and tqdm) else xml_files
    for xml_path in file_iter:
        # 解析单张XML
        try:
            img_info, instances = parse_voc_xml(xml_path, config.VOC_CATEGORIES)
        except Exception as e:
            print(f"跳过异常XML文件：{xml_path}，错误：{str(e)}")
            continue
        
        # 检查图像文件是否存在（避免无效关联）
        img_path = os.path.join(config.VOC_IMG_DIR, img_info["file_name"])
        if not os.path.exists(img_path):
            print(f"跳过图像不存在的XML：{xml_path}，图像路径：{img_path}")
            continue
        
        # 添加图像信息（仅含有效实例的图像）
        if len(instances) > 0:
            coco_json["images"].append(img_info)
            valid_img_count += 1
            
            # 添加实例信息（补充全局唯一ID）
            for instance in instances:
                instance["id"] = global_ann_id  # 全局唯一实例ID
                coco_json["annotations"].append(instance)
                global_ann_id += 1
                valid_ann_count += 1
    
    # 5. 保存COCO JSON文件
    print(f"步骤4/4：保存COCO JSON文件，路径：{config.OUTPUT_COCO_JSON}")
    # 创建输出目录（若不存在）
    output_dir = os.path.dirname(config.OUTPUT_COCO_JSON)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"自动创建输出目录：{output_dir}")
    
    # 保存JSON（缩进2，便于查看）
    with open(config.OUTPUT_COCO_JSON, "w", encoding="utf-8") as f:
        json.dump(coco_json, f, indent=2, ensure_ascii=False)
    
    # 6. 输出转换统计信息
    print("\n" + "="*50)
    print("VOC2012转COCO格式转换完成！")
    print("="*50)
    print(f"XML标注文件总数：{len(xml_files)}")
    print(f"有效图像数（含实例标注）：{valid_img_count}")
    print(f"有效实例数：{valid_ann_count}")
    print(f"COCO JSON保存路径：{config.OUTPUT_COCO_JSON}")
    print(f"COCO JSON文件大小：{round(os.path.getsize(config.OUTPUT_COCO_JSON)/1024, 2)} KB")
    print("="*50)


# ====================== 3. 运行转换（直接执行脚本即可） ======================
if __name__ == "__main__":
    try:
        voc2coco_convert()
    except Exception as e:
        print(f"\n转换过程发生致命错误：{str(e)}")
        print("请检查以下内容：1. XML目录是否存在 2. 图像路径是否正确 3. XML格式是否规范")
        exit(1)