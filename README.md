# SAM-Adaptive-Point: 自适应选点优化的SAM图像分割实验
## 项目概述
该项目针对**Meta Segment Anything Model (SAM)**的点提示（Point Prompt）进行核心优化，提出了**Point(Adaptive)自适应选点策略**，通过筛选COCO val2017数据集的1000张独立样本，对比传统Point-only策略与自适应选点策略的图像分割性能（以mIoU为核心指标），同时实现了「复杂图片/简单图片」的自动分类与统计，最终输出结构化的实验结果表格与详细日志。

核心目标：解决SAM在小实例、不规则形状、遮挡实例上的分割精度不足问题，通过自适应增加关键选点提升难例分割效果，同时量化复杂图片对分割性能的影响。

## 核心功能
1.  **独立样本筛选**：从COCO val2017中随机抽取1000张全新样本，生成独立的图片目录与标注文件，避免与原有实验数据冲突。
2.  **双策略对比实验**：支持Point-only（传统单质心点+负点）与Point(Adaptive)（自适应多核心点+高响应点）两种策略的分割推理。
3.  **实例与图片复杂度分类**：基于实例尺寸、形状、遮挡特征，实现「易例/难例」实例分类，以及「复杂图片/简单图片」图片级分类。
4.  **结构化结果输出**：自动生成核心CSV性能表格，记录单张图片的双策略mIoU与复杂度标签，同时保存详细实验日志。
5.  **鲁棒性优化**：增加空掩码校验、坐标越界校验、除零错误规避、显存自动清空等机制，提升实验稳定性。

## 环境配置
### 1. 基础依赖安装
创建并激活Python虚拟环境后，执行以下命令安装核心依赖：
```bash
# 基础数值计算与图像处理依赖
pip install numpy opencv-python torch torchvision
pip install pycocotools scikit-image csv
# 其他辅助依赖
pip install shutil tqdm
```

### 2. Segment Anything 安装
该项目依赖官方`segment-anything`库，需手动克隆并安装：
```bash
# 克隆官方仓库
git clone https://github.com/facebookresearch/segment-anything.git
# 进入仓库目录并安装
cd segment-anything
pip install -e .
```

### 3. 环境要求
- Python 3.8+
- CUDA 11.3+（推荐，支持GPU加速，大幅提升推理速度）
- 显存≥8GB（运行ViT-B模型，处理1024×1024图像）
- Windows/macOS/Linux 均兼容（注意路径格式差异，项目默认适配Windows）

## 数据与权重准备
### 1. COCO 数据集下载
- 下载地址：[COCO val2017 数据集](https://cocodataset.org/#download)
  - 需下载：`val2017`（图片压缩包）、`annotations_trainval2017`（标注压缩包）
- 解压后目录结构（对应代码`Config`配置）：
```
E:\SAM_Model\
└── datasets\
    └── COCO\
        ├── val2017\ （解压后的图片目录，包含5000张验证集图片）
        └── annotations\ （解压后的标注目录，包含instances_val2017.json）
```

### 2. SAM 权重文件下载
- 选择模型：ViT-B（轻量型，适合快速实验，对应代码`MODEL_TYPE = "vit_b"`）
- 下载地址：[sam_vit_b_01ec64.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)
- 存放路径（对应代码`Config`配置）：
```
E:\SAM_Model\
└── weights\
    └── sam_vit_b_01ec64.pth
```

### 补充：其他SAM权重（可选）
若需更换模型，可下载ViT-L/ViT-H权重，同时修改代码中`MODEL_TYPE`与`SAM_CHECKPOINT_PATH`：
- ViT-L：`sam_vit_l_0b3195.pth`
- ViT-H：`sam_vit_h_4b8939.pth`

## 项目结构
```
SAM-Adaptive-Point/
├── Point_Adaptive.py （核心实验脚本，项目入口）
├── Test1.py ~ Test6.py （辅助测试脚本）
├── func_demo.py （工具函数演示脚本）
├── voc2coco_script.py & voc2coco_test.py （VOC转COCO辅助脚本）
├── README.md （项目说明文档）
├── .gitignore （Git忽略配置文件）
├── datasets/ （COCO数据集目录，手动创建）
├── weights/ （SAM权重目录，手动创建）
└── results/ （实验结果目录，运行脚本后自动生成）
    └── coco_exp_new_1000/
        ├── annotations/ （1000张样本标注文件）
        ├── images/ （1000张样本图片）
        ├── point_adaptive_complex_table.csv （核心性能表格）
        ├── new_sample_list.txt （1000张样本列表）
        └── new_exp_log.txt （详细实验日志）
```

## 运行步骤
### 1. 配置修改（可选）
若需调整实验参数、路径，可修改`Config`类中的相关配置（如样本数量、复杂图片阈值、自适应选点参数等），默认配置已满足基础实验需求。

### 2. 执行核心脚本
在终端（PowerShell/CMD/Terminal）中进入脚本所在目录，执行以下命令：
```powershell
# 进入代码目录
cd E:\SAM_Model\code
# 运行核心实验脚本
python Point_Adaptive.py
```

### 3. 实验进度查看
脚本运行后，终端会实时打印实验进度：
- 每处理50张图片，输出当前图片的双策略mIoU与复杂度标签
- 自动跳过无法读取的图片与无效实例
- GPU环境下，每处理完一张图片自动清空显存，避免内存溢出

### 4. 结果查看
实验完成后，所有结果保存在`E:\SAM_Model\results\coco_exp_new_1000\`目录下，核心文件：
1.  `point_adaptive_complex_table.csv`：核心性能表格，包含1000张图片的img_id、img_name、双策略mIoU、复杂度标签等信息，可直接用Excel/CSV阅读器打开分析。
2.  `new_exp_log.txt`：详细实验日志，包含实验配置、特征阈值、整体结果汇总、错误信息等，可用于问题排查与实验复盘。

## 结果说明
### 核心指标定义
1.  **Point-only_mIoU**：传统单质心点+随机负点策略的图片级平均IoU，反映SAM默认点提示的分割性能。
2.  **Point(Adaptive)_mIoU**：自适应选点策略的图片级平均IoU，难例采用多核心点+高响应点，易例采用核心区域点，反映优化后的分割性能。
3.  **is_complex_image**：布尔值（True/False），标识该图片是否为复杂图片（难例实例占比≥50%）。
4.  **complex_label**：文本标签（「复杂图片」/「简单图片」），方便快速筛选与统计。

### 典型实验结果趋势
正常运行后，实验结果应满足：
- Point(Adaptive)的整体mIoU高于Point-only，提升幅度通常在0.03~0.08之间。
- 复杂图片的双策略mIoU差距大于简单图片，说明自适应选点对复杂图片的优化效果更显著。
- 复杂图片占比通常在20%~40%之间（受COCO样本分布影响）。

## 代码核心优化点
1.  **选点策略优化**
    - 核心区域距离阈值从0.5提升至0.6，更聚焦实例核心区域，减少边缘噪声点。
    - 难例最大选点数量从2增加至3，覆盖更多连通域，提升不规则实例分割精度。
    - 负点生成增加距离校验（≥50像素），避免负点靠近目标实例，提升提示有效性。

2.  **鲁棒性提升**
    - 增加空掩码、无轮廓、除零错误等异常场景的校验与兜底策略。
    - 坐标越界校验，避免映射后的坐标超出图像尺寸，导致推理报错。
    - 掩码缩放采用最近邻精准插值，保持实例边界的完整性。

3.  **性能与效率优化**
    - GPU环境下启用自动混合精度推理，提升推理速度，减少显存占用。
    - 逐张图片清空显存，支持大样本量长时间运行，避免显存溢出。
    - 过滤小实例（面积<100像素），减少无效计算，提升实验效率。

## 注意事项
1.  **路径格式**：项目默认适配Windows系统（路径分隔符为`\`），若在Linux/macOS下运行，需将`Config`类中的所有路径修改为`/`（如`/home/user/SAM_Model/datasets/COCO`）。
2.  **权重与数据集路径**：确保`SAM_CHECKPOINT_PATH`、`COCO_IMG_PATH`等路径配置正确，否则会出现文件找不到错误。
3.  **样本重复生成**：若已生成1000张样本，脚本会直接加载现有数据，避免重复复制图片与生成标注文件，如需重新生成，可删除`new_sample_list.txt`与`new_ann_path`对应的文件。
4.  **依赖版本问题**：若出现`pycocotools`安装失败，可尝试`pip install pycocotools-windows`（Windows专属）或`conda install -c conda-forge pycocotools`。
5.  **敏感信息**：请勿将个人数据集、密钥等敏感信息提交到GitHub，可通过`.gitignore`文件过滤相关目录/文件。

## 后续扩展方向
1.  支持更多数据集（如VOC、Medical Image）的适配与实验。
2.  增加其他提示策略（如Box、Mask）与自适应选点策略的对比。
3.  实现实验结果的可视化（如分割结果对比图、复杂图片分布热力图）。
4.  支持ViT-L/ViT-H模型的批量实验与性能对比。
5.  增加超参数自动调优，优化自适应选点的核心参数。
