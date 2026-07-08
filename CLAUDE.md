# 项目当前状态说明

> 本文件用于让后续开发者或智能编码助手快速理解项目现状、约束和下一步工作。每完成一个可验收阶段，都应同步更新“当前进度”和“下一步任务”。

## 1. 项目概述

- 项目名称：基于视觉的图书盘点系统。
- 实现路线：基于多张有序图像的书脊分割、OCR、馆藏匹配、跨图去重和计数。
- 目标用户：课程验收教师及图书盘点操作人员。
- 开发环境：Windows、PowerShell、Python、RTX 4060 8GB。
- Conda 环境：`BookOCR`，Python 3.10.20。
- 标注环境：`LabelStudio`，Python 3.10.20，Label Studio 1.23.0。
- 界面方案：Streamlit。
- 分割/定位模型：保留 YOLOv8n-seg 作为分割基线，新增 YOLOv8n-OBB 作为倾斜书脊主推路线。
- OCR 方案：PaddleOCR。
- 当前日期：2026-07-02。

## 2. 已具备资源

### 实验材料

- 实验指导书：`2026春季学期项目制实践实验指导书 - 副本.pdf`。
- 原始图像目录：`图像采集/图像采集/`。
- 原始图像数量：102 张 HEIC。
- 图像压缩包：`图像采集.zip`，内容与已解压图片属于同一批数据，不能重复计数。
- 补拍图像目录：`data/raw/supplement_20260625/`。
- 补拍图像数量：100 张 JPG。
- 当前进入中间数据流水线的独立图像总数：202 张，已满足不少于 200 张的最低要求。

### 馆藏目录

- 文件：`泰达西区库-馆藏清单-按册20260522-005/泰达西区库-馆藏清单-按册20260522-005.xlsx`。
- 工作表：`Sheet0`。
- 表头位于第 5 行，数据从第 6 行开始。
- 目录说明总计约 59,078 种、77,595 册。
- OCR 匹配的核心字段：
  - `D列：索书号`
  - `F列：条码号`
  - `G列：题名`
  - `H列：责任者`
  - `J列：出版社`
  - `K列：出版年`
  - `L列：标准号`
  - `V列：馆藏地`
  - `AB列：书刊状态`

原始 Excel 体积较大，后续不得在每次推理时直接全量读取。应由预处理脚本一次性生成经过清洗、去重和索引优化的 `data/processed/catalog/catalog.csv`。

## 3. 实验硬约束

- 图像方案数据集不得少于 200 张。
- 必须标注书脊轮廓、书名及对应数量。
- 必须至少训练一个神经网络模型。
- 软件必须能充分展示算法运行结果。
- 论文必须严格按照《电子学报》模板撰写。
- 数据不可与其他小组共享或高度重复。
- 代码压缩包不得包含原始图片和视频。

## 4. 当前技术决策

- 先以 YOLOv8n-seg、640 像素输入建立基线；Windows 本机训练暂时关闭
  AMP，以减少自动下载和 CUDA 自检带来的不稳定因素。
- 由于书架中存在大量倾斜摆放书脊，新增 YOLOv8n-OBB 旋转框模型；
  后续盘点软件优先考虑用 OBB 做书脊定位、计数和 OCR 旋转裁剪。
- RTX 4060 8GB 可以完成该模型训练；优先使用自动 batch，显存不足时逐步降低 batch。
- YOLO/PyTorch 使用 GPU，PaddleOCR 使用 CPU，避免 Windows 同进程加载不同
  CUDA/cuDNN 版本产生 DLL 冲突，并把显存优先留给分割模型。
- 数据划分必须按书架场景进行，禁止把同一连续书架的相邻图片分到不同集合。
- OCR 只负责提供候选文本，最终书名通过馆藏目录相似度匹配确定。
- 低置信度结果不得强行匹配，应保留为“待确认”。
- 跨图去重必须同时考虑空间关系和书名相似度，不能仅按书名合并。
- 原始文件保持只读，所有衍生数据写入 `data/interim` 和 `data/processed`。
- 所有核心代码必须包含详细中文注释、中文 docstring 和类型标注。

## 5. 目录职责

```text
OCR/
├─ app.py                         # Streamlit 应用入口
├─ configs/                       # 路径、模型和阈值配置
├─ src/book_inventory/
│  ├─ data/                       # 图像、标注、馆藏目录预处理
│  ├─ detection/                  # YOLO 分割训练与推理
│  ├─ ocr/                        # 书脊裁剪、方向校正和 OCR
│  ├─ matching/                   # 文本规范化与馆藏匹配
│  ├─ dedup/                      # 相邻图像配准和跨图去重
│  ├─ pipeline/                   # 完整盘点流水线
│  └─ ui/                         # Streamlit 页面组件
├─ scripts/                       # 可独立运行的数据和训练脚本
├─ tests/                         # 单元测试与小型测试数据
├─ data/
│  ├─ raw/                        # 原始数据入口，不纳入 Git
│  ├─ interim/                    # HEIC 转换等中间结果
│  └─ processed/                  # 标注、划分和精简馆藏目录
├─ models/
│  ├─ configs/                    # YOLO 数据集与训练配置
│  └─ weights/                    # 模型权重，不纳入 Git
├─ outputs/                       # 预测图、CSV、日志和报告数据
└─ docs/                          # 论文和答辩材料
```

## 6. 当前进度

- [x] 阅读并核对实验指导书。
- [x] 确定采用多图图像盘点路线。
- [x] 确认 RTX 4060 8GB 可训练 YOLOv8n-seg。
- [x] 确认已有 102 张 HEIC 图像。
- [x] 找到并识别馆藏 Excel 的核心字段。
- [x] 建立项目目录骨架。
- [x] 编写实施计划和项目状态说明。
- [x] 实现 HEIC 批量转换脚本。
- [x] 将 102 张 HEIC 成功转换为 4032×3024 JPEG，零失败。
- [x] 实现馆藏 Excel 精简与书目合并脚本。
- [x] 将 77,595 册馆藏合并为 59,078 条书目记录。
- [x] 创建独立 Conda 环境 `BookOCR`。
- [x] 验证 PyTorch 2.11.0 CUDA 12.8 可调用 RTX 4060。
- [x] 验证 PaddlePaddle CPU 3.2.2 可与 PyTorch GPU 在同一进程稳定运行。
- [x] 安装并验证 Ultralytics、PaddleOCR 和 Streamlit。
- [x] 完成 102 张 JPEG 的基础质量与相邻特征审计。
- [x] 生成 Label Studio 标注配置和 102 条导入任务。
- [x] 补采 100 张 JPG，并放入 `data/raw/supplement_20260625/`。
- [x] 升级图片整理脚本，使其可统一处理 HEIC、HEIF、JPG、JPEG 和 PNG。
- [x] 将 202 张图片统一整理到 `data/interim/images_jpg/`，零失败。
- [x] 完成 202 张 JPEG 的基础质量与相邻特征审计。
- [x] 重新生成 Label Studio 标注配置和 202 条导入任务。
- [x] 创建独立 Conda 环境 `LabelStudio`。
- [x] 安装并验证 Label Studio 1.23.0。
- [x] 新增 `scripts/start_label_studio.ps1`，用于启动本项目标注服务。
- [x] 启动 Label Studio 并确认 `http://127.0.0.1:8080` 返回 200。
- [x] 完成首批 30 张书脊多边形人工标注。
- [x] 新增 `scripts/export_label_studio_to_yolo.py`，可从 Label Studio SQLite
  导出 YOLOv8-seg 数据集。
- [x] 导出启动数据集 `data/processed/dataset/bootstrap_v1`：
  30 张图片、625 个书脊实例，划分为 train/val/test = 24/3/3。
- [x] 生成数据集配置 `models/configs/book_spine_bootstrap_v1.yaml`。
- [x] 新增 `scripts/train_bootstrap_yolo.py`，用于稳定训练启动模型。
- [x] 训练从零初始化的 YOLOv8n-seg 启动模型，并复制权重到
  `models/weights/book_spine_bootstrap_v1_scratch_fast_best.pt`。
- [x] 手动下载官方 `yolov8n-seg.pt` 预训练权重到
  `models/weights/yolov8n-seg.pt`，解决 Ultralytics 自动下载偶发失败问题。
- [x] 用本地预训练权重重新训练启动模型，并复制权重到
  `models/weights/book_spine_bootstrap_v1_pretrained_best.pt`。
- [x] 在 3 张测试图、53 个书脊实例上完成启动模型测试：
  box mAP@0.5 = 0.935，mask mAP@0.5 = 0.842。
- [x] 人工标注扩展到 65 张。
- [x] 基于 65 张标注重新训练 v2 辅助预标模型，并保存为
  `models/weights/book_spine_v2_65_best.pt`。
- [x] v2 模型验证集结果：6 张图、125 个书脊实例；
  box mAP@0.5 = 0.951，mask mAP@0.5 = 0.887。
- [x] 新增 `scripts/generate_label_studio_predictions.py`，可将 YOLOv8-seg
  预测结果写入 Label Studio prediction 表。
- [x] 曾为 137 个未完成任务生成 `book_spine_v2_65` 预标注；实际人工检查效果较差，
  修改成本高于重画，已从 Label Studio 数据库清除，人工标注保留不变。
- [x] 人工书脊多边形标注扩展到 100 张。
- [x] 重新导出 YOLOv8-seg 数据集并训练 `book_spine_v3_100_best.pt`；
  test 集 10 张图、230 个书脊，mask mAP@0.5 = 0.937。
- [x] 新增 `scripts/export_label_studio_to_yolo_obb.py`，可将人工多边形标注转换为
  YOLOv8-OBB 旋转框数据集。
- [x] 新增 `scripts/train_yolo_obb.py`，用于稳定训练书脊 OBB 模型。
- [x] 下载官方 `yolov8n-obb.pt` 到 `models/weights/yolov8n-obb.pt`。
- [x] 导出 OBB 数据集 `data/processed/dataset/obb_v1`：
  100 张图片、2193 个旋转框，划分为 train/val/test = 80/10/10。
- [x] 训练 OBB 模型并保存为 `models/weights/book_spine_obb_v1_100_best.pt`；
  test 集 10 张图、217 个书脊，OBB mAP@0.5 = 0.974，mAP@0.5:0.95 = 0.740。
- [x] 新增 `scripts/generate_label_studio_obb_predictions.py`，可将 OBB 四点框写入
  Label Studio prediction 表。
- [x] 已用 `book_spine_obb_v1_100_best.pt` 为剩余 102 张未完成任务生成 OBB 预标注，
  共 2329 个预测书脊，平均置信度约 0.831。
- [x] 202 张任务中已完成人工确认 200 张，并导出最终 OBB 数据集：
  train/val/test = 160/20/20，共 4537 个书脊旋转框。
- [x] 训练最终 OBB 模型并保存为 `models/weights/book_spine_obb_final_200_best.pt`；
  test 集 20 张图、456 个书脊，Precision = 0.966，Recall = 0.941，
  mAP@0.5 = 0.988，mAP@0.5:0.95 = 0.826。
- [x] 最终 OBB 模型在 test 集按 `conf=0.60` 推理时，预测 452 本、真值 456 本，
  总计数准确率约 99.1%。
- [x] 清理模型与标注缓存：`models/weights` 仅保留最终 OBB、seg 基线和官方预训练权重；
  `runs` 中重复 `.pt` 已删除但训练曲线与评估图保留；Label Studio 历史中间备份已清理。
- [x] 新增基于最终 OBB 模型的书脊检测封装 `src/book_inventory/detection/obb_detector.py`。
- [x] 新增 OBB 四点框透视矫正与书脊裁剪模块 `src/book_inventory/ocr/spine_cropper.py`。
- [x] 新增 `scripts/demo_obb_crop.py`，已在 2 张样例图上生成检测预览图、书脊裁剪图和裁剪清单。
- [x] 新增 PaddleOCR 书脊识别封装 `src/book_inventory/ocr/paddle_reader.py`。
- [x] 新增 `scripts/demo_spine_ocr.py`，已完成 10 张书脊裁剪图 OCR 冒烟测试并导出
  `outputs/ocr/ocr_demo/spine_ocr_results.csv`。
- [x] 新增馆藏目录匹配模块 `src/book_inventory/matching/catalog_matcher.py`，使用字符片段倒排索引召回候选，
  再结合完整串相似度、最长公共片段覆盖率和字符 Jaccard 分数排序。
- [x] 新增 `scripts/demo_catalog_match.py`，已对 10 条 OCR demo 结果完成馆藏匹配测试；
  明确匹配样例自动标记为 `matched`，不确定样例标记为 `pending`。
- [x] 优化馆藏匹配策略：由单一书名相似度升级为“书名相似度 + 作者片段 + 索书号种次号”的分层证据；
  避免短词和出版社信息误匹配，同时可处理书名 OCR 较差但作者/索书号可用的样例。
- [x] 新增端到端盘点流水线 `src/book_inventory/pipeline/inventory_pipeline.py`。
- [x] 新增 `scripts/run_inventory_demo.py`，已对 1 张样例图完成
  OBB 检测、书脊裁剪、OCR、馆藏匹配和 CSV 汇总；当前检测 19 个书脊，其中 17 个自动匹配、
  2 个待确认，输出 `outputs/inventory/demo/盘点明细.csv` 和 `outputs/inventory/demo/盘点结果.csv`。
- [x] 新增正式盘点入口 `scripts/run_inventory.py`，默认可处理完整 `data/interim/images_jpg/` 目录，
  支持 `--limit`、`--dry-run` 和带时间戳输出目录；`--dry-run` 已确认当前可处理 202 张图。
- [x] 新增运行说明 `docs/INVENTORY_RUN.md`，记录快速测试、完整运行、输出内容和当前边界。
- [x] 新增 `scripts/create_test_title_review.py`，已为最终 OBB test 集 20 张图生成
  `outputs/evaluation/title_review_test20/书名核验表.csv`；当前包含 452 个书脊裁剪图对应记录，
  其中 434 条自动 matched、18 条 pending；人工核验 matched 行无误，当前正式书名识别准确率为
  434/452 = 96.02%，若以 test 集真实 456 个书脊为端到端分母则为 95.18%。
- [ ] 完成标注规范、标注及数据集划分。
- [x] 扩大标注数据后重新训练并评估 YOLOv8n-OBB / YOLOv8n-seg。
- [x] 实现单图/多图独立计数和 CSV 汇总。
- [ ] 实现跨图重叠区域去重计数。
- [ ] 完成 Streamlit 界面。
- [ ] 完成完整测试、论文和答辩材料。

## 7. 下一步任务

下一阶段按以下顺序实施：

1. 实现相邻图像重叠区域的去重计数逻辑，并对未能可靠配准的图像给出提示。
2. 为正式流水线增加 OCR 缓存/断点续跑，避免全量运行时重复 OCR 已处理书脊。
3. 在 Streamlit 中集成 OBB 检测、书脊裁剪、OCR、匹配、人工修正和 CSV 导出。
4. 对独立测试集统计 OCR 文本准确率、书名匹配准确率和最终计数准确率。
5. 整理论文实验表格和 5–8 分钟答辩演示流程。

当前启动模型说明：

- Ultralytics 自动下载 `yolov8n-seg.pt` 时曾出现 GitHub 连接被重置；
  已改为手动下载官方权重并固定为本地路径
  `models/weights/yolov8n-seg.pt`。
- 训练命令使用 `workers=0`、`amp=False` 以避免 Windows 下卡住或触发额外下载。
- 当前推荐定位权重为
  `models/weights/book_spine_obb_final_200_best.pt`。
- 当前 seg 基线权重为
  `models/weights/book_spine_v3_100_best.pt`。
- OBB 对倾斜书脊明显优于 seg 轮廓显示，更适合作为后续 OCR 裁剪入口。
- `book_spine_v2_65` 预标注效果未达到可辅助标注要求，已清除 prediction 表；
  当前 Label Studio 状态为 202 个任务、100 个已人工完成、102 个 OBB 预标注。
- OBB 预标注数据库备份为
  `data/interim/label_studio_data/label_studio_before_obb_predictions_20260702_131638.sqlite3`。
- 最终 OBB 推理建议默认阈值：`conf=0.60`，`iou=0.50`。
- 当前 Label Studio 主数据库为
  `data/interim/label_studio_data/label_studio.sqlite3`；
  清理后额外保留一份最终 200 张标注备份
  `data/interim/label_studio_data/label_studio_final_200_backup_20260702_180337.sqlite3`。
- 清除劣质预标注前的数据库备份为
  `data/interim/label_studio_data/label_studio_before_clear_bad_predictions_20260701_170301.sqlite3`。

Label Studio 当前启动方式：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start_label_studio.ps1
```

访问地址：

```text
http://127.0.0.1:8080
```

## 7.1 已生成的本地数据

以下文件是可以通过脚本重新生成的本地中间产物，已由 `.gitignore` 排除，
不应上传 GitHub：

- JPEG 图像：`data/interim/images_jpg/`，共 202 张。
  - 原 102 张 HEIC 转换图：4032×3024。
  - 补拍 100 张 JPG 整理图：1706×1279。
- 转换清单：`data/interim/heic_conversion_manifest.csv`。
- 精简馆藏：`data/processed/catalog/catalog.csv`，共 59,078 条书目记录。
- 图像审计清单：`data/interim/image_audit.csv`，共 202 条。
- 图像审计汇总：`data/interim/image_audit_summary.json`。
- Label Studio 任务：`data/interim/label_studio_tasks.json`，共 202 条。

自动审计摘要：

- 图像尺寸包含 4032×3024 和 1706×1279 两种规格，训练前由 YOLO 数据加载器统一缩放。
- 平均亮度为 122.957，平均对比度为 66.490。
- 拉普拉斯清晰度中位数为 436.995。
- 相邻图像单应性内点率中位数为 14.6341%。
- 按当前保守阈值未发现明显模糊或极端曝光图像。
- 自动审计不能替代书脊完整性、遮挡和场景覆盖的人工检查。

可重复执行的命令：

```powershell
python scripts/convert_heic.py
python scripts/convert_catalog.py
python scripts/audit_images.py
python scripts/prepare_label_studio_tasks.py
```

HEIC 转换依赖 `pillow-heif`。这批原图使用网格结构，不能使用 FFmpeg 的
默认视频流直接转换，否则只会得到 512×512 的局部图块。

## 8. 已知风险

- 当前已有 202 张图像，已达到最低数量要求；但仍需通过人工复核剔除无效样本，
  确保最终可标注图像数量仍不少于 200 张。
- 自动质量审计未发现明显模糊或极端曝光图像，但书脊完整性、遮挡和场景覆盖
  仍需人工确认。
- 馆藏目录中同名书、分册名、副标题和不同版本可能造成匹配歧义。
- 竖排文字、反光、细小字体和遮挡会降低 OCR 准确率。
- 相邻图片重叠不足时无法可靠估计单应性矩阵。
- 补拍 JPG 文件名不含拍摄顺序，且未保留 EXIF 时间；用于跨图去重演示时，
  需要在界面中手动调整顺序，或后续补充一份人工排序清单。
- 《电子学报》模板目前尚未出现在项目目录中。

## 9. 更新规则

- 完成任务后立即更新第 6 节复选框。
- 技术路线或阈值发生变化时更新第 4 节，并说明原因。
- 新增数据、模型和测试结果时记录准确数量与指标，不能使用模糊表述。
- 不得把密码、密钥、个人隐私或大体积原始数据写入本文件。
