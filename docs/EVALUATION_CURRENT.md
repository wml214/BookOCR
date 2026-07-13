# 当前阶段实验指标评估说明

更新时间：2026-07-03

本文档用于课程报告“实验指标”部分。当前项目已经具备两类指标：

1. **书脊检测/计数指标**：基于最终 YOLOv8n-OBB 模型和独立 test 集，可以作为正式测试集指标。
2. **书名识别/馆藏匹配指标**：当前先基于 1 张端到端样例图进行人工核验式评估；若要作为最终论文指标，应补充逐本书名 ground truth 后重新统计。

## 1. 基础公式

按实验指导图中的定义：

```text
识别准确率 = 正确识别的图书本数 / 测试集中总图书数量
计数准确率 = 正确计数的图书种类数量 / 测试集中图书种类数量
```

检测模型常用指标：

```text
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 * Precision * Recall / (Precision + Recall)
```

## 2. 书脊检测模型指标

最终模型：

```text
models/weights/book_spine_obb_final_200_best.pt
```

测试集：

```text
data/processed/dataset/obb_v1/images/test
```

测试集规模：

```text
20 张图像，456 个书脊标注实例
```

检测指标：

| 指标 | 数值 |
| --- | ---: |
| Precision | 0.966 |
| Recall | 0.941 |
| F1 | 0.953 |
| mAP@0.5 | 0.988 |
| mAP@0.5:0.95 | 0.826 |

可用于报告的图：

- `outputs/reports/book_spine_obb_final_200_test/confusion_matrix.png`
- `outputs/reports/book_spine_obb_final_200_test/confusion_matrix_normalized.png`
- `outputs/reports/book_spine_obb_final_200_test/BoxPR_curve.png`
- `outputs/reports/book_spine_obb_final_200_test/BoxP_curve.png`
- `outputs/reports/book_spine_obb_final_200_test/BoxR_curve.png`
- `outputs/reports/book_spine_obb_final_200_test/BoxF1_curve.png`

说明：当前检测任务只有 `book_spine` 一个类别，因此这里的混淆矩阵是“书脊 / 背景”的检测混淆矩阵，不是多书名类别混淆矩阵。

## 3. 书脊数量计数准确率

在 test 集上，使用最终 OBB 模型按推荐阈值推理：

```text
conf = 0.60
iou  = 0.50
```

统计结果：

| 项目 | 数值 |
| --- | ---: |
| test 集真实书脊数 | 456 |
| 模型预测书脊数 | 452 |
| 数量计数准确率 | 452 / 456 = 99.12% |

该指标反映“书脊检测数量”是否接近真实数量，适合放在报告的检测计数部分。

## 4. 当前端到端书名识别 / 匹配效果

### 4.1 单张 demo 样例

```text
outputs/inventory/demo/盘点结果.csv
```

样例图：

```text
012de8a7a2bb4a14199cc387627b311.jpg
```

统计结果：

| 项目 | 数值 |
| --- | ---: |
| 样例图检测书脊数 | 19 |
| 自动匹配成功数 | 17 |
| 待确认数 | 2 |
| 当前样例识别准确率 | 17 / 19 = 89.47% |

当前仍为 `pending` 的样例：

| OCR 原文 | 当前状态 | 原因 |
| --- | --- | --- |
| 成教第 / 清华大学出版社北京交通大学出 / F272 / 202 | pending | OCR 主标题信息不足，强行匹配容易误匹配 |
| 主编李珍陈 | pending | 只识别到作者片段，缺少书名和可靠索书号 |

说明：这里的 89.47% 是“当前端到端样例图的自动匹配通过率 / 人工核验口径”，还不能直接等同于完整测试集书名识别准确率。最终论文若要更严谨，应在独立测试集中为每本书脊补充标准书名 ground truth。

### 4.2 20 张 test 图核验表

已经为最终 OBB test 集 20 张图生成书名核验表：

```text
outputs/evaluation/title_review_test20/书名核验表.csv
```

同时生成：

```text
outputs/evaluation/title_review_test20/crops/      # 452 个书脊裁剪图
outputs/evaluation/title_review_test20/previews/   # 20 张检测预览图
```

当前自动输出统计与人工核验结论：

| 项目 | 数值 |
| --- | ---: |
| test 集图像数 | 20 |
| test 集真实书脊标注数 | 456 |
| 系统检测并生成的书脊裁剪图 | 452 |
| 自动匹配为 matched | 434 |
| 自动保留为 pending | 18 |
| 自动匹配通过率 | 434 / 452 = 96.02% |
| 人工核验确认错误匹配数 | 0 |
| 以已检测书脊为分母的书名识别准确率 | 434 / 452 = 96.02% |
| 以 test 集真实书脊为分母的端到端识别准确率 | 434 / 456 = 95.18% |

人工核验后，`matched` 行未发现错误匹配；18 条 `pending` 作为未自动识别处理。
因此正式书名识别准确率可按两个口径报告：

```text
书名识别准确率（已检测书脊口径） = 434 / 452 = 96.02%
端到端识别准确率（真实书脊口径） = 434 / 456 = 95.18%
```

报告中建议优先写“已检测书脊口径”的书名识别准确率，同时补充说明检测阶段仍有 4 个书脊未检出。

## 5. 图书种类计数准确率

当前单张 demo 样例中每个书名均为 1 册，因此在人工核验口径下：

```text
图书种类数 = 19
正确计数并匹配的图书种类数 = 17
当前样例图书种类计数准确率 = 17 / 19 = 89.47%
```

该指标会同时受三部分影响：

1. 书脊是否检测出来；
2. OCR 是否读出足够文本；
3. 馆藏目录匹配是否匹配到正确规范书名。

在 20 张 test 图上，由于存在同名书、重复书名和 pending 样本，严格的“图书种类计数准确率”
需要在 `书名核验表.csv` 中补全 18 条 pending 的真实书名后统计。当前可先报告：

```text
书脊数量计数准确率 = 452 / 456 = 99.12%
书名自动识别准确率 = 434 / 452 = 96.02%
```

## 6. 多类别混淆矩阵的当前状态

如果报告只写当前结果，可以展示：

- 单类别书脊检测混淆矩阵；
- 端到端样例中的 matched / pending 统计表；
- 若干正确匹配与 pending 案例。

如果想冲高分，建议补充“书名类别混淆矩阵”。需要额外准备一份人工核验文件，至少包含：

```text
source_image, spine_index, true_title, predicted_title, status
```

其中：

- `true_title`：人工核验的标准书名；
- `predicted_title`：系统输出的规范书名；
- `status`：correct / wrong / pending。

然后可以按书名类别生成混淆矩阵：

- 行：真实书名；
- 列：预测书名；
- 对角线：识别正确；
- 非对角线：误匹配；
- pending 可单独作为一列，表示未自动识别。

## 7. 报告中建议写法

可以在论文中写成：

> 在书脊定位任务上，本文采用 YOLOv8n-OBB 模型。最终模型在 20 张独立测试图像、456 个书脊实例上取得 Precision=0.966、Recall=0.941、mAP@0.5=0.988、mAP@0.5:0.95=0.826。按 conf=0.60、iou=0.50 推理时，预测书脊数为 452，真实书脊数为 456，书脊数量计数准确率为 99.12%。

> 在端到端盘点样例中，系统对 1 张书架图检测到 19 个书脊，其中 17 个自动匹配到馆藏规范书名，2 个因 OCR 信息不足保留为待确认，当前样例识别准确率为 89.47%。后续可通过补充独立测试集书名 ground truth，进一步统计多类别书名识别混淆矩阵。
