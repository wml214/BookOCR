# 跨图像去重运行说明

本阶段新增了相邻图像空间去重能力，用于处理多张有序书架照片之间的重叠区域。它不会只按书名合并，而是同时满足两个条件才合并：

1. 两个书脊都已经自动匹配到同一个馆藏规范书名；
2. 相邻照片通过 ORB + RANSAC 单应性矩阵配准后，两个 OBB 书脊框的空间 IoU 达到阈值。

这样可以避免把“两本同名实体书”误合并，也可以避免在图像配准不可靠时误删书。

## 1. 先重新运行盘点流水线

去重脚本依赖新版 `盘点明细.csv` 中的 `obb_points_json` 字段。如果旧结果中没有这个字段，需要先重新跑一次正式盘点：

```powershell
D:\Downloads\Anaconda\envs\BookOCR\python.exe scripts\run_inventory.py --limit 0 --device 0 --conf 0.60 --iou 0.50
```

如果只是快速检查，可以先跑少量图片：

```powershell
D:\Downloads\Anaconda\envs\BookOCR\python.exe scripts\run_inventory.py --limit 3 --output-dir outputs\inventory\dedup_check --device 0 --conf 0.60 --iou 0.50
```

## 2. 对某次盘点结果执行去重

假设上一步输出目录为 `outputs/inventory/run_YYYYMMDD_HHMMSS`，运行：

```powershell
D:\Downloads\Anaconda\envs\BookOCR\python.exe scripts\apply_inventory_dedup.py --inventory-dir outputs\inventory\run_YYYYMMDD_HHMMSS --image-dir data\interim\images_jpg
```

默认输出到：

```text
outputs/inventory/run_YYYYMMDD_HHMMSS/dedup/
```

## 3. 输出文件

- `去重明细.csv`：在原盘点明细基础上增加 `dedup_status` 和 `duplicate_of_item_id` 字段。
- `重复合并记录.csv`：记录被合并的书脊对、来源图像和空间 IoU。
- `图像配准日志.csv`：记录相邻图像的匹配点数、内点数、内点比例和配准状态。
- `去重盘点结果.csv`：去重后重新按书名汇总册数。

## 4. 当前默认阈值

```text
spatial_iou = 0.45
min_inlier_ratio = 0.15
```

如果 `图像配准日志.csv` 中出现 `not_enough_inliers`，说明相邻图像的公共区域或可匹配特征不足，本次不会强行合并。报告中可以说明：系统会对无法可靠配准的相邻图像保留独立计数，并提示存在潜在重复。

## 5. 已完成烟雾测试

已用 2 张样例图完成链路测试：

```text
outputs/inventory/dedup_smoke/
outputs/inventory/dedup_smoke/dedup/
```

测试结果显示脚本能够正常读取新版明细、执行图像配准、写出四类去重文件。该样例中相邻图像内点比例为 0.0962，低于默认阈值，因此没有执行合并，属于预期的安全行为。
