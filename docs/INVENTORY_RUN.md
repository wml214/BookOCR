# 图书盘点流水线运行说明

当前盘点流水线已经不是“只能跑 demo”。`scripts/run_inventory_demo.py` 用于快速测试，
正式运行建议使用 `scripts/run_inventory.py`。

## 快速检查输入

```powershell
D:\Downloads\Anaconda\envs\BookOCR\python.exe scripts\run_inventory.py --dry-run
```

## 快速跑前 3 张

```powershell
D:\Downloads\Anaconda\envs\BookOCR\python.exe scripts\run_inventory.py --limit 3
```

## 跑完整目录

```powershell
D:\Downloads\Anaconda\envs\BookOCR\python.exe scripts\run_inventory.py --limit 0
```

完整目录会比较慢。当前 1 张图大约几十秒，202 张图可能需要较长时间。

## 输出内容

默认输出到：

```text
outputs/inventory/run_年月日_时分秒/
```

主要文件：

- `盘点明细.csv`：每个书脊的检测置信度、OCR 原文、匹配书名和匹配状态。
- `盘点结果.csv`：按规范书名汇总后的册数、平均匹配置信度、OCR 原文和来源图像。
- `previews/`：带 OBB 框的检测预览图。
- `crops/`：裁剪后的单本书脊图。

## 当前边界

- 已完成：OBB 检测、书脊裁剪、PaddleOCR、馆藏匹配、单图/多图独立计数、CSV 导出。
- 尚未完成：相邻图片重叠区域去重、Streamlit 图形界面、人工修正界面。
- `pending` 表示证据不足，不强行匹配；这是为了避免把错误结果算进自动匹配。
