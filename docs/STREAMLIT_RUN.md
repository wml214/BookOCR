# Streamlit 可视化界面运行说明

当前 `app.py` 已经从工程骨架页升级为可运行的图书盘点界面，包含：

- 多图上传，支持 JPG、JPEG、PNG、HEIC、HEIF；
- OBB 模型权重、馆藏目录、检测阈值和匹配阈值配置；
- 书脊检测、书脊裁剪、PaddleOCR、馆藏目录匹配；
- 可选相邻图像空间去重；
- 汇总结果、识别明细、检测预览图、配准日志展示；
- 原始盘点结果和去重盘点结果 CSV 下载。

## 启动命令

在项目根目录 `D:\IT\OCR` 中运行：

```powershell
D:\Downloads\Anaconda\envs\BookOCR\python.exe -m streamlit run app.py
```

浏览器会自动打开。如果没有自动打开，手动访问终端中显示的地址，通常是：

```text
http://localhost:8501
```

## 推荐默认参数

```text
OBB 模型权重：D:\IT\OCR\models\weights\book_spine_obb_final_200_best.pt
馆藏目录 CSV：D:\IT\OCR\data\processed\catalog\catalog.csv
推理设备：0
检测置信度 conf：0.60
NMS IoU：0.50
推理尺寸 imgsz：640
书名自动匹配阈值：0.72
启用相邻图像去重：是
去重空间 IoU 阈值：0.45
最低配准内点比例：0.15
```

## 使用流程

1. 点击上传区域，按拍摄顺序选择多张书架图片。
2. 在侧边栏确认模型权重和馆藏目录路径。
3. 保持默认阈值，点击“开始盘点”。
4. 等待系统完成检测、OCR、匹配和可选去重。
5. 在“汇总结果”查看书名和册数。
6. 在“识别明细”查看每个书脊的 OCR 原文、匹配结果和置信度。
7. 在“标注预览”查看带 OBB 框的结果图。
8. 在“运行日志”查看相邻图像配准和重复合并记录。
9. 下载 CSV 作为课程验收或论文实验结果材料。

## 输出目录

每次运行都会自动创建一个独立目录：

```text
outputs/inventory/streamlit_YYYYMMDD_HHMMSS/
```

主要文件包括：

- `盘点明细.csv`
- `盘点结果.csv`
- `previews/`
- `crops/`
- `uploads/`

如果启用去重，还会生成：

- `dedup/去重明细.csv`
- `dedup/重复合并记录.csv`
- `dedup/图像配准日志.csv`
- `dedup/去重盘点结果.csv`

## 注意事项

- 首次点击“开始盘点”会加载 PaddleOCR 模型，可能会比较慢。
- 多图去重依赖上传顺序，上传顺序应与拍摄顺序一致。
- 如果相邻图像公共区域太少，系统会保留独立计数，并在配准日志中显示 `not_enough_inliers`。
- 如果只上传 1 张图片，系统会自动跳过去重。
