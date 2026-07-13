# Conda 开发环境配置

## 1. 当前已验证环境

- Conda 环境名：`BookOCR`
- Python：3.10.20
- 显卡：NVIDIA GeForce RTX 4060 Laptop GPU，8GB
- PyTorch：2.11.0+cu128
- PaddlePaddle CPU：3.2.2
- Ultralytics：8.4.75
- PaddleOCR：3.7.0
- Streamlit：1.58.0
- Label Studio：1.23.0，使用独立 Conda 环境 `LabelStudio`

PyTorch 已完成真实 GPU 矩阵运算测试。PaddleOCR 使用 CPU，避免 Windows
同一进程内 PyTorch CUDA 12.8 和 Paddle CUDA 12.6 的 cuDNN DLL 冲突。

## 2. PyCharm 切换解释器

在 PyCharm 中依次打开：

```text
文件 → 设置 → 项目: OCR → Python 解释器
```

点击“添加解释器”，选择“添加本地解释器”或“Conda 环境”，然后选择现有环境。

解释器路径：

```text
D:\Downloads\Anaconda\envs\BookOCR\python.exe
```

应用后，在 PyCharm 终端或 Python 控制台执行：

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

预期输出包含：

```text
True
NVIDIA GeForce RTX 4060 Laptop GPU
```

项目采用“YOLO 使用 GPU、PaddleOCR 使用 CPU”的混合方案。OCR 输入是已裁剪的
单个书脊，CPU 推理速度可接受，同时能把 8GB 显存完整留给分割模型。

## 3. 从零重建环境

以下命令应在 Anaconda Prompt 或已初始化 Conda 的 PowerShell 中执行。

```powershell
conda create -n BookOCR python=3.10 pip -y
conda activate BookOCR

python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip install -r requirements-ml.txt
```

不建议直接使用系统 Python 3.13。深度学习库对新 Python 版本的轮子支持可能滞后，
而 Python 3.10 目前对 PyTorch、Ultralytics 和 PaddleOCR 的兼容性更稳妥。

## 4. 环境验证

### PyTorch GPU

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

### PaddlePaddle CPU

```powershell
python -c "import paddle; print(paddle.__version__); print(paddle.device.get_device()); print(paddle.randn([2, 2]))"
```

还需要验证 PyTorch 与 PaddleOCR 能在同一进程导入：

```powershell
python -c "import torch, paddle; from paddleocr import PaddleOCR; print(torch.cuda.is_available(), paddle.device.get_device())"
```

### 项目关键依赖

```powershell
python -c "from ultralytics import YOLO; from paddleocr import PaddleOCR; import streamlit, cv2; print('环境正常')"
python -m pip check
```

## 5. 常用运行命令

```powershell
# 转换 HEIC
python scripts/convert_heic.py

# 转换馆藏 Excel
python scripts/convert_catalog.py

# 生成图像质量清单
python scripts/audit_images.py

# 生成 Label Studio 任务
python scripts/prepare_label_studio_tasks.py

# 启动当前 Streamlit 工程骨架
streamlit run app.py
```

## 6. YOLO 预训练权重

Ultralytics 在第一次使用 `yolov8n-seg.pt` 时会尝试从 GitHub 自动下载权重。
Windows 环境下该自动下载偶尔会被网络重置，报出类似 `Download failure`
或 `ConnectionResetError` 的错误。为了让训练更稳定，本项目把官方权重手动下载到
本地固定路径：

```text
models/weights/yolov8n-seg.pt
```

如需重新下载，可在项目根目录执行：

```powershell
Invoke-WebRequest `
  -Uri "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n-seg.pt" `
  -OutFile "models\weights\yolov8n-seg.pt"
```

下载后验证：

```powershell
python -c "from ultralytics import YOLO; m=YOLO('models/weights/yolov8n-seg.pt'); print(m.task)"
```

预期输出包含：

```text
segment
```

启动模型训练脚本默认会加载这份本地权重：

```powershell
python scripts/train_bootstrap_yolo.py
```

如果权重确实无法下载，也可以临时从零训练：

```powershell
python scripts/train_bootstrap_yolo.py --model yolov8n-seg.yaml --name book_spine_bootstrap_v1_scratch
```

但在当前书脊数据量较小的阶段，预训练权重效果明显更好。

## 7. Label Studio 标注环境

Label Studio 单独安装在 `LabelStudio` 环境中，避免它的大量 Web 依赖影响
`BookOCR` 训练和推理环境。

已验证解释器和命令位置：

```text
D:\Downloads\Anaconda\envs\LabelStudio\python.exe
D:\Downloads\Anaconda\envs\LabelStudio\Scripts\label-studio.exe
```

启动标注服务：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start_label_studio.ps1
```

启动后访问：

```text
http://127.0.0.1:8080
```

首次打开需要注册一个本地账号。之后创建项目，导入：

```text
configs/label_studio/book_spine.xml
data/interim/label_studio_tasks.json
```

本地图片访问依赖以下环境变量，启动脚本会自动设置：

```powershell
$env:LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED = "true"
$env:LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT = "D:\IT\OCR"
```

Label Studio 数据库和媒体文件保存到：

```text
data/interim/label_studio_data/
```

该目录属于本地中间数据，不上传 GitHub。
