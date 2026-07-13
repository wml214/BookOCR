"""训练 YOLOv8n-OBB 书脊旋转框模型。

本脚本用于解决倾斜书脊场景中普通水平框或分割 mask 不稳定的问题。
OBB 模型输出 4 点旋转矩形，适合后续进行书脊裁剪、方向校正和 OCR。

训练前请先运行：

```powershell
python scripts/export_label_studio_to_yolo_obb.py
```
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    """解析训练参数。"""

    parser = argparse.ArgumentParser(description="训练 YOLOv8n-OBB 书脊旋转框模型。")
    parser.add_argument(
        "--model",
        default=str(PROJECT_ROOT / "models/weights/yolov8n-obb.pt"),
        help="OBB 预训练权重路径，默认使用本地 yolov8n-obb.pt。",
    )
    parser.add_argument(
        "--data",
        default=str(PROJECT_ROOT / "models/configs/book_spine_obb_v1.yaml"),
        help="Ultralytics OBB 数据集 YAML。",
    )
    parser.add_argument("--epochs", type=int, default=80, help="最大训练轮数。")
    parser.add_argument("--imgsz", type=int, default=640, help="输入图像尺寸。")
    parser.add_argument("--batch", type=int, default=4, help="批大小。")
    parser.add_argument("--device", default="0", help="训练设备，RTX 4060 使用 0。")
    parser.add_argument(
        "--name",
        default="book_spine_obb_v1_100",
        help="训练运行名称。",
    )
    return parser.parse_args()


def main() -> None:
    """启动 OBB 模型训练。"""

    args = parse_args()
    model_path = Path(args.model)
    if args.model.endswith(".pt") and not model_path.exists():
        raise FileNotFoundError(
            "找不到本地 OBB 预训练权重："
            f"{model_path}\n"
            "请先下载 yolov8n-obb.pt 到 models/weights/，"
            "或传入 --model yolov8n-obb.yaml 从零训练。"
        )

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=0,
        project=str(PROJECT_ROOT / "runs/obb"),
        name=args.name,
        exist_ok=True,
        patience=20,
        amp=False,
        pretrained=args.model.endswith(".pt"),
        deterministic=False,
        seed=20260702,
    )


if __name__ == "__main__":
    main()
