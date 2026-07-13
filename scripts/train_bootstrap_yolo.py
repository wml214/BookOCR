"""训练第一版 YOLOv8n-seg 书脊分割启动模型。

用途：
    当 Label Studio 中已有一批人工书脊多边形后，先导出为
    ``data/processed/dataset/bootstrap_v1``，再运行本脚本训练启动模型。

注意：
    本脚本默认加载已经手动下载到本地的 ``models/weights/yolov8n-seg.pt``
    预训练权重，避免训练时再由 Ultralytics 自动访问 GitHub 下载。
    如果这份本地权重不存在，脚本会给出明确报错；此时请先按
    ``docs/ENVIRONMENT.md`` 中的说明下载权重，或临时传入
    ``--model yolov8n-seg.yaml`` 从零训练。

Windows 稳定性设置：
    - ``workers=0``：避免 Windows 多进程 DataLoader 偶发卡住；
    - ``amp=False``：避免 Ultralytics AMP 自检触发额外下载或长时间等待；
    - ``pretrained=True``：使用本地 ``.pt`` 权重时继续进行迁移学习。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    """解析训练参数。"""

    parser = argparse.ArgumentParser(description="训练 YOLOv8n-seg 书脊分割启动模型。")
    parser.add_argument(
        "--model",
        default=str(PROJECT_ROOT / "models/weights/yolov8n-seg.pt"),
        help="模型结构或权重路径；默认使用本地预训练 yolov8n-seg.pt。",
    )
    parser.add_argument(
        "--data",
        default=str(PROJECT_ROOT / "models/configs/book_spine_bootstrap_v1.yaml"),
        help="Ultralytics 数据集 YAML。",
    )
    parser.add_argument("--epochs", type=int, default=60, help="最大训练轮数。")
    parser.add_argument("--imgsz", type=int, default=640, help="输入图像尺寸。")
    parser.add_argument("--batch", type=int, default=4, help="批大小。")
    parser.add_argument("--device", default="0", help="训练设备，RTX 4060 使用 0。")
    parser.add_argument(
        "--name",
        default="book_spine_bootstrap_v1_pretrained",
        help="训练运行名称。",
    )
    return parser.parse_args()


def main() -> None:
    """启动训练。"""

    args = parse_args()
    model_path = Path(args.model)
    if args.model.endswith(".pt") and not model_path.exists():
        raise FileNotFoundError(
            "找不到本地预训练权重："
            f"{model_path}\n"
            "请先下载 yolov8n-seg.pt 到 models/weights/，"
            "或使用 --model yolov8n-seg.yaml 从零训练。"
        )

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=0,
        project=str(PROJECT_ROOT / "runs/segment_bootstrap"),
        name=args.name,
        exist_ok=True,
        patience=15,
        amp=False,
        pretrained=args.model.endswith(".pt"),
        deterministic=False,
        seed=20260628,
    )


if __name__ == "__main__":
    main()
