"""演示最终 OBB 模型的书脊检测与透视裁剪。

运行示例：

```powershell
D:\Downloads\Anaconda\envs\BookOCR\python.exe scripts\demo_obb_crop.py --limit 3
```

输出：
    ``outputs/crops/obb_demo/`` 下的检测预览图和单本书脊裁剪图。
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from book_inventory.config import load_config
from book_inventory.detection.obb_detector import OBBBookSpineDetector
from book_inventory.ocr.spine_cropper import (
    crop_all_spines,
    draw_obb_preview,
    read_image,
    save_crops,
    write_image,
)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    config = load_config()
    parser = argparse.ArgumentParser(description="演示 OBB 检测和书脊透视裁剪。")
    parser.add_argument(
        "--source",
        type=Path,
        default=config.path("paths", "converted_images"),
        help="输入图片或图片目录。",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=config.path("paths", "obb_final_weights"),
        help="最终 OBB 模型权重。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.path("paths", "outputs") / "crops/obb_demo",
        help="输出目录。",
    )
    parser.add_argument("--conf", type=float, default=0.60, help="OBB 推理置信度阈值。")
    parser.add_argument("--iou", type=float, default=0.50, help="OBB NMS IoU 阈值。")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO 推理输入尺寸。")
    parser.add_argument("--device", default="0", help="推理设备，GPU 为 0，CPU 为 cpu。")
    parser.add_argument("--limit", type=int, default=3, help="最多处理多少张图片，0 表示全部。")
    parser.add_argument("--padding-ratio", type=float, default=0.03, help="裁剪框扩张比例。")
    return parser.parse_args()


def collect_images(source: Path, limit: int) -> list[Path]:
    """收集待处理图片。"""

    if source.is_file():
        return [source]

    suffixes = {".jpg", ".jpeg", ".png", ".bmp"}
    images = sorted(path for path in source.iterdir() if path.suffix.lower() in suffixes)
    if limit > 0:
        images = images[:limit]
    return images


def main() -> None:
    """脚本入口。"""

    args = parse_args()
    images = collect_images(args.source, args.limit)
    if not images:
        raise RuntimeError(f"没有找到待处理图片：{args.source}")

    detector = OBBBookSpineDetector(
        args.weights,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
    )

    preview_dir = args.output_dir / "previews"
    crop_dir = args.output_dir / "spines"
    preview_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for image_path in images:
        print(f"处理图片：{image_path.name}")
        image = read_image(image_path)
        detections = detector.detect(image_path)
        crops = crop_all_spines(image, detections, padding_ratio=args.padding_ratio)
        saved_crops = save_crops(crops, crop_dir, image_stem=image_path.stem)

        preview = draw_obb_preview(image, detections)
        preview_path = preview_dir / f"{image_path.stem}_obb_preview.jpg"
        write_image(preview_path, preview)

        print(f"  检测书脊：{len(detections)}，成功裁剪：{len(saved_crops)}")
        for crop in saved_crops:
            rows.append(
                {
                    "source_image": image_path.name,
                    "spine_index": crop.detection.index,
                    "confidence": f"{crop.detection.confidence:.4f}",
                    "crop_path": str(crop.output_path),
                    "crop_height": crop.image.shape[0],
                    "crop_width": crop.image.shape[1],
                }
            )

    manifest_path = args.output_dir / "crop_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_image",
                "spine_index",
                "confidence",
                "crop_path",
                "crop_height",
                "crop_width",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"输出目录：{args.output_dir}")
    print(f"裁剪清单：{manifest_path}")


if __name__ == "__main__":
    main()
