"""正式运行图书盘点流水线。

这个脚本用于课程演示和后续 Streamlit 集成前的命令行验收。
它和 `run_inventory_demo.py` 的区别是：

- 默认处理整个输入目录，而不是只处理 1 张样例图；
- 默认输出到带时间戳的新目录，避免覆盖历史盘点结果；
- 提供 `--dry-run`，可以先检查将要处理多少张图片；
- 输出正式的 `盘点明细.csv` 和 `盘点结果.csv`。

示例：

    # 先检查会处理多少张图片，不真正跑模型
    python scripts/run_inventory.py --dry-run

    # 只处理前 3 张图片，用于快速验收
    python scripts/run_inventory.py --limit 3

    # 处理完整目录，耗时较长
    python scripts/run_inventory.py --limit 0
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from book_inventory.detection import OBBBookSpineDetector  # noqa: E402
from book_inventory.matching import CatalogMatcher  # noqa: E402
from book_inventory.ocr import PaddleSpineOCR  # noqa: E402
from book_inventory.pipeline import (  # noqa: E402
    InventoryPipeline,
    write_detail_csv,
    write_summary_csv,
)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="正式运行图书盘点流水线")
    parser.add_argument(
        "--source",
        type=Path,
        default=PROJECT_ROOT / "data/interim/images_jpg",
        help="输入图片或图片目录，默认使用转换后的 JPEG 图像目录。",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=PROJECT_ROOT / "models/weights/book_spine_obb_final_200_best.pt",
        help="最终 OBB 模型权重路径。",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=PROJECT_ROOT / "data/processed/catalog/catalog.csv",
        help="馆藏目录 CSV 路径。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="输出目录；不填写时自动使用 outputs/inventory/run_时间戳。",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="处理图片数量上限；0 表示处理全部图片。",
    )
    parser.add_argument("--conf", type=float, default=0.60, help="OBB 检测置信度阈值。")
    parser.add_argument("--iou", type=float, default=0.50, help="OBB NMS IoU 阈值。")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO 推理图像尺寸。")
    parser.add_argument("--device", default="0", help="推理设备；RTX 4060 用 0，CPU 用 cpu。")
    parser.add_argument("--match-threshold", type=float, default=0.72, help="自动匹配阈值。")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只列出将要处理的图片数量和输出目录，不加载模型、不运行 OCR。",
    )
    return parser.parse_args()


def collect_images(source: Path, limit: int) -> list[Path]:
    """从文件或目录收集待处理图片。"""

    if source.is_file():
        images = [source]
    elif source.is_dir():
        images = sorted(
            path
            for path in source.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        )
    else:
        raise FileNotFoundError(f"找不到输入图片或目录：{source}")

    if limit > 0:
        images = images[:limit]
    if not images:
        raise RuntimeError(f"没有找到可处理图片：{source}")
    return images


def resolve_output_dir(output_dir: Path | None) -> Path:
    """确定本次盘点输出目录。"""

    if output_dir is not None:
        return output_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "outputs/inventory" / f"run_{timestamp}"


def main() -> None:
    """脚本入口。"""

    args = parse_args()
    images = collect_images(args.source, args.limit)
    output_dir = resolve_output_dir(args.output_dir)

    print(f"待处理图片：{len(images)} 张")
    print(f"输出目录：{output_dir}")

    if args.dry_run:
        print("dry-run 模式：仅检查输入，不运行模型。")
        return

    print("加载 OBB 检测模型...")
    detector = OBBBookSpineDetector(
        args.weights,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
    )

    print("加载 PaddleOCR...")
    ocr_reader = PaddleSpineOCR()

    print("加载馆藏目录索引...")
    matcher = CatalogMatcher.from_csv(args.catalog)

    pipeline = InventoryPipeline(
        detector=detector,
        ocr_reader=ocr_reader,
        matcher=matcher,
        output_dir=output_dir,
        match_threshold=args.match_threshold,
    )

    started_at = time.perf_counter()
    details = pipeline.process_images(images)
    summary = pipeline.summarize(details)

    detail_csv = output_dir / "盘点明细.csv"
    summary_csv = output_dir / "盘点结果.csv"
    write_detail_csv(details, detail_csv)
    write_summary_csv(summary, summary_csv)

    matched_count = sum(1 for item in details if item.match_status == "matched")
    pending_count = len(details) - matched_count
    elapsed = time.perf_counter() - started_at

    print(f"识别书脊数：{len(details)}")
    print(f"成功匹配数：{matched_count}")
    print(f"待确认数：{pending_count}")
    print(f"处理时间：{elapsed:.1f}s")
    print(f"盘点明细：{detail_csv}")
    print(f"盘点结果：{summary_csv}")
    print(f"标注预览图目录：{output_dir / 'previews'}")


if __name__ == "__main__":
    main()
