"""运行图书盘点端到端 demo。

默认只处理 `data/interim/images_jpg` 中排序后的第 1 张图，避免一次性跑完整数据集耗时太久。

示例：

    # 处理第一张样例图
    python scripts/run_inventory_demo.py --limit 1

    # 处理指定图片
    python scripts/run_inventory_demo.py --source data/interim/images_jpg/xxx.jpg

    # 处理目录中的前 3 张图片
    python scripts/run_inventory_demo.py --source data/interim/images_jpg --limit 3

输出目录默认：
    outputs/inventory/demo/
"""

from __future__ import annotations

import argparse
import sys
import time
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

    parser = argparse.ArgumentParser(description="图书盘点端到端 demo")
    parser.add_argument(
        "--source",
        type=Path,
        default=PROJECT_ROOT / "data/interim/images_jpg",
        help="输入图片或图片目录。",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=PROJECT_ROOT / "models/weights/book_spine_obb_final_200_best.pt",
        help="最终 OBB 模型权重。",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=PROJECT_ROOT / "data/processed/catalog/catalog.csv",
        help="馆藏目录 CSV。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/inventory/demo",
        help="输出目录。",
    )
    parser.add_argument("--limit", type=int, default=1, help="处理图片数量上限，0 表示全部。")
    parser.add_argument("--conf", type=float, default=0.60, help="OBB 检测置信度阈值。")
    parser.add_argument("--iou", type=float, default=0.50, help="OBB NMS IoU 阈值。")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO 推理图像尺寸。")
    parser.add_argument("--device", default="0", help="推理设备，RTX 4060 用 0，CPU 用 cpu。")
    parser.add_argument("--match-threshold", type=float, default=0.72, help="自动匹配阈值。")
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


def main() -> None:
    """脚本入口。"""

    args = parse_args()
    images = collect_images(args.source, args.limit)

    print(f"待处理图片：{len(images)} 张")
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
        output_dir=args.output_dir,
        match_threshold=args.match_threshold,
    )

    started_at = time.perf_counter()
    details = pipeline.process_images(images)
    summary = pipeline.summarize(details)

    detail_csv = args.output_dir / "盘点明细.csv"
    summary_csv = args.output_dir / "盘点结果.csv"
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
    print(f"标注预览图目录：{args.output_dir / 'previews'}")


if __name__ == "__main__":
    main()
