"""对书脊裁剪图批量执行 PaddleOCR，并导出 CSV。

典型用法：

    # 先用 scripts/demo_obb_crop.py 生成裁剪图，再识别其中前 20 个书脊
    python scripts/demo_spine_ocr.py --limit 20

    # 识别指定裁剪目录
    python scripts/demo_spine_ocr.py --crops-dir outputs/crops/obb_demo/spines

输出文件默认位于：
    outputs/ocr/ocr_demo/spine_ocr_results.csv

这个脚本属于 OCR 阶段的冒烟测试脚本：目标是先确认“裁剪图 -> OCR 文本”链路跑通。
后续馆藏目录匹配、重复计数和 Streamlit 界面会复用同一套 OCR 封装。
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from book_inventory.ocr import PaddleSpineOCR  # noqa: E402


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="批量识别书脊裁剪图中的文字")
    parser.add_argument(
        "--crops-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/crops/obb_demo/spines",
        help="书脊裁剪图目录，默认读取 demo_obb_crop.py 的输出目录。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs/ocr/ocr_demo/spine_ocr_results.csv",
        help="OCR 结果 CSV 输出路径。",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="最多识别多少张裁剪图；0 表示全部识别。",
    )
    parser.add_argument(
        "--min-line-conf",
        type=float,
        default=0.20,
        help="单行 OCR 文本最低保留置信度。",
    )
    parser.add_argument(
        "--no-textline-orientation",
        action="store_true",
        help="关闭文本行方向分类。一般不建议关闭，除非想对比速度和效果。",
    )
    return parser.parse_args()


def collect_crop_images(crops_dir: Path, limit: int = 0) -> list[Path]:
    """收集待 OCR 的书脊裁剪图。"""

    if not crops_dir.exists():
        raise FileNotFoundError(f"找不到裁剪图目录：{crops_dir}")

    images = sorted(
        [
            path
            for path in crops_dir.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        ]
    )
    if limit > 0:
        images = images[:limit]
    return images


def main() -> None:
    """脚本入口。"""

    args = parse_args()
    crop_images = collect_crop_images(args.crops_dir, args.limit)
    if not crop_images:
        raise RuntimeError(f"没有找到可识别的裁剪图：{args.crops_dir}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"待识别书脊裁剪图：{len(crop_images)} 张")
    print("正在初始化 PaddleOCR，第一次运行可能需要加载模型，请稍等...")
    reader = PaddleSpineOCR(
        min_line_confidence=args.min_line_conf,
        use_textline_orientation=not args.no_textline_orientation,
    )

    rows: list[dict[str, str | int | float]] = []
    started_at = time.perf_counter()
    for index, crop_path in enumerate(crop_images, start=1):
        item_started_at = time.perf_counter()
        result = reader.recognize(crop_path)
        elapsed_ms = (time.perf_counter() - item_started_at) * 1000

        rows.append(
            {
                "序号": index,
                "裁剪图": str(crop_path),
                "拼接文本": result.joined_text,
                "可读文本": result.readable_text,
                "平均置信度": round(result.average_confidence, 4),
                "文本行数": len(result.lines),
                "逐行结果JSON": result.lines_as_json(),
                "耗时ms": round(elapsed_ms, 1),
            }
        )

        print(
            f"[{index:>3}/{len(crop_images)}] "
            f"{crop_path.name} -> {result.readable_text or '<未识别>'} "
            f"(conf={result.average_confidence:.3f}, {elapsed_ms:.0f}ms)"
        )

    total_elapsed = time.perf_counter() - started_at
    with args.output.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "序号",
                "裁剪图",
                "拼接文本",
                "可读文本",
                "平均置信度",
                "文本行数",
                "逐行结果JSON",
                "耗时ms",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"OCR 结果已保存：{args.output}")
    print(f"总耗时：{total_elapsed:.1f}s，平均：{total_elapsed / len(crop_images):.2f}s/张")


if __name__ == "__main__":
    main()
