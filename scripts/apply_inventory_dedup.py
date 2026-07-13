"""对一次盘点结果执行跨图像去重。

本脚本适合在 `scripts/run_inventory.py` 生成盘点明细之后运行。
它会读取每个书脊的 OBB 四点坐标，把相邻照片通过 ORB + 单应性矩阵配准到同一坐标系，
再用“同一规范书名 + 空间重合度足够高”的规则合并重复书脊。

推荐运行方式：
    python scripts/apply_inventory_dedup.py --inventory-dir outputs/inventory/run_xxxxxxxx_xxxxxx

如果盘点明细文件不在默认目录，也可以显式指定：
    python scripts/apply_inventory_dedup.py --detail outputs/inventory/run_xxx/盘点明细.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from book_inventory.dedup import (  # noqa: E402
    DedupItem,
    deduplicate_adjacent_images,
    parse_obb_points,
)


REQUIRED_DETAIL_COLUMNS = {"source_image", "spine_index", "matched_title", "match_status", "obb_points_json"}


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="对图书盘点明细执行相邻图像空间去重")
    parser.add_argument(
        "--detail",
        type=Path,
        default=None,
        help="盘点明细 CSV 路径；不填写时从 --inventory-dir 自动查找。",
    )
    parser.add_argument(
        "--inventory-dir",
        type=Path,
        default=None,
        help="某次盘点输出目录，脚本会在其中自动寻找带 OBB 坐标的明细 CSV。",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=PROJECT_ROOT / "data/interim/images_jpg",
        help="原始 JPG/PNG 图片目录，用于相邻图像配准。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="去重结果输出目录；默认写入盘点目录下的 dedup 子目录。",
    )
    parser.add_argument(
        "--spatial-iou",
        type=float,
        default=0.45,
        help="映射到同一坐标系后的 OBB 空间 IoU 阈值。",
    )
    parser.add_argument(
        "--min-inlier-ratio",
        type=float,
        default=0.15,
        help="图像单应性配准的最低内点比例，过低时认为配准不可靠。",
    )
    return parser.parse_args()


def find_detail_csv(inventory_dir: Path) -> Path:
    """在盘点输出目录中自动寻找带 OBB 坐标的明细 CSV。"""

    if not inventory_dir.exists():
        raise FileNotFoundError(f"找不到盘点输出目录：{inventory_dir}")

    candidates = sorted(inventory_dir.glob("*.csv"))
    for candidate in candidates:
        with candidate.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            fieldnames = set(reader.fieldnames or [])
        if REQUIRED_DETAIL_COLUMNS.issubset(fieldnames):
            return candidate

    raise RuntimeError(
        "没有找到可去重的盘点明细 CSV。"
        "请先重新运行 scripts/run_inventory.py，确保明细里包含 obb_points_json 字段。"
    )


def resolve_detail_path(args: argparse.Namespace) -> Path:
    """确定要读取的盘点明细文件。"""

    if args.detail is not None:
        if not args.detail.exists():
            raise FileNotFoundError(f"找不到盘点明细 CSV：{args.detail}")
        return args.detail

    if args.inventory_dir is None:
        raise ValueError("请提供 --inventory-dir 或 --detail")

    return find_detail_csv(args.inventory_dir)


def read_detail_rows(detail_csv: Path) -> list[dict[str, str]]:
    """读取盘点明细 CSV。"""

    with detail_csv.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        fieldnames = set(reader.fieldnames or [])

    missing_columns = REQUIRED_DETAIL_COLUMNS - fieldnames
    if missing_columns:
        raise RuntimeError(
            f"盘点明细缺少字段：{', '.join(sorted(missing_columns))}。"
            "请先重新运行 scripts/run_inventory.py 生成新版明细。"
        )
    return rows


def rows_to_dedup_items(rows: list[dict[str, str]]) -> list[DedupItem]:
    """把 CSV 行转换为去重模块需要的 DedupItem。"""

    items: list[DedupItem] = []
    for row_index, row in enumerate(rows):
        points = parse_obb_points(row.get("obb_points_json", ""))
        if points is None:
            continue

        items.append(
            DedupItem(
                item_id=row_index,
                source_image=row.get("source_image", ""),
                spine_index=int(float(row.get("spine_index", 0) or 0)),
                title_key=row.get("matched_title", "").strip(),
                points=points,
                match_status=row.get("match_status", "").strip(),
            )
        )
    return items


def collect_ordered_image_paths(rows: list[dict[str, str]], image_dir: Path) -> list[Path]:
    """按盘点明细中出现的顺序收集图片路径。

    盘点去重要求相邻照片顺序可信，所以这里不重新按文件名排序，而是尊重明细里的处理顺序。
    """

    seen: set[str] = set()
    image_paths: list[Path] = []
    for row in rows:
        image_name = row.get("source_image", "")
        if not image_name or image_name in seen:
            continue
        seen.add(image_name)
        image_path = image_dir / image_name
        if image_path.exists():
            image_paths.append(image_path)
    return image_paths


def build_duplicate_mapping(duplicate_pairs: list[object]) -> dict[int, int]:
    """生成“被删除行号 -> 保留行号”的映射。"""

    duplicate_to_kept: dict[int, int] = {}
    for pair in duplicate_pairs:
        duplicate_to_kept[pair.removed_item_id] = pair.kept_item_id
    return duplicate_to_kept


def summarize_kept_rows(rows: list[dict[str, str]]) -> list[dict[str, str | int | float]]:
    """按去重后的明细重新汇总册数。"""

    counter: Counter[str] = Counter()
    score_sum: Counter[str] = Counter()
    statuses: dict[str, set[str]] = {}
    source_images: dict[str, set[str]] = {}
    raw_texts: dict[str, set[str]] = {}

    for row in rows:
        if row.get("dedup_status") == "duplicate":
            continue

        if row.get("match_status") == "matched" and row.get("matched_title"):
            key = row["matched_title"]
        else:
            key = f"待确认：{row.get('ocr_text') or row.get('crop_path') or row.get('item_id')}"

        counter[key] += 1
        try:
            score_sum[key] += float(row.get("match_score", 0) or 0)
        except ValueError:
            score_sum[key] += 0
        statuses.setdefault(key, set()).add(row.get("match_status", ""))
        source_images.setdefault(key, set()).add(row.get("source_image", ""))
        raw_texts.setdefault(key, set()).add(row.get("ocr_readable_text") or row.get("ocr_text") or "")

    summary_rows: list[dict[str, str | int | float]] = []
    for title, count in counter.most_common():
        summary_rows.append(
            {
                "规范书名": title,
                "册数": count,
                "平均匹配置信度": round(score_sum[title] / count, 4),
                "状态": ",".join(sorted(status for status in statuses[title] if status)),
                "OCR原文": "；".join(sorted(text for text in raw_texts[title] if text)),
                "来源图像": "；".join(sorted(image for image in source_images[title] if image)),
            }
        )
    return summary_rows


def write_csv(rows: list[dict[str, object]], output_path: Path, fieldnames: list[str]) -> None:
    """写出 UTF-8 BOM CSV，方便 Excel 直接打开不乱码。"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """脚本入口。"""

    args = parse_args()
    detail_csv = resolve_detail_path(args)
    output_dir = args.output_dir or detail_csv.parent / "dedup"

    rows = read_detail_rows(detail_csv)
    items = rows_to_dedup_items(rows)
    image_paths = collect_ordered_image_paths(rows, args.image_dir)

    if len(image_paths) < 2:
        raise RuntimeError("少于 2 张可配准图片，无法执行跨图像去重。")

    kept_ids, duplicate_pairs, registration_logs = deduplicate_adjacent_images(
        items,
        image_paths,
        spatial_iou_threshold=args.spatial_iou,
        min_inlier_ratio=args.min_inlier_ratio,
    )

    duplicate_to_kept = build_duplicate_mapping(duplicate_pairs)
    output_rows: list[dict[str, object]] = []
    for row_index, row in enumerate(rows):
        new_row: dict[str, object] = {"item_id": row_index, **row}
        if row_index in duplicate_to_kept:
            new_row["dedup_status"] = "duplicate"
            new_row["duplicate_of_item_id"] = duplicate_to_kept[row_index]
        else:
            new_row["dedup_status"] = "keep"
            new_row["duplicate_of_item_id"] = ""
        output_rows.append(new_row)

    detail_fieldnames = ["item_id", *list(rows[0].keys()), "dedup_status", "duplicate_of_item_id"]
    duplicate_rows = [pair.__dict__ for pair in duplicate_pairs]
    summary_rows = summarize_kept_rows([dict(row) for row in output_rows])

    write_csv(output_rows, output_dir / "去重明细.csv", detail_fieldnames)
    write_csv(
        duplicate_rows,
        output_dir / "重复合并记录.csv",
        [
            "kept_item_id",
            "removed_item_id",
            "source_image_a",
            "source_image_b",
            "title_key",
            "spatial_iou",
        ],
    )
    write_csv(
        registration_logs,
        output_dir / "图像配准日志.csv",
        ["source_image", "target_image", "matched_points", "inlier_points", "inlier_ratio", "status"],
    )
    write_csv(
        summary_rows,
        output_dir / "去重盘点结果.csv",
        ["规范书名", "册数", "平均匹配置信度", "状态", "OCR原文", "来源图像"],
    )

    print(f"读取明细：{detail_csv}")
    print(f"参与去重书脊：{len(items)} / {len(rows)}")
    print(f"参与配准图片：{len(image_paths)}")
    print(f"合并重复书脊：{len(duplicate_pairs)}")
    print(f"去重后保留书脊：{len(rows) - len(duplicate_pairs)}")
    print(f"输出目录：{output_dir}")


if __name__ == "__main__":
    main()
