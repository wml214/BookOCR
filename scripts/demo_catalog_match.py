"""将 OCR 结果与馆藏目录进行相似度匹配，并导出候选结果 CSV。

典型用法：

    python scripts/demo_catalog_match.py

默认输入：
    outputs/ocr/ocr_demo/spine_ocr_results.csv

默认输出：
    outputs/matching/match_demo/catalog_match_results.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from book_inventory.matching import CatalogMatcher  # noqa: E402


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="OCR 文本与馆藏目录匹配 demo")
    parser.add_argument(
        "--ocr-csv",
        type=Path,
        default=PROJECT_ROOT / "outputs/ocr/ocr_demo/spine_ocr_results.csv",
        help="demo_spine_ocr.py 生成的 OCR 结果 CSV。",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=PROJECT_ROOT / "data/processed/catalog/catalog.csv",
        help="馆藏目录 CSV。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs/matching/match_demo/catalog_match_results.csv",
        help="匹配结果 CSV 输出路径。",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="每条 OCR 文本保留多少个候选。",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.72,
        help="自动接受匹配的最低相似度阈值。",
    )
    return parser.parse_args()


def read_ocr_rows(ocr_csv: Path) -> list[dict[str, str]]:
    """读取 OCR demo 输出的 CSV。"""

    if not ocr_csv.exists():
        raise FileNotFoundError(f"找不到 OCR 结果 CSV：{ocr_csv}")

    with ocr_csv.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def main() -> None:
    """脚本入口。"""

    args = parse_args()
    ocr_rows = read_ocr_rows(args.ocr_csv)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"读取 OCR 结果：{len(ocr_rows)} 条")
    print("正在加载馆藏目录并建立索引...")
    started_at = time.perf_counter()
    matcher = CatalogMatcher.from_csv(args.catalog)
    print(f"馆藏条目：{len(matcher.entries)} 条，建索引用时 {time.perf_counter() - started_at:.1f}s")

    output_rows: list[dict[str, str | int | float]] = []
    for row in ocr_rows:
        query = row.get("拼接文本") or row.get("可读文本") or ""
        candidates = matcher.search(
            query,
            top_k=args.top_k,
            accept_threshold=args.threshold,
        )

        if not candidates:
            output_rows.append(
                {
                    "序号": row.get("序号", ""),
                    "裁剪图": row.get("裁剪图", ""),
                    "OCR文本": query,
                    "候选排名": 0,
                    "规范书名": "",
                    "匹配分数": 0.0,
                    "状态": "pending",
                    "作者": "",
                    "索书号": "",
                    "出版社": "",
                    "馆藏册数": 0,
                }
            )
            continue

        for rank, candidate in enumerate(candidates, start=1):
            entry = candidate.entry
            output_rows.append(
                {
                    "序号": row.get("序号", ""),
                    "裁剪图": row.get("裁剪图", ""),
                    "OCR文本": query,
                    "候选排名": rank,
                    "规范书名": entry.title,
                    "匹配分数": candidate.score,
                    "状态": candidate.status,
                    "作者": entry.author,
                    "索书号": entry.call_number,
                    "出版社": entry.publisher,
                    "馆藏册数": entry.total_copy_count,
                }
            )

        best = candidates[0]
        print(
            f"[{row.get('序号', '?')}] {query or '<空>'} -> "
            f"{best.entry.title} ({best.score:.3f}, {best.status})"
        )

    with args.output.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "序号",
                "裁剪图",
                "OCR文本",
                "候选排名",
                "规范书名",
                "匹配分数",
                "状态",
                "作者",
                "索书号",
                "出版社",
                "馆藏册数",
            ],
        )
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"匹配结果已保存：{args.output}")


if __name__ == "__main__":
    main()
