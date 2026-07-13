"""根据盘点明细生成“书名核验表”。

用途：
    端到端流水线会生成 `盘点明细.csv`，其中包含 OCR 原文、预测书名、匹配分数等字段。
    本脚本在此基础上增加人工核验需要填写的列：

    - true_title：人工确认的真实书名；
    - is_correct：预测是否正确，正确填 1，错误填 0，不可判定可留空；
    - remark：备注，例如“遮挡严重”“馆藏无此书”“无法判断”。

示例：
    python scripts/create_title_review_sheet.py ^
        --detail outputs/evaluation/title_review_test20/盘点明细.csv ^
        --output outputs/evaluation/title_review_test20/书名核验表.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="根据盘点明细生成书名核验表")
    parser.add_argument(
        "--detail",
        type=Path,
        required=True,
        help="盘点明细 CSV 路径。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="书名核验表输出路径。",
    )
    return parser.parse_args()


def main() -> None:
    """脚本入口。"""

    args = parse_args()
    if not args.detail.exists():
        raise FileNotFoundError(f"找不到盘点明细：{args.detail}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.detail.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    output_rows: list[dict[str, str]] = []
    for row in rows:
        predicted_title = row.get("matched_title", "")
        match_status = row.get("match_status", "")

        output_rows.append(
            {
                "source_image": row.get("source_image", ""),
                "spine_index": row.get("spine_index", ""),
                "crop_path": row.get("crop_path", ""),
                "ocr_text": row.get("ocr_readable_text") or row.get("ocr_text", ""),
                "predicted_title": predicted_title,
                "match_score": row.get("match_score", ""),
                "match_status": match_status,
                # 对 matched 的样本，先把 true_title 预填为预测书名，人工只需要改错；
                # 对 pending 的样本，true_title 留空，提示需要人工填写。
                "true_title": predicted_title if match_status == "matched" else "",
                "is_correct": "1" if match_status == "matched" else "",
                "remark": "",
            }
        )

    with args.output.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "source_image",
                "spine_index",
                "crop_path",
                "ocr_text",
                "predicted_title",
                "match_score",
                "match_status",
                "true_title",
                "is_correct",
                "remark",
            ],
        )
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"已生成书名核验表：{args.output}")
    print(f"总行数：{len(output_rows)}")
    print("说明：matched 行已预填 true_title/is_correct；请人工检查并改错。")


if __name__ == "__main__":
    main()
