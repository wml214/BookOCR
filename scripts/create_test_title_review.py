"""为测试集生成书名识别核验表。

本脚本用于正式评估“书名识别准确率”和“图书种类计数准确率”。

特点：
1. 面向 test 集图像批量生成书脊裁剪图；
2. 对每个裁剪图执行 OCR 和馆藏匹配；
3. 逐行写入 `书名核验表.csv`，中途停止后可继续运行；
4. 已写入核验表的 crop 会自动跳过，避免重复 OCR。

生成的核验表字段：
    source_image, spine_index, crop_path, ocr_text, predicted_title,
    match_score, match_status, true_title, is_correct, remark

人工核验方法：
    - 预测正确：true_title 保持不变，is_correct=1；
    - 预测错误：true_title 改为真实书名，is_correct=0；
    - 无法判断：is_correct 留空，remark 写“不可判定”。
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from book_inventory.detection import OBBBookSpineDetector  # noqa: E402
from book_inventory.matching import CatalogMatcher  # noqa: E402
from book_inventory.ocr import (  # noqa: E402
    PaddleSpineOCR,
    crop_all_spines,
    draw_obb_preview,
    read_image,
    save_crops,
    write_image,
)


FIELDNAMES = [
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
]


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="为测试集生成书名核验表")
    parser.add_argument(
        "--source",
        type=Path,
        default=PROJECT_ROOT / "data/processed/dataset/obb_v1/images/test",
        help="测试集图像目录。",
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
        default=PROJECT_ROOT / "outputs/evaluation/title_review_test20",
        help="输出目录。",
    )
    parser.add_argument("--conf", type=float, default=0.60, help="OBB 检测置信度阈值。")
    parser.add_argument("--iou", type=float, default=0.50, help="OBB NMS IoU 阈值。")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO 推理图像尺寸。")
    parser.add_argument("--device", default="0", help="推理设备；RTX 4060 用 0，CPU 用 cpu。")
    parser.add_argument("--match-threshold", type=float, default=0.72, help="自动匹配阈值。")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只检查图像、已有 crop 和已有核验表行数，不运行模型。",
    )
    parser.add_argument(
        "--max-crops",
        type=int,
        default=0,
        help="最多 OCR 多少个 crop；0 表示全部。用于调试或分批处理。",
    )
    return parser.parse_args()


def collect_images(source: Path) -> list[Path]:
    """收集测试集图像。"""

    if not source.exists():
        raise FileNotFoundError(f"找不到测试集图像目录：{source}")
    images = sorted(
        path
        for path in source.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    if not images:
        raise RuntimeError(f"测试集目录中没有图像：{source}")
    return images


def read_done_crop_paths(review_csv: Path) -> set[str]:
    """读取已完成 OCR/匹配的 crop 路径，用于断点续跑。"""

    if not review_csv.exists():
        return set()
    with review_csv.open("r", encoding="utf-8-sig", newline="") as file:
        return {row.get("crop_path", "") for row in csv.DictReader(file)}


def ensure_review_header(review_csv: Path) -> None:
    """如果核验表不存在，则创建表头。"""

    if review_csv.exists():
        return
    review_csv.parent.mkdir(parents=True, exist_ok=True)
    with review_csv.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()


def crop_pattern_for_image(image_path: Path) -> str:
    """生成某张图对应的 crop 文件匹配模式。"""

    return f"{image_path.stem}_spine_*.jpg"


def parse_crop_name(crop_path: Path, image_stem_to_name: dict[str, str]) -> tuple[str, int]:
    """从 crop 文件名中解析来源图像和书脊编号。"""

    match = re.match(r"(.+)_spine_(\d+)_\d+\.\d+\.jpg$", crop_path.name)
    if not match:
        return "", 0
    image_stem = match.group(1)
    spine_index = int(match.group(2))
    return image_stem_to_name.get(image_stem, image_stem), spine_index


def ensure_crops_for_images(
    images: list[Path],
    *,
    output_dir: Path,
    detector: OBBBookSpineDetector,
) -> list[Path]:
    """确保每张 test 图都有书脊裁剪图。已有 crop 时直接复用。"""

    crop_dir = output_dir / "crops"
    preview_dir = output_dir / "previews"
    crop_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    all_crops: list[Path] = []
    for image_path in images:
        existing = sorted(crop_dir.glob(crop_pattern_for_image(image_path)))
        if existing:
            all_crops.extend(existing)
            continue

        print(f"生成裁剪图：{image_path.name}")
        image = read_image(image_path)
        detections = detector.detect(image_path)
        preview = draw_obb_preview(image, detections)
        write_image(preview_dir / f"{image_path.stem}_obb_preview.jpg", preview)

        crops = crop_all_spines(image, detections)
        saved = save_crops(crops, crop_dir, image_stem=image_path.stem)
        all_crops.extend(crop.output_path for crop in saved if crop.output_path is not None)

    return sorted(all_crops)


def append_review_row(review_csv: Path, row: dict[str, str]) -> None:
    """向核验表追加一行。"""

    with review_csv.open("a", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writerow(row)


def main() -> None:
    """脚本入口。"""

    args = parse_args()
    images = collect_images(args.source)
    crop_dir = args.output_dir / "crops"
    review_csv = args.output_dir / "书名核验表.csv"
    image_stem_to_name = {image.stem: image.name for image in images}

    existing_crops = sorted(crop_dir.glob("*.jpg")) if crop_dir.exists() else []
    done_paths = read_done_crop_paths(review_csv)

    print(f"测试集图像：{len(images)} 张")
    print(f"已有裁剪图：{len(existing_crops)} 个")
    print(f"核验表已完成：{len(done_paths)} 行")
    print(f"输出目录：{args.output_dir}")

    if args.dry_run:
        print("dry-run 模式：不运行模型。")
        return

    print("加载 OBB 检测模型...")
    detector = OBBBookSpineDetector(
        args.weights,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
    )

    all_crops = ensure_crops_for_images(images, output_dir=args.output_dir, detector=detector)
    print(f"可用于核验的裁剪图：{len(all_crops)} 个")

    ensure_review_header(review_csv)
    done_paths = read_done_crop_paths(review_csv)
    pending_crops = [crop for crop in all_crops if str(crop) not in done_paths]
    if args.max_crops > 0:
        pending_crops = pending_crops[: args.max_crops]
    print(f"本次需要 OCR/匹配：{len(pending_crops)} 个")

    if not pending_crops:
        print(f"核验表已是最新：{review_csv}")
        return

    print("加载 PaddleOCR...")
    ocr_reader = PaddleSpineOCR()
    print("加载馆藏目录索引...")
    matcher = CatalogMatcher.from_csv(args.catalog)

    for index, crop_path in enumerate(pending_crops, start=1):
        source_image, spine_index = parse_crop_name(crop_path, image_stem_to_name)
        ocr_result = ocr_reader.recognize(crop_path)
        best_match = matcher.best_match(
            ocr_result.joined_text,
            accept_threshold=args.match_threshold,
        )

        if best_match is None:
            predicted_title = ""
            match_score = ""
            match_status = "pending"
        else:
            predicted_title = best_match.entry.title
            match_score = f"{best_match.score:.4f}"
            match_status = best_match.status

        row = {
            "source_image": source_image,
            "spine_index": str(spine_index),
            "crop_path": str(crop_path),
            "ocr_text": ocr_result.readable_text,
            "predicted_title": predicted_title,
            "match_score": match_score,
            "match_status": match_status,
            "true_title": predicted_title if match_status == "matched" else "",
            "is_correct": "1" if match_status == "matched" else "",
            "remark": "",
        }
        append_review_row(review_csv, row)
        print(
            f"[{index}/{len(pending_crops)}] {source_image} #{spine_index} -> "
            f"{predicted_title or '<待确认>'} ({match_status})"
        )

    print(f"书名核验表已生成/更新：{review_csv}")


if __name__ == "__main__":
    main()
