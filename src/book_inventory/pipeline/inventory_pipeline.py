"""图书盘点端到端流水线。

当前版本先完成“单图/多图独立盘点”的基础闭环：
OBB 书脊检测 -> 书脊裁剪 -> PaddleOCR -> 馆藏目录匹配 -> CSV 汇总。

注意：
跨图重叠去重尚未在本文件中实现。因此多张相邻照片如果包含同一本实体书，
当前会先各自计数；后续会在 dedup 模块接入空间配准和重复合并逻辑。
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from book_inventory.detection import OBBBookSpineDetector
from book_inventory.matching import CatalogMatcher
from book_inventory.ocr import (
    PaddleSpineOCR,
    crop_all_spines,
    draw_obb_preview,
    read_image,
    save_crops,
    write_image,
)


@dataclass(frozen=True)
class InventoryDetail:
    """单本书脊的完整识别明细。"""

    source_image: str
    spine_index: int
    detection_confidence: float
    crop_path: str
    ocr_text: str
    ocr_readable_text: str
    ocr_confidence: float
    matched_title: str
    match_score: float
    match_status: str
    author: str
    call_number: str
    publisher: str
    obb_points_json: str = ""


class InventoryPipeline:
    """图书盘点基础流水线。"""

    def __init__(
        self,
        *,
        detector: OBBBookSpineDetector,
        ocr_reader: PaddleSpineOCR,
        matcher: CatalogMatcher,
        output_dir: str | Path,
        crop_padding_ratio: float = 0.03,
        match_threshold: float = 0.72,
    ) -> None:
        """初始化流水线。"""

        self.detector = detector
        self.ocr_reader = ocr_reader
        self.matcher = matcher
        self.output_dir = Path(output_dir)
        self.crop_padding_ratio = crop_padding_ratio
        self.match_threshold = match_threshold

        self.preview_dir = self.output_dir / "previews"
        self.crop_dir = self.output_dir / "crops"

    def process_images(self, image_paths: list[str | Path]) -> list[InventoryDetail]:
        """批量处理多张图像。

        当前多图只是顺序独立处理，尚不做跨图去重。
        """

        all_details: list[InventoryDetail] = []
        for image_path in image_paths:
            all_details.extend(self.process_image(image_path))
        return all_details

    def process_image(self, image_path: str | Path) -> list[InventoryDetail]:
        """处理单张书架图像，并返回每个书脊的识别明细。"""

        path = Path(image_path)
        image = read_image(path)
        detections = self.detector.detect(path)

        preview = draw_obb_preview(image, detections)
        write_image(self.preview_dir / f"{path.stem}_obb_preview.jpg", preview)

        crops = crop_all_spines(
            image,
            detections,
            padding_ratio=self.crop_padding_ratio,
        )
        saved_crops = save_crops(crops, self.crop_dir, image_stem=path.stem)

        details: list[InventoryDetail] = []
        for crop in saved_crops:
            if crop.output_path is None:
                continue

            ocr_result = self.ocr_reader.recognize(crop.output_path)
            best_match = self.matcher.best_match(
                ocr_result.joined_text,
                accept_threshold=self.match_threshold,
            )

            if best_match is None:
                matched_title = ""
                match_score = 0.0
                match_status = "pending"
                author = ""
                call_number = ""
                publisher = ""
            else:
                matched_title = best_match.entry.title
                match_score = best_match.score
                match_status = best_match.status
                author = best_match.entry.author
                call_number = best_match.entry.call_number
                publisher = best_match.entry.publisher

            details.append(
                InventoryDetail(
                    source_image=path.name,
                    spine_index=crop.detection.index,
                    detection_confidence=round(crop.detection.confidence, 4),
                    crop_path=str(crop.output_path),
                    ocr_text=ocr_result.joined_text,
                    ocr_readable_text=ocr_result.readable_text,
                    ocr_confidence=round(ocr_result.average_confidence, 4),
                    matched_title=matched_title,
                    match_score=match_score,
                    match_status=match_status,
                    author=author,
                    call_number=call_number,
                    publisher=publisher,
                    obb_points_json=json.dumps(
                        crop.detection.points.astype(float).round(2).tolist(),
                        ensure_ascii=False,
                    ),
                )
            )

        return details

    @staticmethod
    def summarize(details: list[InventoryDetail]) -> list[dict[str, str | int | float]]:
        """按规范书名汇总册数。

        未自动匹配的书脊不会被强行并入某本书，而是以“待确认：OCR文本”单独保留。
        """

        counter: Counter[str] = Counter()
        score_sum: Counter[str] = Counter()
        source_images: dict[str, set[str]] = {}
        raw_texts: dict[str, set[str]] = {}
        statuses: dict[str, set[str]] = {}

        for item in details:
            if item.match_status == "matched" and item.matched_title:
                key = item.matched_title
            else:
                key = f"待确认：{item.ocr_text or item.crop_path}"

            counter[key] += 1
            score_sum[key] += item.match_score
            source_images.setdefault(key, set()).add(item.source_image)
            raw_texts.setdefault(key, set()).add(item.ocr_readable_text or item.ocr_text)
            statuses.setdefault(key, set()).add(item.match_status)

        rows: list[dict[str, str | int | float]] = []
        for title, count in counter.most_common():
            rows.append(
                {
                    "规范书名": title,
                    "册数": count,
                    "平均匹配置信度": round(score_sum[title] / count, 4),
                    "状态": ",".join(sorted(statuses[title])),
                    "OCR原文": "；".join(sorted(text for text in raw_texts[title] if text)),
                    "来源图像": "；".join(sorted(source_images[title])),
                }
            )
        return rows


def write_detail_csv(details: list[InventoryDetail], output_path: str | Path) -> None:
    """保存单本书脊识别明细。"""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "source_image",
                "spine_index",
                "detection_confidence",
                "crop_path",
                "ocr_text",
                "ocr_readable_text",
                "ocr_confidence",
                "matched_title",
                "match_score",
                "match_status",
                "author",
                "call_number",
                "publisher",
                "obb_points_json",
            ],
        )
        writer.writeheader()
        writer.writerows([detail.__dict__ for detail in details])


def write_summary_csv(rows: list[dict[str, str | int | float]], output_path: str | Path) -> None:
    """保存按书名汇总后的盘点结果。"""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["规范书名", "册数", "平均匹配置信度", "状态", "OCR原文", "来源图像"],
        )
        writer.writeheader()
        writer.writerows(rows)
