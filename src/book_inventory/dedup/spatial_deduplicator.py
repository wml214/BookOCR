"""跨图书脊去重逻辑。

本模块把相邻图像中的书脊检测结果映射到同一坐标系，并按空间重合度和书名一致性合并重复书脊。
它不依赖 OCR 引擎，只处理已经生成的盘点明细。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from book_inventory.dedup.image_registration import estimate_homography, transform_points
from book_inventory.ocr.spine_cropper import order_quad_points


@dataclass(frozen=True)
class DedupItem:
    """参与去重的单个书脊条目。"""

    item_id: int
    source_image: str
    spine_index: int
    title_key: str
    points: np.ndarray
    match_status: str


@dataclass(frozen=True)
class DuplicatePair:
    """判定为重复实体书的一对书脊。"""

    kept_item_id: int
    removed_item_id: int
    source_image_a: str
    source_image_b: str
    title_key: str
    spatial_iou: float


class UnionFind:
    """用于合并重复书脊簇的并查集。"""

    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        """查找根节点。"""

        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, left: int, right: int) -> None:
        """合并两个节点。"""

        root_left = self.find(left)
        root_right = self.find(right)
        if root_left != root_right:
            self.parent[root_right] = root_left


def parse_obb_points(value: str) -> np.ndarray | None:
    """从 CSV 字段中解析 OBB 四点坐标。"""

    if not value:
        return None
    try:
        points = np.asarray(json.loads(value), dtype=np.float32)
    except json.JSONDecodeError:
        return None
    if points.shape != (4, 2):
        return None
    return points


def polygon_iou(points_a: np.ndarray, points_b: np.ndarray) -> float:
    """计算两个 OBB 四边形的 IoU。"""

    poly_a = order_quad_points(points_a).astype(np.float32)
    poly_b = order_quad_points(points_b).astype(np.float32)
    area_a = abs(cv2.contourArea(poly_a))
    area_b = abs(cv2.contourArea(poly_b))
    if area_a <= 0 or area_b <= 0:
        return 0.0

    intersection_area, _ = cv2.intersectConvexConvex(poly_a, poly_b)
    union_area = area_a + area_b - intersection_area
    if union_area <= 0:
        return 0.0
    return float(intersection_area / union_area)


def title_compatible(left: DedupItem, right: DedupItem) -> bool:
    """判断两个书脊标题是否允许合并。

    自动匹配到同一规范书名时可以合并；pending 项不按标题强行合并，避免误删。
    """

    if not left.title_key or not right.title_key:
        return False
    if left.match_status != "matched" or right.match_status != "matched":
        return False
    return left.title_key == right.title_key


def deduplicate_adjacent_images(
    items: list[DedupItem],
    image_paths: list[str | Path],
    *,
    spatial_iou_threshold: float = 0.45,
    min_inlier_ratio: float = 0.15,
) -> tuple[set[int], list[DuplicatePair], list[dict[str, object]]]:
    """对有序相邻图像进行跨图去重。

    Args:
        items: 盘点明细转换后的书脊条目。
        image_paths: 用户上传或处理的有序图像路径。
        spatial_iou_threshold: 映射后 OBB 的空间 IoU 阈值。
        min_inlier_ratio: 单应性矩阵的最低内点率。

    Returns:
        kept_ids: 去重后保留的 item_id 集合。
        duplicate_pairs: 被合并的重复对列表。
        registration_logs: 相邻图像配准日志。
    """

    if not items:
        return set(), [], []

    item_by_image: dict[str, list[DedupItem]] = {}
    for item in items:
        item_by_image.setdefault(item.source_image, []).append(item)

    uf = UnionFind(len(items))
    position_by_item_id = {item.item_id: position for position, item in enumerate(items)}
    duplicates: list[DuplicatePair] = []
    registration_logs: list[dict[str, object]] = []

    paths = [Path(path) for path in image_paths]
    for previous_path, current_path in zip(paths, paths[1:], strict=False):
        result = estimate_homography(current_path, previous_path)
        registration_logs.append(
            {
                "source_image": current_path.name,
                "target_image": previous_path.name,
                "matched_points": result.matched_points,
                "inlier_points": result.inlier_points,
                "inlier_ratio": result.inlier_ratio,
                "status": result.status,
            }
        )

        if not result.is_valid or result.inlier_ratio < min_inlier_ratio:
            continue

        previous_items = item_by_image.get(previous_path.name, [])
        current_items = item_by_image.get(current_path.name, [])
        if not previous_items or not current_items:
            continue

        for current_item in current_items:
            mapped_points = transform_points(current_item.points, result.matrix)
            best_pair: tuple[DedupItem, float] | None = None

            for previous_item in previous_items:
                if not title_compatible(current_item, previous_item):
                    continue
                iou = polygon_iou(mapped_points, previous_item.points)
                if iou >= spatial_iou_threshold and (
                    best_pair is None or iou > best_pair[1]
                ):
                    best_pair = (previous_item, iou)

            if best_pair is None:
                continue

            kept_item, best_iou = best_pair
            uf.union(
                position_by_item_id[kept_item.item_id],
                position_by_item_id[current_item.item_id],
            )
            duplicates.append(
                DuplicatePair(
                    kept_item_id=kept_item.item_id,
                    removed_item_id=current_item.item_id,
                    source_image_a=kept_item.source_image,
                    source_image_b=current_item.source_image,
                    title_key=kept_item.title_key,
                    spatial_iou=round(best_iou, 4),
                )
            )

    root_to_kept: dict[int, int] = {}
    for item in items:
        root = uf.find(position_by_item_id[item.item_id])
        root_to_kept[root] = min(root_to_kept.get(root, item.item_id), item.item_id)
    kept_ids = set(root_to_kept.values())
    return kept_ids, duplicates, registration_logs
