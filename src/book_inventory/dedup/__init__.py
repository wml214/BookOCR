"""跨图去重模块：负责图像配准、空间映射和重复实体书判断。"""

from book_inventory.dedup.image_registration import (
    HomographyResult,
    estimate_homography,
    transform_points,
)
from book_inventory.dedup.spatial_deduplicator import (
    DedupItem,
    DuplicatePair,
    deduplicate_adjacent_images,
    parse_obb_points,
    polygon_iou,
)

__all__ = [
    "DedupItem",
    "DuplicatePair",
    "HomographyResult",
    "deduplicate_adjacent_images",
    "estimate_homography",
    "parse_obb_points",
    "polygon_iou",
    "transform_points",
]
