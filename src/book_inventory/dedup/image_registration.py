"""相邻书架图像配准工具。

跨图去重的核心问题是：相邻两张照片中，同一本实体书可能同时出现。
本模块使用 ORB 特征点 + RANSAC 单应性矩阵，把后一张图映射到前一张图坐标系，
再根据书脊 OBB 的空间重合度判断是否重复。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from book_inventory.ocr import read_image


@dataclass(frozen=True)
class HomographyResult:
    """两张相邻图片之间的单应性估计结果。"""

    source_image: Path
    target_image: Path
    matrix: np.ndarray | None
    matched_points: int
    inlier_points: int
    inlier_ratio: float
    status: str

    @property
    def is_valid(self) -> bool:
        """判断单应性是否足够可靠。"""

        return self.matrix is not None and self.status == "ok"


def estimate_homography(
    source_image: str | Path,
    target_image: str | Path,
    *,
    max_features: int = 4000,
    ratio_threshold: float = 0.75,
    min_matches: int = 12,
    ransac_threshold: float = 5.0,
) -> HomographyResult:
    """估计从 source_image 到 target_image 的单应性矩阵。

    Args:
        source_image: 待映射图像，通常是后一张相邻图。
        target_image: 目标坐标系图像，通常是前一张相邻图。
        max_features: ORB 最大特征点数。
        ratio_threshold: KNN 匹配的 Lowe ratio 阈值。
        min_matches: RANSAC 前需要的最少有效匹配点数。
        ransac_threshold: RANSAC 重投影误差阈值。
    """

    source_path = Path(source_image)
    target_path = Path(target_image)
    source = read_image(source_path)
    target = read_image(target_path)

    source_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=max_features)
    keypoints_src, descriptors_src = orb.detectAndCompute(source_gray, None)
    keypoints_dst, descriptors_dst = orb.detectAndCompute(target_gray, None)

    if descriptors_src is None or descriptors_dst is None:
        return HomographyResult(source_path, target_path, None, 0, 0, 0.0, "no_features")

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw_matches = matcher.knnMatch(descriptors_src, descriptors_dst, k=2)

    good_matches = []
    for pair in raw_matches:
        if len(pair) < 2:
            continue
        first, second = pair
        if first.distance < ratio_threshold * second.distance:
            good_matches.append(first)

    if len(good_matches) < min_matches:
        return HomographyResult(
            source_path,
            target_path,
            None,
            len(good_matches),
            0,
            0.0,
            "not_enough_matches",
        )

    src_points = np.float32([keypoints_src[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_points = np.float32([keypoints_dst[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    matrix, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, ransac_threshold)
    if matrix is None or mask is None:
        return HomographyResult(
            source_path,
            target_path,
            None,
            len(good_matches),
            0,
            0.0,
            "homography_failed",
        )

    inlier_points = int(mask.ravel().sum())
    inlier_ratio = inlier_points / max(len(good_matches), 1)
    status = "ok" if inlier_points >= min_matches else "not_enough_inliers"

    return HomographyResult(
        source_image=source_path,
        target_image=target_path,
        matrix=matrix,
        matched_points=len(good_matches),
        inlier_points=inlier_points,
        inlier_ratio=round(inlier_ratio, 4),
        status=status,
    )


def transform_points(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    """使用单应性矩阵变换 OBB 四点坐标。"""

    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(pts, homography)
    return transformed.reshape(-1, 2)
