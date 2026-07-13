"""基于 OBB 四点框的书脊裁剪与透视矫正。

YOLOv8-OBB 输出的是书脊四个角点。OCR 不适合直接在整张书架图上识别，
因此需要先把每个旋转框裁剪成单本书脊小图。本模块负责：

- 读取和保存包含中文路径的图片；
- 将任意顺序的四点框排序为左上、右上、右下、左下；
- 做透视变换，把倾斜书脊拉正；
- 自动把裁剪结果旋转为“高大于宽”的竖向书脊图；
- 生成带编号和置信度的检测预览图。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from book_inventory.detection.obb_detector import OBBDetection


@dataclass(frozen=True)
class CroppedSpine:
    """单个裁剪出的书脊图像及其元信息。"""

    detection: OBBDetection
    image: np.ndarray
    output_path: Path | None = None


def read_image(image_path: str | Path) -> np.ndarray:
    """读取图片，兼容 Windows 中文路径。"""

    path = Path(image_path)
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取图片：{path}")
    return image


def write_image(image_path: str | Path, image: np.ndarray) -> None:
    """保存图片，兼容 Windows 中文路径。"""

    path = Path(image_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower() or ".jpg"
    success, encoded = cv2.imencode(suffix, image)
    if not success:
        raise ValueError(f"无法编码图片：{path}")
    encoded.tofile(str(path))


def order_quad_points(points: np.ndarray) -> np.ndarray:
    """将四个角点排序为左上、右上、右下、左下。

    透视变换要求源点和目标点顺序一致。这里使用经典的 sum/diff 方法：
    - x+y 最小的是左上；
    - x+y 最大的是右下；
    - x-y 最小的是右上；
    - x-y 最大的是左下。
    """

    pts = np.asarray(points, dtype=np.float32)
    if pts.shape != (4, 2):
        raise ValueError(f"OBB 四点形状应为 (4, 2)，当前为 {pts.shape}")

    ordered = np.zeros((4, 2), dtype=np.float32)
    point_sum = pts.sum(axis=1)
    point_diff = np.diff(pts, axis=1).reshape(-1)

    ordered[0] = pts[np.argmin(point_sum)]
    ordered[2] = pts[np.argmax(point_sum)]
    ordered[1] = pts[np.argmin(point_diff)]
    ordered[3] = pts[np.argmax(point_diff)]
    return ordered


def expand_quad(points: np.ndarray, ratio: float = 0.03) -> np.ndarray:
    """以中心点为基准轻微扩张四边形。

    OBB 框如果贴得太紧，裁剪后可能切掉书脊边缘文字。默认扩张 3%，既能保留边缘，
    又不会引入太多相邻书脊。
    """

    if ratio <= 0:
        return points.astype(np.float32)
    center = points.mean(axis=0, keepdims=True)
    return center + (points - center) * (1.0 + ratio)


def crop_spine_by_obb(
    image: np.ndarray,
    detection: OBBDetection,
    *,
    padding_ratio: float = 0.03,
    min_size: int = 8,
) -> np.ndarray | None:
    """根据单个 OBB 检测结果裁剪并拉正书脊。

    Returns:
        成功时返回 BGR 裁剪图；如果框过小或透视变换失败，返回 ``None``。
    """

    ordered = order_quad_points(detection.points)
    ordered = expand_quad(ordered, ratio=padding_ratio)

    top_width = np.linalg.norm(ordered[1] - ordered[0])
    bottom_width = np.linalg.norm(ordered[2] - ordered[3])
    left_height = np.linalg.norm(ordered[3] - ordered[0])
    right_height = np.linalg.norm(ordered[2] - ordered[1])

    crop_width = int(round(max(top_width, bottom_width)))
    crop_height = int(round(max(left_height, right_height)))
    if crop_width < min_size or crop_height < min_size:
        return None

    destination = np.asarray(
        [
            [0, 0],
            [crop_width - 1, 0],
            [crop_width - 1, crop_height - 1],
            [0, crop_height - 1],
        ],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(ordered.astype(np.float32), destination)
    cropped = cv2.warpPerspective(
        image,
        matrix,
        (crop_width, crop_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )

    # 书脊 OCR 更希望输入为竖向长图。若透视结果横向，自动顺时针旋转 90 度。
    if cropped.shape[1] > cropped.shape[0]:
        cropped = cv2.rotate(cropped, cv2.ROTATE_90_CLOCKWISE)
    return cropped


def crop_all_spines(
    image: np.ndarray,
    detections: list[OBBDetection],
    *,
    padding_ratio: float = 0.03,
) -> list[CroppedSpine]:
    """批量裁剪一张图片中的所有书脊。"""

    crops: list[CroppedSpine] = []
    for detection in detections:
        cropped = crop_spine_by_obb(image, detection, padding_ratio=padding_ratio)
        if cropped is None:
            continue
        crops.append(CroppedSpine(detection=detection, image=cropped))
    return crops


def save_crops(
    crops: list[CroppedSpine],
    output_dir: str | Path,
    *,
    image_stem: str,
) -> list[CroppedSpine]:
    """保存裁剪结果，并返回带输出路径的元信息。"""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved: list[CroppedSpine] = []
    for crop in crops:
        filename = f"{image_stem}_spine_{crop.detection.index:03d}_{crop.detection.confidence:.2f}.jpg"
        output_path = output_dir / filename
        write_image(output_path, crop.image)
        saved.append(
            CroppedSpine(
                detection=crop.detection,
                image=crop.image,
                output_path=output_path,
            )
        )
    return saved


def draw_obb_preview(
    image: np.ndarray,
    detections: list[OBBDetection],
    *,
    color: tuple[int, int, int] = (255, 0, 0),
) -> np.ndarray:
    """绘制带编号和置信度的 OBB 检测预览图。"""

    preview = image.copy()
    for detection in detections:
        points = np.round(detection.points).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(preview, [points], isClosed=True, color=color, thickness=3)

        anchor = tuple(points.reshape(-1, 2).mean(axis=0).astype(int))
        label = f"{detection.index}:{detection.confidence:.2f}"
        cv2.putText(
            preview,
            label,
            anchor,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    return preview
