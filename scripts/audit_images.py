"""分析已转换 JPEG 的基础质量、重复情况和相邻图像重叠程度。

本脚本用于标注前的数据质检，主要输出：

- 每张图片的尺寸、亮度、对比度、清晰度和曝光比例；
- 相邻图片的 ORB 特征匹配数量与单应性内点比例；
- 完全重复和近似重复图片提示；
- 建议人工检查的质量问题；
- 全体数据的汇总 JSON。

注意：自动指标只能帮助筛选，不能代替人工判断。书脊是否完整、文字是否可读、
是否存在严重遮挡等内容仍需要在标注前目视确认。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ImageAudit:
    """单张图像的质量与相邻关系统计。"""

    sequence: int
    file_name: str
    width: int
    height: int
    file_size_bytes: int
    sha256: str
    perceptual_hash: str
    brightness_mean: float
    contrast_std: float
    sharpness_laplacian: float
    dark_pixel_ratio: float
    bright_pixel_ratio: float
    previous_file: str
    adjacent_good_matches: int
    adjacent_inlier_ratio: float
    quality_flags: str
    scene_id: str
    manual_status: str
    manual_notes: str


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """分块计算 SHA-256，用于发现完全相同的图片。"""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def difference_hash(gray: np.ndarray, hash_size: int = 8) -> str:
    """计算简单感知哈希，用于提示视觉上高度相似的图片。

    图像缩放到 ``hash_size + 1`` 列后，比较相邻像素的亮度大小。该哈希速度快，
    适合数据清单初筛，但不能作为删除图片的唯一依据。
    """

    resized = cv2.resize(
        gray,
        (hash_size + 1, hash_size),
        interpolation=cv2.INTER_AREA,
    )
    differences = resized[:, 1:] > resized[:, :-1]
    bits = "".join("1" if value else "0" for value in differences.flatten())
    return f"{int(bits, 2):0{hash_size * hash_size // 4}x}"


def hamming_distance(hash_a: str, hash_b: str) -> int:
    """计算两个十六进制感知哈希之间的汉明距离。"""

    return (int(hash_a, 16) ^ int(hash_b, 16)).bit_count()


def calculate_adjacent_overlap(
    previous_gray: np.ndarray | None,
    current_gray: np.ndarray,
    maximum_features: int,
) -> tuple[int, float]:
    """估计当前图和前一张图的局部特征重叠程度。

    为降低 4032×3024 原图的计算开销，先按长边 1280 像素等比例缩放。然后：

    1. 使用 ORB 提取关键点；
    2. 使用 Hamming 距离做 KNN 匹配；
    3. 使用 Lowe 比率筛选可靠匹配；
    4. 当可靠点足够时估计单应性矩阵并计算 RANSAC 内点比例。

    内点比例越高，说明相邻图越可能包含可稳定配准的公共书脊区域。
    """

    if previous_gray is None:
        return 0, 0.0

    def resize_for_features(image: np.ndarray) -> np.ndarray:
        height, width = image.shape
        scale = min(1.0, 1280.0 / max(height, width))
        if scale == 1.0:
            return image
        return cv2.resize(
            image,
            (round(width * scale), round(height * scale)),
            interpolation=cv2.INTER_AREA,
        )

    previous_small = resize_for_features(previous_gray)
    current_small = resize_for_features(current_gray)

    orb = cv2.ORB_create(nfeatures=maximum_features)
    previous_keypoints, previous_descriptors = orb.detectAndCompute(
        previous_small, None
    )
    current_keypoints, current_descriptors = orb.detectAndCompute(
        current_small, None
    )

    if previous_descriptors is None or current_descriptors is None:
        return 0, 0.0

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn_matches = matcher.knnMatch(
        previous_descriptors,
        current_descriptors,
        k=2,
    )
    good_matches = [
        first
        for pair in knn_matches
        if len(pair) == 2
        for first, second in [pair]
        if first.distance < 0.75 * second.distance
    ]

    if len(good_matches) < 8:
        return len(good_matches), 0.0

    source_points = np.float32(
        [previous_keypoints[match.queryIdx].pt for match in good_matches]
    ).reshape(-1, 1, 2)
    destination_points = np.float32(
        [current_keypoints[match.trainIdx].pt for match in good_matches]
    ).reshape(-1, 1, 2)
    _, mask = cv2.findHomography(
        source_points,
        destination_points,
        cv2.RANSAC,
        4.0,
    )
    if mask is None:
        return len(good_matches), 0.0

    return len(good_matches), float(mask.ravel().mean())


def build_quality_flags(
    brightness: float,
    contrast: float,
    sharpness: float,
    dark_ratio: float,
    bright_ratio: float,
    minimum_sharpness: float,
) -> list[str]:
    """根据保守阈值生成需要人工复核的质量提示。"""

    flags: list[str] = []
    if brightness < 55:
        flags.append("整体偏暗")
    elif brightness > 210:
        flags.append("整体偏亮")
    if contrast < 32:
        flags.append("对比度偏低")
    if sharpness < minimum_sharpness:
        flags.append("可能模糊")
    if dark_ratio > 0.35:
        flags.append("暗部占比高")
    if bright_ratio > 0.20:
        flags.append("高光占比高")
    return flags


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="生成 JPEG 数据质量审计清单。")
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=PROJECT_ROOT / "data/interim/images_jpg",
        help="待审计 JPEG 目录。",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data/interim/image_audit.csv",
        help="逐图审计 CSV 输出路径。",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=PROJECT_ROOT / "data/interim/image_audit_summary.json",
        help="汇总 JSON 输出路径。",
    )
    parser.add_argument(
        "--minimum-sharpness",
        type=float,
        default=80.0,
        help="拉普拉斯清晰度提示阈值，仅用于筛查。",
    )
    parser.add_argument(
        "--maximum-features",
        type=int,
        default=3000,
        help="每张缩略图最多提取的 ORB 特征数量。",
    )
    return parser.parse_args()


def main() -> None:
    """执行数据审计，写出逐图清单和汇总结果。"""

    args = parse_args()
    image_paths = sorted(
        path
        for path in args.image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not image_paths:
        raise RuntimeError(f"没有找到可审计图片：{args.image_dir.resolve()}")

    audits: list[ImageAudit] = []
    exact_hashes: dict[str, str] = {}
    perceptual_hashes: list[tuple[str, str]] = []
    previous_gray: np.ndarray | None = None
    previous_name = ""

    for sequence, image_path in enumerate(image_paths, start=1):
        # imdecode 可以稳定处理 Windows 中文路径；cv2.imread 在部分环境中会失败。
        encoded = np.fromfile(image_path, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"无法解码图片：{image_path}")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        file_hash = sha256_file(image_path)
        perceptual_hash = difference_hash(gray)

        brightness = float(gray.mean())
        contrast = float(gray.std())
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        dark_ratio = float(np.mean(gray <= 30))
        bright_ratio = float(np.mean(gray >= 245))
        good_matches, inlier_ratio = calculate_adjacent_overlap(
            previous_gray,
            gray,
            args.maximum_features,
        )
        flags = build_quality_flags(
            brightness,
            contrast,
            sharpness,
            dark_ratio,
            bright_ratio,
            args.minimum_sharpness,
        )

        if file_hash in exact_hashes:
            flags.append(f"与{exact_hashes[file_hash]}完全重复")
        else:
            exact_hashes[file_hash] = image_path.name

        # 汉明距离不超过 4 时只作“近似重复”提示，仍需人工目视确认。
        similar = [
            name
            for name, known_hash in perceptual_hashes
            if hamming_distance(perceptual_hash, known_hash) <= 4
        ]
        if similar:
            flags.append(f"疑似近似重复:{similar[-1]}")
        perceptual_hashes.append((image_path.name, perceptual_hash))

        audits.append(
            ImageAudit(
                sequence=sequence,
                file_name=image_path.name,
                width=width,
                height=height,
                file_size_bytes=image_path.stat().st_size,
                sha256=file_hash,
                perceptual_hash=perceptual_hash,
                brightness_mean=round(brightness, 3),
                contrast_std=round(contrast, 3),
                sharpness_laplacian=round(sharpness, 3),
                dark_pixel_ratio=round(dark_ratio, 6),
                bright_pixel_ratio=round(bright_ratio, 6),
                previous_file=previous_name,
                adjacent_good_matches=good_matches,
                adjacent_inlier_ratio=round(inlier_ratio, 6),
                quality_flags="；".join(flags),
                # 场景编号必须结合真实书架位置人工填写，不能仅凭文件名臆测。
                scene_id="",
                manual_status="待检查",
                manual_notes="",
            )
        )
        previous_gray = gray
        previous_name = image_path.name
        print(
            f"[{sequence:03d}/{len(image_paths):03d}] {image_path.name} "
            f"清晰度={sharpness:.1f} 相邻匹配={good_matches} "
            f"内点率={inlier_ratio:.2%}"
        )

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=ImageAudit.__dataclass_fields__)
        writer.writeheader()
        writer.writerows(asdict(audit) for audit in audits)

    flagged_count = sum(bool(audit.quality_flags) for audit in audits)
    overlap_values = [
        audit.adjacent_inlier_ratio for audit in audits if audit.previous_file
    ]
    summary = {
        "image_count": len(audits),
        "widths": sorted({audit.width for audit in audits}),
        "heights": sorted({audit.height for audit in audits}),
        "flagged_image_count": flagged_count,
        "brightness_mean": round(mean(audit.brightness_mean for audit in audits), 3),
        "contrast_mean": round(mean(audit.contrast_std for audit in audits), 3),
        "sharpness_median": round(
            median(audit.sharpness_laplacian for audit in audits), 3
        ),
        "adjacent_inlier_ratio_median": round(median(overlap_values), 6),
        "minimum_required_image_count": 200,
        "additional_images_needed": max(0, 200 - len(audits)),
        "notes": [
            "quality_flags 仅表示需要人工复核，不等同于必须删除。",
            "scene_id、manual_status 和 manual_notes 需要人工填写。",
            "数据集划分必须以 scene_id 为单位，不能随机拆散连续书架。",
        ],
    }
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"审计图片数：{len(audits)}")
    print(f"含质量提示图片数：{flagged_count}")
    print(f"距离最低 200 张还需补采：{summary['additional_images_needed']}")
    print(f"逐图清单：{args.manifest.resolve()}")
    print(f"汇总结果：{args.summary.resolve()}")


if __name__ == "__main__":
    main()

