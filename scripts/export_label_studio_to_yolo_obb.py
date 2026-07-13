"""将 Label Studio 书脊多边形标注转换为 YOLOv8-OBB 数据集。

OBB（Oriented Bounding Box，有向/旋转框）比普通水平框更适合当前项目中的
倾斜书脊。Label Studio 中已有的书脊多边形可以自动转换为最小外接旋转矩形，
再导出为 Ultralytics YOLO OBB 标签格式：

```
class_id x1 y1 x2 y2 x3 y3 x4 y4
```

其中 4 个点为旋转框四角，坐标均归一化到 0~1。

注意：
    本脚本只读取人工提交的最新有效标注，不读取 prediction 预标注。
    输出目录会被重新创建，因此可在每次新增标注后重复运行。
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASS_ID = 0
CLASS_NAME = "book_spine"


@dataclass(frozen=True)
class ObbAnnotation:
    """单个书脊的 OBB 四点标注。"""

    points: list[tuple[float, float]]


@dataclass(frozen=True)
class LabeledTask:
    """一张已人工标注图片及其 OBB 标注。"""

    task_id: int
    file_name: str
    source_image: Path
    sequence: int
    obbs: list[ObbAnnotation]


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="导出 Label Studio 标注为 YOLOv8-OBB 数据集。")
    parser.add_argument(
        "--database",
        type=Path,
        default=PROJECT_ROOT / "data/interim/label_studio_data/label_studio.sqlite3",
        help="Label Studio SQLite 数据库路径。",
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        default=PROJECT_ROOT,
        help="解析 /data/local-files/?d=... 时使用的项目根目录。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data/processed/dataset/obb_v1",
        help="YOLO OBB 数据集输出目录。",
    )
    parser.add_argument(
        "--dataset-yaml",
        type=Path,
        default=PROJECT_ROOT / "models/configs/book_spine_obb_v1.yaml",
        help="Ultralytics OBB 数据集 YAML 输出路径。",
    )
    parser.add_argument("--project-id", type=int, default=1, help="Label Studio 项目 ID。")
    parser.add_argument("--seed", type=int, default=20260702, help="数据划分随机种子。")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="训练集比例。")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="验证集比例。")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="测试集比例。")
    return parser.parse_args()


def load_json(value: str | None, default: Any) -> Any:
    """安全解析 JSON 字符串。"""

    if not value:
        return default
    return json.loads(value)


def resolve_image_path(task_data: dict[str, Any], images_root: Path) -> Path:
    """从 Label Studio 任务数据中解析图片绝对路径。"""

    image_value = str(task_data.get("image", ""))
    file_name = str(task_data.get("file_name", ""))
    if "?d=" in image_value:
        relative = image_value.split("?d=", 1)[1]
        relative = relative.replace("%5C", "/").replace("\\", "/")
        return (images_root / relative).resolve()
    return (images_root / "data/interim/images_jpg" / file_name).resolve()


def image_size(image_path: Path) -> tuple[int, int]:
    """读取图片宽高，作为标注原始尺寸缺失时的兜底。"""

    image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取图片：{image_path}")
    height, width = image.shape[:2]
    return width, height


def polygon_to_obb(
    points_percent: list[list[float]],
    width: int,
    height: int,
) -> ObbAnnotation | None:
    """将 Label Studio 百分比多边形转换为最小外接旋转矩形。

    Label Studio 的多边形点为百分比坐标。为了避免宽高比例不同导致角度失真，
    这里先还原为像素坐标，计算旋转矩形后再归一化。
    """

    pixel_points: list[tuple[float, float]] = []
    for point in points_percent:
        if not isinstance(point, list | tuple) or len(point) < 2:
            continue
        x = max(0.0, min(float(width), float(point[0]) / 100.0 * width))
        y = max(0.0, min(float(height), float(point[1]) / 100.0 * height))
        pixel_points.append((x, y))

    if len(pixel_points) < 3:
        return None

    contour = np.asarray(pixel_points, dtype=np.float32).reshape((-1, 1, 2))
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)

    normalized: list[tuple[float, float]] = []
    for x, y in box:
        normalized.append(
            (
                max(0.0, min(1.0, float(x) / width)),
                max(0.0, min(1.0, float(y) / height)),
            )
        )

    return ObbAnnotation(points=normalized)


def extract_obbs(result: list[dict[str, Any]], fallback_width: int, fallback_height: int) -> list[ObbAnnotation]:
    """从 Label Studio result 中提取并转换所有书脊 OBB。"""

    obbs: list[ObbAnnotation] = []
    for item in result:
        if item.get("type") != "polygonlabels":
            continue
        if item.get("from_name") != "spine_polygon":
            continue

        value = item.get("value") or {}
        labels = value.get("polygonlabels") or []
        if CLASS_NAME not in labels:
            continue

        width = int(item.get("original_width") or fallback_width)
        height = int(item.get("original_height") or fallback_height)
        points = value.get("points") or []
        obb = polygon_to_obb(points, width=width, height=height)
        if obb is not None:
            obbs.append(obb)

    return obbs


def load_labeled_tasks(database: Path, images_root: Path, project_id: int) -> list[LabeledTask]:
    """读取每个任务最新的人工有效标注。"""

    if not database.exists():
        raise FileNotFoundError(f"找不到 Label Studio 数据库：{database}")

    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    cursor = connection.cursor()

    rows = cursor.execute(
        """
        SELECT
            t.id AS task_id,
            t.data AS task_data,
            latest.result AS result
        FROM task t
        JOIN (
            SELECT tc.*
            FROM task_completion tc
            JOIN (
                SELECT task_id, MAX(id) AS latest_id
                FROM task_completion
                WHERE project_id = ? AND was_cancelled = 0
                GROUP BY task_id
            ) picked ON picked.latest_id = tc.id
        ) latest ON latest.task_id = t.id
        WHERE t.project_id = ?
          AND t.is_labeled = 1
        ORDER BY t.id
        """,
        (project_id, project_id),
    ).fetchall()
    connection.close()

    tasks: list[LabeledTask] = []
    for row in rows:
        task_data = load_json(row["task_data"], {})
        source_image = resolve_image_path(task_data, images_root)
        if not source_image.exists():
            raise FileNotFoundError(f"任务 {row['task_id']} 对应图片不存在：{source_image}")

        width, height = image_size(source_image)
        result = load_json(row["result"], [])
        obbs = extract_obbs(result, fallback_width=width, fallback_height=height)
        if not obbs:
            continue

        file_name = str(task_data.get("file_name") or source_image.name)
        sequence = int(task_data.get("sequence") or row["task_id"])
        tasks.append(
            LabeledTask(
                task_id=int(row["task_id"]),
                file_name=file_name,
                source_image=source_image,
                sequence=sequence,
                obbs=obbs,
            )
        )

    return tasks


def split_tasks(
    tasks: list[LabeledTask],
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, list[LabeledTask]]:
    """按比例划分 train/val/test。"""

    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"train/val/test 比例之和必须为 1，当前为 {ratio_sum}")

    shuffled = list(tasks)
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def write_dataset(splits: dict[str, list[LabeledTask]], output_dir: Path, dataset_yaml: Path) -> None:
    """写出 YOLO OBB 数据集。"""

    if output_dir.exists():
        shutil.rmtree(output_dir)

    summary_rows: list[dict[str, Any]] = []
    for split_name, split_tasks in splits.items():
        image_dir = output_dir / "images" / split_name
        label_dir = output_dir / "labels" / split_name
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        for task in split_tasks:
            target_image = image_dir / task.file_name
            shutil.copy2(task.source_image, target_image)

            label_path = label_dir / f"{Path(task.file_name).stem}.txt"
            lines: list[str] = []
            for obb in task.obbs:
                values = [str(CLASS_ID)]
                for x, y in obb.points:
                    values.append(f"{x:.6f}")
                    values.append(f"{y:.6f}")
                lines.append(" ".join(values))
            label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

            summary_rows.append(
                {
                    "split": split_name,
                    "task_id": task.task_id,
                    "file_name": task.file_name,
                    "sequence": task.sequence,
                    "book_spine_count": len(task.obbs),
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "split_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split", "task_id", "file_name", "sequence", "book_spine_count"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    dataset_yaml.parent.mkdir(parents=True, exist_ok=True)
    dataset_yaml.write_text(
        "\n".join(
            [
                "# YOLOv8-OBB 书脊旋转框数据集",
                "# 本文件由 scripts/export_label_studio_to_yolo_obb.py 自动生成。",
                f"path: {output_dir.as_posix()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "",
                "names:",
                f"  {CLASS_ID}: {CLASS_NAME}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    """脚本入口。"""

    args = parse_args()
    tasks = load_labeled_tasks(args.database, args.images_root, args.project_id)
    if not tasks:
        raise RuntimeError("没有读取到可用于 OBB 训练的人工标注。")

    splits = split_tasks(
        tasks,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    write_dataset(splits, args.output_dir, args.dataset_yaml)

    print(f"已导出 YOLOv8-OBB 数据集：{args.output_dir}")
    for split_name, tasks_in_split in splits.items():
        instance_count = sum(len(task.obbs) for task in tasks_in_split)
        print(f"{split_name}: {len(tasks_in_split)} 张图片，{instance_count} 个旋转框")
    print(f"数据集配置：{args.dataset_yaml}")


if __name__ == "__main__":
    main()
