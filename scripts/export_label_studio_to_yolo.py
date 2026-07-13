"""将 Label Studio 书脊多边形标注转换为 YOLOv8-seg 数据集。

当前项目第一阶段只训练“书脊实例分割”模型，因此本脚本只读取
``spine_polygon`` / ``book_spine`` 多边形，不读取书名文本字段。

脚本默认直接读取 Label Studio 本地 SQLite 数据库，原因是：

- Label Studio 会保留跳过记录和历史提交，手工导出时容易混入无效数据；
- 我们可以明确筛选 ``task.is_labeled = 1`` 且 ``was_cancelled = 0`` 的最新有效标注；
- 可重复生成 YOLO 数据集，减少界面导出格式变化带来的不确定性。

输出结构符合 Ultralytics YOLOv8 分割训练要求：

```
data/processed/dataset/bootstrap_v1/
├─ images/
│  ├─ train/
│  ├─ val/
│  └─ test/
├─ labels/
│  ├─ train/
│  ├─ val/
│  └─ test/
└─ split_summary.csv

models/configs/book_spine_bootstrap_v1.yaml
```

YOLO 分割标签每行格式为：

```
class_id x1 y1 x2 y2 ... xn yn
```

其中坐标均为 0~1 之间的归一化值。
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASS_NAME = "book_spine"
CLASS_ID = 0


@dataclass(frozen=True)
class PolygonAnnotation:
    """单个书脊多边形标注。"""

    points: list[tuple[float, float]]


@dataclass(frozen=True)
class LabeledTask:
    """一张已完成标注的图片及其所有书脊多边形。"""

    task_id: int
    file_name: str
    source_image: Path
    sequence: int
    polygons: list[PolygonAnnotation]


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="导出 Label Studio 标注为 YOLOv8-seg 数据集。")
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
        default=PROJECT_ROOT / "data/processed/dataset/bootstrap_v1",
        help="YOLO 数据集输出目录。",
    )
    parser.add_argument(
        "--dataset-yaml",
        type=Path,
        default=PROJECT_ROOT / "models/configs/book_spine_bootstrap_v1.yaml",
        help="Ultralytics 数据集 YAML 输出路径。",
    )
    parser.add_argument(
        "--project-id",
        type=int,
        default=1,
        help="Label Studio 项目 ID。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260628,
        help="数据划分随机种子。",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="训练集比例。",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="验证集比例。",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="测试集比例。",
    )
    return parser.parse_args()


def load_json(value: str | None, default: Any) -> Any:
    """安全解析 JSON 字符串，空值时返回默认值。"""

    if not value:
        return default
    return json.loads(value)


def resolve_image_path(task_data: dict[str, Any], images_root: Path) -> Path:
    """从 Label Studio 任务数据中解析本地图片绝对路径。

    当前任务的 ``image`` 字段形如：

    ``/data/local-files/?d=data/interim/images_jpg/xxx.jpg``

    其中 ``d=`` 后面的路径相对于项目根目录。
    """

    image_value = str(task_data.get("image", ""))
    file_name = str(task_data.get("file_name", ""))

    if "?d=" in image_value:
        relative = image_value.split("?d=", 1)[1]
        # Label Studio 源存储同步时可能会把反斜杠 URL 编码为 %5C。
        relative = relative.replace("%5C", "/").replace("\\", "/")
        return (images_root / relative).resolve()

    # 兜底：如果任务数据里没有 image URL，就按 file_name 在标准目录中查找。
    return (images_root / "data/interim/images_jpg" / file_name).resolve()


def extract_polygons(result: list[dict[str, Any]]) -> list[PolygonAnnotation]:
    """从 Label Studio result 中提取 book_spine 多边形。

    Label Studio 多边形点坐标是百分比坐标，范围通常为 0~100。YOLO 需要 0~1，
    因此这里除以 100 并裁剪到合法范围。
    """

    polygons: list[PolygonAnnotation] = []
    for item in result:
        if item.get("type") != "polygonlabels":
            continue
        if item.get("from_name") != "spine_polygon":
            continue

        value = item.get("value") or {}
        labels = value.get("polygonlabels") or []
        if CLASS_NAME not in labels:
            continue

        raw_points = value.get("points") or []
        points: list[tuple[float, float]] = []
        for point in raw_points:
            if not isinstance(point, list | tuple) or len(point) < 2:
                continue
            x = max(0.0, min(1.0, float(point[0]) / 100.0))
            y = max(0.0, min(1.0, float(point[1]) / 100.0))
            points.append((x, y))

        # YOLO 分割至少需要 3 个点才能形成有效多边形。
        if len(points) >= 3:
            polygons.append(PolygonAnnotation(points=points))
    return polygons


def load_labeled_tasks(database: Path, images_root: Path, project_id: int) -> list[LabeledTask]:
    """读取每个任务最新的有效标注。"""

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

    tasks: list[LabeledTask] = []
    for row in rows:
        task_data = load_json(row["task_data"], {})
        result = load_json(row["result"], [])
        polygons = extract_polygons(result)
        if not polygons:
            # 空标注不进入分割训练集。
            continue

        file_name = str(task_data.get("file_name") or f"task_{row['task_id']}.jpg")
        sequence = int(task_data.get("sequence") or row["task_id"])
        source_image = resolve_image_path(task_data, images_root)
        if not source_image.exists():
            raise FileNotFoundError(f"标注任务 {row['task_id']} 对应图片不存在：{source_image}")

        tasks.append(
            LabeledTask(
                task_id=int(row["task_id"]),
                file_name=file_name,
                source_image=source_image,
                sequence=sequence,
                polygons=polygons,
            )
        )

    connection.close()
    return tasks


def split_tasks(
    tasks: list[LabeledTask],
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, list[LabeledTask]]:
    """把已标注图片划分为 train/val/test。

    这只是第一版预标模型的启动集。最终论文实验应在补齐全量标注后，
    根据 ``scene_id`` 按书架场景重新划分，避免连续照片跨集合泄漏。
    """

    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"划分比例之和必须为 1，当前为 {ratio_sum}")

    shuffled = tasks[:]
    random.Random(seed).shuffle(shuffled)

    total = len(shuffled)
    if total < 10:
        raise ValueError("有效标注图片少于 10 张，不建议训练分割模型。")

    train_count = max(1, round(total * train_ratio))
    val_count = max(1, round(total * val_ratio))

    # 确保 test 至少 1 张，同时不让 train/val/test 总数超过 total。
    if train_count + val_count >= total:
        val_count = max(1, total - train_count - 1)
    test_count = total - train_count - val_count
    if test_count < 1:
        test_count = 1
        train_count = max(1, total - val_count - test_count)

    return {
        "train": shuffled[:train_count],
        "val": shuffled[train_count : train_count + val_count],
        "test": shuffled[train_count + val_count :],
    }


def write_yolo_label(task: LabeledTask, label_path: Path) -> None:
    """写入单张图片的 YOLO 分割标签文件。"""

    lines: list[str] = []
    for polygon in task.polygons:
        coords = " ".join(f"{value:.6f}" for point in polygon.points for value in point)
        lines.append(f"{CLASS_ID} {coords}")

    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def reset_output_dirs(output_dir: Path) -> None:
    """清空并重建 YOLO 数据集输出目录。"""

    if output_dir.exists():
        shutil.rmtree(output_dir)

    for split in ("train", "val", "test"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def export_dataset(splits: dict[str, list[LabeledTask]], output_dir: Path) -> None:
    """复制图片并生成 YOLO 标签。"""

    reset_output_dirs(output_dir)

    summary_path = output_dir / "split_summary.csv"
    with summary_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=("split", "task_id", "sequence", "file_name", "polygon_count"),
        )
        writer.writeheader()

        for split, tasks in splits.items():
            for task in tasks:
                image_target = output_dir / "images" / split / task.file_name
                label_target = output_dir / "labels" / split / f"{Path(task.file_name).stem}.txt"
                shutil.copy2(task.source_image, image_target)
                write_yolo_label(task, label_target)
                writer.writerow(
                    {
                        "split": split,
                        "task_id": task.task_id,
                        "sequence": task.sequence,
                        "file_name": task.file_name,
                        "polygon_count": len(task.polygons),
                    }
                )


def write_dataset_yaml(output_dir: Path, dataset_yaml: Path) -> None:
    """写入 Ultralytics 数据集配置 YAML。"""

    dataset_yaml.parent.mkdir(parents=True, exist_ok=True)
    content = f"""# YOLOv8-seg 书脊分割启动数据集
# 本文件由 scripts/export_label_studio_to_yolo.py 自动生成。
path: {output_dir.as_posix()}
train: images/train
val: images/val
test: images/test

names:
  0: {CLASS_NAME}
"""
    dataset_yaml.write_text(content, encoding="utf-8")


def write_export_report(tasks: list[LabeledTask], splits: dict[str, list[LabeledTask]], output_dir: Path) -> None:
    """写入导出摘要，便于后续复核。"""

    report = {
        "labeled_image_count": len(tasks),
        "polygon_count": sum(len(task.polygons) for task in tasks),
        "splits": {
            split: {
                "image_count": len(items),
                "polygon_count": sum(len(task.polygons) for task in items),
            }
            for split, items in splits.items()
        },
        "class_names": [CLASS_NAME],
    }
    (output_dir / "export_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    """脚本入口。"""

    args = parse_args()
    tasks = load_labeled_tasks(args.database, args.images_root, args.project_id)
    if len(tasks) < 30:
        print(f"警告：当前有效标注图片只有 {len(tasks)} 张，建议至少 30 张后再训练第一版模型。")

    splits = split_tasks(tasks, args.seed, args.train_ratio, args.val_ratio, args.test_ratio)
    export_dataset(splits, args.output_dir)
    write_dataset_yaml(args.output_dir.resolve(), args.dataset_yaml)
    write_export_report(tasks, splits, args.output_dir)

    print(f"有效标注图片：{len(tasks)}")
    print(f"书脊多边形总数：{sum(len(task.polygons) for task in tasks)}")
    for split, items in splits.items():
        print(f"{split}: {len(items)} 张，{sum(len(task.polygons) for task in items)} 个书脊")
    print(f"YOLO 数据集：{args.output_dir.resolve()}")
    print(f"数据集配置：{args.dataset_yaml.resolve()}")


if __name__ == "__main__":
    main()
