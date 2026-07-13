"""使用 YOLOv8-seg 模型为 Label Studio 未完成任务生成书脊预标注。

本脚本的目标是把已经训练好的书脊实例分割模型，用来辅助后续人工标注：

1. 从 Label Studio SQLite 数据库读取任务列表；
2. 跳过已经人工提交完成的任务；
3. 对未完成任务运行 YOLOv8-seg 预测；
4. 将预测出的书脊 mask 转换为 Label Studio 的多边形百分比坐标；
5. 写入 Label Studio 的 ``prediction`` 表，使界面中可以直接加载模型预标注。

注意：
    本脚本不会把预测结果直接变成人工标注结果。它只生成“预测/预标注”，
    仍需要在 Label Studio 中逐张检查、删除误检、补充漏检，然后点击 Submit。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASS_NAME = "book_spine"
FROM_NAME = "spine_polygon"
TO_NAME = "image"


@dataclass(frozen=True)
class TaskInfo:
    """Label Studio 中的一条图片任务。"""

    task_id: int
    project_id: int
    image_path: Path
    file_name: str
    sequence: int
    original_width: int
    original_height: int


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="为 Label Studio 生成 YOLO 书脊预标注。")
    parser.add_argument(
        "--database",
        type=Path,
        default=PROJECT_ROOT / "data/interim/label_studio_data/label_studio.sqlite3",
        help="Label Studio SQLite 数据库路径。",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "models/weights/book_spine_v2_65_best.pt",
        help="用于生成预标注的 YOLOv8-seg 权重。",
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        default=PROJECT_ROOT,
        help="解析 /data/local-files/?d=... 图片路径时使用的项目根目录。",
    )
    parser.add_argument("--project-id", type=int, default=1, help="Label Studio 项目 ID。")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO 推理输入尺寸。")
    parser.add_argument("--conf", type=float, default=0.30, help="预测置信度阈值。")
    parser.add_argument("--iou", type=float, default=0.50, help="YOLO NMS IoU 阈值。")
    parser.add_argument(
        "--polygon-mode",
        choices=("min-rect", "mask", "box"),
        default="min-rect",
        help=(
            "预标注多边形形状：min-rect 为 4 点旋转矩形，最适合书脊快速修正；"
            "mask 为原始分割轮廓；box 为水平矩形。"
        ),
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=80,
        help="单个书脊多边形最多保留点数，避免 Label Studio 过卡。",
    )
    parser.add_argument(
        "--model-version",
        default="book_spine_v2_65",
        help="写入 Label Studio prediction 表的模型版本名。",
    )
    parser.add_argument(
        "--include-labeled",
        action="store_true",
        help="默认跳过已人工完成任务；加上该参数会给已完成任务也生成预测。",
    )
    parser.add_argument(
        "--clear-existing",
        action="store_true",
        help="写入前删除同一 model-version 的旧预测，便于重复生成。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只运行预测并统计数量，不写入 Label Studio 数据库。",
    )
    return parser.parse_args()


def load_json(value: str | None, default: Any) -> Any:
    """安全解析 JSON 字符串。"""

    if not value:
        return default
    return json.loads(value)


def resolve_image_path(task_data: dict[str, Any], images_root: Path) -> Path:
    """从 Label Studio 任务数据中解析本地图片路径。"""

    image_value = str(task_data.get("image", ""))
    file_name = str(task_data.get("file_name", ""))

    if "?d=" in image_value:
        relative = image_value.split("?d=", 1)[1]
        relative = relative.replace("%5C", "/").replace("\\", "/")
        return (images_root / relative).resolve()

    return (images_root / "data/interim/images_jpg" / file_name).resolve()


def read_image_size(image_path: Path) -> tuple[int, int]:
    """读取图片宽高。

    使用 OpenCV 只读取必要的图像信息即可；如果图片无法读取，说明任务数据
    或本地文件存在问题，需要立即报错，不能静默跳过。
    """

    image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取图片：{image_path}")
    height, width = image.shape[:2]
    return width, height


def load_tasks(database: Path, images_root: Path, project_id: int, include_labeled: bool) -> list[TaskInfo]:
    """读取需要生成预标注的 Label Studio 任务。"""

    if not database.exists():
        raise FileNotFoundError(f"找不到 Label Studio 数据库：{database}")

    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    cursor = connection.cursor()

    labeled_clause = "" if include_labeled else "AND is_labeled = 0"
    rows = cursor.execute(
        f"""
        SELECT id, project_id, data
        FROM task
        WHERE project_id = ?
          {labeled_clause}
        ORDER BY id
        """,
        (project_id,),
    ).fetchall()
    connection.close()

    tasks: list[TaskInfo] = []
    for row in rows:
        task_data = load_json(row["data"], {})
        image_path = resolve_image_path(task_data, images_root)
        if not image_path.exists():
            raise FileNotFoundError(f"任务 {row['id']} 对应图片不存在：{image_path}")

        width, height = read_image_size(image_path)
        tasks.append(
            TaskInfo(
                task_id=int(row["id"]),
                project_id=int(row["project_id"]),
                image_path=image_path,
                file_name=str(task_data.get("file_name") or image_path.name),
                sequence=int(task_data.get("sequence") or row["id"]),
                original_width=width,
                original_height=height,
            )
        )

    return tasks


def simplify_polygon(points: np.ndarray, max_points: int) -> np.ndarray:
    """简化 YOLO mask 边界点，减少 Label Studio 前端负担。

    YOLO 输出的 mask 边界有时包含几百个点。书脊通常近似长条形，多数情况下
    不需要特别密集的轮廓点。这里逐步增大 ``epsilon``，直到点数不超过上限。
    """

    if len(points) <= max_points:
        return points

    contour = points.astype(np.float32).reshape((-1, 1, 2))
    perimeter = cv2.arcLength(contour, closed=True)
    for ratio in (0.001, 0.002, 0.004, 0.008, 0.012, 0.016, 0.02):
        approx = cv2.approxPolyDP(contour, epsilon=perimeter * ratio, closed=True)
        simplified = approx.reshape((-1, 2))
        if 3 <= len(simplified) <= max_points:
            return simplified

    # 如果常规简化仍然点数过多，就按间隔采样，保证至少留下 3 个点。
    step = max(1, int(np.ceil(len(points) / max_points)))
    sampled = points[::step]
    return sampled if len(sampled) >= 3 else points[:3]


def convert_prediction_points(points: np.ndarray, polygon_mode: str, max_points: int) -> np.ndarray:
    """把 YOLO 输出的 mask 轮廓转换成更适合人工修正的多边形。

    - ``mask``：保留分割轮廓并做适度简化，边界最贴合，但点多，难修改；
    - ``min-rect``：用最小外接旋转矩形表示书脊，只有 4 个点，最适合批量预标注；
    - ``box``：用水平外接矩形表示，只有 4 个点，适合完全竖直且不倾斜的图片。
    """

    if polygon_mode == "mask":
        return simplify_polygon(points, max_points=max_points)

    if polygon_mode == "box":
        min_x = float(np.min(points[:, 0]))
        max_x = float(np.max(points[:, 0]))
        min_y = float(np.min(points[:, 1]))
        max_y = float(np.max(points[:, 1]))
        return np.asarray(
            [
                [min_x, min_y],
                [max_x, min_y],
                [max_x, max_y],
                [min_x, max_y],
            ],
            dtype=np.float32,
        )

    # 默认方案：最小外接旋转矩形。书脊通常是细长四边形，这比原始 mask 更好编辑。
    contour = points.astype(np.float32).reshape((-1, 1, 2))
    rect = cv2.minAreaRect(contour)
    box_points = cv2.boxPoints(rect)
    return box_points.astype(np.float32)


def polygon_to_label_studio_result(
    points: np.ndarray,
    task: TaskInfo,
    score: float,
    max_points: int,
    polygon_mode: str,
) -> dict[str, Any] | None:
    """把单个 YOLO mask 多边形转换成 Label Studio result。"""

    if len(points) < 3:
        return None

    converted = convert_prediction_points(
        points=points,
        polygon_mode=polygon_mode,
        max_points=max_points,
    )
    if len(converted) < 3:
        return None

    percent_points: list[list[float]] = []
    for x, y in converted:
        percent_x = max(0.0, min(100.0, float(x) / task.original_width * 100.0))
        percent_y = max(0.0, min(100.0, float(y) / task.original_height * 100.0))
        percent_points.append([percent_x, percent_y])

    return {
        "id": uuid.uuid4().hex[:10],
        "from_name": FROM_NAME,
        "to_name": TO_NAME,
        "type": "polygonlabels",
        "origin": "prediction",
        "score": float(score),
        "original_width": task.original_width,
        "original_height": task.original_height,
        "image_rotation": 0,
        "value": {
            "points": percent_points,
            "closed": True,
            "polygonlabels": [CLASS_NAME],
        },
    }


def predict_task(model: YOLO, task: TaskInfo, args: argparse.Namespace) -> tuple[list[dict[str, Any]], float]:
    """对单张图片运行 YOLO 预测并返回 Label Studio result 列表。"""

    prediction = model.predict(
        source=str(task.image_path),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=0,
        verbose=False,
    )[0]

    if prediction.masks is None or prediction.boxes is None:
        return [], 0.0

    confidences = prediction.boxes.conf.detach().cpu().numpy().tolist()
    polygons = prediction.masks.xy
    results: list[dict[str, Any]] = []
    used_scores: list[float] = []

    for points, confidence in zip(polygons, confidences, strict=False):
        if points is None or len(points) < 3:
            continue
        item = polygon_to_label_studio_result(
            points=np.asarray(points, dtype=np.float32),
            task=task,
            score=float(confidence),
            max_points=args.max_points,
            polygon_mode=args.polygon_mode,
        )
        if item is None:
            continue
        results.append(item)
        used_scores.append(float(confidence))

    average_score = float(np.mean(used_scores)) if used_scores else 0.0
    return results, average_score


def backup_database(database: Path) -> Path:
    """写入 prediction 前备份数据库。"""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = database.with_name(f"{database.stem}_before_predictions_{timestamp}{database.suffix}")
    shutil.copy2(database, backup_path)
    return backup_path


def write_predictions(
    database: Path,
    predictions: dict[int, tuple[int, list[dict[str, Any]], float]],
    model_version: str,
    project_id: int,
    clear_existing: bool,
) -> None:
    """将预测结果写入 Label Studio SQLite 数据库。"""

    now = datetime.utcnow().replace(microsecond=0).isoformat(sep=" ") + "+00:00"
    connection = sqlite3.connect(database)
    cursor = connection.cursor()

    if clear_existing:
        cursor.execute(
            "DELETE FROM prediction WHERE project_id = ? AND model_version = ?",
            (project_id, model_version),
        )

    for task_id, (project_id, result, score) in predictions.items():
        cursor.execute(
            """
            INSERT INTO prediction (
                result, score, model_version, created_at, updated_at,
                task_id, cluster, mislabeling, neighbors, project_id,
                model_run_id, model_id
            )
            VALUES (?, ?, ?, ?, ?, ?, NULL, 0.0, NULL, ?, NULL, NULL)
            """,
            (
                json.dumps(result, ensure_ascii=False),
                float(score),
                model_version,
                now,
                now,
                int(task_id),
                int(project_id),
            ),
        )

    # Label Studio 列表页会读取 task.total_predictions；同步更新它，界面更直观。
    for task_id in predictions:
        cursor.execute(
            """
            UPDATE task
            SET total_predictions = (
                SELECT COUNT(*) FROM prediction WHERE prediction.task_id = task.id
            )
            WHERE id = ?
            """,
            (int(task_id),),
        )

    connection.commit()
    connection.close()


def main() -> None:
    """脚本入口。"""

    args = parse_args()
    if not args.model.exists():
        raise FileNotFoundError(f"找不到模型权重：{args.model}")

    tasks = load_tasks(
        database=args.database,
        images_root=args.images_root,
        project_id=args.project_id,
        include_labeled=args.include_labeled,
    )
    if not tasks:
        print("没有需要生成预标注的任务。")
        return

    print(f"准备为 {len(tasks)} 个任务生成预标注。")
    model = YOLO(args.model)

    predictions: dict[int, tuple[int, list[dict[str, Any]], float]] = {}
    total_regions = 0
    for index, task in enumerate(tasks, start=1):
        result, score = predict_task(model, task, args)
        predictions[task.task_id] = (task.project_id, result, score)
        total_regions += len(result)
        print(
            f"[{index:03d}/{len(tasks):03d}] "
            f"task={task.task_id} file={task.file_name} regions={len(result)} score={score:.3f}"
        )

    print(f"预标注统计：任务 {len(predictions)} 个，预测书脊 {total_regions} 个。")
    if args.dry_run:
        print("dry-run 模式：未写入 Label Studio 数据库。")
        return

    backup_path = backup_database(args.database)
    print(f"已备份数据库：{backup_path}")
    write_predictions(
        database=args.database,
        predictions=predictions,
        model_version=args.model_version,
        project_id=args.project_id,
        clear_existing=args.clear_existing,
    )
    print("已写入 Label Studio prediction 表。请刷新 Label Studio 页面查看预标注。")


if __name__ == "__main__":
    main()
