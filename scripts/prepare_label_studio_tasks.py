"""为 Label Studio 生成本地图片标注任务 JSON。

Label Studio 在本机运行时，可以通过 ``/data/local-files/?d=...`` 访问
允许目录中的图片。该脚本只生成任务描述，不复制图片，也不修改审计清单。

使用前需要设置环境变量：

``LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true``
``LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=D:\\IT\\OCR``

然后将输出的 JSON 导入 Label Studio 项目，并使用
``configs/label_studio/book_spine.xml`` 作为标注界面配置。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import quote


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(
        description="生成 Label Studio 本地图片任务 JSON。"
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=PROJECT_ROOT / "data/interim/images_jpg",
        help="JPEG 图片目录。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data/interim/label_studio_tasks.json",
        help="任务 JSON 输出路径。",
    )
    return parser.parse_args()


def main() -> None:
    """根据 JPEG 文件生成稳定、有序的 Label Studio 任务。"""

    args = parse_args()
    if not args.image_dir.is_dir():
        raise FileNotFoundError(f"找不到图片目录：{args.image_dir.resolve()}")

    image_paths = sorted(args.image_dir.glob("*.jpg"))
    if not image_paths:
        raise RuntimeError(f"图片目录为空：{args.image_dir.resolve()}")

    tasks: list[dict[str, object]] = []
    for sequence, image_path in enumerate(image_paths, start=1):
        relative_path = image_path.resolve().relative_to(PROJECT_ROOT).as_posix()
        # URL 中保留正斜杠并编码中文或空格，确保 Windows 本地文件服务可访问。
        image_url = f"/data/local-files/?d={quote(relative_path, safe='/')}"
        tasks.append(
            {
                "data": {
                    "image": image_url,
                    "file_name": image_path.name,
                    "sequence": sequence,
                }
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(tasks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"生成任务数：{len(tasks)}")
    print(f"输出文件：{args.output.resolve()}")


if __name__ == "__main__":
    main()

