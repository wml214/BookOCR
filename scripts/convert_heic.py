"""将项目中的原始图片批量整理为高质量 JPEG。

本项目最初采集的 iPhone HEIC 使用了网格图像结构。FFmpeg 会把它识别为多个 512×512
图块，直接选择第一个视频流只会输出一个小图块。因此本脚本使用
``pillow-heif`` 读取并拼装完整主图，再由 Pillow 保存 JPEG。

补拍图片可能已经是 JPG/JPEG/PNG。为了让后续审计、标注和训练流程只面对一种
稳定格式，本脚本也会把这些常见图片重新转存为统一的高质量 JPEG。

脚本具有以下安全特性：

- 永不修改原始图片；
- 默认不覆盖已存在且可正常读取的 JPEG；
- 每张图转换后立即校验尺寸和文件完整性；
- 生成 CSV 清单，记录成功、跳过或失败原因。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps

try:
    from pillow_heif import register_heif_opener
except ImportError as exc:  # pragma: no cover - 仅在缺少依赖时触发
    raise SystemExit(
        "缺少 pillow-heif。请先执行：python -m pip install pillow-heif"
    ) from exc


# 为 Pillow 注册 HEIF/HEIC 解码器。注册后可使用普通 Image.open 读取 HEIC。
register_heif_opener()


# 默认路径始终相对于项目根目录计算，避免 PyCharm 将工作目录设为
# ``scripts`` 后错误地查找 ``scripts/图像采集`` 或把结果写入脚本目录。
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 允许进入数据流水线的原始图片格式。HEIC/HEIF 用于手机原图，JPG/PNG 用于补拍或导出图。
SUPPORTED_SOURCE_SUFFIXES = {".heic", ".heif", ".jpg", ".jpeg", ".png"}

# 默认同时扫描旧的采集目录和新的 data/raw。data/raw 适合继续追加补拍数据，
# 且已被 .gitignore 忽略，不会误提交到 GitHub。
DEFAULT_SOURCE_DIRS = (
    PROJECT_ROOT / "图像采集/图像采集",
    PROJECT_ROOT / "data/raw",
)


@dataclass(frozen=True)
class ConversionResult:
    """单张图片转换结果，用于生成可审计的清单。"""

    source_file: str
    output_file: str
    status: str
    width: int
    height: int
    source_sha256: str
    message: str


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """分块计算文件 SHA-256，避免一次把大文件读入内存。"""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def validate_jpeg(path: Path) -> tuple[int, int]:
    """校验 JPEG 是否可以完整解码，并返回宽高。"""

    with Image.open(path) as image:
        image.verify()

    # ``verify`` 后图像对象不可继续使用，需要重新打开读取尺寸。
    with Image.open(path) as image:
        width, height = image.size
        if width < 1000 or height < 1000:
            raise ValueError(
                f"输出尺寸异常：{width}x{height}，可能错误读取了 HEIC 缩略图"
            )
        return width, height


def convert_one(
    source: Path,
    destination: Path,
    quality: int,
    overwrite: bool,
) -> ConversionResult:
    """转换单张原始图片，并返回详细结果。

    ``ImageOps.exif_transpose`` 会根据 EXIF 方向把像素旋转到正确朝向，
    随后保存时即可去除容易造成不同软件显示不一致的方向依赖。
    """

    source_hash = sha256_file(source)

    if destination.exists() and not overwrite:
        try:
            width, height = validate_jpeg(destination)
            return ConversionResult(
                source.name,
                destination.name,
                "skipped",
                width,
                height,
                source_hash,
                "目标 JPEG 已存在且校验通过",
            )
        except Exception:
            # 已存在但损坏的文件不能视为成功，继续重新转换。
            pass

    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(source) as image:
            # 先应用方向信息，再转换为 JPEG 支持的 RGB 色彩模式。
            converted = ImageOps.exif_transpose(image).convert("RGB")
            converted.save(
                destination,
                format="JPEG",
                quality=quality,
                subsampling=0,
                optimize=True,
            )

        width, height = validate_jpeg(destination)
        return ConversionResult(
            source.name,
            destination.name,
            "success",
            width,
            height,
            source_hash,
            "",
        )
    except Exception as exc:
        # 转换失败时删除不完整输出，防止下次运行误判为有效文件。
        destination.unlink(missing_ok=True)
        return ConversionResult(
            source.name,
            destination.name,
            "failed",
            0,
            0,
            source_hash,
            str(exc),
        )


def write_manifest(results: list[ConversionResult], manifest_path: Path) -> None:
    """将转换结果写入 UTF-8 BOM CSV，便于在 Excel 中直接查看。"""

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=(
                "source_file",
                "output_file",
                "status",
                "width",
                "height",
                "source_sha256",
                "message",
            ),
        )
        writer.writeheader()
        writer.writerows(result.__dict__ for result in results)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="批量将 HEIC/JPG/PNG 整理为 JPEG。")
    parser.add_argument(
        "--source-dir",
        type=Path,
        action="append",
        default=None,
        help=(
            "原始图片目录，可重复传入多个。"
            "不传时默认扫描 图像采集/图像采集 和 data/raw。"
        ),
    )
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否递归扫描 source-dir 子目录，默认开启。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data/interim/images_jpg",
        help="JPEG 输出目录。",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data/interim/heic_conversion_manifest.csv",
        help="转换清单输出路径。",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        choices=range(80, 101),
        metavar="80-100",
        help="JPEG 质量，默认 95。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="重新转换并覆盖已经存在的有效 JPEG。",
    )
    return parser.parse_args()


def iter_source_files(source_dirs: list[Path], recursive: bool) -> list[Path]:
    """收集所有可处理的原始图片路径。

    这里故意只筛选文件后缀，不按文件名推断批次；这样后续继续补拍时，
    只要把图片放进 ``data/raw`` 的任意子目录，再运行脚本即可进入流水线。
    """

    sources: list[Path] = []
    for source_dir in source_dirs:
        if not source_dir.is_dir():
            print(f"跳过不存在的目录：{source_dir.resolve()}")
            continue

        candidates = source_dir.rglob("*") if recursive else source_dir.iterdir()
        sources.extend(
            path
            for path in candidates
            if path.is_file() and path.suffix.lower() in SUPPORTED_SOURCE_SUFFIXES
        )

    # 用 resolve 后的字符串去重，避免同一目录被重复传入时重复处理同一张图。
    unique_sources = {str(path.resolve()): path for path in sources}
    return sorted(unique_sources.values(), key=lambda path: str(path.resolve()).lower())


def build_output_names(sources: list[Path]) -> dict[Path, str]:
    """为每张源图生成输出文件名。

    大多数情况下沿用源文件名的 stem，例如 ``IMG_7832.HEIC`` 输出为
    ``IMG_7832.jpg``。如果不同目录中出现同名源图，则追加短哈希，避免后处理
    结果互相覆盖。
    """

    stem_counts: dict[str, int] = {}
    for source in sources:
        stem_counts[source.stem.lower()] = stem_counts.get(source.stem.lower(), 0) + 1

    output_names: dict[Path, str] = {}
    for source in sources:
        if stem_counts[source.stem.lower()] == 1:
            output_names[source] = f"{source.stem}.jpg"
        else:
            short_hash = sha256_file(source)[:8]
            output_names[source] = f"{source.stem}_{short_hash}.jpg"
    return output_names


def main() -> None:
    """批量转换所有原始图片，生成清单并在失败时返回非零退出码。"""

    args = parse_args()
    source_dirs = args.source_dir or list(DEFAULT_SOURCE_DIRS)
    sources = iter_source_files(source_dirs, args.recursive)
    if not sources:
        readable_dirs = "、".join(str(path.resolve()) for path in source_dirs)
        raise RuntimeError(f"目录中没有可处理图片：{readable_dirs}")

    output_names = build_output_names(sources)
    results: list[ConversionResult] = []
    for index, source in enumerate(sources, start=1):
        destination = args.output_dir / output_names[source]
        result = convert_one(source, destination, args.quality, args.overwrite)
        results.append(result)
        print(
            f"[{index:03d}/{len(sources):03d}] "
            f"{source.name}: {result.status} "
            f"{result.width}x{result.height}"
        )

    write_manifest(results, args.manifest)

    success_count = sum(result.status == "success" for result in results)
    skipped_count = sum(result.status == "skipped" for result in results)
    failed = [result for result in results if result.status == "failed"]
    print(f"成功转换：{success_count}")
    print(f"已存在跳过：{skipped_count}")
    print(f"转换失败：{len(failed)}")
    print(f"转换清单：{args.manifest.resolve()}")

    if failed:
        failed_names = ", ".join(result.source_file for result in failed)
        raise SystemExit(f"存在转换失败文件：{failed_names}")


if __name__ == "__main__":
    main()
