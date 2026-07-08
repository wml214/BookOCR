"""基于视觉的图书盘点系统 Streamlit 应用。

这个界面把已经实现的命令行能力包装成可演示的软件：
1. 支持一次上传多张有序书架图片；
2. 调用最终版 YOLOv8n-OBB 模型检测书脊；
3. 对书脊裁剪图执行 PaddleOCR；
4. 结合馆藏目录进行规范书名匹配；
5. 可选执行相邻图片空间去重；
6. 展示汇总结果、明细、标注预览图，并提供 CSV 下载。

注意：首次启动会加载 YOLO、PaddleOCR 和馆藏索引，速度会慢一些；之后 Streamlit 会缓存资源。
"""

from __future__ import annotations

import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from book_inventory.dedup import DedupItem, deduplicate_adjacent_images, parse_obb_points  # noqa: E402
from book_inventory.detection import OBBBookSpineDetector  # noqa: E402
from book_inventory.matching import CatalogMatcher  # noqa: E402
from book_inventory.ocr import PaddleSpineOCR  # noqa: E402
from book_inventory.pipeline import InventoryDetail, InventoryPipeline, write_detail_csv, write_summary_csv  # noqa: E402


def load_streamlit():
    """延迟导入 Streamlit，避免普通脚本导入 app.py 时强制依赖界面包。"""

    import streamlit as st

    return st


def write_dict_csv(rows: list[dict[str, Any]], output_path: Path, fieldnames: list[str]) -> None:
    """把字典列表写为 CSV。

    使用 utf-8-sig 是为了让 Windows Excel 直接打开时中文不乱码。
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_uploaded_images(uploaded_files: list[Any], output_dir: Path) -> list[Path]:
    """保存用户上传的图片，并按上传顺序返回本地路径。

    YOLO 和 OpenCV 对 HEIC 支持不好，因此 HEIC/HEIF 会先转换成 JPG。
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    for index, uploaded_file in enumerate(uploaded_files, start=1):
        suffix = Path(uploaded_file.name).suffix.lower()
        safe_stem = Path(uploaded_file.name).stem.replace(" ", "_")

        if suffix in {".heic", ".heif"}:
            from PIL import Image
            from pillow_heif import register_heif_opener

            register_heif_opener()
            image = Image.open(uploaded_file)
            image = image.convert("RGB")
            output_path = output_dir / f"{index:03d}_{safe_stem}.jpg"
            image.save(output_path, quality=95)
        else:
            output_path = output_dir / f"{index:03d}_{safe_stem}{suffix}"
            output_path.write_bytes(uploaded_file.getbuffer())

        saved_paths.append(output_path)

    return saved_paths


def detail_to_dedup_items(details: list[InventoryDetail]) -> list[DedupItem]:
    """把盘点明细转换为跨图去重模块需要的结构。"""

    items: list[DedupItem] = []
    for item_id, detail in enumerate(details):
        points = parse_obb_points(detail.obb_points_json)
        if points is None:
            continue
        items.append(
            DedupItem(
                item_id=item_id,
                source_image=detail.source_image,
                spine_index=detail.spine_index,
                title_key=detail.matched_title,
                points=points,
                match_status=detail.match_status,
            )
        )
    return items


def apply_dedup_to_details(
    details: list[InventoryDetail],
    image_paths: list[Path],
    output_dir: Path,
    *,
    spatial_iou_threshold: float,
    min_inlier_ratio: float,
) -> tuple[list[InventoryDetail], list[dict[str, Any]], list[dict[str, Any]]]:
    """对盘点明细执行跨图像去重，并写出日志文件。"""

    items = detail_to_dedup_items(details)
    kept_ids, duplicate_pairs, registration_logs = deduplicate_adjacent_images(
        items,
        image_paths,
        spatial_iou_threshold=spatial_iou_threshold,
        min_inlier_ratio=min_inlier_ratio,
    )

    removed_ids = {pair.removed_item_id for pair in duplicate_pairs}
    kept_details = [detail for index, detail in enumerate(details) if index not in removed_ids]
    duplicate_rows = [pair.__dict__ for pair in duplicate_pairs]

    write_dict_csv(
        duplicate_rows,
        output_dir / "dedup" / "重复合并记录.csv",
        ["kept_item_id", "removed_item_id", "source_image_a", "source_image_b", "title_key", "spatial_iou"],
    )
    write_dict_csv(
        registration_logs,
        output_dir / "dedup" / "图像配准日志.csv",
        ["source_image", "target_image", "matched_points", "inlier_points", "inlier_ratio", "status"],
    )

    detail_rows: list[dict[str, Any]] = []
    duplicate_to_kept = {pair.removed_item_id: pair.kept_item_id for pair in duplicate_pairs}
    for index, detail in enumerate(details):
        row = {"item_id": index, **detail.__dict__}
        row["dedup_status"] = "duplicate" if index in removed_ids else "keep"
        row["duplicate_of_item_id"] = duplicate_to_kept.get(index, "")
        detail_rows.append(row)

    write_dict_csv(
        detail_rows,
        output_dir / "dedup" / "去重明细.csv",
        [
            "item_id",
            "source_image",
            "spine_index",
            "detection_confidence",
            "crop_path",
            "ocr_text",
            "ocr_readable_text",
            "ocr_confidence",
            "matched_title",
            "match_score",
            "match_status",
            "author",
            "call_number",
            "publisher",
            "obb_points_json",
            "dedup_status",
            "duplicate_of_item_id",
        ],
    )

    return kept_details, duplicate_rows, registration_logs


def make_download_button(st: Any, label: str, file_path: Path, mime: str = "text/csv") -> None:
    """如果文件存在，则在界面中生成下载按钮。"""

    if file_path.exists():
        st.download_button(
            label=label,
            data=file_path.read_bytes(),
            file_name=file_path.name,
            mime=mime,
        )


def render_preview_gallery(st: Any, preview_dir: Path) -> None:
    """展示检测预览图。"""

    preview_paths = sorted(preview_dir.glob("*.jpg"))
    if not preview_paths:
        st.info("暂未生成预览图。")
        return

    columns = st.columns(2)
    for index, preview_path in enumerate(preview_paths[:20]):
        columns[index % 2].image(str(preview_path), caption=preview_path.name, use_container_width=True)

    if len(preview_paths) > 20:
        st.caption(f"界面最多预览前 20 张，完整预览图在：{preview_dir}")


def main() -> None:
    """Streamlit 应用入口。"""

    st = load_streamlit()

    st.set_page_config(
        page_title="基于视觉的图书盘点系统",
        page_icon="📚",
        layout="wide",
    )

    @st.cache_resource(show_spinner=False)
    def cached_detector(weights: str, conf: float, iou: float, imgsz: int, device: str) -> OBBBookSpineDetector:
        """缓存 OBB 检测模型，避免每次交互都重新加载权重。"""

        return OBBBookSpineDetector(weights, conf=conf, iou=iou, imgsz=imgsz, device=device)

    @st.cache_resource(show_spinner=False)
    def cached_ocr_reader() -> PaddleSpineOCR:
        """缓存 PaddleOCR 实例。"""

        return PaddleSpineOCR()

    @st.cache_resource(show_spinner=False)
    def cached_matcher(catalog: str) -> CatalogMatcher:
        """缓存馆藏目录匹配索引。"""

        return CatalogMatcher.from_csv(catalog)

    st.title("基于视觉的图书盘点系统")
    st.caption("YOLOv8n-OBB 书脊定位 + PaddleOCR + 馆藏目录匹配 + 相邻图像去重")

    with st.sidebar:
        st.header("运行配置")
        weights_path = st.text_input(
            "OBB 模型权重",
            str(PROJECT_ROOT / "models/weights/book_spine_obb_final_200_best.pt"),
        )
        catalog_path = st.text_input(
            "馆藏目录 CSV",
            str(PROJECT_ROOT / "data/processed/catalog/catalog.csv"),
        )
        device = st.selectbox("推理设备", ["0", "cpu"], index=0)
        conf = st.slider("检测置信度 conf", 0.10, 0.95, 0.60, 0.05)
        iou = st.slider("NMS IoU", 0.10, 0.95, 0.50, 0.05)
        imgsz = st.selectbox("推理尺寸 imgsz", [512, 640, 768, 960], index=1)
        match_threshold = st.slider("书名自动匹配阈值", 0.50, 0.95, 0.72, 0.01)

        st.divider()
        enable_dedup = st.checkbox("启用相邻图像去重", value=True)
        spatial_iou = st.slider("去重空间 IoU 阈值", 0.10, 0.90, 0.45, 0.05)
        min_inlier_ratio = st.slider("最低配准内点比例", 0.01, 0.50, 0.15, 0.01)

    uploaded_files = st.file_uploader(
        "上传按拍摄顺序排列的书架图片",
        type=["jpg", "jpeg", "png", "heic", "heif"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        st.success(f"已选择 {len(uploaded_files)} 张图片。请确认上传顺序就是拍摄顺序。")
        with st.expander("查看上传顺序", expanded=False):
            for index, uploaded_file in enumerate(uploaded_files, start=1):
                st.write(f"{index:03d}. {uploaded_file.name}")
    else:
        st.info("请先上传 JPG、PNG 或 HEIC 图片。")

    run_button = st.button("开始盘点", type="primary", disabled=not uploaded_files)

    if not run_button:
        st.stop()

    run_id = datetime.now().strftime("streamlit_%Y%m%d_%H%M%S")
    output_dir = PROJECT_ROOT / "outputs" / "inventory" / run_id
    upload_dir = output_dir / "uploads"

    with st.status("正在保存上传图片...", expanded=True) as status:
        image_paths = save_uploaded_images(uploaded_files, upload_dir)
        st.write(f"图片已保存到：{upload_dir}")

        st.write("加载模型、OCR 和馆藏索引...")
        detector = cached_detector(weights_path, conf, iou, imgsz, device)
        ocr_reader = cached_ocr_reader()
        matcher = cached_matcher(catalog_path)

        st.write("开始执行书脊检测、OCR 和馆藏匹配...")
        started_at = time.perf_counter()
        pipeline = InventoryPipeline(
            detector=detector,
            ocr_reader=ocr_reader,
            matcher=matcher,
            output_dir=output_dir,
            match_threshold=match_threshold,
        )
        details = pipeline.process_images(image_paths)
        summary_rows = pipeline.summarize(details)

        detail_csv = output_dir / "盘点明细.csv"
        summary_csv = output_dir / "盘点结果.csv"
        write_detail_csv(details, detail_csv)
        write_summary_csv(summary_rows, summary_csv)

        dedup_summary_rows = summary_rows
        duplicate_rows: list[dict[str, Any]] = []
        registration_logs: list[dict[str, Any]] = []
        if enable_dedup and len(image_paths) >= 2:
            st.write("执行相邻图像空间去重...")
            kept_details, duplicate_rows, registration_logs = apply_dedup_to_details(
                details,
                image_paths,
                output_dir,
                spatial_iou_threshold=spatial_iou,
                min_inlier_ratio=min_inlier_ratio,
            )
            dedup_summary_rows = pipeline.summarize(kept_details)
            write_summary_csv(dedup_summary_rows, output_dir / "dedup" / "去重盘点结果.csv")
        elif enable_dedup:
            st.write("只有 1 张图片，跳过去重。")

        elapsed = time.perf_counter() - started_at
        status.update(label="盘点完成", state="complete", expanded=False)

    matched_count = sum(1 for item in details if item.match_status == "matched")
    pending_count = len(details) - matched_count

    metric_cols = st.columns(5)
    metric_cols[0].metric("上传图片", len(image_paths))
    metric_cols[1].metric("识别书脊", len(details))
    metric_cols[2].metric("成功匹配", matched_count)
    metric_cols[3].metric("待确认", pending_count)
    metric_cols[4].metric("处理时间", f"{elapsed:.1f}s")

    if enable_dedup:
        st.caption(
            f"去重合并 {len(duplicate_rows)} 个重复书脊；"
            f"配准日志 {len(registration_logs)} 条。输出目录：{output_dir}"
        )
    else:
        st.caption(f"未启用去重。输出目录：{output_dir}")

    tab_summary, tab_detail, tab_preview, tab_logs = st.tabs(["汇总结果", "识别明细", "标注预览", "运行日志"])

    with tab_summary:
        st.subheader("按书名汇总")
        st.dataframe(dedup_summary_rows, use_container_width=True)
        col_a, col_b = st.columns(2)
        with col_a:
            make_download_button(st, "下载原始盘点结果 CSV", summary_csv)
        with col_b:
            make_download_button(st, "下载去重盘点结果 CSV", output_dir / "dedup" / "去重盘点结果.csv")

    with tab_detail:
        st.subheader("单本书脊识别明细")
        st.dataframe([detail.__dict__ for detail in details], use_container_width=True)
        make_download_button(st, "下载盘点明细 CSV", detail_csv)

    with tab_preview:
        st.subheader("OBB 检测预览图")
        render_preview_gallery(st, output_dir / "previews")

    with tab_logs:
        st.subheader("去重与配准日志")
        if registration_logs:
            st.dataframe(registration_logs, use_container_width=True)
        else:
            st.info("本次没有生成配准日志。")

        if duplicate_rows:
            st.subheader("重复合并记录")
            st.dataframe(duplicate_rows, use_container_width=True)
        else:
            st.info("本次没有合并重复书脊。")

        st.code(
            json.dumps(
                {
                    "output_dir": str(output_dir),
                    "weights": weights_path,
                    "catalog": catalog_path,
                    "conf": conf,
                    "iou": iou,
                    "imgsz": imgsz,
                    "match_threshold": match_threshold,
                    "enable_dedup": enable_dedup,
                },
                ensure_ascii=False,
                indent=2,
            ),
            language="json",
        )


if __name__ == "__main__":
    main()
