"""OCR 模块：负责书脊裁剪、方向校正、图像增强和文字识别。"""

from book_inventory.ocr.paddle_reader import (
    OCRTextLine,
    PaddleSpineOCR,
    SpineOCRResult,
    normalize_ocr_text,
)
from book_inventory.ocr.spine_cropper import (
    CroppedSpine,
    crop_all_spines,
    crop_spine_by_obb,
    draw_obb_preview,
    read_image,
    save_crops,
    write_image,
)

__all__ = [
    "CroppedSpine",
    "OCRTextLine",
    "PaddleSpineOCR",
    "SpineOCRResult",
    "crop_all_spines",
    "crop_spine_by_obb",
    "draw_obb_preview",
    "normalize_ocr_text",
    "read_image",
    "save_crops",
    "write_image",
]
