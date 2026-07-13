"""PaddleOCR 书脊文字识别封装。

本项目的图像流程是：
1. 先用 YOLOv8-OBB 找到每一本书的倾斜书脊；
2. 再把每个 OBB 书脊透视裁剪为单本书的小图；
3. 最后把这些小图交给 PaddleOCR 识别文字。

这里单独封装 PaddleOCR，是为了让上层脚本不直接依赖 PaddleOCR 的复杂返回结构。
后续如果要更换 OCR 引擎、调整过滤阈值、加入方向判断，只需要改这个文件。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class OCRTextLine:
    """PaddleOCR 识别出的一行文本。

    Attributes:
        text: OCR 原始文本。
        confidence: 文本识别置信度，范围通常为 0~1。
        box: 该行文本在裁剪图中的四点框，格式为 [[x1, y1], ...]。
    """

    text: str
    confidence: float
    box: list[list[int]]


@dataclass(frozen=True)
class SpineOCRResult:
    """单个书脊裁剪图的 OCR 结果。"""

    image_path: Path
    lines: list[OCRTextLine]

    @property
    def joined_text(self) -> str:
        """将多行 OCR 文本按从上到下的顺序拼接，便于后续书名匹配。"""

        return "".join(line.text for line in self.lines)

    @property
    def readable_text(self) -> str:
        """保留分隔符的文本，便于人工检查 OCR 中间结果。"""

        return " | ".join(line.text for line in self.lines)

    @property
    def average_confidence(self) -> float:
        """按文本行简单平均的 OCR 置信度。没有有效文本时返回 0。"""

        if not self.lines:
            return 0.0
        return sum(line.confidence for line in self.lines) / len(self.lines)

    def lines_as_json(self) -> str:
        """把逐行识别结果序列化为 JSON 字符串，便于写入 CSV。"""

        payload = [
            {
                "text": line.text,
                "confidence": round(line.confidence, 4),
                "box": line.box,
            }
            for line in self.lines
        ]
        return json.dumps(payload, ensure_ascii=False)


class PaddleSpineOCR:
    """面向书脊裁剪图的 PaddleOCR 识别器。

    PaddleOCR 初始化会加载检测、方向分类和识别模型，耗时较长，所以本类在构造时只初始化一次。
    实际项目中应复用同一个实例，不要每张图都重新创建。
    """

    def __init__(
        self,
        *,
        lang: str = "ch",
        min_line_confidence: float = 0.20,
        use_textline_orientation: bool = True,
    ) -> None:
        """初始化 OCR 引擎。

        Args:
            lang: PaddleOCR 语言配置，中文书籍默认使用 ch。
            min_line_confidence: 过滤极低置信度文本行的阈值。
            use_textline_orientation: 是否启用文本行方向分类。书脊文字常有旋转，建议开启。
        """

        # 延迟导入：只有真的跑 OCR 时才需要 PaddleOCR，避免普通工具脚本也强制加载重模型。
        from paddleocr import PaddleOCR

        self.min_line_confidence = min_line_confidence
        self.ocr = PaddleOCR(
            lang=lang,
            # 书脊裁剪图已经经过 OBB 透视校正，不需要整页文档方向分类和文档去弯曲。
            # 关闭这两项可以明显减少无关处理，也降低误把书脊当文档版面的概率。
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            # 单行文字方向仍然保留，因为书脊上经常出现竖排、横排、倒置文字。
            use_textline_orientation=use_textline_orientation,
        )

    def recognize(self, image_path: str | Path) -> SpineOCRResult:
        """识别单张书脊裁剪图。"""

        path = Path(image_path)
        raw_result = self.ocr.predict(str(path))
        lines = self._parse_result(raw_result)
        return SpineOCRResult(image_path=path, lines=lines)

    def _parse_result(self, raw_result: Any) -> list[OCRTextLine]:
        """解析 PaddleOCR 3.x 的 OCRResult 返回结构。

        当前环境中的 PaddleOCR 返回 list[OCRResult]，每个 OCRResult 的 `.json["res"]`
        中包含 rec_texts、rec_scores、rec_polys。这里写得稍微宽松一些，方便兼容小版本差异。
        """

        if isinstance(raw_result, list) and raw_result:
            item = raw_result[0]
        else:
            item = raw_result

        if hasattr(item, "json"):
            data = item.json.get("res", item.json)
        elif isinstance(item, dict):
            data = item.get("res", item)
        else:
            return []

        texts = data.get("rec_texts", []) or []
        scores = data.get("rec_scores", []) or []
        boxes = data.get("rec_polys", data.get("dt_polys", [])) or []

        parsed: list[OCRTextLine] = []
        for index, text in enumerate(texts):
            cleaned = normalize_ocr_text(str(text))
            score = float(scores[index]) if index < len(scores) else 0.0
            box = boxes[index] if index < len(boxes) else []

            # 空文本、识别置信度太低的噪声行不参与后续拼接和匹配。
            if not cleaned or score < self.min_line_confidence:
                continue

            parsed.append(
                OCRTextLine(
                    text=cleaned,
                    confidence=score,
                    box=_to_int_box(box),
                )
            )

        # PaddleOCR 返回顺序通常已经接近阅读顺序；这里再按文本框中心 y 坐标排序一次，
        # 对竖向书脊来说可以保证从上到下拼接。
        parsed.sort(key=lambda line: _box_center_y(line.box))
        return parsed


def normalize_ocr_text(text: str) -> str:
    """清理 OCR 文本中的空白和常见无意义符号。

    注意：这里不做激进纠错，只处理明显的格式噪声。
    真正的书名纠错应该放到“馆藏目录匹配”模块中完成。
    """

    text = text.strip()
    text = re.sub(r"\s+", "", text)
    return text


def _to_int_box(box: Any) -> list[list[int]]:
    """把 PaddleOCR 输出的 numpy/list 坐标统一转为普通 Python int。"""

    result: list[list[int]] = []
    for point in box:
        if len(point) < 2:
            continue
        result.append([int(round(float(point[0]))), int(round(float(point[1])))])
    return result


def _box_center_y(box: list[list[int]]) -> float:
    """计算文本框中心 y 坐标，用于排序。"""

    if not box:
        return 0.0
    return sum(point[1] for point in box) / len(box)
