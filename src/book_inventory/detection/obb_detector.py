"""YOLOv8-OBB 书脊检测器。

本模块封装最终选定的 OBB 路线。它不关心 OCR、馆藏匹配或界面展示，只负责：

1. 加载训练好的 YOLOv8-OBB 权重；
2. 对输入图片进行推理；
3. 返回每个书脊的 4 个角点和置信度。

OBB 四点框相比普通水平框更适合倾斜摆放的书脊，也方便后续做透视裁剪。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class OBBDetection:
    """单个书脊旋转框检测结果。"""

    index: int
    points: np.ndarray
    confidence: float
    class_id: int = 0
    class_name: str = "book_spine"

    def to_dict(self) -> dict[str, object]:
        """转换为便于日志、CSV 或 JSON 保存的字典。"""

        return {
            "index": self.index,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "points": self.points.astype(float).round(2).tolist(),
        }


class OBBBookSpineDetector:
    """基于 YOLOv8-OBB 的书脊检测器。"""

    def __init__(
        self,
        weights_path: str | Path,
        *,
        conf: float = 0.60,
        iou: float = 0.50,
        imgsz: int = 640,
        device: str | int = 0,
    ) -> None:
        """初始化检测器。

        Args:
            weights_path: 训练好的 OBB 权重路径。
            conf: 置信度阈值，最终模型当前推荐 0.60。
            iou: NMS IoU 阈值。
            imgsz: YOLO 推理输入尺寸。
            device: 推理设备，RTX 4060 通常为 0；CPU 可传入 ``"cpu"``。
        """

        self.weights_path = Path(weights_path)
        if not self.weights_path.exists():
            raise FileNotFoundError(f"找不到 OBB 模型权重：{self.weights_path}")

        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device

        # 延迟到运行时导入 Ultralytics，避免普通数据处理脚本导入项目包时强依赖深度学习环境。
        from ultralytics import YOLO

        self.model = YOLO(self.weights_path)

    def detect(self, image_path: str | Path) -> list[OBBDetection]:
        """检测单张图片中的书脊旋转框。"""

        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"找不到待检测图片：{image_path}")

        prediction = self.model.predict(
            source=str(image_path),
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )[0]

        if prediction.obb is None:
            return []

        points_array = prediction.obb.xyxyxyxy.detach().cpu().numpy()
        confidences = prediction.obb.conf.detach().cpu().numpy()
        classes = prediction.obb.cls.detach().cpu().numpy().astype(int)

        detections: list[OBBDetection] = []
        for index, (points, confidence, class_id) in enumerate(
            zip(points_array, confidences, classes, strict=False),
            start=1,
        ):
            detections.append(
                OBBDetection(
                    index=index,
                    points=np.asarray(points, dtype=np.float32),
                    confidence=float(confidence),
                    class_id=int(class_id),
                )
            )
        return detections
