"""书脊检测模块：负责 YOLO 分割/OBB 模型的训练、评估和推理。"""

from book_inventory.detection.obb_detector import OBBBookSpineDetector, OBBDetection

__all__ = ["OBBBookSpineDetector", "OBBDetection"]
