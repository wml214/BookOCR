"""盘点流水线模块：串联检测、OCR、匹配、去重、计数和结果导出。"""

from book_inventory.pipeline.inventory_pipeline import (
    InventoryDetail,
    InventoryPipeline,
    write_detail_csv,
    write_summary_csv,
)

__all__ = [
    "InventoryDetail",
    "InventoryPipeline",
    "write_detail_csv",
    "write_summary_csv",
]
