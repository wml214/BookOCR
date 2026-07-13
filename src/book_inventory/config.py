"""项目配置读取工具。

本模块只做一件事：从 ``configs/project.yaml`` 读取路径和参数，并把相对路径
转换为基于项目根目录的绝对路径。这样模型推理、OCR 和 Streamlit 页面都可以
使用同一份配置，避免到处写死路径。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ProjectConfig:
    """项目运行配置。"""

    raw: dict[str, Any]
    project_root: Path = PROJECT_ROOT

    def path(self, *keys: str) -> Path:
        """读取配置中的路径项并转换为绝对路径。

        参数示例：
            ``config.path("paths", "converted_images")``
        """

        value: Any = self.raw
        for key in keys:
            value = value[key]
        path = Path(str(value))
        if path.is_absolute():
            return path
        return (self.project_root / path).resolve()

    def get(self, *keys: str, default: Any = None) -> Any:
        """按层级读取普通配置项，缺失时返回默认值。"""

        value: Any = self.raw
        for key in keys:
            if not isinstance(value, dict) or key not in value:
                return default
            value = value[key]
        return value


def load_config(config_path: str | Path | None = None) -> ProjectConfig:
    """加载项目 YAML 配置。"""

    path = Path(config_path) if config_path else PROJECT_ROOT / "configs/project.yaml"
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return ProjectConfig(raw=raw)
