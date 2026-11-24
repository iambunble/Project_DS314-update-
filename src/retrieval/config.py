"""Đọc config.yaml, trả về dict; đường dẫn được resolve theo thư mục gốc dự án."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict[str, Any]:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if overrides:
        _deep_update(cfg, overrides)
    if os.environ.get("IMAGE_ROOT"):          # đặt thư mục ảnh theo máy mà không phải sửa config.yaml
        cfg["paths"]["image_root"] = os.environ["IMAGE_ROOT"]
    if os.environ.get("INDEX_DIR"):           # dùng chỉ mục đã build ở thư mục khác (vd bản sao code để thử nghiệm)
        cfg["paths"]["index_dir"] = os.environ["INDEX_DIR"]
    for key, value in cfg["paths"].items():
        p = Path(value)
        cfg["paths"][key] = p if p.is_absolute() else (PROJECT_ROOT / p)
    return cfg


def _deep_update(base: dict, upd: dict) -> None:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
