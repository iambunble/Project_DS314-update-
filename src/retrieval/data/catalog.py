"""Catalog = bảng metadata đã chuẩn hoá + đường dẫn ảnh. Mọi module khác dùng chỉ số hàng (0..N-1)
của catalog làm khoá chung, nên embedding/BM25/filter luôn thẳng hàng với nhau."""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass
class Catalog:
    df: pd.DataFrame  # cột: imageID, owner, day, day_num, hour, frame, classes, counts, description, timestamp, image_path

    def __len__(self) -> int:
        return len(self.df)

    @property
    def ids(self) -> list[str]:
        return self.df["imageID"].tolist()

    def index_of(self, image_id: str) -> int:
        return self._id2idx[image_id]

    def __post_init__(self):
        self.df = self.df.reset_index(drop=True)
        self._id2idx = {iid: i for i, iid in enumerate(self.df["imageID"])}
        self.owners = sorted(self.df["owner"].unique())
        self.days = sorted(self.df["day"].unique())
        self.all_classes = sorted({c for cs in self.df["classes"] for c in cs})
        # ma trận đếm đồ vật [N, C] để lọc / chấm điểm nhanh bằng numpy
        self.class_index = {c: j for j, c in enumerate(self.all_classes)}
        m = np.zeros((len(self.df), len(self.all_classes)), dtype=np.int16)
        for i, counts in enumerate(self.df["counts"]):
            for c, n in counts.items():
                m[i, self.class_index[c]] = n
        self.count_matrix = m

    def record(self, idx: int) -> dict:
        r = self.df.iloc[idx]
        return {
            "imageID": r.imageID, "owner": r.owner, "day": r.day, "hour": int(r.hour),
            "frame": int(r.frame), "time": r.time_str, "objects": r.counts,
            "description": r.description,
            "image_path": str(r.image_path) if isinstance(r.image_path, (str, Path)) else None,
        }


def load_catalog(cfg: dict) -> Catalog:
    df = pd.read_csv(cfg["paths"]["metadata_csv"])
    df["classes"] = df["class"].apply(_safe_json(list))
    df["counts"] = df["class_count"].apply(_safe_json(dict))
    df["day_num"] = df["day"].str.extract(r"(\d+)").astype(int)
    fps = cfg.get("frames_per_second", 1) or 1
    # cột `hour` là chuẩn; frame ≈ giây trong giờ (vài frame giờ 13 vượt 3600 -> kẹp lại 3599 để không lấn sang giờ sau)
    sec_in_hour = (df["frame"] / fps).astype(int).clip(0, 3599)
    df["sec_of_day"] = df["hour"] * 3600 + sec_in_hour
    df["timestamp"] = (df["day_num"] - 1) * 86400 + df["sec_of_day"]  # giây, để sắp xếp sự kiện
    df["time_str"] = [f"{d} {h:02d}:{s // 60:02d}:{s % 60:02d}"
                      for d, h, s in zip(df["day"], df["hour"].astype(int), sec_in_hour)]
    df["image_path"] = _attach_image_paths(df, cfg)
    df = df.sort_values(["day_num", "hour", "frame"]).reset_index(drop=True)
    return Catalog(df)


def _safe_json(kind):
    def f(x):
        try:
            v = json.loads(x)
            return v if isinstance(v, kind) else kind()
        except (TypeError, ValueError):
            return kind()
    return f


def _attach_image_paths(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """Quét đệ quy image_root (Batch1/2/3...). Ghép theo:
       1) tên file == imageID, hoặc 2) tên file dạng 'HH_frame' (pattern trong config) + duy nhất trong CSV."""
    root = Path(cfg["paths"]["image_root"])
    paths = pd.Series([None] * len(df), index=df.index, dtype=object)
    if not root.exists():
        log.warning("Không thấy thư mục ảnh %s — vẫn tìm được theo mô tả, nhưng không có CLIP/hiển thị ảnh.", root)
        return paths

    by_id = {iid: i for i, iid in zip(df.index, df["imageID"])}
    by_hf: dict[tuple[int, int], list[int]] = {}
    for i, h, fr in zip(df.index, df["hour"], df["frame"]):
        by_hf.setdefault((int(h), int(fr)), []).append(i)
    pat = re.compile(cfg.get("image_name_pattern", r"^(?P<hour>\d{1,2})_(?P<frame>\d+)$"))

    n_found, unmatched, ambiguous = 0, [], []
    for p in root.rglob("*"):
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        stem = p.stem
        if stem in by_id:
            paths[by_id[stem]] = str(p); n_found += 1; continue
        m = pat.match(stem)
        if m:
            rows = by_hf.get((int(m["hour"]), int(m["frame"])), [])
            if len(rows) == 1:
                paths[rows[0]] = str(p); n_found += 1; continue
            if len(rows) > 1:
                ambiguous.append(str(p)); continue
        unmatched.append(str(p))

    missing = int(paths.isna().sum())
    log.info("Ghép ảnh: %d khớp, %d ảnh thừa không có trong CSV, %d ảnh mơ hồ, %d dòng CSV thiếu ảnh",
             n_found, len(unmatched), len(ambiguous), missing)
    if ambiguous:
        log.warning("Nhiều dòng CSV trùng giờ+frame (khác ngày/địa điểm) — hãy đổi tên ảnh theo imageID. Ví dụ: %s",
                    ambiguous[:3])
    return paths
