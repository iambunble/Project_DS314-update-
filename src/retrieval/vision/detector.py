"""Nhận diện đồ vật bằng YOLO11 (80 lớp COCO) + màu áo cho từng người.

Kết quả mỗi ảnh là danh sách: {"c": lớp, "s": độ tin cậy, "b": [x1,y1,x2,y2] (tỉ lệ 0..1), "col": [màu chính, màu phụ?]}
Lưu ở artifacts/detections.json, thẳng hàng với thứ tự catalog.
"""
from __future__ import annotations

import logging
import time

import numpy as np

from .colors import torso_colors

log = logging.getLogger(__name__)


class YoloDetector:
    def __init__(self, model: str = "yolo11m.pt", conf: float = 0.25, iou: float = 0.5, imgsz: int = 640,
                 device: str | None = None, batch: int = 8):
        from ultralytics import YOLO
        self.model_name = model
        self.model = YOLO(model)
        self.kw = dict(conf=conf, iou=iou, imgsz=imgsz, verbose=False)
        if device:
            self.kw["device"] = device
        self.batch = batch

    @property
    def name(self) -> str:
        return f"{self.model_name}@{self.kw['imgsz']}"

    def run(self, paths: list[str | None]) -> list[list[dict]]:
        out: list[list[dict]] = [[] for _ in paths]
        todo = [(i, p) for i, p in enumerate(paths) if p]
        t0 = time.time()
        for k in range(0, len(todo), self.batch):
            chunk = todo[k:k + self.batch]
            results = self.model.predict([p for _, p in chunk], **self.kw)
            for (i, _), r in zip(chunk, results):
                out[i] = self._convert(r)
            done = k + len(chunk)
            if (k // self.batch) % 25 == 0 or done == len(todo):
                el = time.time() - t0
                log.info("YOLO: %d/%d ảnh (%.0fs, còn ~%.0fs)", done, len(todo), el,
                         el / done * (len(todo) - done))
        return out

    @staticmethod
    def _convert(r) -> list[dict]:
        if r.boxes is None or len(r.boxes) == 0:
            return []
        H, W = r.orig_shape
        rgb = r.orig_img[..., ::-1]  # ultralytics trả BGR
        xyxy = r.boxes.xyxy.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        conf = r.boxes.conf.cpu().numpy()
        dets = []
        for b, c, s in zip(xyxy, cls, conf):
            name = r.names[int(c)]
            d = {"c": name, "s": round(float(s), 3),
                 "b": [round(float(b[0] / W), 4), round(float(b[1] / H), 4),
                       round(float(b[2] / W), 4), round(float(b[3] / H), 4)]}
            if name == "person":
                d["col"] = [c_ for c_, _ in torso_colors(rgb, b)]
            dets.append(d)
        return dets


def summarize(dets: list[dict], count_conf: float) -> tuple[dict[str, int], list[list[str]]]:
    """-> (số lượng mỗi lớp với conf >= count_conf, danh sách màu áo của từng người)."""
    counts: dict[str, int] = {}
    colors: list[list[str]] = []
    for d in dets:
        if d["s"] < count_conf:
            continue
        counts[d["c"]] = counts.get(d["c"], 0) + 1
        if d["c"] == "person" and d.get("col"):
            colors.append(d["col"])
    return counts, colors


def count_matrix(all_dets: list[list[dict]], classes: list[str], count_conf: float) -> np.ndarray:
    idx = {c: j for j, c in enumerate(classes)}
    m = np.zeros((len(all_dets), len(classes)), np.int16)
    for i, dets in enumerate(all_dets):
        for d in dets:
            if d["s"] >= count_conf and d["c"] in idx:
                m[i, idx[d["c"]]] += 1
    return m
