"""Bố cục cảnh của camera cố định: chia khung hình thành các VÙNG (bếp nấu, bồn rửa, bàn ăn, sofa...).

Cả 5000 ảnh đều từ 1 camera gắn cố định ở góc trên căn bếp, nên 1 toạ độ trong ảnh luôn ứng với cùng 1
chỗ trong phòng. Mỗi người YOLO tìm được -> lấy "điểm chân" (giữa cạnh dưới khung người) -> rơi vào vùng nào
thì người đó đang ở vùng đó. Nhờ vậy trả lời được "2 người đứng ở bếp nấu", "có người ngồi ở sofa",
"không ai ở bồn rửa"... mà không cần model mới (dùng lại detections.json).

Toạ độ vùng nằm trong config.yaml (scene.zones), tỉ lệ 0..1 theo chiều rộng/cao ảnh. Đổi camera / căn phòng
thì chỉ cần vẽ lại vùng: `python scripts/show_zones.py <ảnh>` để xem vùng chồng lên ảnh.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Zone:
    name: str
    label: str                                   # tên hiển thị tiếng Việt
    rects: list[tuple[float, float, float, float]]
    words: list[str] = field(default_factory=list)        # cụm từ khớp ở bất kỳ đâu trong câu
    gated_words: list[str] = field(default_factory=list)  # chỉ khớp khi đứng sau giới từ vị trí ("ở bếp")

    def contains(self, x: float, y: float) -> bool:
        return any(x1 <= x <= x2 and y1 <= y <= y2 for x1, y1, x2, y2 in self.rects)


@dataclass
class SceneLayout:
    zones: list[Zone]

    @property
    def names(self) -> list[str]:
        return [z.name for z in self.zones]

    def zone(self, name: str) -> Zone:
        return next(z for z in self.zones if z.name == name)

    def label(self, name: str) -> str:
        return self.zone(name).label

    def zones_of_box(self, box) -> list[str]:
        """box = [x1, y1, x2, y2] tỉ lệ 0..1 -> các vùng chứa điểm chân của người."""
        x, y = foot_point(box)
        return [z.name for z in self.zones if z.contains(x, y)]

    def zone_count_matrix(self, dets: list[list[dict]], count_conf: float) -> np.ndarray:
        """[N, Z] số người trong từng vùng của mỗi ảnh."""
        m = np.zeros((len(dets), len(self.zones)), np.int16)
        for i, ds in enumerate(dets):
            for d in ds:
                if d["c"] != "person" or d["s"] < count_conf:
                    continue
                x, y = foot_point(d["b"])
                for j, z in enumerate(self.zones):
                    if z.contains(x, y):
                        m[i, j] += 1
        return m


def foot_point(box) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, y2


def load_layout(cfg: dict) -> SceneLayout | None:
    sc = (cfg.get("scene") or {})
    zs = sc.get("zones") or {}
    if not sc.get("enabled", True) or not zs:
        return None
    from ..query.lexicon import ZONE_SYNONYMS
    zones = []
    for name, z in zs.items():
        rects = z["rects"] if "rects" in z else [z["rect"]]
        syn = ZONE_SYNONYMS.get(name, {})
        zones.append(Zone(name=name, label=z.get("label", name), rects=[tuple(map(float, r)) for r in rects],
                          words=list(dict.fromkeys(syn.get("words", []) + z.get("words", []))),
                          gated_words=list(dict.fromkeys(syn.get("gated", []) + z.get("gated_words", [])))))
    return SceneLayout(zones)
