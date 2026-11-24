"""Bằng chứng thị giác bổ sung cho SearchEngine — mỗi nguồn bật/tắt riêng trong config.yaml:

  objects.clip_tags  [PA1] CLIP theo vùng: so embedding vùng cắt sẵn (region_emb.npy: đảo bếp, bếp nấu...) với cặp câu
                     "a pot on the stove" / "an empty stove" -> có/không cho từng ảnh. Không cần build thêm.
  objects.yoloe      [PA2] khung đồ vật từ YOLOE (objects.json) -> số lượng + độ tin cậy.
  pose               [PA2] tư thế từng người (đứng / ngồi / cúi) + màu áo theo vai–hông (pose.json).
  hands              [PA2] vùng bàn tay (hands_emb.npy) -> "người cầm dao / điện thoại".

KẾT HỢP 2 NGUỒN cho câu hỏi có/không ("có nồi"): điểm có mặt = trung bình các nguồn đang có
  p_yoloe = độ tin cậy cao nhất của lớp đó trong ảnh, đổi về 0..1 (conf 0.15 -> 0, >= 0.5 -> 1)
  p_clip  = sigmoid((chênh_lệch − ngưỡng) / độ_dốc), ngưỡng tự chọn theo phân bố (Otsu) hoặc đặt tay trong config
-> hai nguồn cùng nói "có" thì điểm cao; một nguồn nói có thì nửa điểm.

CẦM GÌ: điểm bàn tay = max(
  khung YOLOE của vật đó nằm ngay chỗ bàn tay  (liên kết hình học),
  CLIP thấy bàn tay giống "a hand holding a knife" hơn "an empty hand" — xếp hạng trong mọi bàn tay, chỉ
  top `hands.top_frac` được điểm)  -> điểm ảnh = bàn tay cao nhất.
"""
from __future__ import annotations

import logging

import numpy as np

from ..vision.pose import POSTURES

log = logging.getLogger(__name__)

# [PA1] đồ vật -> (vùng cắt trong scene.regions, câu "có", câu "không")
DEFAULT_CLIP_TAGS: dict[str, dict] = {
    "pot": {"region": "stove",
            "pos": ["a cooking pot on the stove", "a pot on a stovetop", "a saucepan on the hob"],
            "neg": ["an empty stove", "a clean stovetop with nothing on it"]},
    "frying pan": {"region": "stove",
                   "pos": ["a frying pan on the stove", "a skillet on the stovetop"],
                   "neg": ["an empty stove", "a clean stovetop with nothing on it"]},
    "cutting board": {"region": "island",
                      "pos": ["a cutting board on the kitchen counter", "a chopping board with food on the counter"],
                      "neg": ["an empty kitchen counter", "a clean countertop with nothing on it"]},
    "plate": {"region": "island",
              "pos": ["plates on the kitchen counter", "a plate of food on the counter"],
              "neg": ["an empty kitchen counter", "a clean countertop with nothing on it"]},
    "knife": {"region": "island",
              "pos": ["a kitchen knife on a cutting board", "a knife on the counter"],
              "neg": ["an empty kitchen counter", "a clean countertop with nothing on it"]},
    "cup": {"region": "island",
            "pos": ["cups on the kitchen counter", "a mug on the counter"],
            "neg": ["an empty kitchen counter", "a clean countertop with nothing on it"]},
}

# Nguồn cho câu có/không từng đồ vật — chọn theo AUC đo trên 5000 ảnh (nhãn yếu = mô tả gốc có nhắc đồ vật):
#   pot: YOLOE 0.68 · CLIP vùng 0.43 | frying pan 0.69 · 0.55 | cup 0.74 · 0.56     -> chỉ YOLOE
#   knife 0.54 · 0.63 | cutting board 0.60 · 0.60 (gộp 0.63) | plate 0.71 · 0.69     -> cả hai
DEFAULT_SOURCES: dict[str, list[str]] = {
    "pot": ["yoloe"], "frying pan": ["yoloe"], "cup": ["yoloe"], "cell phone": ["yoloe"],
    "knife": ["yoloe", "clip"], "cutting board": ["yoloe", "clip"], "plate": ["yoloe", "clip"],
}

# [PA2] câu CLIP cho vùng bàn tay
DEFAULT_HAND_PROMPTS: dict[str, list[str]] = {
    "knife": ["a hand holding a knife", "a hand cutting food with a kitchen knife"],
    "cell phone": ["a hand holding a smartphone", "a person looking at a mobile phone in their hand"],
    "cup": ["a hand holding a cup", "a hand holding a mug"],
    "plate": ["a hand holding a plate", "a hand carrying a plate of food"],
    "pot": ["a hand holding a cooking pot", "a hand holding a saucepan"],
    "frying pan": ["a hand holding a frying pan"],
    "cutting board": ["a hand holding a cutting board"],
}
HAND_NEG = ["an empty hand", "a hand with nothing in it"]


def clip_tag_specs(cfg: dict, region_names: list[str]) -> dict[str, dict]:
    """[PA1] đồ vật -> {region, pos, neg}: mặc định + phần ghi đè trong config, chỉ giữ vùng có trong region_emb."""
    tc = (cfg.get("objects") or {}).get("clip_tags") or {}
    specs = {**DEFAULT_CLIP_TAGS, **(tc.get("tags") or {})}
    return {k: v for k, v in specs.items() if v.get("region") in region_names}


def tags_name(clip_name: str, specs: dict, crops_model: str) -> str:
    import hashlib
    h = hashlib.md5(repr(sorted((k, v["region"], tuple(v["pos"]), tuple(v["neg"])) for k, v in specs.items()))
                    .encode()).hexdigest()[:8]
    return f"{clip_name}|{h}|{hashlib.md5(crops_model.encode()).hexdigest()[:8]}"


def clip_margins(region_emb: np.ndarray, region_names: list[str], specs: dict, encode) -> dict[str, np.ndarray]:
    """chênh lệch = sim(vùng, trung bình câu "có") − sim(vùng, trung bình câu "không"); vùng trống (ảnh thiếu) = nan."""
    out = {}
    for obj, sp in specs.items():
        reg = region_emb[:, region_names.index(sp["region"]), :]
        pos = np.stack([encode(t) for t in sp["pos"]]).mean(axis=0)
        neg = np.stack([encode(t) for t in sp["neg"]]).mean(axis=0)
        mg = (reg @ pos - reg @ neg).astype(np.float32)
        mg[np.linalg.norm(reg, axis=1) == 0] = np.nan
        out[obj] = mg
    return out


def hand_margins(hands_emb: np.ndarray, cfg: dict, encode) -> dict[str, np.ndarray]:
    prompts = {**DEFAULT_HAND_PROMPTS, **((cfg.get("hands") or {}).get("prompts") or {})}
    neg = np.stack([encode(t) for t in HAND_NEG]).mean(axis=0)
    return {c: (hands_emb @ np.stack([encode(t) for t in ps]).mean(axis=0) - hands_emb @ neg).astype(np.float32)
            for c, ps in prompts.items()}


def otsu(x: np.ndarray, bins: int = 128) -> float:
    """Ngưỡng chia 2 nhóm sao cho phương sai trong nhóm nhỏ nhất (Otsu) — dùng khi chưa đặt ngưỡng tay."""
    x = x[np.isfinite(x)]
    if x.size < 2 or x.max() - x.min() < 1e-9:
        return float(x.mean()) if x.size else 0.0
    hist, edges = np.histogram(x, bins=bins)
    mids = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * mids) / np.maximum(w0, 1)
    m1 = (np.sum(hist * mids) - np.cumsum(hist * mids)) / np.maximum(w1, 1)
    between = w0 * w1 * (m0 - m1) ** 2
    return float(mids[int(np.argmax(between[:-1]))])


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class VisionExtras:
    def __init__(self, cfg: dict, n: int, index, detections: list[list[dict]] | None, clip_encoder=None,
                 count_conf: float = 0.35, layout=None):
        self.cfg, self.n, self.index, self.clip = cfg, n, index, clip_encoder
        oc = cfg.get("objects") or {}
        self.yc = oc.get("yoloe") or {}
        self.tc = oc.get("clip_tags") or {}
        self.hc = cfg.get("hands") or {}
        self.pc = cfg.get("pose") or {}
        self.count_conf = count_conf
        # -------- YOLOE
        self.obj_classes: list[str] = []
        self.obj_counts = np.zeros((n, 0), np.int16)
        self.obj_maxconf = np.zeros((n, 0), np.float32)
        if getattr(index, "objects", None) is not None:
            self.obj_classes = list(index.objects.get("classes", []))
            ci = {c: j for j, c in enumerate(self.obj_classes)}
            cc = self.yc.get("count_conf", 0.30)
            self.obj_counts = np.zeros((n, len(ci)), np.int16)
            self.obj_maxconf = np.zeros((n, len(ci)), np.float32)
            for i, ds in enumerate(index.objects["dets"][:n]):
                for d in ds:
                    j = ci.get(d["c"])
                    if j is None:
                        continue
                    self.obj_maxconf[i, j] = max(self.obj_maxconf[i, j], d["s"])
                    if d["s"] >= cc:
                        self.obj_counts[i, j] += 1
        # -------- PA1: CLIP theo vùng (tính lười khi cần, vì phải mã hoá câu bằng CLIP)
        self.tag_specs: dict[str, dict] = {}
        self.saved_tags: dict = getattr(index, "clip_tags", None) or {}
        if self.tc.get("enabled", False):
            if self.saved_tags:                                     # đã build bước tags
                self.tag_specs = {k: {} for k in self.saved_tags}
            elif getattr(index, "region_emb", None) is not None and clip_encoder is not None:   # tính lúc chạy
                self.tag_specs = clip_tag_specs(cfg, getattr(index, "region_names", None) or [])
        self._tag_cache: dict[str, tuple[np.ndarray, float, float]] = {}
        # -------- pose: (ảnh, người) -> tư thế
        self.posture: dict[tuple[int, int], str] = {}
        self.pose_colors: dict[tuple[int, int], list[str]] = {}
        if getattr(index, "pose", None) is not None:
            from ..vision.pose import posture_of
            kpc = self.pc.get("kp_conf", 0.3)
            seat = tuple(self.pc.get("sitting_zones", ["living", "dining"]))
            for i, people in enumerate(index.pose["persons"][:n]):
                for k, p in people.items():
                    zs = None
                    if layout is not None and detections is not None and int(k) < len(detections[i]):
                        zs = layout.zones_of_box(detections[i][int(k)]["b"])
                    # tính lại lúc chạy (có vùng trong bếp) -> chỉnh luật không cần build lại
                    self.posture[(i, int(k))] = posture_of(p["kp"], kpc, zones=zs, sitting_zones=seat)
                    if p.get("col"):
                        self.pose_colors[(i, int(k))] = p["col"]
        # -------- hands
        self.h_img = np.zeros(0, np.int64)
        self.h_link: dict[str, np.ndarray] = {}
        self._hand_cache: dict[str, np.ndarray] = {}
        if getattr(index, "hands_emb", None) is not None and len(index.hands_emb):
            self.h_img = index.hands_ref[:, 0].astype(np.int64)
            self._build_hand_links()

    # ------------------------------------------------------------------ thông tin chung
    @property
    def classes(self) -> list[str]:
        """Lớp đồ vật mới mà parser nên nhận ra (từ YOLOE và/hoặc PA1)."""
        return list(dict.fromkeys(self.obj_classes + list(self.tag_specs)))

    def sources(self) -> list[str]:
        out = []
        if self.tag_specs:
            out.append("clip_tags")
        if self.obj_classes:
            out.append("yoloe")
        if self.posture:
            out.append("pose")
        if len(self.h_img):
            out.append("hands")
        return out

    def counts(self, cls: str) -> np.ndarray | None:
        if cls in self.obj_classes:
            return self.obj_counts[:, self.obj_classes.index(cls)]
        return None

    # ------------------------------------------------------------------ có / không
    def p_yoloe(self, cls: str) -> np.ndarray | None:
        if cls not in self.obj_classes:
            return None
        lo, hi = self.yc.get("p_low", 0.15), self.yc.get("p_high", 0.5)
        return np.clip((self.obj_maxconf[:, self.obj_classes.index(cls)] - lo) / max(1e-6, hi - lo), 0, 1)

    def clip_margin(self, cls: str):
        """[PA1] -> (chênh lệch [N], ngưỡng, độ dốc) hoặc None."""
        if cls not in self.tag_specs:
            return None
        if cls not in self._tag_cache:
            if cls in self.saved_tags:
                m = self.saved_tags[cls].astype(np.float32).copy()
            else:
                m = clip_margins(self.index.region_emb, self.index.region_names, {cls: self.tag_specs[cls]},
                                 self.clip.encode_query)[cls]
            ok = np.isfinite(m)
            thr = (self.tc.get("thresholds") or {}).get(cls)
            thr = float(thr) if thr is not None else otsu(m[ok])
            slope = float(np.std(m[ok]) / 3) if ok.any() else 1.0
            m[~ok] = np.nan
            self._tag_cache[cls] = (m, thr, max(slope, 1e-4))
        return self._tag_cache[cls]

    def p_clip(self, cls: str) -> np.ndarray | None:
        r = self.clip_margin(cls)
        if r is None:
            return None
        m, thr, slope = r
        return np.nan_to_num(_sigmoid((m - thr) / slope), nan=0.0).astype(np.float32)

    def sources_for(self, cls: str) -> list[str]:
        """Nguồn dùng cho câu có/không của từng đồ vật (objects.sources trong config). Mặc định chọn theo kết quả đo
        trên 5000 ảnh (AUC so với mô tả gốc): CLIP vùng đoán nồi còn kém hơn ngẫu nhiên (0.43) nên nồi/chảo/cốc chỉ
        dùng YOLOE; dao, thớt, đĩa thì CLIP vùng ngang hoặc tốt hơn YOLOE nên dùng cả hai."""
        src = (self.cfg.get("objects") or {}).get("sources") or DEFAULT_SOURCES
        return list(src.get(cls, ["yoloe", "clip"]))

    def presence(self, cls: str, coco_counts: np.ndarray | None = None) -> tuple[np.ndarray | None, list[str]]:
        """Điểm có mặt 0..1 = trung bình các nguồn được phép cho đồ vật này. coco_counts: số đếm YOLO COCO."""
        parts, used = [], []
        allow = self.sources_for(cls)
        py = self.p_yoloe(cls) if "yoloe" in allow else None
        if "yoloe" in allow and coco_counts is not None and np.any(coco_counts > 0):
            py = np.maximum(py if py is not None else 0, (coco_counts > 0).astype(np.float32))
        if py is not None:
            parts.append(py); used.append("YOLOE")
        pc = self.p_clip(cls) if "clip" in allow else None
        if pc is not None:
            parts.append(pc); used.append("CLIP vùng")
        if not parts:
            return None, used
        return (np.mean(parts, axis=0)).astype(np.float32), used

    # ------------------------------------------------------------------ tư thế
    def posture_score(self, posture: str, need: int, person_rows: list[tuple[int, int]]) -> np.ndarray:
        """Điểm ảnh: đủ `need` người đúng tư thế = 1 · có ít hơn = 0.5 · không ai đúng nhưng có người không rõ
        tư thế (bị che) = 0.3 · còn lại 0. person_rows: (ảnh, chỉ số detection) của mọi người."""
        n = self.n
        hit = np.zeros(n, np.int32)
        unk = np.zeros(n, bool)
        for i, k in person_rows:
            p = self.posture.get((i, k), "unknown")
            if p == posture:
                hit[i] += 1
            elif p == "unknown":
                unk[i] = True
        s = np.where(hit >= max(1, need), 1.0, np.where(hit >= 1, 0.5, np.where(unk, 0.3, 0.0)))
        return s.astype(np.float32)

    def posture_zero(self, posture: str, person_rows) -> np.ndarray:
        hit = np.zeros(self.n, np.int32)
        for i, k in person_rows:
            if self.posture.get((i, k)) == posture:
                hit[i] += 1
        return (hit == 0).astype(np.float32)

    # ------------------------------------------------------------------ cầm gì
    def _build_hand_links(self) -> None:
        """Khung YOLOE nằm ngay chỗ bàn tay (tâm khung trong ô bàn tay nới 1.5 lần) -> liên kết = độ tin cậy."""
        if self.index.objects is None:
            return
        box = self.index.hands_box
        cx, cy = (box[:, 0] + box[:, 2]) / 2, (box[:, 1] + box[:, 3]) / 2
        hw, hh = (box[:, 2] - box[:, 0]) * 0.75, (box[:, 3] - box[:, 1]) * 0.75
        dets = self.index.objects["dets"]
        lo = self.yc.get("hand_link_conf", 0.2)
        for cls in self.obj_classes:
            link = np.zeros(len(box), np.float32)
            for h, i in enumerate(self.h_img):
                for d in dets[i] if i < len(dets) else []:
                    if d["c"] != cls or d["s"] < lo:
                        continue
                    ox, oy = (d["b"][0] + d["b"][2]) / 2, (d["b"][1] + d["b"][3]) / 2
                    if abs(ox - cx[h]) <= hw[h] and abs(oy - cy[h]) <= hh[h]:
                        link[h] = max(link[h], min(1.0, d["s"] / 0.5))
            self.h_link[cls] = link

    def hand_clip(self, cls: str) -> np.ndarray | None:
        """CLIP: bàn tay giống 'cầm <vật>' hơn 'tay không' -> xếp hạng, top_frac bàn tay đầu được điểm 0..1."""
        if not len(self.h_img):
            return None
        prompts = {**DEFAULT_HAND_PROMPTS, **(self.hc.get("prompts") or {})}.get(cls)
        saved = (getattr(self.index, "hands_tags", None) or {}).get(cls)
        if not prompts and saved is None:
            return None
        if cls not in self._hand_cache:
            if self.clip is not None and prompts:
                e = self.index.hands_emb
                pos = np.stack([self.clip.encode_query(t) for t in prompts]).mean(axis=0)
                neg = np.stack([self.clip.encode_query(t) for t in HAND_NEG]).mean(axis=0)
                m = e @ pos - e @ neg
            elif saved is not None:
                m = saved.astype(np.float32)
            else:
                return None
            pct = np.argsort(np.argsort(m)) / max(1, len(m) - 1)          # hạng phần trăm 0..1
            top = float(self.hc.get("top_frac", 0.12))
            self._hand_cache[cls] = np.clip((pct - (1 - top)) / top, 0, 1).astype(np.float32)
        return self._hand_cache[cls]

    def holding_score(self, cls: str) -> tuple[np.ndarray | None, list[str]]:
        """Điểm ảnh 0..1: bàn tay có khả năng cầm `cls` cao nhất trong ảnh."""
        if not len(self.h_img):
            return None, []
        parts, used = [], []
        if cls in self.h_link:
            parts.append(self.h_link[cls]); used.append("YOLOE ở bàn tay")
        hc = self.hand_clip(cls)
        if hc is not None:
            parts.append(hc); used.append("CLIP bàn tay")
        if not parts:
            return None, used
        per_hand = np.max(parts, axis=0)
        img = np.zeros(self.n, np.float32)
        np.maximum.at(img, self.h_img, per_hand)
        return img, used

    def hands_of(self, idx: int) -> list[dict]:
        if not len(self.h_img):
            return []
        sel = np.flatnonzero(self.h_img == idx)
        ref, box = self.index.hands_ref, self.index.hands_box
        return [{"det": int(ref[h, 1]), "side": int(ref[h, 2]), "box": [float(v) for v in box[h]],
                 "holding": {c: round(float(v[h]), 2) for c, v in self.h_link.items() if v[h] > 0}} for h in sel]


__all__ = ["VisionExtras", "POSTURES", "DEFAULT_CLIP_TAGS", "DEFAULT_HAND_PROMPTS", "otsu", "clip_tag_specs",
           "clip_margins", "hand_margins", "tags_name"]
