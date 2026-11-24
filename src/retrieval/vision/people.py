"""Người đã biết (data/people.yaml): đề nhắc tên ("thầy Linh") -> ưu tiên ảnh có đúng người đó.

Cách làm: mỗi người có vài trăm khung "chắc chắn là người này" (pos) và "người khác hay bị nhầm" (neg) gán nhãn
bằng mắt trong bộ ảnh, cộng ảnh chụp riêng (photos). Huấn luyện 1 bộ phân loại nhỏ (hồi quy logistic) trên
CLIP cả người + CLIP thân trên + biểu đồ màu áo / quần -> xác suất "là người này" cho mọi người trong 5000 ảnh.

Vì sao không chỉ so độ giống CLIP: nhìn từ trên cao, CLIP ViT-B/32 lẫn người cùng dáng / cùng chỗ đứng (đo trên
kết quả thật: ~½ ảnh nhận nhầm). Màu áo là dấu hiệu mạnh nhất (áo len xanh dương + quần jean) nhưng một mình màu
thì lẫn với sơ mi xanh / áo thể thao xanh -> kết hợp cả hai, và neg chọn đúng những người hay bị nhầm.
Gán nhãn theo kiểu học chủ động: build -> sheet (xem 100 người máy tin nhất) -> ô sai thêm vào neg -> build lại.

  build : python scripts/people.py build  -> artifacts/person_color.npy  biểu đồ màu áo/quần [P, 48]
                                             artifacts/people_sim.npz    {tên: xác suất [P] theo thứ tự person_ref}
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

SIM_FILE, COLOR_FILE = "people_sim.npz", "person_color.npy"


# ------------------------------------------------------------------ đặc trưng màu
def color_hist(rgb: np.ndarray) -> np.ndarray:
    """Biểu đồ màu [24] của 1 vùng ảnh: 12 sắc độ (điểm có màu) + 4 mức xám + 4 mức bão hoà + 4 mức sáng (điểm có
    màu). Chia theo tổng số điểm nên vùng ít màu (áo xám / đen) có phần sắc độ nhỏ."""
    if rgb.size == 0:
        return np.zeros(24, np.float32)
    a = rgb.reshape(-1, 3).astype(np.float32) / 255
    mx, mn = a.max(1), a.min(1)
    d = mx - mn
    r, g, b = a[:, 0], a[:, 1], a[:, 2]
    h = np.zeros_like(mx)
    nz = d > 1e-6
    rm = nz & (mx == r)
    gm = nz & (mx == g) & ~rm
    bm = nz & ~rm & ~gm
    h[rm] = ((g - b)[rm] / d[rm]) % 6
    h[gm] = (b - r)[gm] / d[gm] + 2
    h[bm] = (r - g)[bm] / d[bm] + 4
    h *= 60
    s = np.where(mx > 1e-6, d / np.maximum(mx, 1e-6), 0)
    chrom = (s > 0.25) & (mx > 0.12)
    out = np.r_[np.bincount((h[chrom] / 30).astype(int) % 12, minlength=12),
                np.bincount(np.minimum((mx[~chrom] * 4).astype(int), 3), minlength=4),
                np.bincount(np.minimum((s[chrom] * 4).astype(int), 3), minlength=4),
                np.bincount(np.minimum((mx[chrom] * 4).astype(int), 3), minlength=4)].astype(np.float32)
    return out / max(1, len(a))


def body_color(im: np.ndarray, box, upper) -> np.ndarray:
    """[48] = màu áo (vai–hông) + màu quần (hông–chân), chỉ lấy dải giữa thân (bỏ tay, nền hai bên)."""
    H, W = im.shape[:2]
    x1, x2 = box[0] + 0.25 * (box[2] - box[0]), box[2] - 0.25 * (box[2] - box[0])
    cx = slice(int(x1 * W), int(x2 * W))
    up = im[int(max(0, upper[1]) * H):int(min(1, upper[3]) * H), cx]
    lo = im[int(min(1, upper[3]) * H):int(box[3] * H), cx]
    return np.r_[color_hist(up[::2, ::2]), color_hist(lo[::2, ::2])]


def build_colors(paths, dets, pose_persons, person_ref: np.ndarray) -> np.ndarray:
    from PIL import Image
    from .outfits import upper_box
    out = np.zeros((len(person_ref), 48), np.float32)
    cur, im = None, None
    for p, (i, k) in enumerate(person_ref):
        if not paths[i]:
            continue
        if cur != i:
            with Image.open(paths[i]) as f:
                im = np.asarray(f.convert("RGB"))
            cur = i
        rec = (pose_persons[i] or {}).get(str(k)) if pose_persons else None
        b = dets[i][k]["b"]
        out[p] = body_color(im, b, upper_box(b, rec["kp"] if rec else None))
    return out


def features(F: np.ndarray, U: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Đặc trưng cho bộ phân loại: CLIP cả người, CLIP thân trên (đã chuẩn hoá), √biểu đồ màu (×3 cho cùng cỡ)."""
    return np.hstack([_unit(F), _unit(U), 3 * np.sqrt(np.maximum(C, 0))]).astype(np.float32)


def classify(X: np.ndarray, pos: list[int], neg: list[int], extra_pos: np.ndarray | None = None,
             C: float = 3.0) -> tuple[np.ndarray, float | None]:
    """Hồi quy logistic pos vs neg -> (xác suất [P], độ chính xác kiểm tra chéo 5 phần hoặc None nếu ít nhãn).
    Khung đã gán nhãn giữ đúng nhãn (pos = 1, neg = 0)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    Xt = X[pos + neg]
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    if extra_pos is not None and len(extra_pos):
        Xt, y = np.vstack([Xt, extra_pos]), np.r_[y, np.ones(len(extra_pos))]
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000)
    cv = None
    if min(int(y.sum()), int(len(y) - y.sum())) >= 10:
        cv = float(cross_val_score(clf, Xt, y, cv=5).mean())
    prob = clf.fit(Xt, y).predict_proba(X)[:, 1].astype(np.float32)
    prob[pos], prob[neg] = 1.0, 0.0
    return prob, cv


def load_people(cfg: dict) -> dict:
    import yaml
    from ..config import PROJECT_ROOT
    pc = cfg.get("people") or {}
    if not pc.get("enabled", False):
        return {}
    f = Path(pc.get("file", "data/people.yaml"))
    f = f if f.is_absolute() else PROJECT_ROOT / f
    if not f.exists():
        return {}
    return (yaml.safe_load(f.read_text(encoding="utf-8")) or {}).get("people") or {}


def parse_ref(s: str) -> tuple[str, list[float]]:
    iid, box = s.rsplit(":", 1)
    return iid.strip(), [float(v) for v in box.split(",")]


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def resolve_refs(refs: list[str], ids: list[str], dets, person_ref: np.ndarray, min_iou: float = 0.5):
    """'imageID:x1,y1,x2,y2' -> chỉ số người (hàng person_ref) có khung trùng nhất. -> (chỉ số, số không tìm thấy)"""
    pos = {x: i for i, x in enumerate(ids)}
    rows: dict[int, list[tuple[int, int]]] = {}
    for p, (i, k) in enumerate(person_ref):
        rows.setdefault(int(i), []).append((p, int(k)))
    out, missing = [], 0
    for s in refs:
        iid, box = parse_ref(s)
        i = pos.get(iid)
        best = max(((_iou(box, dets[i][k]["b"]), p) for p, k in rows.get(i, [])), default=(0, None)) \
            if i is not None else (0, None)
        if best[1] is not None and best[0] >= min_iou:
            out.append(best[1])
        else:
            missing += 1
    return out, missing


def _unit(a):
    a = np.asarray(a, np.float32)
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-6)


def alias_pattern(names: list[str]) -> re.Pattern:
    alt = "|".join(sorted((re.escape(n.lower()) for n in names), key=len, reverse=True))
    return re.compile(r"(?<!\w)(?:" + alt + r")(?!\w)")


class PeopleBook:
    """Điểm người đã biết theo từng người trong ảnh + quy đổi sang điểm thưởng [0, 1] cho từng ảnh."""

    def __init__(self, people: dict, sims: dict[str, np.ndarray], lo: float = 0.5, hi: float = 0.9):
        self.people, self.sims, self.lo, self.hi = people, sims, lo, hi

    @classmethod
    def load(cls, cfg: dict, index_dir: Path, n_persons: int) -> tuple["PeopleBook | None", str | None]:
        pc = cfg.get("people") or {}
        people = load_people(cfg)
        if not people:
            return None, None
        f = index_dir / SIM_FILE
        if not f.exists():
            return None, "Chưa build người đã biết (data/people.yaml) -> chạy: python scripts/people.py build"
        with np.load(f) as z:
            sims = {k: z[k].astype(np.float32) for k in z.files if k in people}
        if any(len(v) != n_persons for v in sims.values()):
            return None, "people_sim.npz không khớp số người hiện tại -> chạy lại: python scripts/people.py build"
        missing = sorted(set(people) - set(sims))
        warn = (f"Người chưa build: {', '.join(missing)} -> chạy: python scripts/people.py build" if missing else None)
        return cls(people, sims, float(pc.get("lo", 0.5)), float(pc.get("hi", 0.9))), warn

    def person_bonus(self, name: str) -> np.ndarray:
        """[P] trong [0,1] từ xác suất: ≤ lo -> 0 (không chắc), ≥ hi -> 1 (gần như chắc chắn)."""
        return np.clip((self.sims[name] - self.lo) / max(self.hi - self.lo, 1e-6), 0, 1)
