"""[PA2] YOLO-pose: khung xương 17 điểm (vai, khuỷu, cổ tay, hông, gối, cổ chân...) cho từng người.

Dùng để:
  1. MÀU ÁO CHÍNH XÁC HƠN: lấy vùng thân áo theo tứ giác vai–hông thật (thu nhỏ vào giữa để tránh tay, nền)
     thay vì 1 ô cố định 20–52% chiều cao khung người — ô cố định hay trúng đầu/tóc khi camera nhìn từ trên xuống.
  2. TƯ THẾ: đứng / ngồi / cúi theo góc thân và độ dài đùi trên ảnh (luật đơn giản, xem posture_of()).
  3. CỔ TAY: để cắt vùng bàn tay (bước hands) -> biết người đó đang cầm gì.

Người của pose được GHÉP với người trong detections.json theo IoU khung (giữ nguyên chỉ số người cũ để
person_emb, vùng, "tìm người này"... vẫn khớp). Người pose thấy mà YOLO cũ không thấy được lưu riêng ở "extra".

artifacts/pose.json = {"model", "persons": [ {"<chỉ số detection>": {"kp", "posture", "torso", "col", "iou"}} mỗi ảnh ],
                       "extra": [[{"b", "s", "kp"}] mỗi ảnh]}
  kp = 17 × [x, y, conf] tỉ lệ 0..1 · torso = 4 điểm [vai trái, vai phải, hông phải, hông trái] (đã thu nhỏ) hoặc null
"""
from __future__ import annotations

import logging
import math
import time

import numpy as np

from .colors import COLORS, classify_pixels

log = logging.getLogger(__name__)

# chỉ số điểm COCO-17
NOSE, L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN = 0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
POSTURES = ("standing", "sitting", "bending")


def pose_name(cfg: dict) -> str:
    pc = cfg.get("pose") or {}
    from pathlib import Path
    return f"{Path(pc.get('model', 'yolo26s-pose.pt')).name}@{pc.get('imgsz', 640)}|kp={pc.get('kp_conf', 0.3)}|v1"


def _iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _mid(kp, i, j, c):
    ok = [k for k in (i, j) if kp[k][2] >= c]
    if not ok:
        return None
    return (sum(kp[k][0] for k in ok) / len(ok), sum(kp[k][1] for k in ok) / len(ok))


def posture_of(kp, kp_conf: float = 0.3, aspect: float = 16 / 9, zones: list[str] | None = None,
               sitting_zones: tuple[str, ...] = ("living", "dining")) -> str:
    """kp: 17×[x, y, conf] tỉ lệ 0..1. aspect = W/H để đo góc đúng tỉ lệ. zones: các vùng chứa điểm chân người đó.
    Camera nhìn từ trên xuống nên đùi người đứng trông ngắn hơn thực (đo trên 158 người: chiều dọc đùi / chiều dài
    thân trung bình 0.62, người đứng sát camera chỉ ~0.45–0.5) -> luật dè dặt, không chắc thì "unknown":
    - cúi  : thân (giữa hông -> giữa vai) nghiêng > 35° so với phương đứng, hoặc vai thấp hơn/ngang hông
    - ngồi : đùi gần như nằm ngang (< 0.25 thân); hoặc < 0.4 thân / không thấy gối mà đang ở vùng sofa, bàn ăn
    - đứng : thấy gối, đùi >= 0.35 thân và thân gần thẳng (< 30°)
    - unknown: thiếu điểm (bị đảo bếp che, quay lưng...) hoặc không rơi vào luật nào"""
    sh, hip = _mid(kp, L_SH, R_SH, kp_conf), _mid(kp, L_HIP, R_HIP, kp_conf)
    if sh is None or hip is None:
        return "unknown"
    dx, dy = (sh[0] - hip[0]) * aspect, hip[1] - sh[1]     # dy > 0: vai cao hơn hông
    torso = math.hypot(dx, dy)
    if torso < 1e-3:
        return "unknown"
    angle = math.degrees(math.atan2(abs(dx), dy)) if dy > 0 else 90.0
    if angle > 35:
        return "bending"
    in_seat = bool(zones) and any(z in sitting_zones for z in zones) and "foreground" not in (zones or [])
    knee = _mid(kp, L_KN, R_KN, kp_conf)
    if knee is None:
        return "sitting" if in_seat else "unknown"
    thigh = (knee[1] - hip[1]) / torso
    if thigh < 0.25 or (thigh < 0.4 and in_seat):
        return "sitting"
    if thigh >= 0.35 and angle < 30:
        return "standing"
    return "unknown"


def torso_quad(kp, kp_conf: float = 0.3, shrink: float = 0.2):
    """Tứ giác vai trái – vai phải – hông phải – hông trái, thu nhỏ `shrink` về tâm. Thiếu hông thì ước lượng
    hông = vai + 1.3 × độ rộng vai theo chiều dọc (người bị đảo bếp che nửa dưới)."""
    if kp[L_SH][2] < kp_conf or kp[R_SH][2] < kp_conf:
        return None
    ls, rs = np.array(kp[L_SH][:2]), np.array(kp[R_SH][:2])
    if kp[L_HIP][2] >= kp_conf and kp[R_HIP][2] >= kp_conf:
        lh, rh = np.array(kp[L_HIP][:2]), np.array(kp[R_HIP][:2])
    else:
        w = float(np.linalg.norm(ls - rs))
        if w < 1e-3:
            return None
        down = np.array([0.0, 1.3 * w * 16 / 9])        # độ rộng vai theo x -> chiều dọc theo tỉ lệ ảnh
        lh, rh = ls + down * 0.75, rs + down * 0.75      # chỉ lấy phần ngực (an toàn khi bị che)
    q = np.stack([ls, rs, rh, lh])
    c = q.mean(axis=0)
    q = c + (q - c) * (1 - shrink)
    return [[round(float(x), 4), round(float(y), 4)] for x, y in q]


def quad_colors(image_rgb: np.ndarray, quad, min_frac: float = 0.22, min_px: int = 30) -> list[str]:
    """Màu chính (+ màu phụ nếu chiếm >= min_frac) của các điểm ảnh nằm trong tứ giác thân áo."""
    from PIL import Image, ImageDraw
    H, W = image_rgb.shape[:2]
    pts = [(x * W, y * H) for x, y in quad]
    x1, y1 = max(0, int(min(p[0] for p in pts))), max(0, int(min(p[1] for p in pts)))
    x2, y2 = min(W, int(max(p[0] for p in pts)) + 1), min(H, int(max(p[1] for p in pts)) + 1)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return []
    mask = Image.new("L", (x2 - x1, y2 - y1), 0)
    ImageDraw.Draw(mask).polygon([(x - x1, y - y1) for x, y in pts], fill=1)
    m = np.asarray(mask, bool)
    px = image_rgb[y1:y2, x1:x2][m]
    if len(px) < min_px:
        return []
    if len(px) > 4000:                                   # lấy mẫu thưa cho nhanh
        px = px[:: len(px) // 4000 + 1]
    idx = classify_pixels(px[None, :, :], dark_blue=True).ravel()
    frac = np.bincount(idx, minlength=len(COLORS)).astype(np.float32)
    frac /= max(1.0, frac.sum())
    order = np.argsort(-frac)
    out = [COLORS[order[0]]]
    if frac[order[1]] >= min_frac:
        out.append(COLORS[order[1]])
    return out


class PoseEstimator:
    def __init__(self, model: str = "yolo26s-pose.pt", imgsz: int = 640, conf: float = 0.25, kp_conf: float = 0.3,
                 match_iou: float = 0.4, device: str | None = None, batch: int = 8):
        from ultralytics import YOLO
        self.model = YOLO(model)
        self.kw = dict(conf=conf, imgsz=imgsz, verbose=False)
        if device:
            self.kw["device"] = device
        self.kp_conf, self.match_iou, self.batch = kp_conf, match_iou, batch

    def run(self, paths: list[str | None], dets: list[list[dict]], count_conf: float = 0.35):
        persons: list[dict] = [{} for _ in paths]
        extra: list[list[dict]] = [[] for _ in paths]
        todo = [(i, p) for i, p in enumerate(paths) if p]
        t0 = time.time()
        for k in range(0, len(todo), self.batch):
            chunk = todo[k:k + self.batch]
            results = self.model.predict([p for _, p in chunk], **self.kw)
            for (i, _), r in zip(chunk, results):
                persons[i], extra[i] = self._convert(r, dets[i] if dets else [], count_conf)
            done = k + len(chunk)
            if (k // self.batch) % 25 == 0 or done == len(todo):
                el = time.time() - t0
                log.info("pose: %d/%d ảnh (%.0fs, còn ~%.0fs)", done, len(todo), el, el / done * (len(todo) - done))
        return persons, extra

    def _convert(self, r, dets: list[dict], count_conf: float):
        if r.boxes is None or len(r.boxes) == 0:
            return {}, []
        H, W = r.orig_shape
        rgb = r.orig_img[..., ::-1]
        boxes = r.boxes.xyxy.cpu().numpy() / np.array([W, H, W, H])
        confs = r.boxes.conf.cpu().numpy()
        kps = r.keypoints.data.cpu().numpy()                     # [n, 17, 3] pixel
        kps[..., 0] /= W
        kps[..., 1] /= H
        people = [(k, d["b"]) for k, d in enumerate(dets) if d["c"] == "person" and d["s"] >= count_conf]
        # ghép tham lam theo IoU giảm dần
        pairs = sorted(((_iou(b, pb), j, k) for j, b in enumerate(boxes) for k, pb in people), reverse=True)
        used_p, used_k, out = set(), set(), {}
        for iou, j, k in pairs:
            if iou < self.match_iou or j in used_p or k in used_k:
                continue
            used_p.add(j); used_k.add(k)
            out[str(k)] = self._person(kps[j], rgb, iou)
        extra = [{"b": [round(float(v), 4) for v in boxes[j]], "s": round(float(confs[j]), 3),
                  "kp": [[round(float(x), 4), round(float(y), 4), round(float(c), 3)] for x, y, c in kps[j]]}
                 for j in range(len(boxes)) if j not in used_p]
        return out, extra

    def _person(self, kp: np.ndarray, rgb: np.ndarray, iou: float) -> dict:
        kpl = [[round(float(x), 4), round(float(y), 4), round(float(c), 3)] for x, y, c in kp]
        quad = torso_quad(kpl, self.kp_conf)
        H, W = rgb.shape[:2]
        return {"kp": kpl, "posture": posture_of(kpl, self.kp_conf, W / H), "torso": quad,
                "col": quad_colors(rgb, quad) if quad else [], "iou": round(float(iou), 3)}


def wrists(kp, kp_conf: float = 0.3, size: float = 0.22, box=None, aspect: float = 16 / 9):
    """Ô vuông quanh mỗi bàn tay (tỉ lệ 0..1): tâm = cổ tay đẩy thêm 35% theo hướng khuỷu -> cổ tay
    (bàn tay nằm sau cổ tay), cạnh = size × chiều cao khung người (tối thiểu ~40 px ở ảnh 720p).
    -> [(bên: 0 trái / 1 phải, [x1, y1, x2, y2])]"""
    out = []
    bh = (box[3] - box[1]) if box is not None else 0.3
    s_y = max(40 / 720, size * bh)
    s_x = s_y / aspect
    for side, (w, e) in enumerate(((L_WR, L_EL), (R_WR, R_EL))):
        if kp[w][2] < kp_conf:
            continue
        wx, wy = kp[w][0], kp[w][1]
        if kp[e][2] >= kp_conf:
            cx, cy = wx + 0.35 * (wx - kp[e][0]), wy + 0.35 * (wy - kp[e][1])
        else:
            cx, cy = wx, wy
        out.append((side, [max(0.0, cx - s_x / 2), max(0.0, cy - s_y / 2), min(1.0, cx + s_x / 2),
                           min(1.0, cy + s_y / 2)]))
    return out
