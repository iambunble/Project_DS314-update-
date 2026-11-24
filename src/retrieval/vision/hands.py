"""[PA2] Vùng BÀN TAY: cắt ô nhỏ quanh mỗi cổ tay (từ pose.json) rồi cho CLIP đọc -> biết người đó đang cầm gì.

Vì sao: dao, điện thoại... chỉ vài chục điểm ảnh nên CLIP toàn ảnh hay ảnh cắt cả người đều không "thấy". Cắt sát
bàn tay rồi phóng lên 224×224 thì vật cầm trên tay chiếm phần lớn khung.

Lúc TÌM: so embedding bàn tay với câu "a hand holding a knife" và "an empty hand" (zero-shot, không cần huấn
luyện) + kiểm tra có khung YOLOE của vật đó ngay chỗ bàn tay không (xem search/vision_extras.py).

artifacts/hands_emb.npy [H, d] float16 · hands_ref.npy [H, 3] = (chỉ số ảnh, chỉ số detection người, bên 0/1)
           hands_box.npy [H, 4] = ô bàn tay tỉ lệ 0..1
"""
from __future__ import annotations

import logging
import time

import numpy as np

from .pose import wrists

log = logging.getLogger(__name__)


def hands_name(cfg: dict, clip_name: str, pose_model: str) -> str:
    hc = cfg.get("hands") or {}
    return f"{clip_name}|{pose_model}|size={hc.get('size', 0.22)}|kp={(cfg.get('pose') or {}).get('kp_conf', 0.3)}"


def hand_boxes(pose_persons: list[dict], dets: list[list[dict]], kp_conf: float = 0.3, size: float = 0.22):
    """-> list (ảnh, người, bên, ô) cho mọi bàn tay nhìn thấy."""
    out = []
    for i, people in enumerate(pose_persons):
        for k, p in people.items():
            box = dets[i][int(k)]["b"] if dets and int(k) < len(dets[i]) else None
            for side, hb in wrists(p["kp"], kp_conf, size, box):
                if hb[2] - hb[0] > 1e-3 and hb[3] - hb[1] > 1e-3:
                    out.append((i, int(k), side, hb))
    return out


def build_hands(paths: list[str | None], pose_persons: list[dict], dets: list[list[dict]], encoder,
                kp_conf: float = 0.3, size: float = 0.22, batch: int = 64):
    from PIL import Image
    items = hand_boxes(pose_persons, dets, kp_conf, size)
    by_img: dict[int, list[int]] = {}
    for h, (i, _, _, _) in enumerate(items):
        by_img.setdefault(i, []).append(h)
    vecs: list[np.ndarray | None] = [None] * len(items)
    buf, slots = [], []
    t0 = time.time()

    def flush():
        if not buf:
            return
        for h, v in zip(slots, encoder.encode_pil(buf)):
            vecs[h] = v.astype(np.float16)
        buf.clear(); slots.clear()

    imgs = sorted(by_img)
    for done, i in enumerate(imgs, 1):
        if not paths[i]:
            continue
        with Image.open(paths[i]) as im:
            im = im.convert("RGB")
            W, H = im.size
            for h in by_img[i]:
                x1, y1, x2, y2 = items[h][3]
                buf.append(im.crop((int(x1 * W), int(y1 * H), max(int(x1 * W) + 2, int(x2 * W)),
                                    max(int(y1 * H) + 2, int(y2 * H)))))
                slots.append(h)
        if len(buf) >= batch:
            flush()
        if done % 250 == 0 or done == len(imgs):
            el = time.time() - t0
            log.info("hands: %d/%d ảnh, %d bàn tay (%.0fs, còn ~%.0fs)", done, len(imgs), len(items), el,
                     el / done * (len(imgs) - done))
    flush()
    keep = [h for h, v in enumerate(vecs) if v is not None]
    d = vecs[keep[0]].shape[0] if keep else 512
    emb = np.stack([vecs[h] for h in keep]) if keep else np.zeros((0, d), np.float16)
    ref = np.array([items[h][:3] for h in keep], np.int32).reshape(-1, 3)
    box = np.array([items[h][3] for h in keep], np.float32).reshape(-1, 4)
    return emb, ref, box
