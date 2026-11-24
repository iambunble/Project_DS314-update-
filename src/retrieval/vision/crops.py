"""Bước build "crops": CLIP trên ẢNH CẮT thay vì cả khung hình.

Vì sao: camera cố định nên nền bếp giống hệt nhau ở mọi ảnh (cosine CLIP trung bình giữa 2 ảnh bất kỳ ≈ 0.85).
Một người chỉ chiếm ~3% diện tích ảnh, đồ ăn trên đảo bếp còn nhỏ hơn -> CLIP toàn ảnh gần như không "thấy"
áo sọc, người đang cầm dao, nồi trên bếp... Cắt riêng ra rồi mới mã hoá thì chi tiết đó chiếm cả khung 224×224.

  - regions : các vùng cố định trong config (đảo bếp, bếp nấu, phòng khách) -> region_emb.npy [N, R, d]
  - persons : mỗi người YOLO tìm được (nới khung 10%)                      -> person_emb.npy [P, d]
                                                                              person_ref.npy [P, 2] = (ảnh, chỉ số detection)
Mỗi ảnh chỉ đọc từ đĩa 1 lần. CPU: ~10–15 phút cho 5000 ảnh (~12k người + 15k vùng).
"""
from __future__ import annotations

import logging
import time

import numpy as np

log = logging.getLogger(__name__)


def person_boxes(dets: list[dict], count_conf: float, min_area: float) -> list[tuple[int, list[float]]]:
    out = []
    for k, d in enumerate(dets):
        if d["c"] != "person" or d["s"] < count_conf:
            continue
        x1, y1, x2, y2 = d["b"]
        if (x2 - x1) * (y2 - y1) < min_area:
            continue
        out.append((k, d["b"]))
    return out


def pad_box(box, pad: float, W: int, H: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    return (int(max(0, (x1 - pad * w)) * W), int(max(0, (y1 - pad * h)) * H),
            int(min(1, (x2 + pad * w)) * W), int(min(1, (y2 + pad * h)) * H))


def build_crops(paths: list[str | None], dets: list[list[dict]] | None, encoder, regions: dict[str, list[float]],
                count_conf: float = 0.35, min_area: float = 0.004, pad: float = 0.10, batch: int = 64,
                with_persons: bool = True):
    from PIL import Image
    n = len(paths)
    names = list(regions)
    R = len(names)
    region_emb = None
    person_vecs: list[np.ndarray] = []
    person_ref: list[tuple[int, int]] = []

    buf, slots = [], []          # ảnh cắt chờ mã hoá + nơi ghi kết quả: ("r", i, j) hoặc ("p", i, k)

    def flush():
        nonlocal region_emb
        if not buf:
            return
        vecs = encoder.encode_pil(buf)
        if region_emb is None and R:
            region_emb = np.zeros((n, R, vecs.shape[1]), np.float16)
        for (kind, i, j), v in zip(slots, vecs):
            if kind == "r":
                region_emb[i, j] = v
            else:
                person_vecs.append(v.astype(np.float16))
                person_ref.append((i, j))
        buf.clear(); slots.clear()

    todo = [i for i, p in enumerate(paths) if p]
    t0 = time.time()
    for done, i in enumerate(todo, 1):
        with Image.open(paths[i]) as im:
            im = im.convert("RGB")
            W, H = im.size
            for j, nm in enumerate(names):
                x1, y1, x2, y2 = regions[nm]
                buf.append(im.crop((int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H))))
                slots.append(("r", i, j))
            if with_persons and dets is not None:
                for k, b in person_boxes(dets[i], count_conf, min_area):
                    buf.append(im.crop(pad_box(b, pad, W, H)))
                    slots.append(("p", i, k))
        if len(buf) >= batch:
            flush()
        if done % 250 == 0 or done == len(todo):
            el = time.time() - t0
            log.info("crops: %d/%d ảnh, %d người (%.0fs, còn ~%.0fs)", done, len(todo), len(person_ref) + sum(
                s[0] == "p" for s in slots), el, el / done * (len(todo) - done))
    flush()
    d = (region_emb.shape[2] if region_emb is not None else (person_vecs[0].shape[0] if person_vecs else 0))
    pe = np.stack(person_vecs) if person_vecs else np.zeros((0, d), np.float16)
    pr = np.array(person_ref, np.int32).reshape(-1, 2)
    return names, region_emb, pe, pr
