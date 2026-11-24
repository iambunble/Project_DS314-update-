"""[Tính năng mở rộng] Tìm ảnh liên quan tới 1 ảnh, và tìm 1 NGƯỜI ở các ảnh khác.

Ảnh–ảnh: vì camera cố định, CLIP thô của 2 ảnh bất kỳ đã giống nhau ~0.85 (cùng nền bếp). Nên:
  - clip   : CLIP đã trừ "ảnh trung bình" (nền) -> so phần khác biệt
  - region : CLIP các vùng cắt (đảo bếp, bếp nấu...) cũng trừ trung bình theo từng vùng
  - people : với mỗi người trong ảnh gốc, tìm người giống nhất trong ảnh kia (cùng nhóm người?)
  - layout : số người ở từng vùng giống nhau không (cùng "cảnh": 2 người ở bếp + 1 người ở bàn ăn...)
  - text   : mô tả giống nhau
Có thể kèm câu điều kiện ("buổi chiều", "có 2 người") — phần lọc dùng lại parser của engine.

Người–người (similar_person): so CLIP ảnh cắt của người đó với mọi người khác -> nhận ra cùng 1 người (áo,
dáng) ở thời điểm khác. Không phải nhận diện khuôn mặt, chỉ dựa vào ngoại hình trong khung hình.
"""
from __future__ import annotations

import numpy as np

from ..encoders.base import l2norm
from ..search.engine import SearchEngine, _minmax


def _centered(x: np.ndarray, ok: np.ndarray) -> np.ndarray:
    c = x - x[ok].mean(axis=0, keepdims=True)
    return c / np.clip(np.linalg.norm(c, axis=-1, keepdims=True), 1e-6, None)


def _per_image_max(engine: SearchEngine, person_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """điểm từng người [P] -> (điểm ảnh = người cao nhất [N] (nan nếu không có người), chỉ số người đó [N])."""
    n = len(engine.catalog)
    best = np.full(n, -np.inf, np.float32)
    np.maximum.at(best, engine.person_img, person_scores)
    arg = np.full(n, -1, np.int64)
    hit = person_scores >= best[engine.person_img] - 1e-7
    arg[engine.person_img[hit]] = np.flatnonzero(hit)
    best[~np.isfinite(best)] = np.nan
    return best, arg


def similar_to_id(engine: SearchEngine, image_id: str, top_k: int = 20, condition: str = "",
                  exclude_near_seconds: int = 0, weights: dict | None = None) -> list[dict]:
    i = engine.catalog.index_of(image_id)
    w = {"clip": 0.35, "region": 0.15, "people": 0.20, "layout": 0.10, "text": 0.20, **(weights or {})}
    idx = engine.index
    parts, ws = [], []
    cc = engine.clip_centered()
    if cc is not None and (idx.clip_mask is None or idx.clip_mask[i]):
        parts.append(cc @ cc[i]); ws.append(w["clip"])
    if idx.region_emb is not None and engine.has_image[i]:
        rc = _centered(idx.region_emb, engine.has_image)            # [N, R, d]
        parts.append(np.einsum("nrd,rd->n", rc, rc[i]) / rc.shape[1]); ws.append(w["region"])
    if idx.person_emb is not None and len(engine.person_img):
        mine = np.flatnonzero(engine.person_img == i)
        if len(mine):
            acc = np.zeros(len(engine.catalog), np.float32)
            for p in mine:
                best, _ = _per_image_max(engine, idx.person_emb @ idx.person_emb[p])
                acc += np.nan_to_num(best, nan=0.0)
            parts.append(acc / len(mine)); ws.append(w["people"])
    if len(engine.zone_names):
        z = engine.zone_matrix.astype(np.float32)
        z = np.c_[z, engine._counts("person")[:, None].astype(np.float32) * 0.5]
        zn = z / np.clip(np.linalg.norm(z, axis=1, keepdims=True), 1e-6, None)
        lay = zn @ zn[i] if z[i].any() else (z.sum(1) == 0).astype(np.float32)   # ảnh trống ~ ảnh trống
        parts.append(lay); ws.append(w["layout"])
    if idx.text_emb is not None:
        parts.append(idx.text_emb @ idx.text_emb[i]); ws.append(w["text"])
    return _rank(engine, parts, ws, top_k, condition, exclude={i}, ref=i,
                 exclude_near_seconds=exclude_near_seconds)


def similar_to_image(engine: SearchEngine, image_path: str, top_k: int = 20, condition: str = "") -> list[dict]:
    if engine.index.clip_emb is None or engine.clip_encoder is None:
        raise RuntimeError("Cần build CLIP embedding (có ảnh) để tìm bằng ảnh tải lên.")
    q = l2norm(engine.clip_encoder.encode_images([image_path]))[0]
    return _rank(engine, [engine.index.clip_emb @ q], [1.0], top_k, condition)


def similar_person(engine: SearchEngine, pid: int, top_k: int = 20, condition: str = "",
                   exclude_near_seconds: int = 60, color_weight: float = 0.08) -> list[dict]:
    """Tìm các ảnh có người giống người số `pid` (chỉ số trong person_emb). Mỗi kết quả kèm `match` =
    người khớp nhất trong ảnh đó (khung, vùng) để giao diện khoanh lại."""
    idx = engine.index
    if idx.person_emb is None:
        raise RuntimeError("Chưa build CLIP từng người. Chạy: python scripts/build_index.py --stages crops")
    i_ref, k_ref = (int(v) for v in idx.person_ref[pid])
    best, arg = _per_image_max(engine, idx.person_emb @ idx.person_emb[pid])
    ref_col = (engine.index.detections[i_ref][k_ref].get("col") or [None])[0]
    if ref_col and color_weight:
        same = np.zeros(len(engine.catalog), np.float32)
        for j in np.flatnonzero(arg >= 0):
            ii, kk = (int(v) for v in idx.person_ref[arg[j]])
            col = (engine.index.detections[ii][kk].get("col") or [None])[0]
            same[j] = 1.0 if col == ref_col else 0.0
        best = best + color_weight * same
    res = _rank(engine, [best], [1.0], top_k, condition, exclude={i_ref}, ref=i_ref,
                exclude_near_seconds=exclude_near_seconds, require=~np.isnan(best))
    for r in res:
        j = engine.catalog.index_of(r["imageID"])
        p = int(arg[j])
        if p >= 0:
            ii, kk = (int(v) for v in idx.person_ref[p])
            r["match"] = {"pid": p, "box": engine.index.detections[ii][kk]["b"]}
    return res


def _rank(engine, parts, weights, top_k, condition, exclude=(), ref=None, exclude_near_seconds=0, require=None):
    pq = engine.parser.parse(condition) if condition else None
    mask = engine.filter_mask(pq) if pq else np.ones(len(engine.catalog), bool)
    if require is not None:
        mask &= require
    for e in exclude:
        mask[e] = False
    if ref is not None and exclude_near_seconds:
        mask &= np.abs(engine.timestamp - engine.timestamp[ref]) > exclude_near_seconds  # bỏ khung hình liền kề
    score = sum(w * _minmax(p, mask) for p, w in zip(parts, weights)) / max(sum(weights), 1e-9)
    if pq and pq.objects:
        score = score + engine._object_bonus(pq)
    if pq and pq.colors:
        score = score + engine._color_bonus(pq)
    if pq and pq.zones:
        score = score + engine._zone_bonus(pq)
    cand = np.flatnonzero(mask)
    order = cand[np.argsort(-score[cand])][:top_k]
    return [{"imageID": engine.catalog.df.at[j, "imageID"], "score": round(float(score[j]), 4),
             "record": engine.record(int(j))} for j in order]
