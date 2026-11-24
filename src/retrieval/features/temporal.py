"""[Tính năng mở rộng] Tìm chuỗi ảnh theo thứ tự sự kiện.

Ví dụ: "một người cắt dứa rồi mọi người ngồi ăn ở bàn" -> tìm cặp (ảnh A, ảnh B) sao cho A khớp sự kiện 1,
B khớp sự kiện 2 và B xảy ra SAU A (trong khoảng max_gap_seconds).

Thuật toán: lấy top-K ứng viên mỗi sự kiện (dùng chung engine.score) rồi quy hoạch động theo thời gian,
O(S·K²) — với K=100 chỉ vài ms.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from ..search.engine import SearchEngine

SPLIT_RE = re.compile(r"\s*(?:->|→|=>|;|\brồi sau đó\b|\bsau đó\b|\btiếp theo là\b|\btiếp theo\b|\btiếp đó\b"
                      r"|\brồi\b|\bthen\b|\bafter that\b|\bfollowed by\b|\bafterwards\b)\s*", re.I)


def split_events(query: str) -> list[str]:
    parts = [p.strip(" ,.") for p in SPLIT_RE.split(query)]
    return [p for p in parts if p]


@dataclass
class SequenceResult:
    score: float
    steps: list[dict]   # mỗi bước: imageID, time, score, record


def search_sequence(engine: SearchEngine, events: list[str] | str, top_n: int = 10) -> list[SequenceResult]:
    if isinstance(events, str):
        events = split_events(events)
    if len(events) < 2:
        raise ValueError("Cần ít nhất 2 sự kiện (ngăn cách bằng 'rồi', 'sau đó', '->', 'then'...).")
    tc = engine.cfg.get("temporal", {})
    K = tc.get("candidates_per_event", 100)
    gmin, gmax = tc.get("min_gap_seconds", 1), tc.get("max_gap_seconds", 3600)
    ts = engine.timestamp

    cands, scores = [], []
    for ev in events:
        pq = engine.parser.parse(ev)
        final, mask, _, _ = engine.score(pq)
        idx = np.flatnonzero(mask)
        idx = idx[np.argsort(-final[idx])][:K]
        cands.append(idx)
        scores.append(final[idx])

    # best[s][j] = điểm tốt nhất của chuỗi kết thúc tại ứng viên j của bước s
    best = [scores[0].astype(np.float64)]
    back = [np.full(len(cands[0]), -1)]
    for s in range(1, len(events)):
        prev_t, cur_t = ts[cands[s - 1]], ts[cands[s]]
        gap = cur_t[:, None] - prev_t[None, :]                      # [K_cur, K_prev]
        valid = (gap >= gmin) & (gap <= gmax) & np.isfinite(best[-1])[None, :]
        cand_score = np.where(valid, best[-1][None, :], -np.inf)
        j_prev = cand_score.argmax(axis=1)
        b = cand_score[np.arange(len(cur_t)), j_prev] + scores[s]
        best.append(b)
        back.append(np.where(np.isfinite(b), j_prev, -1))

    last = best[-1]
    order = [j for j in np.argsort(-last) if np.isfinite(last[j])][:top_n]
    out = []
    for j in order:
        chain, s, jj = [], len(events) - 1, j
        while s >= 0 and jj >= 0:
            i = int(cands[s][jj])
            rec = engine.record(i)
            chain.append({"event": events[s], "imageID": rec["imageID"], "time": rec["time"],
                          "score": round(float(scores[s][jj]), 4), "record": rec})
            jj = int(back[s][jj]); s -= 1
        out.append(SequenceResult(float(last[j]) / len(events), chain[::-1]))
    return out
