"""Câu hỏi theo thời gian (gộp nhiều ảnh, không phải 1 ảnh): "mấy lần", "bao lâu", "lần đầu / lần cuối",
"lúc nào đông nhất", "sau khi X thì ...".

Ảnh chụp dày (trung bình ~9 giây / ảnh, khoảng cách lớn nhất ~1.5 phút) nên 1 sự việc kéo dài = 1 dãy ảnh liền
nhau khớp câu hỏi. Gộp các ảnh khớp liền nhau thành 1 "lần" (đoạn), rồi đếm số đoạn / cộng độ dài / lấy đoạn sớm
nhất. Các hàm ở đây không phụ thuộc model -> test được bằng số giả.
"""
from __future__ import annotations

import re

import numpy as np

# "mấy lần", "bao nhiêu lần", "số lần"
TIMES = re.compile(r"(?<!\w)(?:(?:bao\s+nhiêu|mấy)\s+lần|số\s+lần)(?!\w)")
# "bao lâu", "mấy phút", "bao nhiêu phút / tiếng", "kéo dài"
DURATION = re.compile(r"(?<!\w)(?:(?:trong\s+)?bao\s+lâu|(?:bao\s+nhiêu|mấy)\s+(?:phút|tiếng)|kéo\s+dài|"
                      r"thời\s+gian\s+bao\s+nhiêu)(?!\w)")
# "lúc nào đông người nhất", "giờ nào vắng nhất", "khi nào nhiều người nhất"
EXTREME = re.compile(r"(?<!\w)(?:(?P<max>đông(?:\s+đúc)?|nhiều\s+người|đông\s+người)|"
                     r"(?P<min>vắng(?:\s+vẻ)?|ít\s+người|vắng\s+người))\s+(?:người\s+)?nhất(?!\w)")
FIRST = re.compile(r"(?<!\w)(?:lần\s+đầu(?:\s+tiên)?|đầu\s+tiên|sớm\s+nhất)(?!\w)")
LAST = re.compile(r"(?<!\w)(?:lần\s+cuối(?:\s+cùng)?|cuối\s+cùng|muộn\s+nhất|trễ\s+nhất)(?!\w)")
# bếp không có ai (dùng cho "bếp trống bao lâu")
EMPTY = re.compile(r"(?<!\w)(?:trống(?:\s+người)?|không\s+(?:có\s+)?(?:ai|người)|vắng\s+tanh|không\s+một\s+ai)(?!\w)")
# "sau khi X thì Y", "trước khi X, Y", "Y sau khi X"
_REL_HEAD = re.compile(r"^\s*(?P<rel>sau|trước)\s+khi\s+(?P<x>.+?)(?:\s*,\s*|\s+thì\s+)(?P<y>.+)$", re.IGNORECASE)
_REL_TAIL = re.compile(r"^(?P<y>.+?)\s+(?P<rel>sau|trước)\s+khi\s+(?P<x>.+?)\s*\??\s*$", re.IGNORECASE)
# bỏ khỏi phần sự việc trước khi tìm ảnh
_TIME_WORDS = re.compile(r"(?<!\w)(?:(?:có\s+)?(?:bao\s+nhiêu|mấy)\s+(?:lần|phút|tiếng)|số\s+lần|(?:trong\s+)?bao\s+lâu|"
                         r"kéo\s+dài|thời\s+gian|lần\s+đầu(?:\s+tiên)?|đầu\s+tiên|sớm\s+nhất|lần\s+cuối(?:\s+cùng)?|"
                         r"cuối\s+cùng|muộn\s+nhất|trễ\s+nhất|(?:lúc|vào|khoảng)?\s*mấy\s+giờ|lúc\s+nào|khi\s+nào|"
                         r"giờ\s+nào|thời\s+điểm\s+nào|(?<!\w)lần)(?!\w)|\?", re.IGNORECASE)


def split_relative(question: str) -> tuple[str, str, str] | None:
    """"Sau khi cắt dứa thì mọi người làm gì?" -> ("sau", "cắt dứa", "mọi người làm gì?"). Không có -> None."""
    m = _REL_HEAD.match(question.strip()) or _REL_TAIL.match(question.strip())
    if not m:
        return None
    return m.group("rel").lower(), m.group("x").strip(" ,.?"), m.group("y").strip(" ,.")


def strip_time_words(text: str) -> str:
    return re.sub(r"\s+", " ", _TIME_WORDS.sub(" ", text)).strip(" ,.")


def episodes(t: np.ndarray, hit: np.ndarray, max_miss: int = 1, max_gap: float = 120) -> list[tuple[int, int]]:
    """t: thời điểm (giây) các ảnh đã sắp tăng dần; hit: ảnh nào khớp. -> [(đầu, cuối)] chỉ số trong t của từng đoạn.
    Trong 1 đoạn cho phép xen tối đa max_miss ảnh không khớp liên tiếp (1 ảnh nhận sai không làm vỡ đoạn), và 2 ảnh
    khớp liền nhau cách nhau không quá max_gap giây."""
    out: list[tuple[int, int]] = []
    start = last = -1
    miss = 0
    for i in range(len(t)):
        if start >= 0 and t[i] - t[last] > max_gap:
            out.append((start, last))
            start = -1
        if hit[i]:
            if start < 0:
                start = i
            last, miss = i, 0
        elif start >= 0:
            miss += 1
            if miss > max_miss:
                out.append((start, last))
                start = -1
    if start >= 0:
        out.append((start, last))
    return out


def episode_seconds(t: np.ndarray, ep: tuple[int, int], step: float) -> float:
    """Độ dài 1 đoạn: từ ảnh đầu đến ảnh cuối + 1 bước chụp (đoạn 1 ảnh vẫn kéo dài ~1 bước)."""
    return float(t[ep[1]] - t[ep[0]] + step)


def fmt_duration(sec: float) -> str:
    m = int(round(sec / 60))
    if sec < 60:
        return f"{int(round(sec))} giây"
    if m < 60:
        return f"{m} phút"
    return f"{m // 60} giờ" + (f" {m % 60} phút" if m % 60 else "")


def hit_threshold(scores: np.ndarray, rel: float) -> float:
    """Ảnh "khớp sự việc": điểm >= trung vị + rel * (cao nhất - trung vị). Điểm của mỗi câu một thang khác nhau
    nên dùng ngưỡng tương đối trong chính câu đó thay vì 1 con số cố định."""
    best, med = float(np.max(scores)), float(np.median(scores))
    return med + rel * (best - med)


def intersect(intervals: list[tuple[int, int]], window: tuple[int, int]) -> list[tuple[int, int]]:
    """Giao các khoảng [a, b) của câu hỏi với cửa sổ thời gian; rỗng -> [(0, 0)] (không ảnh nào thoả)."""
    if not intervals:
        return [window]
    out = [(max(a, window[0]), min(b, window[1])) for a, b in intervals]
    out = [(a, b) for a, b in out if a < b]
    return out or [(0, 0)]
