"""Nhận màu áo của một người từ vùng thân trên trong khung YOLO.

Cách làm: cắt phần giữa thân (bỏ đầu, tay, nền hai bên), đổi sang HSV, gán mỗi điểm ảnh vào 1 trong
11 màu cơ bản rồi lấy màu chiếm nhiều nhất. Nhanh (~1 ms/người), không cần model.
"""
from __future__ import annotations

import numpy as np

COLORS = ["black", "white", "gray", "red", "orange", "yellow", "green", "blue", "purple", "pink", "brown"]

# vùng thân áo, tính theo tỉ lệ khung người (x1, y1, x2, y2)
TORSO = (0.28, 0.20, 0.72, 0.52)


def _rgb_to_hsv(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb = rgb.astype(np.float32) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx, mn = rgb.max(-1), rgb.min(-1)
    d = mx - mn
    h = np.zeros_like(mx)
    nz = d > 1e-6
    rm = nz & (mx == r)
    gm = nz & (mx == g) & ~rm
    bm = nz & ~rm & ~gm
    h[rm] = ((g - b)[rm] / d[rm]) % 6
    h[gm] = (b - r)[gm] / d[gm] + 2
    h[bm] = (r - g)[bm] / d[bm] + 4
    h = h * 60.0
    s = np.where(mx > 1e-6, d / np.maximum(mx, 1e-6), 0)
    return h, s, mx


def classify_pixels(rgb: np.ndarray, dark_blue: bool = False) -> np.ndarray:
    """rgb [..., 3] uint8 -> chỉ số màu trong COLORS.
    dark_blue=True: điểm ảnh tối nhưng vẫn rõ sắc xanh (áo navy, xanh than) được tính là xanh dương thay vì đen —
    bản cũ gán đen hết mọi điểm có độ sáng < 0.22 nên áo navy thành "black" (dùng cho màu áo theo pose)."""
    h, s, v = _rgb_to_hsv(rgb)
    out = np.full(h.shape, COLORS.index("gray"), np.int8)
    black = v < 0.22
    if dark_blue:
        navy = black & (v >= 0.09) & (s >= 0.45) & (h >= 195) & (h < 255)
        black &= ~navy
    achrom = ~black & (s < 0.18)
    out[achrom & (v > 0.72)] = COLORS.index("white")
    out[achrom & (v <= 0.72)] = COLORS.index("gray")
    chrom = ~black & ~achrom
    # màu nhạt, gần trắng/xám
    pale = chrom & (s < 0.28) & (v > 0.6)
    out[pale] = COLORS.index("white")
    chrom &= ~pale
    hue_bins = [(0, 12, "red"), (12, 38, "orange"), (38, 68, "yellow"), (68, 165, "green"),
                (165, 255, "blue"), (255, 290, "purple"), (290, 340, "pink"), (340, 361, "red")]
    for lo, hi, name in hue_bins:
        out[chrom & (h >= lo) & (h < hi)] = COLORS.index(name)
    # cam/đỏ tối -> nâu ; đỏ nhạt sáng -> hồng
    brown = chrom & (h >= 8) & (h < 45) & (v < 0.55)
    out[brown] = COLORS.index("brown")
    pinkish = chrom & ((h < 12) | (h >= 340)) & (s < 0.5) & (v > 0.7)
    out[pinkish] = COLORS.index("pink")
    out[black] = COLORS.index("black")
    return out


def torso_colors(image_rgb: np.ndarray, box_xyxy, min_frac: float = 0.22) -> list[tuple[str, float]]:
    """Trả về [(màu, tỉ lệ)] — màu chính + màu phụ (nếu chiếm >= min_frac). Rỗng nếu vùng quá nhỏ."""
    H, W = image_rgb.shape[:2]
    x1, y1, x2, y2 = box_xyxy
    w, h = x2 - x1, y2 - y1
    tx1, ty1, tx2, ty2 = TORSO
    cx1, cx2 = int(max(0, x1 + tx1 * w)), int(min(W, x1 + tx2 * w))
    cy1, cy2 = int(max(0, y1 + ty1 * h)), int(min(H, y1 + ty2 * h))
    if cx2 - cx1 < 6 or cy2 - cy1 < 6:
        return []
    crop = image_rgb[cy1:cy2, cx1:cx2]
    step = max(1, int(max(crop.shape[:2]) / 48))   # lấy mẫu thưa cho nhanh
    crop = crop[::step, ::step]
    idx = classify_pixels(crop).ravel()
    counts = np.bincount(idx, minlength=len(COLORS)).astype(np.float32)
    frac = counts / max(1.0, counts.sum())
    order = np.argsort(-frac)
    out = [(COLORS[order[0]], float(frac[order[0]]))]
    if frac[order[1]] >= min_frac:
        out.append((COLORS[order[1]], float(frac[order[1]])))
    return out
