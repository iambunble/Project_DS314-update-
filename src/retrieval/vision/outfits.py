"""Sổ tay trang phục: gom mọi người trong 5000 ảnh thành nhóm theo bộ đồ, mỗi nhóm có 1 mô tả do người ghi.

Vì sao: CLIP ViT-B/32 nhìn ảnh cắt từng người vẫn yếu chi tiết quần áo (lẫn sọc / trơn, áo len / sơ mi).
Nhưng video chỉ có ~15–20 người, mỗi người mặc nguyên 1 bộ cả ngày -> gom nhóm theo ngoại hình rồi ghi tên
bộ đồ 1 lần cho cả nhóm (data/outfits.yaml). Khi tìm, câu tả quần áo được so với các mô tả này bằng model chữ
e5 (chính xác hơn nhiều so với để CLIP nhìn ảnh); khung bị che / quay lưng vẫn mang nhãn của nhóm.

  build  : python scripts/outfits.py build   -> artifacts/person_upper_emb.npy  CLIP phần thân trên (vai–hông)
                                                artifacts/outfit_ids.npy        nhóm của từng người [P]
                                                artifacts/outfit_conf.npy       độ gần tâm nhóm [P]
                                                artifacts/outfits/sheet_*.png   ảnh mẫu để ghi nhãn
  nhãn   : data/outfits.yaml  (nhóm -> mô tả; "" = nhóm lẫn, bỏ qua)
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

UPPER_FILE, IDS_FILE, CONF_FILE, SHEET_DIR = "person_upper_emb.npy", "outfit_ids.npy", "outfit_conf.npy", "outfits"


# ------------------------------------------------------------------ build
def upper_box(box, kp=None, kp_conf: float = 0.3) -> list[float]:
    """Khung thân trên (toạ độ 0..1): từ trên vai tới dưới hông theo khung xương; không có thì theo tỉ lệ khung."""
    x1, y1, x2, y2 = box
    h = y2 - y1
    out = [x1, y1 + 0.12 * h, x2, y1 + 0.58 * h]
    if kp is not None:
        kp = np.asarray(kp)
        ok = kp[:, 2] > kp_conf
        if ok[5] and ok[6] and (ok[11] or ok[12]):
            sy = kp[[5, 6], 1].mean()
            hy = kp[[i for i in (11, 12) if ok[i]], 1].mean()
            th = hy - sy
            if th > 0.02:
                out = [x1, sy - 0.12 * th, x2, hy + 0.10 * th]
    return out


def build_upper(paths: list[str | None], dets: list[list[dict]], pose_persons: list[dict] | None,
                person_ref: np.ndarray, encoder, batch: int = 64) -> np.ndarray:
    """CLIP ảnh cắt thân trên của từng người, cùng thứ tự person_ref. Người không cắt được -> vector 0."""
    from PIL import Image
    out = np.zeros((len(person_ref), 0), np.float16)
    buf, slot = [], []

    def flush():
        nonlocal out
        if not buf:
            return
        v = encoder.encode_pil(buf)
        if out.shape[1] == 0:
            out = np.zeros((len(person_ref), v.shape[1]), np.float16)
        out[slot] = v
        buf.clear(); slot.clear()

    t0, cur, im = time.time(), None, None
    for p, (i, k) in enumerate(person_ref):
        if not paths[i]:
            continue
        if cur != i:
            if im is not None:
                im.close()
            im, cur = Image.open(paths[i]).convert("RGB"), i
        rec = (pose_persons[i] or {}).get(str(k)) if pose_persons else None
        x1, y1, x2, y2 = upper_box(dets[i][k]["b"], rec["kp"] if rec else None)
        W, H = im.size
        b = (int(max(0, x1) * W), int(max(0, y1) * H), int(min(1, x2) * W), int(min(1, y2) * H))
        if b[2] - b[0] >= 4 and b[3] - b[1] >= 4:
            buf.append(im.crop(b)); slot.append(p)
        if len(buf) >= batch:
            flush()
        if p % 2000 == 0 and p:
            el = time.time() - t0
            log.info("outfits: %d/%d người (%.0fs, còn ~%.0fs)", p, len(person_ref), el, el / p * (len(person_ref) - p))
    flush()
    return out


def _unit(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32)
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-6)


def cluster(person_emb: np.ndarray, upper_emb: np.ndarray, k: int = 60, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """KMeans trên [CLIP cả người, CLIP thân trên] -> (nhóm [P], cosine tới tâm nhóm [P])."""
    from sklearn.cluster import KMeans
    X = np.hstack([_unit(person_emb), _unit(upper_emb)])
    km = KMeans(k, n_init=4, random_state=seed).fit(X)
    ids = km.labels_.astype(np.int16)
    cen = _unit(km.cluster_centers_)
    conf = np.einsum("pd,pd->p", X / np.sqrt(2), cen[ids])      # X/√2 và tâm đều có độ dài 1
    return ids, conf.astype(np.float16)


def render_sheets(out_dir: Path, ids: np.ndarray, person_ref: np.ndarray, paths, dets, per: int = 10,
                  rows: int = 12, min_area: float = 0.025, seed: int = 1) -> list[Path]:
    """Ảnh mẫu mỗi nhóm (1 hàng / nhóm) để người xem rồi ghi mô tả vào data/outfits.yaml."""
    import random
    from PIL import Image, ImageDraw
    out_dir.mkdir(parents=True, exist_ok=True)
    rng, TW, TH = random.Random(seed), 90, 150
    k = int(ids.max()) + 1
    files = []
    for g in range(0, k, rows):
        cs = range(g, min(k, g + rows))
        sheet = Image.new("RGB", (per * TW + 50, len(cs) * TH), "white")
        d = ImageDraw.Draw(sheet)
        for r, c in enumerate(cs):
            mem = np.where(ids == c)[0]
            area = {m: (lambda b: (b[2] - b[0]) * (b[3] - b[1]))(dets[person_ref[m][0]][person_ref[m][1]]["b"])
                    for m in mem}
            pool = [m for m in mem if area[m] > min_area] or list(mem)
            d.text((2, r * TH + 5), f"c{c}\n{len(mem)}", fill="black")
            for j, m in enumerate(rng.sample(pool, min(per, len(pool)))):
                i, kk = person_ref[m]
                if not paths[i]:
                    continue
                with Image.open(paths[i]) as im:
                    im = im.convert("RGB")
                    W, H = im.size
                    x1, y1, x2, y2 = dets[i][kk]["b"]
                    cr = im.crop((int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H)))
                cr.thumbnail((TW - 4, TH - 4))
                sheet.paste(cr, (50 + j * TW, r * TH + 2))
        f = out_dir / f"sheet_{g // rows}.png"
        sheet.save(f)
        files.append(f)
    return files


def fingerprint(ids: np.ndarray, k: int) -> list[int]:
    return np.bincount(ids.astype(np.int64), minlength=k).tolist()


def write_template(path: Path, ids: np.ndarray, k: int) -> None:
    lines = ["# Sổ tay trang phục — xem artifacts/outfits/sheet_*.png rồi ghi mô tả bộ đồ cho từng nhóm.",
             '# "" = nhóm lẫn nhiều bộ đồ (bỏ qua). Không sửa fingerprint.',
             f"k: {k}", f"fingerprint: {fingerprint(ids, k)}", "labels:"]
    lines += [f'  {c}: ""' for c in range(k)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ------------------------------------------------------------------ tìm kiếm
class OutfitBook:
    """Nhãn bộ đồ cho từng người + chấm điểm câu tả quần áo.

    person_scores(q): q = vector e5 của câu -> điểm [P] của từng người. Điểm e5 giữa câu và các mô tả rất sát
    nhau (0.85–0.90), nên đổi sang z-score trên các mô tả rồi bỏ phần âm: mô tả khớp nhất ~2–3, không khớp 0.
    Người thuộc nhóm lẫn / nằm xa tâm nhóm -> 0 (không cộng, không trừ)."""

    def __init__(self, person_label: np.ndarray, label_texts: list[str], label_emb: np.ndarray):
        self.person_label = person_label          # [P] chỉ số mô tả, -1 = không có nhãn
        self.label_texts = label_texts
        self.label_emb = label_emb                # [L, d]

    @classmethod
    def load(cls, cfg: dict, index_dir: Path, text_encoder, n_persons: int) -> tuple["OutfitBook | None", str | None]:
        """-> (sổ tay, cảnh báo). Thiếu file / không khớp -> (None, lý do)."""
        import yaml
        from ..config import PROJECT_ROOT
        oc = cfg.get("outfits") or {}
        if not oc.get("enabled", False) or text_encoder is None or not n_persons:
            return None, None
        ids_f, conf_f = index_dir / IDS_FILE, index_dir / CONF_FILE
        lab_f = Path(oc.get("labels", "data/outfits.yaml"))
        lab_f = lab_f if lab_f.is_absolute() else PROJECT_ROOT / lab_f
        if not ids_f.exists() or not lab_f.exists():
            return None, "Chưa có sổ tay trang phục -> chạy: python scripts/outfits.py build"
        ids = np.load(ids_f).astype(np.int64)
        if len(ids) != n_persons:
            return None, "Sổ tay trang phục không khớp số người hiện tại -> chạy lại: python scripts/outfits.py build"
        y = yaml.safe_load(lab_f.read_text(encoding="utf-8")) or {}
        k = int(y.get("k", ids.max() + 1))
        if y.get("fingerprint") and list(y["fingerprint"]) != fingerprint(ids, k):
            return None, (f"Nhóm trong {IDS_FILE} khác lúc ghi nhãn ({lab_f.name}) -> xem lại ảnh mẫu "
                          "artifacts/outfits/ và ghi nhãn lại")
        labels = {int(c): str(t).strip() for c, t in (y.get("labels") or {}).items() if t and str(t).strip()}
        if not labels:
            return None, f"{lab_f.name} chưa có nhãn nào"
        texts = sorted(set(labels.values()))
        tid = {t: j for j, t in enumerate(texts)}
        cmap = np.full(k, -1, np.int64)
        for c, t in labels.items():
            if 0 <= c < k:
                cmap[c] = tid[t]
        person_label = cmap[ids]
        if conf_f.exists():                       # người nằm xa tâm nhóm (dễ bị gom nhầm) -> bỏ nhãn
            conf = np.load(conf_f).astype(np.float32)
            pct = float(oc.get("drop_far_pct", 15))
            for c in range(k):
                sel = ids == c
                if sel.any() and pct > 0:
                    person_label[sel & (conf < np.percentile(conf[sel], pct))] = -1
        emb = np.asarray(text_encoder.encode_passages(texts), np.float32)
        return cls(person_label, texts, emb), None

    def label_scores(self, q: np.ndarray) -> np.ndarray:
        s = self.label_emb @ np.asarray(q, np.float32).ravel()
        return np.maximum((s - s.mean()) / (s.std() + 1e-6), 0)

    def person_scores(self, q: np.ndarray) -> np.ndarray:
        v = np.append(self.label_scores(q), 0.0)  # chỉ số -1 -> 0
        return v[self.person_label].astype(np.float32)

    def best(self, q: np.ndarray, top: int = 3) -> list[tuple[str, float]]:
        s = self.label_scores(q)
        return [(self.label_texts[j], float(s[j])) for j in np.argsort(-s)[:top]]
