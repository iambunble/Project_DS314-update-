"""Người đã biết (data/people.yaml) — vd thầy Linh. Xem src/retrieval/vision/people.py.

    python scripts/people.py build           # huấn luyện + chấm mọi người -> artifacts/people_sim.npz (lần đầu ~5 phút)
    python scripts/people.py sheet linh      # 100 người CHƯA gán nhãn mà máy tin nhất -> artifacts/people/linh_top.png

Muốn nhận đúng hơn (học chủ động): mở ảnh sheet, ô nào đúng là người đó thì chép dòng khung tương ứng trong
artifacts/people/linh_top.txt vào pos, ô nào nhầm thì vào neg, rồi build lại. Vài vòng là đủ.
Cần CLIP thân trên (python scripts/outfits.py build) và IMAGE_ROOT/ảnh nếu có photos hoặc muốn xem sheet.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np  # noqa: E402

from retrieval.config import PROJECT_ROOT, load_config  # noqa: E402
from retrieval.vision import people as pp  # noqa: E402
from retrieval.vision.outfits import UPPER_FILE, _unit, upper_box  # noqa: E402


def _load(cfg):
    out = Path(cfg["paths"]["index_dir"])
    ids = json.loads((out / "manifest.json").read_text(encoding="utf-8"))["ids"]
    dets = json.loads((out / "detections.json").read_text(encoding="utf-8"))["dets"]
    ref = np.load(out / "person_ref.npy")
    if not (out / UPPER_FILE).exists():
        sys.exit("Chưa có CLIP thân trên -> chạy trước: python scripts/outfits.py build")
    return out, ids, dets, ref, _unit(np.load(out / "person_emb.npy")), _unit(np.load(out / UPPER_FILE))


def _photo_features(cfg, photos) -> np.ndarray:
    """Ảnh chụp riêng -> đặc trưng như người trong bộ ảnh (cả ảnh = khung người)."""
    from PIL import Image
    from retrieval.encoders.factory import make_clip_encoder
    enc = make_clip_encoder(cfg)
    ims = [Image.open(f).convert("RGB") for f in photos]
    up = upper_box([0, 0, 1, 1])
    ups = [im.crop((int(up[0] * im.width), int(up[1] * im.height), int(up[2] * im.width), int(up[3] * im.height)))
           for im in ims]
    col = np.stack([pp.body_color(np.asarray(im), [0, 0, 1, 1], up) for im in ims])
    return pp.features(enc.encode_pil(ims), enc.encode_pil(ups), col)


def build(cfg, force_colors: bool = False):
    out, ids, dets, ref, F, U = _load(cfg)
    people = pp.load_people({**cfg, "people": {**(cfg.get("people") or {}), "enabled": True}})
    if not people:
        sys.exit("data/people.yaml chưa có ai")
    cf = out / pp.COLOR_FILE
    if force_colors or not cf.exists() or len(np.load(cf, mmap_mode="r")) != len(ref):
        from retrieval.data.catalog import load_catalog
        paths = list(load_catalog(cfg).df["image_path"])
        if not any(paths):
            sys.exit("Không thấy ảnh -> đặt IMAGE_ROOT hoặc sửa paths.image_root trong config.yaml")
        pose = json.loads((out / "pose.json").read_text(encoding="utf-8"))["persons"]             if (out / "pose.json").exists() else None
        print(f"Tính màu áo / quần cho {len(ref)} người (~5 phút)...")
        np.save(cf, pp.build_colors(paths, dets, pose, ref))
    X = pp.features(F, U, np.load(cf))
    pc = cfg.get("people") or {}
    lo, hi = float(pc.get("lo", 0.5)), float(pc.get("hi", 0.9))
    res = {}
    for name, p in people.items():
        pos, miss_p = pp.resolve_refs(p.get("pos") or [], ids, dets, ref)
        neg, miss_n = pp.resolve_refs(p.get("neg") or [], ids, dets, ref)
        neg = [x for x in neg if x not in set(pos)]
        photos = [PROJECT_ROOT / f if not Path(f).is_absolute() else Path(f) for f in (p.get("photos") or [])]
        photos = [f for f in photos if f.exists()]
        extra = _photo_features(cfg, photos) if photos else None
        if len(pos) < 5 or len(neg) < 5:
            print(f"{name}: cần ít nhất 5 khung pos và 5 khung neg (đang có {len(pos)}/{len(neg)}) -> bỏ qua")
            continue
        prob, cv = pp.classify(X, pos, neg, extra)
        res[name] = prob.astype(np.float16)
        print(f"{name}: {len(pos)} pos, {len(neg)} neg, {len(photos)} ảnh riêng"
              + (f" (không tìm thấy {miss_p + miss_n} khung — YOLO đã đổi?)" if miss_p + miss_n else "")
              + (f" | kiểm tra chéo: đúng {cv:.0%}" if cv is not None else "")
              + f" | chắc chắn (≥{hi}): {(prob >= hi).sum()} người, có thể (>{lo}): {(prob > lo).sum()}")
    np.savez(out / pp.SIM_FILE, **res)
    print("->", out / pp.SIM_FILE)


def sheet(cfg, name: str, n: int = 100):
    from PIL import Image, ImageDraw
    from retrieval.data.catalog import load_catalog
    out, ids, dets, ref, _, _ = _load(cfg)
    with np.load(out / pp.SIM_FILE) as z:
        if name not in z.files:
            sys.exit(f"Chưa có '{name}' -> python scripts/people.py build")
        s = z[name].astype(np.float32)
    paths = list(load_catalog(cfg).df["image_path"])
    p = pp.load_people({**cfg, "people": {**(cfg.get("people") or {}), "enabled": True}}).get(name) or {}
    known = set(pp.resolve_refs((p.get("pos") or []) + (p.get("neg") or []), ids, dets, ref)[0])
    order = [q for q in np.argsort(-s) if q not in known][:n]      # chỉ người chưa gán nhãn
    TW, TH, per = 90, 150, 10
    im2 = Image.new("RGB", (per * TW, ((n + per - 1) // per) * TH), "white")
    d = ImageDraw.Draw(im2)
    lines = []
    for j, p in enumerate(order):
        i, k = ref[p]
        b = dets[i][k]["b"]
        lines.append(f'{j}: "{ids[i]}:{",".join(f"{v:.3f}" for v in b)}"  điểm={s[p]:.3f}')
        if not paths[i]:
            continue
        with Image.open(paths[i]) as im:
            W, H = im.size
            cr = im.convert("RGB").crop((int(b[0] * W), int(b[1] * H), int(b[2] * W), int(b[3] * H)))
        cr.thumbnail((TW - 4, TH - 18))
        im2.paste(cr, ((j % per) * TW, (j // per) * TH + 16))
        d.text(((j % per) * TW + 2, (j // per) * TH + 2), f"{j}: {s[p]:.3f}", fill="black")
    od = out / "people"
    od.mkdir(exist_ok=True)
    im2.save(od / f"{name}_top.png")
    (od / f"{name}_top.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("->", od / f"{name}_top.png", "và", od / f"{name}_top.txt")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["build", "sheet"])
    ap.add_argument("name", nargs="?")
    ap.add_argument("-n", type=int, default=100)
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.cmd == "build":
        build(cfg)
    else:
        if not a.name:
            sys.exit("Cách dùng: python scripts/people.py sheet linh")
        sheet(cfg, a.name, a.n)


if __name__ == "__main__":
    main()
