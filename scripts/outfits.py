"""Sổ tay trang phục (gom người theo bộ đồ + ghi nhãn). Xem src/retrieval/vision/outfits.py.

    python scripts/outfits.py build            # CLIP thân trên + gom nhóm + ảnh mẫu artifacts/outfits/sheet_*.png
    python scripts/outfits.py build --force    # tính lại CLIP thân trên (khi đã build lại bước crops)
    python scripts/outfits.py show             # liệt kê nhãn hiện có + số người mỗi nhãn
    python scripts/outfits.py try "người mặc áo len sọc ngang"   # câu này khớp nhãn nào

Sau khi build: mở ảnh mẫu, ghi mô tả bộ đồ cho từng nhóm vào data/outfits.yaml ("" = nhóm lẫn, bỏ qua).
Lưu ý: build lại có thể chia nhóm khác đi -> nhãn cũ không còn đúng (app sẽ cảnh báo và tắt sổ tay).
Cần IMAGE_ROOT (hoặc paths.image_root đúng) để đọc ảnh.
"""
from __future__ import annotations

import argparse
import json
import logging
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
from retrieval.vision import outfits as ob  # noqa: E402


def _labels_path(cfg) -> Path:
    p = Path((cfg.get("outfits") or {}).get("labels", "data/outfits.yaml"))
    return p if p.is_absolute() else PROJECT_ROOT / p


def build(cfg, force: bool) -> None:
    from retrieval.data.catalog import load_catalog
    from retrieval.encoders.factory import make_clip_encoder
    out = Path(cfg["paths"]["index_dir"])
    k = int((cfg.get("outfits") or {}).get("k", 60))
    ref = np.load(out / "person_ref.npy")
    dets = json.loads((out / "detections.json").read_text(encoding="utf-8"))["dets"]
    paths = list(load_catalog(cfg).df["image_path"])
    if not any(paths):
        sys.exit("Không thấy ảnh -> đặt IMAGE_ROOT hoặc sửa paths.image_root trong config.yaml")
    up_f = out / ob.UPPER_FILE
    if force or not up_f.exists() or len(np.load(up_f, mmap_mode="r")) != len(ref):
        pose = None
        if (out / "pose.json").exists():
            pose = json.loads((out / "pose.json").read_text(encoding="utf-8"))["persons"]
        enc = make_clip_encoder(cfg)
        logging.info("CLIP thân trên cho %d người...", len(ref))
        np.save(up_f, ob.build_upper(paths, dets, pose, ref, enc, cfg["encoders"]["clip"].get("batch_size", 32) * 2))
    ids, conf = ob.cluster(np.load(out / "person_emb.npy"), np.load(up_f), k)
    np.save(out / ob.IDS_FILE, ids)
    np.save(out / ob.CONF_FILE, conf)
    sheets = ob.render_sheets(out / ob.SHEET_DIR, ids, ref, paths, dets)
    print(f"Đã gom {len(ref)} người thành {k} nhóm. Ảnh mẫu:")
    for f in sheets:
        print("  ", f)
    lab = _labels_path(cfg)
    if not lab.exists():
        ob.write_template(lab, ids, k)
        print(f"Đã tạo {lab} — ghi mô tả bộ đồ cho từng nhóm vào đó.")
    else:
        import yaml
        y = yaml.safe_load(lab.read_text(encoding="utf-8")) or {}
        if list(y.get("fingerprint") or []) == ob.fingerprint(ids, k):
            print(f"Nhóm giống hệt lúc ghi nhãn -> {lab.name} vẫn dùng được.")
        else:
            print(f"!! Nhóm đã khác lúc ghi nhãn -> nhãn trong {lab.name} KHÔNG còn đúng. Xem ảnh mẫu và ghi lại;\n"
                  f"   fingerprint mới: {ob.fingerprint(ids, k)}")


def _book(cfg):
    from retrieval.encoders.factory import make_text_encoder
    out = Path(cfg["paths"]["index_dir"])
    n = len(np.load(out / "person_ref.npy", mmap_mode="r"))
    cfg.setdefault("outfits", {})["enabled"] = True
    enc = make_text_encoder(cfg)
    book, warn = ob.OutfitBook.load(cfg, out, enc, n)
    if book is None:
        sys.exit(warn)
    return book, enc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["build", "show", "try"])
    ap.add_argument("text", nargs="?")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config(a.config)
    if a.cmd == "build":
        build(cfg, a.force)
    elif a.cmd == "show":
        book, _ = _book(cfg)
        cnt = np.bincount(book.person_label[book.person_label >= 0], minlength=len(book.label_texts))
        print(f"{(book.person_label >= 0).sum()}/{len(book.person_label)} người có nhãn, {len(book.label_texts)} bộ đồ:")
        for t, c in sorted(zip(book.label_texts, cnt), key=lambda x: -x[1]):
            print(f"  {c:5d}  {t}")
    else:
        if not a.text:
            sys.exit('Cách dùng: python scripts/outfits.py try "người mặc áo len sọc"')
        book, enc = _book(cfg)
        for t, s in book.best(enc.encode_query(a.text), 5):
            print(f"  {s:4.2f}  {t}")


if __name__ == "__main__":
    main()
