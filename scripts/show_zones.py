"""Vẽ các vùng (scene.zones) + vùng cắt CLIP (scene.regions) lên 1 ảnh để kiểm tra / hiệu chỉnh toạ độ.

    python scripts/show_zones.py                          # ảnh đầu tiên có trong bộ, kèm điểm chân mọi người
    python scripts/show_zones.py day1_Kitchen_10_1504     # ảnh theo imageID
    python scripts/show_zones.py --no-feet -o zones.jpg

Điểm chân (chấm xanh) lấy từ detections.json: nếu đám chấm của 1 khu (vd trước bếp nấu) lệch khỏi khung,
sửa toạ độ trong config.yaml rồi chạy lại. Không cần build lại index — vùng được tính lúc nạp engine.
"""
import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from retrieval.config import load_config  # noqa: E402
from retrieval.data.catalog import load_catalog  # noqa: E402
from retrieval.scene.layout import foot_point, load_layout  # noqa: E402

PALETTE = [(255, 80, 80), (80, 160, 255), (255, 200, 0), (0, 220, 120), (220, 100, 255), (255, 140, 0),
           (0, 220, 220), (255, 255, 255)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image_id", nargs="?")
    ap.add_argument("-o", "--out", default="zones_preview.jpg")
    ap.add_argument("--no-feet", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    cat = load_catalog(cfg)
    layout = load_layout(cfg)
    df = cat.df
    if a.image_id:
        path = df.at[cat.index_of(a.image_id), "image_path"]
    else:
        path = next(p for p in df["image_path"] if isinstance(p, str))
    im = Image.open(path).convert("RGB")
    W, H = im.size
    over = Image.new("RGBA", im.size, (0, 0, 0, 0))
    dots = Image.new("RGBA", im.size, (0, 0, 0, 0))
    dr, dd = ImageDraw.Draw(over), ImageDraw.Draw(dots)
    det_file = Path(cfg["paths"]["index_dir"]) / "detections.json"
    if not a.no_feet and det_file.exists():
        dets = json.loads(det_file.read_text(encoding="utf-8"))["dets"]
        cc = cfg.get("vision", {}).get("count_conf", 0.35)
        for ds in dets[::2]:
            for d in ds:
                if d["c"] == "person" and d["s"] >= cc:
                    x, y = foot_point(d["b"])
                    dd.ellipse([x * W - 2, y * H - 2, x * W + 2, y * H + 2], fill=(0, 255, 0, 170))
    for k, z in enumerate(layout.zones if layout else []):
        col = PALETTE[k % len(PALETTE)]
        for x1, y1, x2, y2 in z.rects:
            dr.rectangle([x1 * W, y1 * H, x2 * W, y2 * H], outline=col + (255,), width=4, fill=col + (40,))
            dr.rectangle([x1 * W + 4, y1 * H + 4, x1 * W + 12 + 7 * len(z.name), y1 * H + 20], fill=(0, 0, 0, 200))
            dr.text((x1 * W + 8, y1 * H + 6), z.name, fill=col + (255,))
    for name, r in (cfg.get("scene", {}).get("regions") or {}).items():
        x1, y1, x2, y2 = r
        for k in range(0, int((x2 - x1) * W), 16):        # viền nét đứt
            dr.line([(x1 * W + k, y1 * H), (x1 * W + k + 8, y1 * H)], fill=(255, 255, 255, 220), width=2)
            dr.line([(x1 * W + k, y2 * H), (x1 * W + k + 8, y2 * H)], fill=(255, 255, 255, 220), width=2)
        dr.text((x1 * W + 4, y2 * H - 14), f"region:{name}", fill=(255, 255, 255, 255))
    Image.alpha_composite(Image.alpha_composite(im.convert("RGBA"), over), dots).convert("RGB").save(a.out, quality=90)
    print("Đã lưu", a.out)


if __name__ == "__main__":
    main()
