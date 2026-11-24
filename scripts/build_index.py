"""Build chỉ mục 1 lần (hoặc khi đổi CSV / ảnh / model).

    python scripts/build_index.py                 # đầy đủ: text + CLIP ảnh + YOLO + CLIP ảnh cắt (người, vùng)
    python scripts/build_index.py --skip-clip     # chỉ mô tả (nhanh, khi chưa có ảnh)
    python scripts/build_index.py --offline       # TF-IDF, không tải model (test nhanh)
    python scripts/build_index.py --stages crops  # chỉ chạy 1 bước (text | clip | detect | crops | objects | pose | hands)
    python scripts/build_index.py --stages objects,pose,hands   # nâng cấp nhận diện đồ vật / tư thế / cầm gì
    python scripts/build_index.py --force         # build lại cả bước đã có

Thứ tự phụ thuộc: crops cần detect (để biết người ở đâu) và dùng cùng model CLIP với bước clip.
pose cần detect (ghép khung xương với người đã có); hands cần pose + CLIP.

Bước nào đã build đúng model trong config sẽ tự bỏ qua.
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retrieval.config import load_config  # noqa: E402
from retrieval.data.catalog import load_catalog  # noqa: E402
from retrieval.index.store import STAGES, build_index  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--skip-clip", action="store_true")
    ap.add_argument("--offline", action="store_true", help="dùng TF-IDF thay cho model embedding")
    ap.add_argument("--stages", default=",".join(STAGES), help="vd: text,clip,detect,crops,objects,pose,hands")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    overrides = ({"encoders": {"text": {"backend": "tfidf"}, "clip": {"enabled": False}}, "vision": {"enabled": False},
                  "objects": {"yoloe": {"enabled": False}}, "pose": {"enabled": False}, "hands": {"enabled": False}}
                 if args.offline else None)
    if args.skip_clip:
        args.stages = ",".join(s for s in args.stages.split(",") if s.strip() not in ("clip", "crops"))
    cfg = load_config(args.config, overrides)
    catalog = load_catalog(cfg)
    has_img = int(catalog.df["image_path"].notna().sum())
    logging.info("Catalog: %d dòng, %d có ảnh", len(catalog), has_img)
    stages = tuple(s.strip() for s in args.stages.split(",") if s.strip())
    info = build_index(cfg, catalog, stages=stages, force=args.force)
    logging.info("Xong. Lưu tại %s | %s", cfg["paths"]["index_dir"], info)


if __name__ == "__main__":
    main()
