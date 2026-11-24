"""Build & lưu chỉ mục theo từng bước (stage). Bước nào đã build đúng model rồi thì tự bỏ qua,
nên đổi/thêm 1 model chỉ phải chạy lại đúng bước đó.

artifacts/
  manifest.json        # thứ tự imageID + model đã dùng cho từng bước
  text_emb.npy         # [N, d]   bước "text":   embedding mô tả (đã bỏ câu nền nếu scene.clean_descriptions)
  sent_emb.npy         # [S, d]   bước "text":   embedding từng câu mô tả (scene.sentence_index)
  sent_doc.npy         # [S]      câu thứ s thuộc ảnh nào
  clip_img_emb.npy     # [N, d]   bước "clip":   embedding ảnh (hàng không có ảnh = 0)
  clip_mask.npy        # [N] bool
  detections.json      #          bước "detect": YOLO + màu áo từng người
  region_emb.npy       # [N, R, d] bước "crops": CLIP các vùng cắt cố định (đảo bếp, bếp nấu, phòng khách)
  person_emb.npy       # [P, d]   bước "crops":  CLIP từng người
  person_ref.npy       # [P, 2]   (chỉ số ảnh, chỉ số detection của người đó)
  tfidf.pkl            # (chỉ khi backend=tfidf)
  --- nâng cấp nhận diện (mỗi phần bật/tắt riêng trong config, file riêng, không ghi đè file cũ) ---
  clip_tags.npz        # [N] mỗi đồ vật  bước "tags": [PA1] CLIP vùng cắt so với câu "có nồi"/"bếp trống" (vài giây)
  objects.json         #          bước "objects": YOLOE — dao, nồi, chảo, thớt, đĩa, cốc, điện thoại (vision/objects.py)
  pose.json            #          bước "pose":    YOLO-pose — khung xương, tư thế, màu áo theo vai–hông (vision/pose.py)
  hands_emb.npy        # [H, d]   bước "hands":   CLIP vùng bàn tay (vision/hands.py)
  hands_ref.npy        # [H, 3]   (ảnh, người, bên trái/phải)
  hands_box.npy        # [H, 4]   ô bàn tay tỉ lệ 0..1
  hands_tags.npz       # [H] mỗi đồ vật  điểm CLIP "tay cầm <vật>" − "tay không" (để kiểm tra / chỉnh ngưỡng)
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..data.catalog import Catalog
from .bm25 import BM25Index

log = logging.getLogger(__name__)
STAGES = ("text", "clip", "detect", "crops", "tags", "objects", "pose", "hands")


@dataclass
class IndexBundle:
    text_emb: np.ndarray | None
    clip_emb: np.ndarray | None
    clip_mask: np.ndarray | None
    detections: list[list[dict]] | None
    bm25: BM25Index
    manifest: dict
    sent_emb: np.ndarray | None = None
    sent_doc: np.ndarray | None = None
    region_emb: np.ndarray | None = None       # [N, R, d]
    region_names: list[str] = field(default_factory=list)
    person_emb: np.ndarray | None = None       # [P, d]
    person_ref: np.ndarray | None = None       # [P, 2]
    warnings: list[str] = field(default_factory=list)
    objects: dict | None = None                # objects.json: {"model", "classes", "dets"}
    pose: dict | None = None                   # pose.json:    {"model", "persons", "extra"}
    hands_emb: np.ndarray | None = None        # [H, d]
    hands_ref: np.ndarray | None = None        # [H, 3]
    hands_box: np.ndarray | None = None        # [H, 4]
    clip_tags: dict | None = None              # clip_tags.npz: đồ vật -> chênh lệch CLIP [N]
    hands_tags: dict | None = None             # hands_tags.npz: đồ vật -> chênh lệch CLIP bàn tay [H]


def detector_name(cfg: dict) -> str:
    v = cfg.get("vision", {})
    return f"{v.get('model', 'yolo11m.pt')}@{v.get('imgsz', 640)}"


def text_variant(cfg: dict) -> str:
    sc = cfg.get("scene") or {}
    clean = sc.get("enabled", True) and sc.get("clean_descriptions", False)
    sent = (sc.get("enabled", True) and sc.get("sentence_index", False)
            and cfg["encoders"]["text"].get("backend") != "tfidf")     # tfidf: ma trận câu quá lớn, bỏ qua
    return ("clean" if clean else "raw") + ("+sent" if sent else "") + "-v1"


def crops_name(cfg: dict) -> str:
    from ..encoders.factory import clip_encoder_name
    sc = cfg.get("scene") or {}
    regions = sc.get("regions") or {}
    reg = ";".join(f"{k}={','.join(f'{v:.3f}' for v in r)}" for k, r in regions.items())
    pers = (f"persons(conf={cfg.get('vision', {}).get('count_conf', 0.35)},area={sc.get('person_min_area', 0.004)},"
            f"pad={sc.get('person_pad', 0.1)})" if sc.get("person_crops", True) else "nopersons")
    return f"{clip_encoder_name(cfg)}|{reg}|{pers}"


def _read_manifest(out: Path, catalog: Catalog) -> dict:
    mf = out / "manifest.json"
    if mf.exists():
        m = json.loads(mf.read_text(encoding="utf-8"))
        if m.get("ids") == catalog.ids:
            return m
        log.warning("Metadata đã đổi so với lần build trước -> build lại tất cả các bước.")
    return {"ids": catalog.ids, "n": len(catalog)}


def _write_manifest(out: Path, m: dict) -> None:
    m["built_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    tmp = out / "manifest.json.tmp"
    tmp.write_text(json.dumps(m, ensure_ascii=False), encoding="utf-8")
    tmp.replace(out / "manifest.json")


def build_index(cfg: dict, catalog: Catalog, stages: tuple[str, ...] = STAGES, force: bool = False) -> dict:
    from ..encoders.factory import clip_encoder_name, make_clip_encoder, make_text_encoder, text_encoder_name
    from ..scene.text_clean import content_sentences, content_text
    out = Path(cfg["paths"]["index_dir"]); out.mkdir(parents=True, exist_ok=True)
    m = _read_manifest(out, catalog)
    sc = cfg.get("scene") or {}
    descs = catalog.df["description"].fillna("").tolist()
    paths = catalog.df["image_path"].tolist()
    has_img = np.array([isinstance(p, str) for p in paths])

    # ---------------------------------------------------------------- text
    if "text" in stages and cfg["encoders"]["text"]["enabled"]:
        want, variant = text_encoder_name(cfg), text_variant(cfg)
        if (not force and m.get("text_model") == want and m.get("text_variant", "raw-v0") == variant
                and (out / "text_emb.npy").exists()):
            log.info("[text] đã có (%s, %s) -> bỏ qua", want, variant)
        else:
            enc = make_text_encoder(cfg)
            clean = variant.startswith("clean")
            passages = [content_text(d) for d in descs] if clean else descs
            enc.fit(passages)
            t = time.time()
            emb = enc.encode_passages(passages).astype(np.float32)
            np.save(out / "text_emb.npy", emb)
            if enc.name == "tfidf":
                enc.save(out / "tfidf.pkl")
            if "+sent" in variant:
                sents, owner = [], []
                for i, d in enumerate(descs):
                    for s in (content_sentences(d) if clean else [x for x in d.split(". ") if x]):
                        sents.append(s); owner.append(i)
                log.info("[text] mã hoá %d câu mô tả...", len(sents))
                np.save(out / "sent_emb.npy", enc.encode_passages(sents).astype(np.float16))
                np.save(out / "sent_doc.npy", np.array(owner, np.int32))
            else:
                for f in ("sent_emb.npy", "sent_doc.npy"):
                    (out / f).unlink(missing_ok=True)
            m["text_model"], m["text_variant"] = enc.name, variant
            _write_manifest(out, m)
            log.info("[text] %s xong trong %.0fs", emb.shape, time.time() - t)

    # ---------------------------------------------------------------- clip
    if "clip" in stages and cfg["encoders"]["clip"]["enabled"]:
        want = clip_encoder_name(cfg)
        if not force and m.get("clip_model") == want and (out / "clip_img_emb.npy").exists():
            log.info("[clip] đã có (%s) -> bỏ qua", want)
        elif not has_img.any():
            log.warning("[clip] không có ảnh nào khớp -> bỏ qua")
        else:
            enc = make_clip_encoder(cfg)
            t = time.time()
            vecs = enc.encode_images([p for p in paths if isinstance(p, str)])
            emb = np.zeros((len(paths), vecs.shape[1]), np.float32)
            emb[has_img] = vecs
            np.save(out / "clip_img_emb.npy", emb)
            np.save(out / "clip_mask.npy", has_img)
            m["clip_model"] = enc.name
            _write_manifest(out, m)
            log.info("[clip] %d ảnh xong trong %.0fs", int(has_img.sum()), time.time() - t)

    # ---------------------------------------------------------------- detect
    vc = cfg.get("vision", {})
    if "detect" in stages and vc.get("enabled", False):
        want = detector_name(cfg)
        if not force and m.get("detector") == want and (out / "detections.json").exists():
            log.info("[detect] đã có (%s) -> bỏ qua", want)
        elif not has_img.any():
            log.warning("[detect] không có ảnh nào khớp -> bỏ qua")
        else:
            from ..vision.detector import YoloDetector
            det = YoloDetector(vc.get("model", "yolo11m.pt"), vc.get("conf", 0.25), vc.get("iou", 0.5),
                               vc.get("imgsz", 640), vc.get("device"), vc.get("batch", 8))
            t = time.time()
            dets = det.run(paths)
            with open(out / "detections.json", "w", encoding="utf-8") as f:
                json.dump({"model": det.name, "dets": dets}, f, separators=(",", ":"))
            m["detector"] = det.name
            m.pop("crops_detector", None)        # người thay đổi -> ảnh cắt người phải làm lại
            _write_manifest(out, m)
            log.info("[detect] %d ảnh xong trong %.0fs", int(has_img.sum()), time.time() - t)

    # ---------------------------------------------------------------- crops (vùng + từng người)
    if "crops" in stages and cfg["encoders"]["clip"]["enabled"] and sc.get("enabled", True):
        regions = sc.get("regions") or {}
        persons = bool(sc.get("person_crops", True))
        det_ok = (out / "detections.json").exists() and m.get("detector")
        want = crops_name(cfg)
        if not regions and not persons:
            log.info("[crops] scene.regions rỗng và person_crops=false -> bỏ qua")
        elif (not force and m.get("crops_model") == want and m.get("crops_detector") == m.get("detector")
              and (out / "person_emb.npy").exists()):
            log.info("[crops] đã có -> bỏ qua")
        elif not has_img.any():
            log.warning("[crops] không có ảnh nào khớp -> bỏ qua")
        else:
            if persons and not det_ok:
                log.warning("[crops] chưa có detections.json -> chỉ cắt vùng, chưa cắt từng người "
                            "(chạy bước detect trước rồi build lại crops)")
            from ..vision.crops import build_crops
            dets = (json.loads((out / "detections.json").read_text(encoding="utf-8"))["dets"]
                    if det_ok else None)
            enc = make_clip_encoder(cfg)
            t = time.time()
            names, reg, pe, pr = build_crops(
                paths, dets, enc, regions, vc.get("count_conf", 0.35), sc.get("person_min_area", 0.004),
                sc.get("person_pad", 0.10), batch=cfg["encoders"]["clip"].get("batch_size", 32) * 2,
                with_persons=persons and bool(det_ok))
            if reg is not None:
                np.save(out / "region_emb.npy", reg)
            else:
                (out / "region_emb.npy").unlink(missing_ok=True)
            np.save(out / "person_emb.npy", pe)
            np.save(out / "person_ref.npy", pr)
            m["crops_model"], m["crops_regions"] = want, names
            m["crops_detector"] = m.get("detector") if det_ok else None
            _write_manifest(out, m)
            log.info("[crops] %d vùng/ảnh, %d người xong trong %.0fs", len(names), len(pe), time.time() - t)

    _build_extras(cfg, out, m, stages, force, paths, has_img)
    _write_manifest(out, m)
    return {k: v for k, v in m.items() if k != "ids"}


def _build_extras(cfg: dict, out: Path, m: dict, stages, force: bool, paths: list, has_img: np.ndarray) -> None:
    """Các bước nâng cấp nhận diện: objects (YOLOE), pose (YOLO-pose), hands (CLIP vùng bàn tay)."""
    vc = cfg.get("vision", {})
    count_conf = vc.get("count_conf", 0.35)

    # ---------------------------------------------------------------- tags ([PA1] CLIP theo vùng, không cần ảnh)
    tc = (cfg.get("objects") or {}).get("clip_tags") or {}
    if "tags" in stages and tc.get("enabled", False) and cfg["encoders"]["clip"]["enabled"]:
        from ..encoders.factory import clip_encoder_name, make_clip_encoder
        from ..search.vision_extras import clip_tag_specs, tags_name
        specs = clip_tag_specs(cfg, m.get("crops_regions") or [])
        want = tags_name(clip_encoder_name(cfg), specs, m.get("crops_model", ""))
        if not specs:
            log.warning("[tags] chưa có region_emb.npy (bước crops) hoặc không có đồ vật nào khớp vùng -> bỏ qua")
        elif not force and m.get("tags_model") == want and (out / "clip_tags.npz").exists():
            log.info("[tags] đã có -> bỏ qua")
        else:
            enc = make_clip_encoder(cfg, load_image_model=False)
            reg = np.load(out / "region_emb.npy").astype(np.float32)
            from ..search.vision_extras import clip_margins
            margins = clip_margins(reg, m.get("crops_regions") or [], specs, enc.encode_query)
            np.savez(out / "clip_tags.npz", **margins)
            m["tags_model"] = want
            _write_manifest(out, m)
            log.info("[tags] %d đồ vật: %s", len(margins), ", ".join(margins))

    def load_dets():
        f = out / "detections.json"
        return json.loads(f.read_text(encoding="utf-8"))["dets"] if f.exists() and m.get("detector") else None

    # ---------------------------------------------------------------- objects (YOLOE)
    oc = (cfg.get("objects") or {}).get("yoloe") or {}
    if "objects" in stages and oc.get("enabled", False):
        from ..vision.objects import DEFAULT_CLASSES, YoloeObjectDetector, objects_name
        want = objects_name(cfg)
        if not force and m.get("objects_model") == want and (out / "objects.json").exists():
            log.info("[objects] đã có (%s) -> bỏ qua", want)
        elif not has_img.any():
            log.warning("[objects] không có ảnh nào khớp -> bỏ qua")
        else:
            classes = oc.get("classes") or DEFAULT_CLASSES
            det = YoloeObjectDetector(_resolve(oc.get("model", "weights/yoloe-26s-seg.pt")), classes,
                                      _resolve(oc.get("text_emb", "weights/yoloe_text_emb.pt")),
                                      oc.get("imgsz", 960), oc.get("conf", 0.15), oc.get("iou", 0.5),
                                      oc.get("crop", [0.20, 0.08, 0.80, 0.66]), oc.get("device"), oc.get("batch", 8))
            t = time.time()
            dets = det.run(paths)
            with open(out / "objects.json", "w", encoding="utf-8") as f:
                json.dump({"model": want, "classes": list(classes), "dets": dets}, f, separators=(",", ":"))
            m["objects_model"] = want
            _write_manifest(out, m)
            log.info("[objects] %d ảnh xong trong %.0fs", int(has_img.sum()), time.time() - t)

    # ---------------------------------------------------------------- pose (YOLO-pose)
    pc = cfg.get("pose") or {}
    if "pose" in stages and pc.get("enabled", False):
        from ..vision.pose import PoseEstimator, pose_name
        want = pose_name(cfg)
        dets = load_dets()
        if not force and m.get("pose_model") == want and m.get("pose_detector") == m.get("detector") \
                and (out / "pose.json").exists():
            log.info("[pose] đã có (%s) -> bỏ qua", want)
        elif dets is None:
            log.warning("[pose] chưa có detections.json -> chạy bước detect trước")
        else:
            est = PoseEstimator(_resolve(pc.get("model", "weights/yolo26s-pose.pt")), pc.get("imgsz", 640),
                                pc.get("conf", 0.25), pc.get("kp_conf", 0.3), pc.get("match_iou", 0.4),
                                pc.get("device"), pc.get("batch", 8))
            t = time.time()
            persons, extra = est.run(paths, dets, count_conf)
            with open(out / "pose.json", "w", encoding="utf-8") as f:
                json.dump({"model": want, "persons": persons, "extra": extra}, f, separators=(",", ":"))
            m["pose_model"], m["pose_detector"] = want, m.get("detector")
            m.pop("hands_model", None)                       # khung xương đổi -> vùng bàn tay phải làm lại
            _write_manifest(out, m)
            log.info("[pose] %d ảnh, %d người ghép được trong %.0fs", int(has_img.sum()),
                     sum(len(p) for p in persons), time.time() - t)

    # ---------------------------------------------------------------- hands (CLIP vùng bàn tay)
    hc = cfg.get("hands") or {}
    if "hands" in stages and hc.get("enabled", False) and cfg["encoders"]["clip"]["enabled"]:
        from ..encoders.factory import clip_encoder_name, make_clip_encoder
        from ..vision.hands import build_hands, hands_name
        want = hands_name(cfg, clip_encoder_name(cfg), m.get("pose_model", ""))
        if not force and m.get("hands_model") == want and (out / "hands_emb.npy").exists():
            log.info("[hands] đã có -> bỏ qua")
        elif not (out / "pose.json").exists() or not m.get("pose_model"):
            log.warning("[hands] chưa có pose.json -> chạy bước pose trước")
        else:
            pose = json.loads((out / "pose.json").read_text(encoding="utf-8"))
            enc = make_clip_encoder(cfg)
            t = time.time()
            emb, ref, box = build_hands(paths, pose["persons"], load_dets(), enc, pc.get("kp_conf", 0.3),
                                        hc.get("size", 0.22), batch=cfg["encoders"]["clip"].get("batch_size", 32) * 2)
            np.save(out / "hands_emb.npy", emb)
            np.save(out / "hands_ref.npy", ref)
            np.save(out / "hands_box.npy", box)
            from ..search.vision_extras import hand_margins
            np.savez(out / "hands_tags.npz", **hand_margins(emb.astype(np.float32), cfg, enc.encode_query))
            m["hands_model"] = want
            _write_manifest(out, m)
            log.info("[hands] %d bàn tay xong trong %.0fs", len(emb), time.time() - t)


def _resolve(p: str | None) -> str | None:
    """Đường dẫn model tương đối -> theo thư mục dự án (nếu file có ở đó), không thì để ultralytics tự tải."""
    if not p:
        return p
    from ..config import PROJECT_ROOT
    q = Path(p)
    if not q.is_absolute() and (PROJECT_ROOT / q).exists():
        return str(PROJECT_ROOT / q)
    return p


def load_index(cfg: dict, catalog: Catalog) -> IndexBundle:
    out = Path(cfg["paths"]["index_dir"])
    mf = out / "manifest.json"
    if not mf.exists():
        raise FileNotFoundError(f"Chưa có chỉ mục ở {out}. Chạy: python scripts/build_index.py")
    manifest = json.loads(mf.read_text(encoding="utf-8"))
    if manifest["ids"] != catalog.ids:
        raise RuntimeError("Chỉ mục không khớp metadata hiện tại (CSV đã đổi?). Hãy build lại index.")
    load = lambda name: np.load(out / name) if (out / name).exists() else None
    warn: list[str] = []
    sc = cfg.get("scene") or {}

    text_emb = load("text_emb.npy") if manifest.get("text_model") else None
    sent_emb = sent_doc = None
    if text_emb is not None:
        built = manifest.get("text_variant", "raw-v0")
        if built != text_variant(cfg):
            warn.append(f"Embedding mô tả build theo kiểu '{built}', config muốn '{text_variant(cfg)}' -> "
                        "chạy: python scripts/build_index.py --stages text")
        if "+sent" in built and "+sent" in text_variant(cfg) and (out / "sent_emb.npy").exists():
            sent_emb = load("sent_emb.npy").astype(np.float32)
            sent_doc = load("sent_doc.npy")

    clip_emb = load("clip_img_emb.npy") if manifest.get("clip_model") else None
    clip_mask = load("clip_mask.npy") if clip_emb is not None else None

    detections = None
    if cfg.get("vision", {}).get("enabled", False) and manifest.get("detector") and (out / "detections.json").exists():
        detections = json.loads((out / "detections.json").read_text(encoding="utf-8"))["dets"]

    region_emb = person_emb = person_ref = None
    region_names: list[str] = []
    if manifest.get("crops_model") and cfg["encoders"]["clip"]["enabled"] and sc.get("enabled", True):
        if manifest["crops_model"].split("|")[0] != manifest.get("clip_model"):
            warn.append("Ảnh cắt (crops) build bằng model CLIP khác -> chạy: python scripts/build_index.py --stages crops")
        else:
            if (out / "region_emb.npy").exists():
                region_emb = load("region_emb.npy").astype(np.float32)
                region_names = manifest.get("crops_regions", [])
            if (manifest.get("crops_detector") and manifest.get("crops_detector") == manifest.get("detector")
                    and detections is not None and (out / "person_emb.npy").exists()):
                person_emb = load("person_emb.npy").astype(np.float32)
                person_ref = load("person_ref.npy")
            elif detections is not None and sc.get("person_crops", True):
                warn.append("Chưa có CLIP từng người khớp với detections hiện tại -> "
                            "chạy: python scripts/build_index.py --stages crops")
    elif sc.get("enabled", True) and cfg["encoders"]["clip"]["enabled"] and clip_emb is not None:
        warn.append("Chưa build bước 'crops' (CLIP từng người + vùng đảo bếp) -> "
                    "chạy: python scripts/build_index.py --stages crops")

    # ---- nâng cấp nhận diện: chỉ nạp phần đang bật trong config (tắt = như chưa từng build)
    objects = pose = hands_emb = hands_ref = hands_box = clip_tags = hands_tags = None
    if ((cfg.get("objects") or {}).get("clip_tags") or {}).get("enabled", False) and manifest.get("tags_model") \
            and (out / "clip_tags.npz").exists() and manifest["tags_model"].split("|")[0] == manifest.get("clip_model"):
        with np.load(out / "clip_tags.npz") as z:
            clip_tags = {k: z[k] for k in z.files}
    if ((cfg.get("objects") or {}).get("yoloe") or {}).get("enabled", False):
        if manifest.get("objects_model") and (out / "objects.json").exists():
            objects = json.loads((out / "objects.json").read_text(encoding="utf-8"))
        else:
            warn.append("Đã bật objects.yoloe nhưng chưa build -> chạy: python scripts/build_index.py --stages objects")
    if (cfg.get("pose") or {}).get("enabled", False) and detections is not None:
        if manifest.get("pose_model") and manifest.get("pose_detector") == manifest.get("detector") \
                and (out / "pose.json").exists():
            pose = json.loads((out / "pose.json").read_text(encoding="utf-8"))
        else:
            warn.append("Đã bật pose nhưng chưa build (hoặc detections đã đổi) -> "
                        "chạy: python scripts/build_index.py --stages pose")
    if (cfg.get("hands") or {}).get("enabled", False) and pose is not None and clip_emb is not None:
        if manifest.get("hands_model") and (out / "hands_emb.npy").exists() \
                and manifest["hands_model"].split("|")[0] == manifest.get("clip_model"):
            hands_emb = load("hands_emb.npy").astype(np.float32)
            hands_ref, hands_box = load("hands_ref.npy"), load("hands_box.npy")
            if (out / "hands_tags.npz").exists():
                with np.load(out / "hands_tags.npz") as z:
                    hands_tags = {k: z[k] for k in z.files}
        else:
            warn.append("Đã bật hands nhưng chưa build -> chạy: python scripts/build_index.py --stages hands")

    for w in warn:
        log.warning(w)
    clean = sc.get("enabled", True) and sc.get("clean_descriptions", False)
    return IndexBundle(text_emb, clip_emb, clip_mask, detections,
                       BM25Index.from_catalog(catalog, detections, cfg.get("vision", {}).get("count_conf", 0.35),
                                              clean=clean),
                       manifest, sent_emb, sent_doc, region_emb, region_names, person_emb, person_ref, warn,
                       objects, pose, hands_emb, hands_ref, hands_box, clip_tags, hands_tags)
