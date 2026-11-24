"""Test phần nâng cấp nhận diện: đồ vật (YOLOE + CLIP theo vùng), tư thế + màu áo theo pose, đang cầm gì.
Không cần torch / model: dựng chỉ mục giả nhỏ, bộ mã hoá giả."""
import copy
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from retrieval.config import load_config  # noqa: E402
from retrieval.data.catalog import Catalog  # noqa: E402
from retrieval.encoders.base import l2norm  # noqa: E402
from retrieval.index.bm25 import BM25Index  # noqa: E402
from retrieval.index.store import IndexBundle  # noqa: E402
from retrieval.query.lexicon import OBJECT_SYNONYMS  # noqa: E402
from retrieval.query.parser import QueryParser  # noqa: E402
from retrieval.search.engine import ManualFilters, SearchEngine  # noqa: E402
from retrieval.search.vision_extras import clip_margins, otsu  # noqa: E402
from retrieval.vision.colors import COLORS, classify_pixels  # noqa: E402
from retrieval.vision.objects import YoloeObjectDetector  # noqa: E402
from retrieval.vision.pose import posture_of, quad_colors, torso_quad, wrists  # noqa: E402

CFG = load_config()
HOLD = {"knife", "cell phone", "cup", "plate", "pot", "frying pan", "cutting board"}
P = QueryParser(known_classes=set(OBJECT_SYNONYMS), holdable=HOLD, postures=True)


# ------------------------------------------------------------------ parser
@pytest.mark.parametrize("q,holding", [
    ("người cầm dao", {"knife": 1.0}),
    ("someone holding a phone", {"cell phone": 1.0}),
    ("người đang cầm một cái cốc", {"cup": 1.0}),
    ("người đang cắt rau", {"knife": 0.5}),
    ("a man chopping onions", {"knife": 0.5}),
    ("người đang gọi điện thoại", {"cell phone": 0.7}),
    ("có dao trên bàn", {}),                       # có dao nhưng không ai cầm
])
def test_parse_holding(q, holding):
    pq = P.parse(q)
    assert pq.holding == holding
    if holding:
        assert pq.about_people


@pytest.mark.parametrize("q,postures", [
    ("hai người đang ngồi", {"sitting": ">=2"}),
    ("người cúi xuống bồn rửa", {"bending": ">=1"}),
    ("someone standing at the stove", {"standing": ">=1"}),
    ("không ai ngồi", {"sitting": "==0"}),
])
def test_parse_postures(q, postures):
    assert {k: str(v) for k, v in P.parse(q).postures.items()} == postures


@pytest.mark.parametrize("q,cls", [("có nồi trên bếp", "pot"), ("cái chảo", "frying pan"), ("thớt gỗ", "cutting board"),
                                   ("ba cái đĩa", "plate"), ("dĩa trái cây", "plate"), ("a frying pan", "frying pan")])
def test_parse_new_objects(q, cls):
    assert cls in P.parse(q).objects


def test_no_holding_without_hands():
    p = QueryParser(known_classes=set(OBJECT_SYNONYMS))           # chưa bật hands / pose -> như cũ
    pq = p.parse("hai người đang ngồi cầm dao")
    assert pq.holding == {} and pq.postures == {}


# ------------------------------------------------------------------ pose
def kp_person(sh=(0.5, 0.30), hip=(0.5, 0.45), knee=(0.5, 0.55), ankle=(0.5, 0.65), wr=(0.56, 0.47), el=(0.55, 0.40),
              c=0.9):
    kp = [[0, 0, 0.0] for _ in range(17)]
    for i, dx in ((5, -0.02), (6, 0.02)):
        kp[i] = [sh[0] + dx, sh[1], c]
    for i, dx in ((11, -0.015), (12, 0.015)):
        kp[i] = [hip[0] + dx, hip[1], c]
    for i in (13, 14):
        kp[i] = [knee[0], knee[1], c] if knee else [0, 0, 0.0]
    for i in (15, 16):
        kp[i] = [ankle[0], ankle[1], c] if ankle else [0, 0, 0.0]
    kp[9], kp[10] = [wr[0], wr[1], c], [wr[0] - 0.1, wr[1], c]
    kp[7], kp[8] = [el[0], el[1], c], [el[0] - 0.1, el[1], c]
    return kp


def test_posture_rules():
    assert posture_of(kp_person()) == "standing"                                   # đùi dài, thân thẳng
    assert posture_of(kp_person(knee=(0.55, 0.47))) == "sitting"                   # đùi nằm ngang
    assert posture_of(kp_person(sh=(0.62, 0.40))) == "bending"                     # thân nghiêng mạnh
    assert posture_of(kp_person(knee=None)) == "unknown"                           # bị che gối
    assert posture_of(kp_person(knee=None), zones=["living"]) == "sitting"         # ở sofa, khuất gối
    assert posture_of(kp_person(knee=(0.5, 0.50)), zones=["dining"]) == "sitting"  # đùi ngắn + ở bàn ăn
    assert posture_of(kp_person(knee=(0.5, 0.50)), zones=["dining", "foreground"]) != "sitting"


def test_torso_quad_and_navy_color():
    kp = kp_person()
    q = torso_quad(kp)
    assert q is not None and len(q) == 4
    img = np.full((720, 1280, 3), 230, np.uint8)                   # nền sáng
    img[200:340, 600:680] = (15, 22, 48)                           # áo xanh navy tối (bản cũ ra "black")
    assert quad_colors(img, q)[0] == "blue"
    assert classify_pixels(np.array([[[15, 22, 48]]], np.uint8))[0, 0] == COLORS.index("black")
    assert classify_pixels(np.array([[[15, 22, 48]]], np.uint8), dark_blue=True)[0, 0] == COLORS.index("blue")
    assert classify_pixels(np.array([[[15, 15, 15]]], np.uint8), dark_blue=True)[0, 0] == COLORS.index("black")


def test_wrist_boxes_follow_forearm():
    boxes = dict(wrists(kp_person(), box=[0.45, 0.25, 0.6, 0.7]))
    x1, y1, x2, y2 = boxes[0]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    assert cy > 0.47 and cx > 0.56                       # tâm nằm sau cổ tay theo hướng khuỷu -> cổ tay
    assert 1 in boxes and len(wrists(kp_person(c=0.1))) == 0


# ------------------------------------------------------------------ YOLOE: đổi toạ độ ảnh cắt về cả ảnh
def test_yoloe_convert_maps_crop_to_full():
    det = YoloeObjectDetector.__new__(YoloeObjectDetector)
    det.names, det.crop = ["knife", "pot"], (0.25, 0.10, 0.75, 0.60)
    T = lambda a: SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: np.array(a)))

    class Boxes(SimpleNamespace):
        def __len__(self):
            return 1
    r = SimpleNamespace(boxes=Boxes(xyxy=T([[0, 0, 64, 36]]), cls=T([1.0]), conf=T([0.8])))
    out = det._convert(r, 1280, 720)
    assert out[0]["c"] == "pot" and out[0]["b"] == [0.25, 0.1, 0.3, 0.15]


def test_otsu_splits_bimodal():
    x = np.r_[np.random.default_rng(0).normal(0, 0.01, 500), np.random.default_rng(1).normal(0.1, 0.01, 200)]
    assert 0.02 < otsu(x) < 0.08


# ------------------------------------------------------------------ engine với dữ liệu giả
class FakeClip:
    name = "fake"

    def encode_query(self, t):
        r = np.random.default_rng(int(hashlib.md5(t.encode()).hexdigest()[:8], 16))
        return l2norm(r.normal(size=8))


def person(box, col="black"):
    return {"c": "person", "s": 0.9, "b": box, "col": [col]}


STOVE_P = [0.63, 0.15, 0.73, 0.50]
FRONT_P = [0.40, 0.50, 0.55, 0.95]


def _engine(dets, objects=None, pose=None, hands=None, cfg=None, clip_tags=None):
    n = len(dets)
    df = pd.DataFrame({"imageID": [f"day1_Kitchen_12_{f}" for f in range(0, 100 * n, 100)],
                       "owner": "Kitchen", "day": "day1", "day_num": 1, "hour": 12,
                       "frame": list(range(0, 100 * n, 100)), "classes": [[] for _ in range(n)],
                       "counts": [{} for _ in range(n)], "description": ["a kitchen"] * n,
                       "image_path": [f"fake_{i}.jpg" for i in range(n)]})
    df["sec_of_day"] = df["hour"] * 3600 + df["frame"]
    df["timestamp"] = df["sec_of_day"]
    df["time_str"] = df["imageID"]
    idx = IndexBundle(None, None, None, copy.deepcopy(dets), BM25Index(list(df["description"])), {})
    if objects is not None:
        idx.objects = {"model": "fake", "classes": ["knife", "pot", "cell phone"], "dets": objects}
    if pose is not None:
        idx.pose = {"model": "fake", "persons": pose, "extra": [[] for _ in range(n)]}
    if hands is not None:
        idx.hands_emb, idx.hands_ref, idx.hands_box = hands
    if clip_tags is not None:
        idx.clip_tags = clip_tags
    return SearchEngine(cfg or CFG, Catalog(df), idx, None, FakeClip())


def test_presence_fuses_yoloe_and_clip():
    dets = [[person(FRONT_P)]] * 4
    K = {"c": "knife", "s": 0.8, "b": [0.4, 0.4, 0.42, 0.42]}
    objs = [[K], [K], [], []]
    tags = {"knife": np.array([0.10, -0.10, 0.10, -0.10], np.float32)}   # CLIP: ảnh 0, 2 có dao
    eng = _engine(dets, objects=objs, clip_tags=tags)
    assert "knife" in eng.extras.classes and "knife" not in eng.fixed_classes
    p, used = eng.extras.presence("knife")
    assert used == ["YOLOE", "CLIP vùng"]
    assert p[0] > p[1] > p[3] and p[0] > p[2] > p[3]                     # cả 2 nguồn đồng ý > 1 nguồn > không nguồn nào
    res = eng.search("có dao", 4, ManualFilters(dedup_seconds=0))
    assert res.results[0].idx == 0
    none = eng.search("không có dao", 4, ManualFilters(dedup_seconds=0))
    assert none.results[0].idx == 3


def test_pot_uses_only_yoloe_by_default():
    dets = [[person(FRONT_P)]] * 2
    objs = [[{"c": "pot", "s": 0.8, "b": [0.6, 0.2, 0.65, 0.25]}], []]
    tags = {"pot": np.array([-0.10, 0.10], np.float32)}                    # CLIP vùng nói ngược lại -> bị bỏ qua
    eng = _engine(dets, objects=objs, clip_tags=tags)
    p, used = eng.extras.presence("pot")
    assert used == ["YOLOE"] and p[0] == pytest.approx(1.0) and p[1] == 0
    both = _engine(dets, objects=objs, clip_tags=tags,
                   cfg=load_config(overrides={"objects": {"sources": {"pot": ["yoloe", "clip"]}}}))
    assert both.extras.presence("pot")[1] == ["YOLOE", "CLIP vùng"]


def test_holding_bonus_prefers_knife_in_hand():
    dets = [[person(FRONT_P)], [person(FRONT_P)], [person(FRONT_P)]]
    kp = kp_person(wr=(0.50, 0.80), el=(0.48, 0.70))
    pose = [{"0": {"kp": kp, "posture": "standing", "torso": None, "col": []}} for _ in range(3)]
    hb = wrists(kp, box=FRONT_P)
    emb = l2norm(np.random.default_rng(0).normal(size=(6, 8))).astype(np.float32)
    ref = np.array([[i, 0, s] for i in range(3) for s, _ in hb], np.int32)
    box = np.array([b for _ in range(3) for _, b in hb], np.float32)
    hx, hy = (box[0, 0] + box[0, 2]) / 2, (box[0, 1] + box[0, 3]) / 2
    objs = [[{"c": "knife", "s": 0.6, "b": [hx - 0.01, hy - 0.01, hx + 0.01, hy + 0.01]}],   # dao ngay trên tay
            [{"c": "knife", "s": 0.6, "b": [0.30, 0.20, 0.32, 0.22]}],                     # dao nằm trên bàn
            []]
    eng = _engine(dets, objects=objs, pose=pose, hands=(emb, ref, box))
    assert "knife" in eng.parser.holdable
    s, used = eng.extras.holding_score("knife")
    assert "YOLOE ở bàn tay" in used and s[0] == pytest.approx(1.0) and s[0] > s[1]
    res = eng.search("người cầm dao", 3, ManualFilters(dedup_seconds=0))
    assert res.results[0].idx == 0 and res.results[0].components["holding_bonus"] > 0


def test_posture_bonus_and_pose_colors():
    dets = [[person(FRONT_P, "black")], [person(FRONT_P, "black")]]
    pose = [{"0": {"kp": kp_person(knee=(0.55, 0.47)), "posture": "sitting", "torso": None, "col": ["blue"]}},
            {"0": {"kp": kp_person(), "posture": "standing", "torso": None, "col": []}}]
    eng = _engine(dets, pose=pose)
    assert eng.parser.postures_allowed == {"standing", "bending"}         # "ngồi" đang tắt theo config
    assert eng.search("người đang ngồi", 2).parsed.get("postures") is None
    eng = _engine(dets, pose=pose, cfg=load_config(overrides={"pose": {"postures": ["sitting", "standing", "bending"]}}))
    res = eng.search("người đang ngồi", 2, ManualFilters(dedup_seconds=0))
    assert res.results[0].idx == 0 and res.results[0].components["posture_bonus"] > 0
    assert eng.people(0)[0]["colors"] == ["blue"]                 # màu theo pose thay màu cũ
    assert eng.people(1)[0]["colors"] == ["black"]                # pose không có màu -> giữ màu cũ
    assert eng.people(0)[0]["posture"] == "sitting"


def test_switches_off_restore_old_behaviour():
    dets = [[person(FRONT_P, "black")], [person(STOVE_P)]]
    pose = [{"0": {"kp": kp_person(), "posture": "standing", "torso": None, "col": ["blue"]}}, {}]
    cfg = load_config(overrides={"objects": {"clip_tags": {"enabled": False}, "yoloe": {"enabled": False}},
                                 "pose": {"enabled": False}, "hands": {"enabled": False}, "colors": {"source": "box"}})
    eng = _engine(dets, cfg=cfg)                                    # load_index sẽ không nạp gì khi tắt
    assert eng.extras.sources() == [] and eng.parser.holdable == set()
    assert eng.people(0)[0]["colors"] == ["black"]
    eng2 = _engine(dets, pose=pose, cfg=load_config(overrides={"colors": {"source": "box"}}))
    assert eng2.people(0)[0]["colors"] == ["black"]                 # colors.source=box -> giữ màu cũ dù có pose


def test_clip_margins_shape_and_nan_for_missing():
    reg = np.zeros((3, 2, 8), np.float32)
    reg[:2] = l2norm(np.random.default_rng(0).normal(size=(2, 2, 8)))
    m = clip_margins(reg, ["island", "stove"], {"pot": {"region": "stove", "pos": ["a"], "neg": ["b"]}},
                     FakeClip().encode_query)
    assert m["pot"].shape == (3,) and np.isnan(m["pot"][2]) and np.isfinite(m["pot"][:2]).all()
