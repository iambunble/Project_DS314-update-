"""Test phần hiểu bố cục cảnh (camera cố định): vùng, bỏ câu nền, gộp khung hình, bỏ qua nội thất cố định.
Không cần torch / model: dựng catalog + chỉ mục giả nhỏ."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from retrieval.config import load_config  # noqa: E402
from retrieval.data.catalog import Catalog  # noqa: E402
from retrieval.index.bm25 import BM25Index  # noqa: E402
from retrieval.index.store import IndexBundle  # noqa: E402
from retrieval.query.lexicon import OBJECT_SYNONYMS  # noqa: E402
from retrieval.query.parser import QueryParser  # noqa: E402
from retrieval.scene.layout import load_layout  # noqa: E402
from retrieval.scene.text_clean import content_sentences, is_boilerplate  # noqa: E402
from retrieval.search.engine import ManualFilters, SearchEngine  # noqa: E402

CFG = load_config()
LAYOUT = load_layout(CFG)
P = QueryParser(known_hours=set(range(8, 21)), known_owners={"Kitchen"}, known_classes=set(OBJECT_SYNONYMS),
                zones=LAYOUT.zones)


# ------------------------------------------------------------------ parser: vùng
@pytest.mark.parametrize("q,zones", [
    ("hai người đứng ở bếp", {"stove": ">=2"}),
    ("người rửa bát", {"sink": ">=1"}),
    ("có người ngồi trên sofa", {"living": ">=1"}),
    ("không có ai ở bếp nấu", {"stove": "==0"}),
    ("người ngồi ở bàn ăn", {"dining": ">=1"}),
    ("someone sitting on the couch", {"living": ">=1"}),
    ("a man washing dishes at the sink", {"sink": ">=1"}),
    ("người đi ngang gần camera", {"foreground": ">=1"}),
    ("đang nấu ăn ở bếp", {"stove": ">=1"}),
])
def test_parse_zones(q, zones):
    assert P.parse(q).describe()["zones"] == zones


@pytest.mark.parametrize("q", ["trong bếp có 3 người", "nhà bếp buổi sáng", "căn bếp không có người",
                               "bàn ăn có nhiều trái cây", "pizza in the oven"])
def test_no_zone(q):
    """'trong bếp' = cả căn bếp; nói về đồ vật (không có người/hành động) thì không ràng buộc vị trí người."""
    assert P.parse(q).zones == {}


def test_zone_keeps_people_constraint():
    pq = P.parse("không có ai ở bếp nấu")
    assert "person" not in pq.objects        # chỗ khác vẫn có thể có người
    pq = P.parse("căn bếp không có người")
    assert pq.objects["person"].n == 0


# ------------------------------------------------------------------ layout
def test_zone_of_box():
    # người đứng trước bếp nấu: điểm chân ~ (0.68, 0.50)
    assert "stove" in LAYOUT.zones_of_box([0.63, 0.15, 0.73, 0.50])
    # người sát camera
    assert LAYOUT.zones_of_box([0.40, 0.50, 0.55, 0.95]) == ["foreground"]


# ------------------------------------------------------------------ bỏ câu nền
def test_boilerplate():
    assert is_boilerplate("The kitchen features white cabinets, red-tiled backsplash, and a black countertop.")
    assert is_boilerplate("The image shows a modern kitchen with white cabinets and red tiled backsplash.")
    assert not is_boilerplate("A man in a blue shirt is washing dishes at the sink.")
    assert not is_boilerplate("The kitchen features white cabinets and a bowl of popcorn on the island.")
    d = ("Two people are cooking. The kitchen features white cabinets, red backsplash tiles, and a black "
         "countertop. A pineapple sits on the island.")
    assert content_sentences(d) == ["Two people are cooking.", "A pineapple sits on the island."]
    assert content_sentences("The kitchen has white cabinets.") == ["The kitchen has white cabinets."]


# ------------------------------------------------------------------ engine (chỉ mục giả)
def _engine(dets, descs=None, hours=None, frames=None):
    n = len(dets)
    hours = hours or [12] * n
    frames = frames or list(range(0, 100 * n, 100))
    df = pd.DataFrame({"imageID": [f"day1_Kitchen_{h:02d}_{f}" for h, f in zip(hours, frames)],
                       "owner": "Kitchen", "day": "day1", "day_num": 1, "hour": hours, "frame": frames,
                       "classes": [[] for _ in range(n)], "counts": [{} for _ in range(n)],
                       "description": descs or ["a kitchen"] * n,
                       "image_path": [f"fake_{i}.jpg" for i in range(n)]})   # có "ảnh" -> dùng số đếm YOLO
    df["sec_of_day"] = df["hour"] * 3600 + df["frame"]
    df["timestamp"] = df["sec_of_day"]
    df["time_str"] = df["imageID"]
    cat = Catalog(df)
    idx = IndexBundle(None, None, None, dets, BM25Index(list(df["description"])), {})
    return SearchEngine(CFG, cat, idx)


def person(box, col="blue"):
    return {"c": "person", "s": 0.9, "b": box, "col": [col]}


STOVE = [0.63, 0.15, 0.73, 0.50]
FRONT = [0.40, 0.50, 0.55, 0.95]
COUCH = {"c": "couch", "s": 0.9, "b": [0.11, 0.25, 0.26, 0.47]}


def test_zone_bonus_ranks_people_at_stove_first():
    eng = _engine([[person(FRONT), COUCH], [person(STOVE), COUCH], [COUCH], [person(STOVE), person(STOVE)]])
    res = eng.search("hai người đứng ở bếp nấu", 4, ManualFilters(dedup_seconds=0))
    ids = [r.idx for r in res.results]
    assert ids[0] == 3 and ids[1] == 1


def test_fixed_objects_ignored():
    eng = _engine([[COUCH]] * 4 + [[]])
    assert "couch" in eng.fixed_classes and "person" not in eng.fixed_classes
    res = eng.search("có ghế sofa", 5)
    assert res.results[0].components["object_bonus"] == 0


def test_dedup_groups_neighbouring_frames():
    dets = [[person(STOVE)]] * 6
    eng = _engine(dets, frames=[0, 10, 20, 500, 510, 2000])
    res = eng.search("người ở bếp nấu", 10, ManualFilters(dedup_seconds=30))
    assert len(res.results) == 3
    assert sorted(len(r.group) for r in res.results) == [0, 1, 2]
    res = eng.search("người ở bếp nấu", 10, ManualFilters(dedup_seconds=0))
    assert len(res.results) == 6


def test_record_has_people_and_zones():
    eng = _engine([[person(STOVE, "red"), person(FRONT)]])
    rec = eng.record(0)
    assert rec["zone_counts"] == {"stove": 1, "foreground": 1}
    assert [p["colors"] for p in rec["people"]] == [["red"], ["blue"]]


# ------------------------------------------------------------------ bước crops (encoder giả, ảnh tổng hợp)
def test_build_crops(tmp_path):
    from PIL import Image
    from retrieval.vision.crops import build_crops

    class FakeClip:
        def encode_pil(self, ims):
            v = np.array([np.asarray(im.resize((4, 4))).astype(np.float32).ravel() for im in ims])
            return v / np.linalg.norm(v, axis=1, keepdims=True).clip(1e-6)

    paths = []
    for k, col in enumerate([(200, 30, 30), (30, 30, 200)]):
        p = tmp_path / f"{k}.jpg"
        Image.new("RGB", (320, 180), col).save(p)
        paths.append(str(p))
    dets = [[person(STOVE), person(FRONT), {"c": "bowl", "s": 0.9, "b": [0, 0, .1, .1]}], [person(FRONT)]]
    names, reg, pe, pr = build_crops(paths + [None], dets + [[]], FakeClip(), {"island": [.2, .2, .6, .6]},
                                     batch=2)
    assert names == ["island"] and reg.shape == (3, 1, 48)
    assert pr.tolist() == [[0, 0], [0, 1], [1, 0]] and pe.shape == (3, 48)
    assert not reg[2].any()                       # ảnh thiếu file -> hàng 0


def test_color_and_zone_same_person():
    """'người áo đỏ ở bếp nấu': ảnh có người áo đỏ ĐỨNG Ở BẾP xếp trên ảnh có áo đỏ ở chỗ khác."""
    eng = _engine([[person(STOVE, "blue"), person(FRONT, "red")], [person(STOVE, "red"), person(FRONT, "blue")]])
    res = eng.search("người mặc áo đỏ ở bếp nấu", 2, ManualFilters(dedup_seconds=0))
    assert res.results[0].idx == 1
    assert res.results[0].components["color_bonus"] > res.results[1].components["color_bonus"]


def test_zone_negation_only_in_its_clause():
    pq = P.parse("phòng khách không có ai nhưng bếp có người")
    assert pq.describe()["zones"] == {"living": "==0"} and pq.objects["person"].op == ">="
    pq = P.parse("không có ai ở bếp nấu")
    assert pq.describe()["zones"] == {"stove": "==0"} and "person" not in pq.objects


def test_color_count_and_negation_scoring():
    eng = _engine([[person(STOVE, "black"), person(FRONT, "red")],
                   [person(STOVE, "black"), person(FRONT, "black")],
                   [person(STOVE, "blue"), person(FRONT, "white")]])
    r = eng.search("hai người mặc áo đen", 3, ManualFilters(dedup_seconds=0))
    assert r.results[0].idx == 1
    r = eng.search("không có ai mặc áo đen", 3, ManualFilters(dedup_seconds=0))
    assert r.results[0].idx == 2
    r = eng.search("người không mặc áo đen", 3, ManualFilters(dedup_seconds=0))
    assert r.results[-1].idx == 1                    # ảnh toàn người áo đen xếp cuối


def test_alternatives_scoring():
    eng = _engine([[person(STOVE)] * 3, [], [person(STOVE)] * 3, []], hours=[12, 12, 19, 19],
                  frames=[0, 100, 0, 100])
    r = eng.search("buổi trưa có ba người, buổi tối không ai", 4, ManualFilters(dedup_seconds=0))
    assert sorted(x.idx for x in r.results[:2]) == [0, 3]          # mỗi vế 1 ảnh đúng ở đầu
    assert any("vế" in n for n in r.notes)
