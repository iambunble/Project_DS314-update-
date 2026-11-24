"""Chạy: python -m pytest -q"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from retrieval.query.parser import QueryParser  # noqa: E402

H = 3600
P = QueryParser(known_hours=set(range(8, 21)), known_owners={"Kitchen"},
                known_classes={"person", "chair", "dining table", "bowl", "cup", "bottle", "oven", "sink", "couch"})


@pytest.mark.parametrize("q,iv", [
    ("ảnh lúc 9h", [(9 * H, 10 * H)]),
    ("lúc 3 giờ chiều", [(15 * H, 16 * H)]),
    ("3 giờ", [(15 * H, 16 * H)]),                          # dữ liệu 8–20h -> hiểu là 15h
    ("9h30", [(9 * H + 15 * 60, 9 * H + 45 * 60 + 1)]),
    ("9 giờ rưỡi sáng", [(9 * H + 15 * 60, 9 * H + 45 * 60 + 1)]),
    ("từ 9h đến 11h", [(9 * H, 12 * H)]),
    ("từ 2 đến 4 giờ chiều", [(14 * H, 17 * H)]),
    ("trước 10h", [(0, 10 * H)]),
    ("sau 7 giờ tối", [(19 * H, 24 * H)]),
    ("khoảng 12h trưa", [(11 * H, 14 * H)]),
    ("buổi sáng", [(5 * H, 11 * H)]),
    ("vào buổi chiều", [(13 * H, 18 * H)]),
    ("at 4pm", [(16 * H, 17 * H)]),
    ("in the evening", [(17 * H, 22 * H)]),
    ("between 10:00 and 11:30", [(10 * H, 11 * H + 31 * 60)]),
])
def test_time(q, iv):
    assert P.parse(q).intervals == iv


@pytest.mark.parametrize("q", ["chiều cao của tủ", "căn bếp sáng sủa", "ánh sáng chiếu vào", "ba người", "3 ghế"])
def test_no_false_time(q):
    assert P.parse(q).intervals == []


@pytest.mark.parametrize("q,cls,cons", [
    ("ba người đang nấu ăn", "person", "==3"),
    ("3 people cooking", "person", "==3"),
    ("không có người", "person", "==0"),
    ("căn bếp vắng người", "person", "==0"),
    ("nobody in the kitchen", "person", "==0"),
    ("một người đứng một mình", "person", "==1"),
    ("nhiều người tụ tập", "person", ">=3"),
    ("ít nhất 2 người", "person", ">=2"),
    ("hơn 4 người", "person", ">=5"),
    ("hai cái ghế", "chair", "==2"),
    ("không có ghế", "chair", "==0"),
    ("bàn ăn có bát", "bowl", ">=1"),
    ("ghế sofa", "couch", ">=1"),
])
def test_objects(q, cls, cons):
    assert str(P.parse(q).objects[cls]) == cons


def test_sofa_not_chair():
    assert "chair" not in P.parse("ghế sofa").objects


def test_filters_and_text():
    pq = P.parse("Tìm ảnh ngày 1 trong nhà bếp lúc 13h có ba người ngồi ăn")
    assert pq.days == {1} and pq.owners == {"Kitchen"} and pq.intervals == [(13 * H, 14 * H)]
    assert "13" not in pq.text and "ba người ngồi ăn" in pq.text
    assert pq.is_vietnamese


# ------------------------------------------------------------------ các lỗi đã sửa (mục 1)
from retrieval.query.lexicon import OBJECT_SYNONYMS as _OBJ  # noqa: E402

P2 = QueryParser(known_hours=set(range(8, 21)), known_owners={"Kitchen"}, known_classes=set(_OBJ))


@pytest.mark.parametrize("q,person", [
    ("ba người trở lên", ">=3"),
    ("từ 3 người trở lên", ">=3"),
    ("3 people or more", ">=3"),
    ("từ 2 đến 4 người", "2..4"),
    ("2-4 người", "2..4"),
    ("between 2 and 4 people", "2..4"),
    ("ít nhất 2 người nhưng không quá 4 người", "2..4"),
    ("không quá 3 người", "<=3"),
    ("một người áo đỏ và hai người áo đen", "==3"),
    ("có 3 người lúc 12 giờ trưa", "==3"),
])
def test_person_counts(q, person):
    assert P2.parse(q).describe()["objects"]["person"] == person


@pytest.mark.parametrize("q,interval", [
    ("lúc 3 rưỡi chiều", ["15:15–15:45"]),
    ("khoảng 3 rưỡi chiều", ["15:00–16:00"]),
    ("lúc 3 chiều", ["15:00–15:59"]),
    ("8 sáng", ["08:00–08:59"]),
])
def test_bare_hour(q, interval):
    assert P2.parse(q).describe()["time"] == interval


def test_day_not_taken_as_hour():
    d = P2.parse("ngày 1 lúc 2 chiều").describe()
    assert d["days"] == [1] and d["time"] == ["14:00–14:59"]


def test_negated_colors():
    pq = P2.parse("không có ai mặc áo đỏ")
    assert pq.colors_none == [frozenset({"red"})] and pq.colors == []
    assert pq.objects["person"].op == ">=" and pq.objects["person"].n == 1     # có người, chỉ là không ai mặc đỏ
    assert P2.parse("nobody wearing red").colors_none == [frozenset({"red"})]
    assert P2.parse("no one in red").colors_none == [frozenset({"red"})]
    pq = P2.parse("người không mặc áo đen")
    assert pq.colors_except == [frozenset({"black"})] and pq.colors == []


def test_color_counts():
    pq = P2.parse("một người áo đỏ và hai người áo đen")
    assert dict(zip(pq.colors, pq.color_counts)) == {frozenset({"red"}): 1, frozenset({"black"}): 2}
    assert P2.parse("two people in black").color_counts == [2]


@pytest.mark.parametrize("q", ["áo sọc", "áo kẻ caro", "đeo kính", "a striped sweater"])
def test_garment_implies_person(q):
    pq = P2.parse(q)
    assert pq.objects.get("person") is not None and pq.about_people


def test_time_clauses_become_alternatives():
    pq = P2.parse("buổi trưa có ba người, buổi tối không ai")
    assert len(pq.alternatives) == 2
    a, b = pq.alternatives
    assert a.objects["person"].n == 3 and a.intervals == [(11 * 3600, 13 * 3600)]
    assert b.objects["person"].n == 0 and b.intervals == [(18 * 3600, 23 * 3600)]
    # 1 mốc thời gian + vế không có giờ -> KHÔNG tách
    assert P2.parse("lúc 9h, hai người đang nấu ăn").alternatives == []
    assert P2.parse("sau 7 giờ tối, bàn có khăn trải caro đỏ").alternatives == []
