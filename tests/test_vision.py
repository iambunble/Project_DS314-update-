"""Test màu áo (ảnh tổng hợp) + phân tích màu trong câu truy vấn. Không cần YOLO/torch."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from retrieval.query.lexicon import OBJECT_SYNONYMS  # noqa: E402
from retrieval.query.parser import QueryParser  # noqa: E402
from retrieval.vision.colors import classify_pixels, torso_colors  # noqa: E402

RGB = {"black": (20, 20, 25), "white": (235, 235, 230), "gray": (128, 128, 128), "red": (200, 30, 35),
       "orange": (240, 140, 30), "yellow": (235, 220, 40), "green": (40, 150, 60), "blue": (40, 80, 200),
       "purple": (120, 50, 160), "pink": (245, 150, 190), "brown": (110, 70, 35)}


@pytest.mark.parametrize("name", list(RGB))
def test_pixel_colors(name):
    px = np.array([[RGB[name]]], np.uint8)
    from retrieval.vision.colors import COLORS
    assert COLORS[classify_pixels(px)[0, 0]] == name


def test_torso_crop():
    img = np.full((400, 400, 3), 200, np.uint8)           # nền xám sáng
    img[100:300, 150:250] = RGB["blue"]                    # "người" mặc áo xanh dương
    cols = torso_colors(img, (150, 60, 250, 380))
    assert cols[0][0] == "blue"


P = QueryParser(known_hours=set(range(8, 21)), known_classes=set(OBJECT_SYNONYMS))


@pytest.mark.parametrize("q,colors", [
    ("người mặc áo xanh dương", [{"blue"}]),
    ("người mặc áo xanh", [{"blue", "green"}]),
    ("hai người áo đen", [{"black"}]),
    ("cô gái mặc váy hồng và người mặc đồ trắng", [{"pink"}, {"white"}]),
    ("a man in a blue shirt", [{"blue"}]),
    ("woman wearing red", [{"red"}]),
    ("someone in black attire", [{"black"}]),
])
def test_parse_colors(q, colors):
    assert [set(c) for c in P.parse(q).colors] == colors


def test_color_not_object():
    pq = P.parse("người mặc áo cam")
    assert "orange" not in pq.objects and pq.colors == [frozenset({"orange"})]
    assert "orange" in P.parse("có quả cam trên bàn").objects


def test_no_color_from_background():
    assert P.parse("bếp có gạch ốp màu đỏ").colors == []
    assert P.parse("bông cải xanh trong bát").colors == []
