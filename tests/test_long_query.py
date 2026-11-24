"""Đề thi DRES dạng đoạn nhiều câu: đọc điều kiện đúng câu, không bắt lỗi mạo từ "một", vùng tủ lạnh."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from retrieval.config import load_config  # noqa: E402
from retrieval.query.lexicon import OBJECT_SYNONYMS  # noqa: E402
from retrieval.query.parser import QueryParser  # noqa: E402
from retrieval.scene.layout import load_layout  # noqa: E402

CFG = load_config()
P = QueryParser(known_hours=set(range(8, 21)), known_owners={"Kitchen"}, known_classes=set(OBJECT_SYNONYMS),
                zones=load_layout(CFG).zones)
BTC = ("Một người đàn ông mở cửa tủ lạnh. Anh ấy mặc áo len. Phần trên áo màu sáng trong khi phần dưới màu đen. "
       "Thầy Linh cũng có trong bức ảnh đó mặc áo len màu xanh. Những người khác như đang rửa chén. "
       "Áo len có vài sọc ngang.")


def test_btc_question():
    pq = P.parse(BTC)
    assert str(pq.objects["person"]) == ">=3"          # không phải "==1": "một" là mạo từ, còn "những người khác"
    assert "fridge" in pq.zones and "sink" in pq.zones
    assert pq.zone_weights.get("sink") == 0.5 and "fridge" not in pq.zone_weights   # "như đang rửa chén" = đoán
    k = pq.colors.index(frozenset({"blue", "green"})) if frozenset({"blue", "green"}) in pq.colors else 0
    assert pq.color_zones[k] == frozenset()             # áo xanh của thầy Linh không gắn với bồn rửa
    assert len(pq.sentences) >= 4
    assert "linh" not in pq.text and "thầy" not in pq.text


def test_mot_nguoi_is_indefinite():
    assert str(P.parse("một người đàn ông mở tủ lạnh").objects["person"]) == ">=1"
    assert str(P.parse("chỉ có một người trong bếp").objects["person"]) == "==1"
    assert str(P.parse("một người đứng một mình").objects["person"]) == "==1"
    assert str(P.parse("hai người đang nấu ăn").objects["person"]) == "==2"


def test_names_only_when_capitalised():
    assert "người đàn ông" in P.parse("Thầy Linh đứng ở bếp nấu").text
    assert "anh ấy" in P.parse("anh ấy mặc áo len").text
    assert "cô gái" in P.parse("cô gái tóc dài").text


def test_short_query_single_sentence():
    pq = P.parse("người áo xanh dương đang rửa bát")
    assert pq.sentences == [pq.text] and pq.color_zones == [frozenset({"sink"})]
    assert pq.zone_weights == {}


def test_fridge_zone():
    assert "fridge" in P.parse("người đứng ở tủ lạnh").zones
    assert "fridge" in P.parse("someone opening the fridge").zones


# ---------------------------------------------------------------- đề thi 23/11 (thêm gợi ý mỗi 30 giây)
def test_no_one_else_means_exactly_one():
    pq = P.parse("Một người đàn ông đang xem điện thoại. Không có ai khác.")
    assert str(pq.objects["person"]) == "==1"                 # không phải "bếp trống" (==0)
    assert str(P.parse("không có ai trong bếp").objects["person"]) == "==0"
    assert str(P.parse("a man alone, no one else").objects["person"]) == "==1"


def test_frame_position():
    pq = P.parse("Anh ấy đứng ở góc dưới bên trái khung hình. Lúc đó là buổi chiều")
    assert pq.frame_pos == [(0.0, 0.5, 0.45, 1.0)]
    assert "khung hình" not in pq.text and "góc" not in pq.text
    assert P.parse("người ở bên phải bức ảnh").frame_pos == [(0.55, 0.0, 1.0, 1.0)]
    assert P.parse("Thầy Linh cũng có trong bức ảnh đó").frame_pos == []


def test_empty_sentences_dropped():
    pq = P.parse("Một người đàn ông đang xem điện thoại. Anh ấy đeo kính, mặc áo len màu xanh đậm. "
                 "Không có ai khác. Lúc đó là buổi chiều.")
    assert pq.sentences == ["một người đàn ông đang xem điện thoại", "anh ấy đeo kính, mặc áo len màu xanh đậm"]
    assert "lúc đó là" not in pq.text


def test_skirt_colour_not_shirt_colour():
    pq = P.parse("Cô ấy có mái tóc dài, mặc áo len màu kem và váy đen.")
    assert pq.colors == [frozenset({"brown", "white"})]       # màu áo = kem; váy đen không tính
    assert P.parse("cô gái mặc váy hồng và người mặc đồ trắng").colors == [frozenset({"pink"}), frozenset({"white"})]


def test_colour_word_not_fruit():
    pq = P.parse("Một chiếc khăn màu cam được vắt trên vòi nước.")
    assert "orange" not in pq.objects
    assert "orange" in P.parse("trên bàn có quả cam").objects


def test_sports_jersey_colour():
    assert P.parse("một người mặc áo thể thao màu xanh dương").colors == [frozenset({"blue"})]
