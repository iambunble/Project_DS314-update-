"""Dịch truy vấn: bảng thuật ngữ (không cần tải model) + tách câu thành cụm thuộc tính + gom câu theo người."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from retrieval.query.translate import QueryTranslator, add_glossary  # noqa: E402
from retrieval.search.engine import SearchEngine  # noqa: E402


def test_glossary_inserts_english_terms():
    s = add_glossary("đang cầm một chiếc thớt màu đỏ ở đảo bếp")
    assert "thớt (cutting board)" in s and "đảo bếp (kitchen island)" in s
    s = add_glossary("ngồi trên ghế sofa phủ khăn caro đỏ, đội mũ lưỡi trai đen")
    assert "(plaid blanket)" in s and "(baseball cap)" in s
    assert add_glossary("áo thể thao màu xanh").count("(") == 1       # không chèn chồng
    assert "cream" not in add_glossary("bánh kem")


def test_translator_disabled_by_config():
    assert QueryTranslator.from_config({"translation": {"enabled": False}}) is None
    assert QueryTranslator.from_config({}) is None


def test_translator_keeps_english_without_model():
    t = QueryTranslator(model="khong-co-model-nay")
    assert t.translate(["a man holding a phone"]) == ["a man holding a phone"]   # câu tiếng Anh: không gọi model


def test_phrases_split_attributes():
    ph = SearchEngine.phrases("a man in a blue sports jersey wearing a head-mounted camera holding a red cutting board")
    assert ph == ["a man in a blue sports jersey", "a man wearing a head-mounted camera",
                  "a man holding a red cutting board"]
    assert SearchEngine.phrases("a woman holding a phone") == ["a woman holding a phone"]


class _PQ:
    def __init__(self, sentences):
        self.sentences = sentences


def test_entities_merge_pronoun_sentences():
    ents = SearchEngine.entities(SearchEngine, _PQ(["một người đàn ông rửa tay ở bồn rửa",
                                                    "anh ấy mặc áo sơ mi xanh dương",
                                                    "ở phòng khách có người áo trắng ngồi sofa"]))
    assert ents == ["một người đàn ông rửa tay ở bồn rửa. anh ấy mặc áo sơ mi xanh dương",
                    "ở phòng khách có người áo trắng ngồi sofa"]
