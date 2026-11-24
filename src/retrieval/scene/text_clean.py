"""Bỏ "câu nền" khỏi mô tả trước khi embed / BM25.

Mô tả của bộ dữ liệu thường có 1 câu tả nội thất cố định, gần như giống hệt nhau ở mọi ảnh, vd:
    "The kitchen features white cabinets, red-tiled backsplash, and a black countertop."
Câu này chiếm ~15% số câu nhưng không giúp phân biệt ảnh nào với ảnh nào; ngược lại còn kéo embedding của
mọi ảnh về cùng một chỗ (cosine trung bình giữa 2 mô tả bất kỳ ≈ 0.90). Module này:
  - tách mô tả thành câu,
  - xem câu nào chỉ toàn chi tiết nội thất cố định (sau khi bỏ các cụm "white cabinets", "red backsplash"...
    mà không còn người / hành động / đồ vật nào) -> bỏ,
  - giữ NGUYÊN VĂN các câu còn lại (không cắt giữa câu để embedding vẫn đọc được tự nhiên).
"""
from __future__ import annotations

import re

# Cụm mô tả nội thất cố định của căn bếp (rút ra từ thống kê tần suất từ trong metadata.csv)
SCENE_PHRASES = [
    r"(?:white|cream|beige)(?:[- ]colou?red)? (?:kitchen )?(?:cabinets?|cupboards?)",
    r"(?:a |the )?red(?:[- ](?:tiled|tile|brick|subway))?(?: subway)? (?:tiled? )?(?:backsplash|walls?)(?: tiles?)?",
    r"red (?:subway |brick )?tiles?(?: (?:on|above) (?:the )?(?:backsplash|wall|stove|it))?",
    r"(?:a |the )?black (?:granite |marble )?(?:counter ?tops?|counters?|island countertop|countertop island)",
    r"(?:hanging |three )?pendant lights?(?: (?:hanging )?(?:above|over) (?:the )?(?:island|counter|table|dining area))?",
    r"(?:a |the )?black (?:stove|oven|range|range cooker|appliances)",
    r"(?:a |the )?(?:modern|spacious|well-lit|bright|cozy|large|stylish)(?:,? (?:modern|spacious|well-lit|bright|cozy))* kitchen",
    r"(?:white )?tiled? floors?",
    r"red accents",
    r"modern appliances",
    r"(?:a )?tiled floor",
]
SCENE_RE = re.compile(r"\b(?:" + "|".join(SCENE_PHRASES) + r")\b", re.I)

PERSON_RE = re.compile(
    r"\b(?:person|people|persons|man|men|woman|women|individuals?|someone|anyone|group|guy|girl|boy|lady|he|she|"
    r"they|others?|another|figure|child|children|nobody|no one)\b", re.I)

_GENERIC = set("""the a an and with of in on at to is are be appears appear seems seem image shows show features
feature featuring has have there kitchen kitchens area room scene its it this that also which as well including include
includes various visible some several few overall setup setting space design decor style lit while above over by near
hanging is are has well""".split())

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split((text or "").strip()) if s.strip()]


def is_boilerplate(sentence: str, min_content_tokens: int = 2) -> bool:
    """True nếu câu chỉ tả nội thất cố định (không có người và gần như không còn từ nội dung)."""
    rest = SCENE_RE.sub(" ", sentence)
    if PERSON_RE.search(rest):
        return False
    toks = [t for t in re.findall(r"[a-z]+", rest.lower()) if len(t) > 2 and t not in _GENERIC]
    return len(toks) < min_content_tokens


def content_sentences(text: str) -> list[str]:
    """Các câu mang thông tin riêng của ảnh. Nếu bỏ hết thì giữ lại cả mô tả (không để rỗng)."""
    sents = split_sentences(text)
    kept = [s for s in sents if not is_boilerplate(s)]
    return kept or sents


def content_text(text: str) -> str:
    return " ".join(content_sentences(text))
