"""Phân tích câu truy vấn (tiếng Việt / tiếng Anh) thành:
  - text      : phần mô tả nội dung để đưa vào mô hình embedding
  - intervals : khoảng thời gian trong ngày (giây) -> LỌC CỨNG (metadata chính xác)
  - days, owners : ngày / địa điểm -> LỌC CỨNG
  - objects   : ràng buộc đồ vật + số lượng -> CỘNG ĐIỂM MỀM (detector có thể sai)
  - colors    : màu quần áo của người trong ảnh ("áo xanh", "blue shirt") -> CỘNG ĐIỂM MỀM
  - zones     : người đang ở vùng nào của căn bếp ("ở bếp nấu", "trên sofa", "gần camera") -> CỘNG ĐIỂM MỀM
                (camera cố định nên mỗi vùng là 1 khung toạ độ cố định trong ảnh — xem scene/layout.py)
  - holding   : người đang CẦM vật gì ("cầm dao", "holding a phone"; "đang cắt" ngầm hiểu cầm dao) -> CỘNG ĐIỂM MỀM
  - postures  : tư thế ("ngồi", "đứng", "cúi") + số người -> CỘNG ĐIỂM MỀM (cần bật pose)

Thiết kế dựa trên luật (regex) để nhanh, minh bạch, dễ kiểm thử. Sau này có thể thay bằng LLM
mà không ảnh hưởng phần còn lại, miễn trả về đúng ParsedQuery.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from .lexicon import (COLOR_SYNONYMS, FILLER_PATTERNS, HOLD_BEFORE, IMPLIED_HOLD, LOCATION_SYNONYMS, NUMBER_WORDS,
                      OBJECT_SYNONYMS, PERIODS, POSTURE_WORDS, ZONE_GATE)

DAY_SEC = 86400
VI_CHARS = re.compile(r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]")


@dataclass
class CountConstraint:
    op: str   # '==', '>=', '<=', '..' (khoảng n..hi, gồm cả 2 đầu)
    n: int
    hi: int | None = None

    def ok(self, value: int) -> bool:
        if self.op == "..":
            return self.n <= value <= self.hi
        return {"==": value == self.n, ">=": value >= self.n, "<=": value <= self.n}[self.op]

    def __str__(self) -> str:
        return f"{self.n}..{self.hi}" if self.op == ".." else f"{self.op}{self.n}"


@dataclass
class ParsedQuery:
    raw: str
    text: str = ""
    is_vietnamese: bool = False
    intervals: list[tuple[int, int]] = field(default_factory=list)   # [start, end) giây trong ngày
    days: set[int] = field(default_factory=set)
    owners: set[str] = field(default_factory=set)
    objects: dict[str, CountConstraint] = field(default_factory=dict)
    colors: list[frozenset[str]] = field(default_factory=list)   # mỗi phần tử: các màu chấp nhận cho 1 người
    zones: dict[str, CountConstraint] = field(default_factory=dict)  # vùng -> số người cần có ở vùng đó
    color_counts: list[int] = field(default_factory=list)          # song song với colors: "hai người áo đen" -> 2
    colors_none: list[frozenset[str]] = field(default_factory=list)    # "không ai mặc áo đỏ"
    colors_except: list[frozenset[str]] = field(default_factory=list)  # "người không mặc áo đen"
    alternatives: list["ParsedQuery"] = field(default_factory=list)   # câu nhiều vế thời gian: mỗi vế 1 truy vấn
    holding: dict[str, float] = field(default_factory=dict)          # vật đang cầm -> trọng số (1 rõ ràng, <1 ngầm định)
    postures: dict[str, CountConstraint] = field(default_factory=dict)  # tư thế -> số người cần có
    # [câu dài kiểu đề thi DRES] mỗi câu con có nội dung -> chấm ngữ nghĩa riêng từng câu rồi lấy trung bình
    sentences: list[str] = field(default_factory=list)
    color_zones: list[frozenset[str]] = field(default_factory=list)  # song song colors: vùng nhắc CÙNG CÂU với màu
    zone_weights: dict[str, float] = field(default_factory=dict)     # vùng nói kiểu phỏng đoán ("như đang") -> < 1
    people: list[str] = field(default_factory=list)                  # người đã biết được nhắc tên (data/people.yaml)
    # vị trí người trong khung hình (tỉ lệ 0..1 theo TÂM khung người): [(x1, y1, x2, y2), ...] -> CỘNG ĐIỂM MỀM
    frame_pos: list[tuple[float, float, float, float]] = field(default_factory=list)
    # vị trí trong câu (dùng nội bộ khi phân tích)
    _zero_spans: list = field(default_factory=list, repr=False)
    _zone_spans: dict = field(default_factory=dict, repr=False)
    _person_spans: list = field(default_factory=list, repr=False)
    _color_spans: list = field(default_factory=list, repr=False)

    @property
    def about_people(self) -> bool:
        """Câu hỏi nhắm vào người (có người / màu áo / vị trí người), không phải 'không có ai'."""
        p = self.objects.get("person")
        return bool(self.colors or self.colors_none or self.colors_except or self.holding or self.people
                    or any(not (c.op == "==" and c.n == 0) for c in self.postures.values())
                    or self.zones and any(c.n > 0 or c.op != "==" for c in self.zones.values())
                    or (p is not None and not (p.op == "==" and p.n == 0)))

    @property
    def has_filters(self) -> bool:
        return bool(self.intervals or self.days or self.owners)

    def describe(self) -> dict:
        fmt = lambda s: f"{s // 3600:02d}:{s % 3600 // 60:02d}"
        return {
            "text": self.text,
            "time": [f"{fmt(a)}–{fmt(max(a, b - 1))}" for a, b in self.intervals],
            "days": sorted(self.days), "locations": sorted(self.owners),
            "objects": {k: str(v) for k, v in self.objects.items()},
            "clothing": [("/".join(sorted(c)) + (f" ×{n}" if n > 1 else ""))
                         for c, n in zip(self.colors, self.color_counts or [1] * len(self.colors))]
                        + ["không ai " + "/".join(sorted(c)) for c in self.colors_none]
                        + ["có người không mặc " + "/".join(sorted(c)) for c in self.colors_except],
            "zones": {k: str(v) for k, v in self.zones.items()},
            **({"holding": dict(self.holding)} if self.holding else {}),
            **({"postures": {k: str(v) for k, v in self.postures.items()}} if self.postures else {}),
            **({"sentences": list(self.sentences)} if len(self.sentences) > 1 else {}),
            **({"people": list(self.people)} if self.people else {}),
            **({"frame_pos": [tuple(round(v, 2) for v in r) for r in self.frame_pos]} if self.frame_pos else {}),
            **({"alternatives": [a.describe() for a in self.alternatives]} if self.alternatives else {}),
        }


# ----------------------------------------------------------------- regex thời gian
_PER = r"(?:buổi\s*)?(?P<per{i}>sáng|trưa|chiều|tối|đêm)(?!\s*(?:cao|dài|rộng|sủa))"
_TP = r"""
(?P<h{i}>\d{{1,2}})
(?:
    \s*(?:giờ|h|g)(?![^\W\d_])\s*(?:(?P<m{i}>\d{{1,2}})\s*(?:phút|p(?![^\W\d_]))?)?
  | :(?P<mm{i}>\d{{2}})
  | (?=\s*(?:am|pm|a\.m|p\.m|o'clock))
  | (?=\s*(?:rưỡi|kém\s*\d|(?:buổi\s*)?(?:sáng|trưa|chiều|tối|đêm)(?!\s*(?:cao|dài|rộng|sủa))))
)
(?:\s*(?P<ruoi{i}>rưỡi))?
(?:\s*kém\s*(?P<kem{i}>\d{{1,2}})(?:\s*(?:phút|p(?![^\W\d_])))?)?
(?:\s*(?P<ampm{i}>a\.?m\.?|p\.?m\.?|o'clock)(?![^\W\d_]))?
(?:\s*""" + _PER + r""")?
"""


def _tp(i: int) -> str:
    return _TP.format(i=i).replace("{i}", str(i))


_SEP = r"\s*(?:-|–|~|đến|tới|cho\s*đến|to|until|till|and)\s*"
RE_RANGE = re.compile(r"(?:(?:từ|from|between)\s*)?" + _tp(1) + _SEP + _tp(2), re.X)
RE_RANGE_BARE = re.compile(r"(?:(?:từ|from|between)\s*)?(?P<a>\d{1,2})" + _SEP + _tp(2), re.X)
RE_BEFORE = re.compile(r"(?:trước|before|earlier\s+than)\s*" + _tp(1), re.X)
RE_AFTER = re.compile(r"(?:sau|after|later\s+than)\s*" + _tp(1), re.X)
RE_POINT = re.compile(
    r"(?:(?P<approx>khoảng|tầm|chừng|xấp\s*xỉ|around|about|approximately|~)\s*)?"
    r"(?:(?:vào\s*)?lúc\s*|vào\s*|at\s*)?" + _tp(1), re.X)
RE_PERIOD_VI = re.compile(
    r"(?:(?:vào|ban|lúc|hồi|trong|giữa)\s+)?(?:buổi\s+)(?P<p>sáng\s+sớm|sáng|trưa|chiều|tối|đêm)"
    r"|(?:(?:vào|ban|lúc|hồi|giữa)\s+)(?P<p2>sáng\s+sớm|sáng|trưa|chiều|tối|đêm)(?!\s*(?:cao|dài|rộng|sủa))"
    r"|(?P<p3>sáng\s+sớm|giữa\s+trưa|buổi\s+trưa)")
RE_PERIOD_EN = re.compile(
    r"\b(?:in\s+the\s+|during\s+the\s+|at\s+|this\s+)?(?P<p>early\s+morning|morning|noon|midday|lunchtime|afternoon|evening|night)\b")
RE_DAY = re.compile(r"(?:ngày\s*(?:thứ\s*)?|day\s*)(?P<d>\d{1,3})\b")

# ----------------------------------------------------------------- regex người / đồ vật
RE_PEOPLE_ZERO = re.compile(
    r"không\s+(?:có\s+)?(?:một\s+)?(?:ai|bóng\s+người|người\s+nào|người)|vắng\s+(?:người|vẻ|tanh|bóng\s+người)"
    r"|trống\s+(?:trơn|không|vắng)|không\s+một\s+bóng\s+người|chưa\s+có\s+ai"
    r"|\bno\s+(?:one|people|person|humans?)\b|\bnobody\b|\bwithout\s+(?:any\s+)?(?:people|anyone|persons?)\b"
    r"|\b(?:empty|unoccupied|deserted)\s+(?:kitchen|room|scene|space)\b|\bis\s+empty\b")
RE_PEOPLE_ONE = re.compile(
    r"một\s+mình|chỉ\s+(?:có\s+)?(?:một|1)\s+người|\balone\b|\bby\s+(?:him|her)self\b|\bonly\s+one\s+person\b")
# "Không có ai khác" (đề thi 23/11) = chỉ có người vừa được tả, KHÔNG phải "bếp trống" -> phải bắt trước RE_PEOPLE_ZERO
RE_NO_OTHERS = re.compile(
    r"không\s+(?:có\s+|còn\s+|thấy\s+)?(?:một\s+)?(?:ai|người\s+nào|người)\s+(?:khác|nữa)"
    r"|chỉ\s+(?:có\s+)?(?:một\s+)?mình\s+(?:anh|cô|chị|ông|bà|người\s+đó|người\s+này)(?:\s+ấy)?"
    r"|\b(?:no\s+one|nobody|no\s+other\s+(?:people|person))\s+else\b|\bno\s+other\s+(?:people|person)\b")
# vị trí người trong KHUNG HÌNH: "ở góc dưới bên trái khung hình", "bên phải bức ảnh", "giữa khung hình"
_FRAME_NOUN = r"(?:khung\s+hình|bức\s+ảnh|bức\s+hình|tấm\s+ảnh|ảnh|hình)"
RE_FRAME_POS_VI = re.compile(
    r"(?:(?:ở|tại|nằm\s+ở)\s+)?(?:(?:góc|phía|bên|nửa|mép|cạnh)\s+)?(?P<v>dưới|trên)?\s*(?:cùng\s+)?"
    r"(?:(?:bên|phía)\s+)?(?P<h>trái|phải)?\s*(?P<mid>chính\s+giữa|giữa)?\s*(?:của\s+)?" + _FRAME_NOUN + r"(?!\w)")
RE_FRAME_POS_EN = re.compile(
    r"\b(?:(?:in|at|on)\s+the\s+)?(?P<v>bottom|lower|top|upper)?[\s-]*(?P<h>left|right)?[\s-]*(?P<mid>cent(?:er|re)|middle)?\s*"
    r"(?:corner|side|part|half|edge)?\s*of\s+the\s+(?:frame|image|picture|photo)\b")
# câu chỉ còn từ rỗng sau khi bóc giờ/điều kiện ("lúc đó là", "không có ai khác") -> không đem đi so nghĩa
STOPWORDS = set("""anh chị cô ông bà ấy đó này kia lúc khi là có không ai khác một những các người đang đã sẽ ở tại
trong ngoài và với của thì mà cũng rất nữa còn chỉ được bị cái chiếc con hai ba vào ra lại đây đấy vẫn hết toàn bộ
the a an is are was were be it this that there he she they his her their of in on at to and with no one else other
nobody only just then time""".split())
RE_PEOPLE_MANY = re.compile(
    r"nhiều\s+người|đông\s+(?:người|đúc)|(?:một\s+|cả\s+)?nhóm\s+(?:người|bạn)|tụ\s+tập|quây\s+quần"
    r"|\bmany\s+people\b|\b(?:a\s+)?(?:large\s+)?group\s+of\s+(?:people|persons|friends)\b|\bcrowd(?:ed)?\b"
    r"|\bseveral\s+people\b|\blots?\s+of\s+people\b|\bgathering\b")

_NUM_ALT = "|".join(sorted(map(re.escape, NUMBER_WORDS), key=len, reverse=True))
_NUM = r"(?:\d{1,2}|" + _NUM_ALT + r")"
RE_COUNT_BEFORE = re.compile(
    r"(?:(?P<mod>ít\s+nhất|tối\s+thiểu|không\s+ít\s+hơn|nhiều\s+hơn|hơn|trên|ít\s+hơn|dưới|tối\s+đa|nhiều\s+nhất"
    r"|không\s+(?:vượt\s+)?quá|không\s+hơn|đúng|chỉ\s+có"
    r"|at\s+least|no\s+(?:fewer|less)\s+than|more\s+than|over|fewer\s+than|less\s+than|under|at\s+most"
    r"|no\s+more\s+than|not\s+more\s+than|exactly|only)\s+)?"
    r"(?P<num>\d{1,2}|" + _NUM_ALT + r")\s+(?:cái|chiếc|con|bạn|vị|cá\s+nhân)?\s*$")
# "từ 2 đến 4 người", "2-4 người", "between 2 and 4 people"
RE_COUNT_RANGE_BEFORE = re.compile(
    r"(?:(?:từ|from|between)\s+)?(?P<a>" + _NUM + r")\s*(?:-|–|đến|tới|to|and)\s*(?P<b>" + _NUM + r")\s+"
    r"(?:cái|chiếc|con|bạn|vị)?\s*$")
# "ba người trở lên", "3 people or more"
RE_COUNT_AFTER = re.compile(r"^\s*(?P<m>trở\s+lên|trở\s+xuống|or\s+more|or\s+fewer|or\s+less|and\s+up|\+)")
# nối 2 nhóm người khác nhau: "một người áo đỏ VÀ hai người áo đen" -> cộng số người
RE_JOIN = re.compile(r"(?<!\w)(?:và|cùng\s+với|cùng|với|and|plus|along\s+with)(?!\w)")
# vế câu (dùng cho phủ định theo vế và câu nhiều mốc thời gian)
RE_CLAUSE_SEP = re.compile(r"\s*(?:[,;]|(?<!\w)(?:nhưng|còn|trong\s+khi|but|while|whereas)(?!\w))\s*")
RE_NEG_BEFORE = re.compile(
    r"(?:không\s+(?:có|thấy|hề\s+có)|chẳng\s+có|chưa\s+có|\bno\b|\bwithout\b|\bnot\s+any\b|\bzero\b)"
    r"\s+(?:một\s+)?(?:cái|chiếc|any)?\s*$")
_COLOR_ALT = "|".join(sorted(map(re.escape, COLOR_SYNONYMS), key=len, reverse=True))
_GARMENT_VI = (r"(?:(?:mặc|khoác|diện)\s+)?(?:(?:chiếc|cái|bộ)\s+)?(?:áo(?:\s+(?:thun|phông|sơ\s*mi|khoác|len|hoodie|"
               r"ba\s*lỗ|polo|gió|vest|thể\s*thao|jersey|đấu|nỉ))?|đồ|trang\s+phục|quần\s+áo|váy|đầm|tạp\s+dề)")
RE_COLOR = re.compile(
    r"(?<!\w)(?:" + _GARMENT_VI + r"\s+(?:màu\s+)?(?P<c1>" + _COLOR_ALT + r")"
    r"|mặc\s+(?:đồ\s+)?(?:màu\s+)?(?P<c2>" + _COLOR_ALT + r")"
    r"|(?P<c3>" + _COLOR_ALT + r")\s+(?:t-?shirts?|shirts?|tops?|jackets?|hoodies?|sweaters?|dress|clothes|clothing"
    r"|outfits?|coats?|blouses?|apron|attire|uniform)"
    r"|(?:wearing|dressed\s+in)\s+(?:an?\s+)?(?P<c4>" + _COLOR_ALT + r")"
    r"|(?:people|persons|person|men|man|women|woman|guys|guy|someone|girl|boy|lady|one)\s+"
    r"(?P<in5>in\s+(?:an?\s+)?(?P<c5>" + _COLOR_ALT + r")))(?!\w)")
_NOBODY = (r"(?:không\s+(?:có\s+)?(?:một\s+)?(?:ai|người\s+nào)|chẳng\s+(?:có\s+)?ai|chưa\s+có\s+ai"
           r"|nobody|no\s+one|no\s+people|not\s+anyone|none\s+of\s+them)")
# trước cụm màu: "không có ai [đang] mặc áo đỏ", "nobody [is] wearing red", "no one in red"
RE_NOBODY_BEFORE = re.compile(_NOBODY + r"\s+(?:(?:đang|is|are|was|were)\s+)?$")
RE_NOBODY_IN = re.compile(r"(?<!\w)no\s+$")                       # "no one in red": c5 bắt đầu từ "in"
RE_NOT_WEARING_BEFORE = re.compile(r"(?<!\w)(?:không|chẳng|chưa|not|isn't|aren't|doesn't)\s+(?:đang\s+)?$")
# số người ngay trước cụm màu: "hai người áo đen", "two people in black"
RE_NUM_PERSON_BEFORE = re.compile(
    r"(?<!\w)(?P<num>" + _NUM + r")\s+(?:người|bạn|anh|chị|cô|people|persons|men|women|guys|girls)\s+"
    r"(?:(?:đang|đều|cùng|is|are|all)\s+)?$")
# ranh giới câu (đề thi thường là nhiều câu: "Một người mở tủ lạnh. Anh ấy mặc áo len. ...")
RE_SENT_SEP = re.compile(r"[.!?;\n]+")
# câu phỏng đoán: "những người khác NHƯ ĐANG rửa chén" -> vùng/hành động chỉ là gợi ý, không chắc
RE_HEDGE = re.compile(r"(?<!\w)(?:như\s+(?:đang|là)|có\s+vẻ|dường\s+như|hình\s+như|có\s+lẽ|có\s+thể|trông\s+như"
                      r"|possibly|probably|perhaps|maybe|seems?\s+to|appears?\s+to|might|may\s+be)(?!\w)")
# "những người khác", "mọi người khác", "other people", "the others" -> ngoài người được tả còn >= 2 người nữa
RE_OTHERS = re.compile(r"(?<!\w)(?:(?:những|các|mọi|mấy)\s+người\s+(?:còn\s+lại|khác)|người\s+còn\s+lại"
                       r"|other\s+(?:people|persons|individuals)|the\s+others|others)(?!\w)")
# tên riêng trong đề ("Thầy Linh", "cô Hoa"): hệ thống không biết ai là ai -> thay bằng "người đàn ông/phụ nữ"
RE_NAMED = re.compile(r"(?<!\w)(?P<t>[Tt]hầy|[Cc]ô|[Aa]nh|[Cc]hị|[Bb]ạn|[Ôô]ng|[Bb]à)\s+(?P<n>\w+)")
_NAMED_AS = {"thầy": "người đàn ông", "anh": "người đàn ông", "ông": "người đàn ông",
             "cô": "người phụ nữ", "chị": "người phụ nữ", "bà": "người phụ nữ", "bạn": "người"}


def _unname(m: re.Match) -> str:
    """Chỉ thay khi sau danh xưng là TÊN RIÊNG viết hoa ("Thầy Linh"), không đụng "anh ấy", "cô gái"."""
    return _NAMED_AS[m["t"].lower()] if m["n"][0].isupper() else m.group(0)


# nói về quần áo / ngoại hình -> chắc chắn câu hỏi về người ("áo sọc", "áo kẻ caro", "đeo kính")
RE_GARMENT = re.compile(
    r"(?<!\w)(?:áo|quần|váy|đầm|tạp\s+dề|mũ|nón|đeo\s+kính|kính\s+mắt|tóc|shirts?|t-shirts?|sweaters?|jackets?|"
    r"hoodies?|dress|apron|hats?|caps?|glasses|hair|wearing|dressed)(?!\w)")

# Có động từ hành động -> câu đang nói về người dù không có chữ "người" ("đang nấu ăn ở bếp")
RE_ACTION = re.compile(
    r"(?<!\w)(?:nấu|rửa|đứng|ngồi|đi\s+lại|đi\s+ngang|đi|cắt|thái|băm|gọt|(?<!bàn\s)(?<!ghế\s)(?<!đồ\s)(?<!thức\s)(?<!món\s)ăn|uống|nói\s+chuyện|trò\s+chuyện|làm|chuẩn\s+bị|"
    r"dọn|lau|cầm|bưng|nằm|cúi|với\s+tay|nhìn|xem|cooking|cook|washing|wash|standing|stands?|sitting|sits?|seated|"
    r"walking|walks?|cutting|chopping|slicing|eating|drinking|talking|chatting|preparing|cleaning|holding|lying|"
    r"bending|reaching|leaning|working)(?!\w)")
RE_GATE_BEFORE = re.compile(r"(?<!\w)" + ZONE_GATE + r"\s+(?:(?:cái|chiếc|khu|khu\s+vực|chỗ|phía|the|a)\s+)?$")
RE_HOLD_BEFORE = re.compile(r"(?<!\w)" + HOLD_BEFORE)
RE_IMPLIED_HOLD = [(re.compile(pat), cls, w) for pat, (cls, w) in IMPLIED_HOLD.items()]
RE_POSTURE = [(name, re.compile(pat)) for name, pat in POSTURE_WORDS.items()]

_MOD_OP = {"ít nhất": ">=0", "tối thiểu": ">=0", "at least": ">=0",
           "hơn": ">=1", "trên": ">=1", "nhiều hơn": ">=1", "more than": ">=1", "over": ">=1",
           "ít hơn": "<=-1", "dưới": "<=-1", "fewer than": "<=-1", "less than": "<=-1", "under": "<=-1",
           "tối đa": "<=0", "nhiều nhất": "<=0", "at most": "<=0",
           "không quá": "<=0", "không vượt quá": "<=0", "không hơn": "<=0", "no more than": "<=0",
           "not more than": "<=0", "không ít hơn": ">=0", "no fewer than": ">=0", "no less than": ">=0"}


class QueryParser:
    def __init__(self, known_hours: set[int] | None = None, known_owners: set[str] | None = None,
                 known_classes: set[str] | None = None, approx_hours: int = 1, minute_window: int = 15,
                 zones=None, holdable: set[str] | None = None, postures=False, people: dict | None = None):
        # người đã biết: tên -> (mẫu các cách gọi, cụm thay thế khi so ngữ nghĩa)
        self.people_res = []
        for name, p in (people or {}).items():
            names = [n for n in (p.get("names") or [name]) if n]
            alt = "|".join(sorted((re.escape(n.lower()) for n in names), key=len, reverse=True))
            # từ ghép không phải tên: "linh tinh", "linh hoạt", "linh kiện"...
            nb = "|".join(map(re.escape, p.get("not_before") or []))
            tail = r"(?!\w)" + (r"(?!\s+(?:" + nb + r")(?!\w))" if nb else "")
            self.people_res.append((name, re.compile(r"(?<!\w)(?:" + alt + r")" + tail), p.get("as", "người")))
        self.holdable = set(holdable or [])      # vật nhận được "đang cầm" (cần bật hands); rỗng = bỏ qua
        # tư thế được nhận: True = cả 3, tập tên = chỉ những tư thế đó (vd {"standing", "bending"}), False = bỏ qua
        self.postures_allowed = (set(POSTURE_WORDS) if postures is True else set(postures or []))
        self.use_postures = bool(self.postures_allowed)
        self.known_hours = known_hours or set(range(24))
        self.approx_hours = approx_hours
        self.minute_window = minute_window
        self.owner_patterns = []
        for owner in (known_owners or []):
            words = LOCATION_SYNONYMS.get(owner, []) + [owner.lower()]
            alt = "|".join(sorted(map(re.escape, set(words)), key=len, reverse=True))
            self.owner_patterns.append((owner, re.compile(r"(?<!\w)(?:" + alt + r")(?!\w)")))
        syn2cls = {}
        for cls, words in OBJECT_SYNONYMS.items():
            if known_classes is None or cls in known_classes:
                for w in words:
                    syn2cls.setdefault(w, cls)
        self.syn2cls = syn2cls
        alt = "|".join(sorted(map(re.escape, syn2cls), key=len, reverse=True))
        self.re_obj = re.compile(r"(?<!\w)(?:" + alt + r")(?!\w)")
        # vùng: cụm từ -> (tên vùng, cần giới từ đứng trước?)
        self.zone_words: dict[str, tuple[str, bool]] = {}
        for z in zones or []:
            for w in z.words:
                self.zone_words.setdefault(w.lower(), (z.name, False))
            for w in z.gated_words:
                self.zone_words.setdefault(w.lower(), (z.name, True))
        self.re_zone = None
        if self.zone_words:
            alt = "|".join(sorted(map(re.escape, self.zone_words), key=len, reverse=True))
            self.re_zone = re.compile(r"(?<!\w)(?:" + alt + r")(?!\w)")

    # ------------------------------------------------------------------ public
    def parse(self, query: str, _split: bool = True) -> ParsedQuery:
        q = unicodedata.normalize("NFC", query)
        named = [n for n, pat, _ in self.people_res if pat.search(q.lower())]
        q = RE_NAMED.sub(_unname, q).lower()
        for _, pat, as_text in self.people_res:          # "thầy" / "linh" còn sót -> cụm chung khi so ngữ nghĩa
            q = pat.sub(as_text, q)
        q = re.sub(r"\s+", " ", q).strip()
        pq = ParsedQuery(raw=query, is_vietnamese=bool(VI_CHARS.search(q)), people=named)
        if _split:
            alts = self._split_time_clauses(q)
            if alts:
                res = self._combine_alternatives(query, q, alts)
                for a in [res, *res.alternatives]:
                    a.people = list(named)
                return res

        work = q  # bản sao để "xoá" dần các cụm đã dùng (thời gian, ngày)
        work = self._sub(RE_DAY, work, lambda m: pq.days.add(int(m["d"])))   # trước giờ: "ngày 2 chiều" là ngày 2
        work = self._extract_time(work, pq)
        for owner, pat in self.owner_patterns:
            if pat.search(q):
                pq.owners.add(owner)
        work = self._extract_frame_pos(work, pq)
        taken = self._extract_colors(work, pq)
        taken += self._extract_zones(work, pq, taken)
        self._extract_objects(work, pq, taken)
        self._finish_zones(work, pq)
        self._extract_holding(work, pq)
        self._extract_postures(work, pq)
        if "person" not in pq.objects and RE_GARMENT.search(work):
            pq.objects["person"] = CountConstraint(">=", 1)    # "áo sọc", "áo kẻ caro": đang tả người
        self._sentence_scope(work, pq)

        text = work
        for pat in FILLER_PATTERNS:
            text = re.sub(pat, " ", text)
        text = re.sub(r"(?<!\w)(?:lúc|vào|vào lúc|at|on|in|during|around|khoảng|tầm|trong|ngày|day)\s*(?=[,.;]|$)",
                      " ", text)
        text = re.sub(r"\s*([,.;])\s*(?=[,.;]|$)", " ", text)
        pq.text = re.sub(r"\s+", " ", text).strip(" ,.;-")
        pq.sentences = [x for x in (re.sub(r"\s+", " ", t).strip(" ,.;-") for t in RE_SENT_SEP.split(text))
                        if len(x.split()) >= 3 and content_words(x) >= 2]
        if pq.sentences and len(pq.sentences) < len([t for t in RE_SENT_SEP.split(text) if t.strip()]):
            pq.text = ". ".join(pq.sentences)          # bỏ câu rỗng ("lúc đó là") khỏi cả đoạn đem so nghĩa
        pq.intervals = _merge(pq.intervals)
        return pq

    @staticmethod
    def _extract_frame_pos(s: str, pq: ParsedQuery) -> str:
        """'đứng ở góc dưới bên trái khung hình' -> frame_pos [(0, 0.5, 0.45, 1)] và xoá cụm đó khỏi câu."""
        def add(m):
            v, h, mid = (m["v"] or ""), (m["h"] or ""), m["mid"]
            if not (v or h or mid):
                return m.group(0)
            x = (0.0, 0.45) if h in ("trái", "left") else (0.55, 1.0) if h in ("phải", "right") else \
                (0.3, 0.7) if mid and not v else (0.0, 1.0)
            y = (0.5, 1.0) if v in ("dưới", "bottom", "lower") else (0.0, 0.5) if v in ("trên", "top", "upper") else \
                (0.25, 0.75) if mid and not h else (0.0, 1.0)
            pq.frame_pos.append((x[0], y[0], x[1], y[1]))
            return " "
        s = RE_FRAME_POS_VI.sub(add, s)
        return RE_FRAME_POS_EN.sub(add, s)

    @staticmethod
    def _sentence_scope(s: str, pq: ParsedQuery) -> None:
        """Gắn điều kiện vào ĐÚNG câu của nó (đề thi nhiều câu):
          - màu áo chỉ ghép với vùng nhắc trong cùng câu ("thầy Linh mặc áo xanh" ≠ "người khác rửa chén")
          - vùng trong câu phỏng đoán ("như đang rửa chén") chỉ được nửa điểm
          - "những người khác" -> có >= 3 người (người được tả + ít nhất 2 người nữa)"""
        cuts = [m.end() for m in RE_SENT_SEP.finditer(s)]
        sent = lambda pos: sum(1 for c in cuts if c <= pos)
        bounds = [0] + cuts + [len(s)]
        zone_sent = {z: sent(sp[0]) for z, sp in pq._zone_spans.items() if z in pq.zones}
        pq.color_zones = [frozenset(z for z, k in zone_sent.items() if k == sent(sp[0])) for sp in pq._color_spans]
        for z, k in zone_sent.items():
            if RE_HEDGE.search(s[bounds[k]:bounds[k + 1]]):
                pq.zone_weights[z] = 0.5
        p = pq.objects.get("person")
        if RE_OTHERS.search(s) and (p is None or (p.op == ">=" and p.n < 3)):
            pq.objects["person"] = CountConstraint(">=", 3)

    # ------------------------------------------------------------------ câu nhiều vế thời gian
    def _split_time_clauses(self, q: str) -> list[str]:
        """'buổi trưa có ba người, buổi tối không ai' -> ['buổi trưa có ba người', 'buổi tối không ai'].
        Chỉ tách khi có >= 2 vế mỗi vế có mốc thời gian RIÊNG; vế không có giờ gộp vào vế đứng trước nó."""
        parts = [p for p in RE_CLAUSE_SEP.split(q) if p.strip()]
        if len(parts) < 2:
            return []
        groups: list[list[str]] = []
        times = []
        for p in parts:
            iv = _merge(self.parse(p, _split=False).intervals)
            if iv:
                groups.append([p]); times.append(iv)
            elif groups:
                groups[-1].append(p)
            else:
                groups.append([p]); times.append([])
        if not groups[0] or not times[0]:              # vế đầu không có giờ -> gộp vào vế sau
            if len(groups) > 1:
                groups[1] = groups[0] + groups[1]; times[1] = times[1]
                groups, times = groups[1:], times[1:]
        timed = [g for g, t in zip(groups, times) if t]
        if len(timed) < 2 or len({tuple(t) for t in times}) < 2:
            return []
        return [", ".join(g) for g in groups]

    def _combine_alternatives(self, raw: str, q: str, clauses: list[str]) -> ParsedQuery:
        alts = [self.parse(c, _split=False) for c in clauses]
        pq = ParsedQuery(raw=raw, is_vietnamese=bool(VI_CHARS.search(q)), alternatives=alts)
        pq.intervals = _merge([iv for a in alts for iv in a.intervals])
        pq.text = " | ".join(a.text for a in alts if a.text)
        return pq

    # ------------------------------------------------------------------ thời gian
    def _extract_time(self, s: str, pq: ParsedQuery) -> str:
        def add_range(m):
            a = self._to_sec(m, 1)
            b = self._to_sec(m, 2, end=True)
            if a is not None and b is not None:
                if b <= a:  # "từ 9 đến 2 giờ chiều" -> phần đầu cũng chiều
                    a = (a + 12 * 3600) if a + 12 * 3600 < b else a
                pq.intervals.append((a, max(a + 1, b)))

        def add_range_bare(m):
            b_start = self._to_sec(m, 2)
            b = self._to_sec(m, 2, end=True)
            if b is None:
                return
            ah = int(m["a"])
            bh = b_start // 3600
            if bh >= 12 and ah < 12 and ah + 12 <= bh:  # buổi của mốc 2 áp cho mốc 1
                ah += 12
            pq.intervals.append((ah * 3600, b))

        s = self._sub(RE_RANGE, s, add_range)
        s = self._sub(RE_RANGE_BARE, s, add_range_bare)
        s = self._sub(RE_BEFORE, s, lambda m: pq.intervals.append((0, self._to_sec(m, 1))))
        s = self._sub(RE_AFTER, s, lambda m: pq.intervals.append((self._to_sec(m, 1), DAY_SEC)))

        def add_point(m):
            start = self._to_sec(m, 1)
            if start is None:
                return
            has_min = bool(m["m1"] or m["mm1"] or m["ruoi1"] or m["kem1"])
            if has_min:
                w = self.minute_window * 60
                a, b = start - w, start + w + 1
            else:
                a, b = start, start + 3600
            if m["approx"]:   # "khoảng 9h" -> ±approx_hours ; "khoảng 3 rưỡi" (có phút) -> ±30 phút
                pad = 1800 - self.minute_window * 60 if has_min else self.approx_hours * 3600
                a, b = a - pad, b + pad
            pq.intervals.append((max(0, a), min(DAY_SEC, b)))

        s = self._sub(RE_POINT, s, add_point)

        def add_period(m, key_groups):
            name = next(m[g] for g in key_groups if m[g])
            name = re.sub(r"\s+", " ", name).replace("giữa ", "").replace("buổi ", "")
            h0, h1 = PERIODS[name]
            pq.intervals.append((h0 * 3600, h1 * 3600))

        s = self._sub(RE_PERIOD_VI, s, lambda m: add_period(m, ["p", "p2", "p3"]))
        s = self._sub(RE_PERIOD_EN, s, lambda m: add_period(m, ["p"]))
        return s

    def _to_sec(self, m: re.Match, i: int, end: bool = False) -> int | None:
        h = int(m[f"h{i}"])
        minute = m[f"m{i}"] or m[f"mm{i}"]
        mins = int(minute) if minute else 0
        if m[f"ruoi{i}"]:
            mins = 30
        if m[f"kem{i}"]:
            h, mins = h - 1, 60 - int(m[f"kem{i}"])
        per = m[f"per{i}"]
        ampm = (m[f"ampm{i}"] or "").replace(".", "")
        if per in ("chiều", "tối") or ampm == "pm":
            if h < 12:
                h += 12
        elif per == "đêm":
            if 7 <= h < 12:
                h += 12
        elif per == "trưa":
            if h <= 2:
                h += 12
        elif per == "sáng" or ampm == "am":
            if h == 12:
                h = 0
        elif h not in self.known_hours and (h + 12) in self.known_hours:
            h += 12  # "3 giờ" mà dữ liệu chỉ có 8h–20h -> hiểu là 15h
        if not (0 <= h <= 24 and 0 <= mins < 60):
            return None
        sec = h * 3600 + mins * 60
        if end:  # mốc cuối của khoảng: gồm trọn giờ/phút đó
            sec += 60 if minute or m[f"ruoi{i}"] or m[f"kem{i}"] else 3600
        return min(sec, DAY_SEC)

    # ------------------------------------------------------------------ đồ vật / số lượng
    def _extract_colors(self, s: str, pq: ParsedQuery) -> list[tuple[int, int]]:
        """Trả về các đoạn đã dùng cho màu, để 'cam' trong 'áo cam' không bị hiểu là quả cam.
          'không có ai mặc áo đỏ'   -> colors_none   (có người, nhưng không ai mặc đỏ)
          'người không mặc áo đen'  -> colors_except (có ít nhất 1 người mặc màu khác đen)
          'hai người áo đen'        -> colors + color_counts = 2"""
        spans = []
        found = list(RE_COLOR.finditer(s))
        lower = lambda m: bool(re.search(r"(?<!\w)(?:váy|đầm|dress|skirt)(?!\w)", m.group(0)))
        for m in found:
            if lower(m) and any(not lower(o) and 0 <= m.start() - o.end() <= 12 for o in found):
                # "áo len màu kem và váy đen": màu áo đo ở thân trên -> màu váy không tính là màu áo
                spans.append(m.span())
                continue
            g = next(g for g in ("c1", "c2", "c3", "c4", "c5") if m[g])
            cs = frozenset(COLOR_SYNONYMS[re.sub(r"\s+", " ", m[g])])
            start = m.start("in5") if g == "c5" else m.start()
            span = (start, m.end())
            before = s[max(0, start - 50):start]
            if RE_NOBODY_BEFORE.search(before) or (g == "c5" and RE_NOBODY_IN.search(s[max(0, m.start() - 10):m.start()])):
                nb = RE_NOBODY_BEFORE.search(before)
                if nb:                                   # cụm "không có ai" thuộc về màu, không phải "bếp trống"
                    span = (start - (len(before) - nb.start()), m.end())
                else:
                    span = (m.start() - 3, m.end())
                if cs not in pq.colors_none:
                    pq.colors_none.append(cs)
            elif RE_NOT_WEARING_BEFORE.search(before) or re.search(r"(?<!\w)(?:không|chẳng|chưa)\s+(?:đang\s+)?mặc",
                                                                   s[start:start + 12]) and g in ("c1", "c2"):
                if cs not in pq.colors_except:
                    pq.colors_except.append(cs)
            else:
                cnt = RE_NUM_PERSON_BEFORE.search(before)
                n = _num(cnt["num"]) if cnt else 1
                if cs in pq.colors:
                    k = pq.colors.index(cs)
                    pq.color_counts[k] = max(pq.color_counts[k], n)
                else:
                    pq.colors.append(cs); pq.color_counts.append(n); pq._color_spans.append(span)
            spans.append(span)
        if (pq.colors or pq.colors_none or pq.colors_except) and "person" not in pq.objects:
            pq.objects["person"] = CountConstraint(">=", max(pq.color_counts or [1]))
        return spans

    # ------------------------------------------------------------------ vùng trong bếp
    def _extract_zones(self, s: str, pq: ParsedQuery, taken: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """'ở bếp nấu', 'trên sofa', 'rửa bát' -> pq.zones. Trả về các đoạn đã dùng để 'sofa'/'bồn rửa' không bị
        hiểu thêm là đồ vật (đồ nội thất cố định, ảnh nào cũng có)."""
        if self.re_zone is None:
            return []
        spans = []
        for m in self.re_zone.finditer(s):
            if _overlaps(m.span(), taken):
                continue
            name, gated = self.zone_words[m.group(0)]
            if gated and not RE_GATE_BEFORE.search(s[:m.start()]):
                continue
            pq.zones.setdefault(name, CountConstraint(">=", 1))
            pq._zone_spans.setdefault(name, m.span())
            spans.append(m.span())
        return spans

    @staticmethod
    def _finish_zones(s: str, pq: ParsedQuery) -> None:
        if not pq.zones:
            return
        person = pq.objects.get("person")
        if person is None and not RE_ACTION.search(s):
            pq.zones.clear()             # "bàn ăn có nhiều trái cây": nói về đồ vật, không phải vị trí người
            return
        if person is not None and person.op == "==" and person.n == 0:
            # "không có ai ở bếp nấu" -> vùng đó trống, các chỗ khác có thể có người.
            # Phủ định chỉ áp cho vùng CÙNG VẾ: "phòng khách không có ai nhưng bếp có người"
            clause = _clause_index(s)
            zero = {clause(a) for a, _ in pq._zero_spans} or {None}
            for z in pq.zones:
                sp = pq._zone_spans.get(z)
                if None in zero or sp is None or clause(sp[0]) in zero:
                    pq.zones[z] = CountConstraint("==", 0)
            del pq.objects["person"]
            positive = [sp for sp in pq._person_spans if None not in zero and clause(sp[0]) not in zero]
            if positive or any(c.op != "==" or c.n > 0 for c in pq.zones.values()):
                pq.objects["person"] = CountConstraint(">=", 1)
            return
        if person is None:
            pq.objects["person"] = CountConstraint(">=", 1)
        elif len(pq.zones) == 1 and person.op in ("==", ">=", "..") and person.n > 1:
            z = next(iter(pq.zones))
            pq.zones[z] = CountConstraint(">=", person.n)   # "hai người đứng ở bếp" -> bếp có >= 2 người

    def _extract_objects(self, s: str, pq: ParsedQuery, taken: list[tuple[int, int]] | None = None) -> None:
        taken = list(taken or [])
        if "person" in self.syn2cls.values():
            for pat, cons in ((RE_NO_OTHERS, CountConstraint("==", 1)),
                              (RE_PEOPLE_ZERO, CountConstraint("==", 0)),
                              (RE_PEOPLE_ONE, CountConstraint("==", 1)),
                              (RE_PEOPLE_MANY, CountConstraint(">=", 3))):
                for m in pat.finditer(s):
                    if not _overlaps(m.span(), taken):
                        pq.objects["person"] = cons
                        taken.append(m.span())
                        if cons.n == 0:
                            pq._zero_spans.append(m.span())

        exact_people: list[tuple[int, int, int]] = []        # (start, end, n) các cụm "N người" chính xác
        for m in self.re_obj.finditer(s):
            if _overlaps(m.span(), taken):
                continue
            cls = self.syn2cls[m.group(0)]
            before = s[max(0, m.start() - 40):m.start()]
            if re.search(r"(?<!\w)màu\s+$", before):
                continue                      # "khăn màu cam" là màu, không phải quả cam
            if cls == "person":
                pq._person_spans.append(m.span())
            after = s[m.end():m.end() + 15]
            cons = None
            if RE_NEG_BEFORE.search(before):
                cons = CountConstraint("==", 0)
            elif (r := RE_COUNT_RANGE_BEFORE.search(before)):
                a, b = sorted((_num(r["a"]), _num(r["b"])))
                cons = CountConstraint("..", a, b)
            elif (c := RE_COUNT_BEFORE.search(before)):
                n = _num(c["num"])
                mod = re.sub(r"\s+", " ", c["mod"] or "")
                if mod in _MOD_OP:
                    op, delta = _MOD_OP[mod][:2], int(_MOD_OP[mod][2:])
                    cons = CountConstraint(op, max(0, n + delta))
                elif n == 1:
                    # "MỘT người đàn ông mở tủ lạnh", "a man": mạo từ, không phải "chỉ 1 người".
                    # Muốn đúng 1 thì nói "chỉ có một người" / "một mình" / "only one".
                    cons = CountConstraint(">=", 1) if cls not in pq.objects else None
                    if cls == "person":
                        exact_people.append((m.start() - (len(before) - c.start()), m.end(), n))
                else:
                    cons = CountConstraint("==", n)
                    if cls == "person":
                        exact_people.append((m.start() - (len(before) - c.start()), m.end(), n))
                if (a := RE_COUNT_AFTER.search(after)):            # "ba người trở lên"
                    up = re.sub(r"\s+", " ", a["m"]) in ("trở lên", "or more", "and up", "+")
                    cons = CountConstraint(">=" if up else "<=", cons.n)
                    if exact_people and exact_people[-1][1] == m.end():
                        exact_people.pop()
            prev = pq.objects.get(cls)
            if cons is not None and prev is not None and cons.op in (">=", "<=") and prev.op in (">=", "<=") \
                    and cons.op != prev.op:
                lo, hi = (prev.n, cons.n) if prev.op == ">=" else (cons.n, prev.n)   # "ít nhất 2 ... không quá 4"
                pq.objects[cls] = CountConstraint("..", lo, hi) if lo <= hi else cons
            elif cons is not None or prev is None:
                pq.objects[cls] = cons or CountConstraint(">=", 1)

        # "một người áo đỏ VÀ hai người áo đen" -> tổng 3 người
        if len(exact_people) >= 2 and all(RE_JOIN.search(s[e1:s2]) for (_, e1, _), (s2, _, _)
                                          in zip(exact_people, exact_people[1:])):
            pq.objects["person"] = CountConstraint("==", sum(n for _, _, n in exact_people))

    # ------------------------------------------------------------------ cầm gì / tư thế
    def _extract_holding(self, s: str, pq: ParsedQuery) -> None:
        """'người cầm dao', 'holding a phone' -> holding[knife]=1 · 'đang cắt rau' -> holding[knife]=0.5 (ngầm định)."""
        if not self.holdable:
            return
        for m in self.re_obj.finditer(s):
            cls = self.syn2cls[m.group(0)]
            if cls in self.holdable and RE_HOLD_BEFORE.search(s[max(0, m.start() - 30):m.start()]):
                pq.holding[cls] = 1.0
        for pat, cls, w in RE_IMPLIED_HOLD:
            if cls in self.holdable and cls not in pq.holding and pat.search(s):
                pq.holding[cls] = w
        if pq.holding and "person" not in pq.objects and "person" in self.syn2cls.values():
            pq.objects["person"] = CountConstraint(">=", 1)

    def _extract_postures(self, s: str, pq: ParsedQuery) -> None:
        """'hai người đang ngồi' -> sitting>=2 · 'không ai ngồi' -> sitting==0 · 'người cúi xuống' -> bending>=1."""
        if not self.use_postures:
            return
        for name, pat in RE_POSTURE:
            if name not in self.postures_allowed:
                continue
            for m in pat.finditer(s):
                before = s[max(0, m.start() - 40):m.start()]
                if RE_NOBODY_BEFORE.search(before) or re.search(r"(?<!\w)(?:không|chẳng|chưa)\s+(?:có\s+)?ai\s+(?:đang\s+)?$", before):
                    pq.postures[name] = CountConstraint("==", 0)
                    break
                c = RE_NUM_PERSON_BEFORE.search(before)
                n = _num(c["num"]) if c else 1
                prev = pq.postures.get(name)
                pq.postures[name] = CountConstraint(">=", max(n, prev.n if prev else 0))
                break
        if any(not (c.op == "==" and c.n == 0) for c in pq.postures.values()) and "person" not in pq.objects \
                and "person" in self.syn2cls.values():
            pq.objects["person"] = CountConstraint(">=", 1)

    @staticmethod
    def _sub(pattern: re.Pattern, s: str, fn) -> str:
        def repl(m):
            fn(m)
            return " "
        return pattern.sub(repl, s)


def content_words(s: str) -> int:
    """Số từ có nội dung (bỏ đại từ, giới từ, 'lúc đó là'...): câu < 2 từ nội dung không đem đi so nghĩa."""
    return sum(1 for w in re.findall(r"\w+", s.lower()) if w not in STOPWORDS and not w.isdigit())


def _num(word: str) -> int:
    return int(word) if word.isdigit() else NUMBER_WORDS[re.sub(r"\s+", " ", word)]


def _clause_index(s: str):
    """-> hàm vị trí ký tự -> số thứ tự vế câu (ngăn bởi dấu phẩy, 'nhưng', 'còn', 'but'...)."""
    cuts = [m.end() for m in RE_CLAUSE_SEP.finditer(s) if m.group(0).strip()]
    return lambda pos: sum(1 for c in cuts if c <= pos)


def _overlaps(span, spans) -> bool:
    return any(span[0] < b and a < span[1] for a, b in spans)


def _merge(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    iv = sorted((a, b) for a, b in intervals if a is not None and b is not None and b > a)
    out: list[tuple[int, int]] = []
    for a, b in iv:
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out
