"""Từ điển Việt/Anh -> trường metadata. Thêm dữ liệu mới (phòng khác, đồ vật khác) chỉ cần sửa file này."""

# Từ đồng nghĩa -> lớp COCO (đúng tên trong cột `class`). Ghép dài trước, ngắn sau.
OBJECT_SYNONYMS: dict[str, list[str]] = {
    "person": ["người", "ai đó", "mọi người", "cô gái", "chàng trai", "đàn ông", "phụ nữ", "người đàn ông",
               "người phụ nữ", "cô ấy", "anh ấy", "bạn nam", "bạn nữ",
               "person", "persons", "people", "man", "men", "woman", "women", "individual", "individuals",
               "someone", "guy", "girl", "boy", "lady"],
    "chair": ["ghế", "cái ghế", "chiếc ghế", "ghế đẩu", "chair", "chairs", "stool", "stools"],
    "dining table": ["bàn ăn", "bàn", "cái bàn", "chiếc bàn", "dining table", "table", "tables"],
    "bowl": ["bát", "tô", "chén", "cái bát", "bowl", "bowls"],
    "cup": ["cốc", "ly", "tách", "cái cốc", "cái ly", "cup", "cups", "mug", "mugs"],
    "bottle": ["chai", "cái chai", "chai nước", "bottle", "bottles"],
    "oven": ["lò nướng", "lò", "bếp lò", "oven", "ovens", "stove"],
    "sink": ["bồn rửa", "chậu rửa", "bồn rửa bát", "sink", "sinks"],
    "couch": ["ghế sofa", "sofa", "ghế bành", "couch"],
    "tv": ["tivi", "ti vi", "tv", "television"],
    "laptop": ["laptop", "máy tính xách tay", "máy tính"],
    "cell phone": ["điện thoại", "điện thoại di động", "smartphone", "cell phone", "phone", "mobile phone"],
    "teddy bear": ["gấu bông", "teddy bear"],
    "handbag": ["túi xách", "handbag", "purse"],
    "wine glass": ["ly rượu", "ly vang", "wine glass"],
    "bench": ["ghế dài", "băng ghế", "bench"],
    "frisbee": ["đĩa bay", "frisbee"],
}

# Địa điểm (giá trị cột `owner`) — chỉ áp dụng nếu giá trị có thật trong CSV
LOCATION_SYNONYMS: dict[str, list[str]] = {
    "Kitchen": ["nhà bếp", "phòng bếp", "gian bếp", "căn bếp", "trong bếp", "kitchen"],
    "Living room": ["phòng khách", "living room", "livingroom"],
    "Bedroom": ["phòng ngủ", "bedroom"],
    "Bathroom": ["phòng tắm", "nhà vệ sinh", "bathroom", "toilet"],
    "Office": ["văn phòng", "phòng làm việc", "office"],
    "Garden": ["sân vườn", "khu vườn", "garden", "yard"],
}

NUMBER_WORDS: dict[str, int] = {
    "không": 0, "một": 1, "hai": 2, "ba": 3, "bốn": 4, "tư": 4, "năm": 5, "sáu": 6, "bảy": 7, "tám": 8,
    "chín": 9, "mười": 10,
    "zero": 0, "one": 1, "a single": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "a couple of": 2, "a pair of": 2,
}

# Khoảng giờ của buổi: (giờ bắt đầu, giờ kết thúc — không gồm)
PERIODS: dict[str, tuple[int, int]] = {
    "sáng sớm": (5, 8), "sáng": (5, 11), "trưa": (11, 13), "chiều": (13, 18), "tối": (18, 23), "đêm": (21, 24),
    "early morning": (5, 8), "morning": (5, 12), "noon": (11, 13), "midday": (11, 13), "lunchtime": (11, 13),
    "afternoon": (12, 18), "evening": (17, 22), "night": (20, 24),
}

# Cụm "lấp đầy" bỏ đi khỏi câu trước khi embed (không mang nghĩa nội dung ảnh)
FILLER_PATTERNS = [
    r"(hãy\s+)?(tìm|tìm kiếm|kiếm|cho tôi xem|cho mình xem|cho xem|hiển thị|lấy)(\s+(giúp|cho)\s+(tôi|mình))?",
    r"(những|các|mấy)?\s*(bức|tấm)?\s*(ảnh|hình ảnh|hình|khung hình)(\s+(có|về|chụp|mà))?",
    r"\b(find|search( for)?|show( me)?|give me|get|look for)\b",
    r"\b(images?|photos?|pictures?|frames?|scenes?)( of| with| where| showing| that)?\b",
]

# ---------------------------------------------------------------- đồ vật YOLO (COCO-80) bổ sung
# Chỉ những lớp có trong dữ liệu (CSV hoặc kết quả YOLO) mới được dùng khi phân tích câu.
OBJECT_SYNONYMS_EXTRA: dict[str, list[str]] = {
    "refrigerator": ["tủ lạnh", "fridge", "refrigerator"],
    "microwave": ["lò vi sóng", "lò vi ba", "microwave"],
    "toaster": ["máy nướng bánh mì", "toaster"],
    "knife": ["con dao", "dao", "knife", "knives"],
    "fork": ["cái nĩa", "nĩa", "fork", "forks"],
    "spoon": ["cái thìa", "thìa", "muỗng", "spoon", "spoons"],
    "banana": ["quả chuối", "chuối", "banana", "bananas"],
    "apple": ["quả táo", "táo", "apple", "apples"],
    "orange": ["quả cam", "trái cam", "cam", "orange", "oranges"],
    "broccoli": ["bông cải xanh", "súp lơ xanh", "bông cải", "broccoli"],
    "carrot": ["cà rốt", "carrot", "carrots"],
    "sandwich": ["bánh mì kẹp", "sandwich", "sandwiches"],
    "pizza": ["pizza", "bánh pizza"],
    "donut": ["bánh donut", "bánh vòng", "donut", "doughnut", "donuts"],
    "cake": ["bánh kem", "bánh ngọt", "bánh sinh nhật", "cake", "cakes"],
    "hot dog": ["xúc xích", "hot dog", "hotdog"],
    "potted plant": ["chậu cây", "cây cảnh", "chậu hoa", "potted plant", "plant", "plants"],
    "vase": ["bình hoa", "lọ hoa", "vase"],
    "clock": ["đồng hồ treo tường", "đồng hồ", "clock"],
    "book": ["quyển sách", "cuốn sách", "sách", "book", "books"],
    "scissors": ["cái kéo", "chiếc kéo", "scissors"],
    "backpack": ["ba lô", "balo", "backpack"],
    "umbrella": ["cái ô", "chiếc ô", "cây dù", "umbrella"],
    "suitcase": ["vali", "va li", "suitcase"],
    "remote": ["điều khiển từ xa", "remote"],
    "keyboard": ["bàn phím", "keyboard"],
    "mouse": ["chuột máy tính", "computer mouse"],
    "sports ball": ["quả bóng", "trái bóng", "ball"],
    "tie": ["cà vạt", "necktie"],
    "dog": ["con chó", "chú chó", "chó", "dog"],
    "cat": ["con mèo", "chú mèo", "mèo", "cat"],
    "bed": ["giường", "bed"],
    "toilet": ["bồn cầu", "toilet bowl"],
    "bicycle": ["xe đạp", "bicycle", "bike"],
}

# ---------------------------------------------------------------- đồ vật YOLOE / CLIP theo vùng (nâng cấp nhận diện)
# Chỉ được dùng khi đã bật objects.yoloe hoặc objects.clip_tags (engine báo lớp nào có dữ liệu).
# "dĩa" theo cách nói miền Nam = đĩa (bản cũ xếp vào nĩa; bộ ảnh không có nĩa nào được nhận ra).
OBJECT_SYNONYMS_VISION: dict[str, list[str]] = {
    "pot": ["cái nồi", "chiếc nồi", "nồi canh", "nồi súp", "nồi", "xoong", "soong", "xoang",
            "cooking pot", "saucepan", "pots", "pot"],
    "frying pan": ["cái chảo", "chiếc chảo", "chảo chiên", "chảo rán", "chảo", "frying pan", "skillet", "wok", "pan",
                   "pans"],
    "cutting board": ["cái thớt", "tấm thớt", "thớt", "cutting board", "chopping board", "cutting boards"],
    "plate": ["cái đĩa", "chiếc đĩa", "đĩa thức ăn", "đĩa", "cái dĩa", "dĩa", "plate", "plates"],
}

# "cầm / đang cầm / holding" + (lượng từ) + đồ vật  -> người đang cầm vật đó
HOLD_BEFORE = (r"(?:cầm|cầm\s+theo|tay\s+cầm|nắm|giữ|bưng|xách|holding|holds?|carrying|carries|with\s+(?:a|an)\s+"
               r"(?=knife|phone|cell|mobile|cup|mug|plate))\s+(?:(?:một|1|cái|chiếc|con|tấm|a|an|the|his|her|their)\s+)"
               r"{0,2}$")
# hành động ngầm định đang cầm vật (trọng số thấp hơn cầm rõ ràng)
IMPLIED_HOLD: dict[str, tuple[str, float]] = {
    r"(?<!\w)(?:cắt|thái|băm|chặt|gọt|lạng|cutting|chopping|slicing|dicing|peeling)(?!\w)": ("knife", 0.5),
    r"(?:gọi\s+điện|nghe\s+điện\s+thoại|bấm\s+điện\s+thoại|lướt\s+điện\s+thoại|xem\s+điện\s+thoại|nhắn\s+tin|"
    r"on\s+the\s+phone|texting|scrolling\s+(?:on\s+)?(?:a|the|his|her)?\s*phone|looking\s+at\s+(?:a|the|his|her)\s+phone)":
        ("cell phone", 0.7),
}

# tư thế (cần bật pose)
POSTURE_WORDS: dict[str, str] = {
    "sitting": r"(?<!\w)(?:ngồi|đang\s+ngồi|sitting|sits|seated)(?!\w)",
    "standing": r"(?<!\w)(?:đứng|đang\s+đứng|standing|stands)(?!\w)",
    "bending": r"(?<!\w)(?:cúi\s+người|cúi\s+xuống|cúi|khom\s+người|khom|bending(?:\s+over)?|bent\s+over|"
               r"leaning\s+over|stooping|crouching)(?!\w)",
}

# ---------------------------------------------------------------- vùng trong căn bếp (camera cố định)
# Tên vùng phải trùng với scene.zones trong config.yaml (ở đó có toạ độ).
#   words : khớp ở bất kỳ đâu trong câu
#   gated : từ chung chung, chỉ tính là vùng khi đứng ngay sau giới từ vị trí ("ở bếp", "cạnh bàn", "at the table")
#           -> "trong bếp" / "nhà bếp" vẫn là cả căn bếp, không phải khu bếp nấu.
ZONE_SYNONYMS: dict[str, dict[str, list[str]]] = {
    "stove": {"words": ["bếp nấu", "bếp ga", "bếp gas", "bếp từ", "bếp điện", "bếp lò", "lò nướng", "khu bếp nấu",
                        "stove", "stovetop", "cooktop", "range cooker", "oven"],
              "gated": ["bếp", "lò"]},
    "sink": {"words": ["bồn rửa bát", "bồn rửa chén", "bồn rửa", "chậu rửa bát", "chậu rửa", "vòi nước",
                       "rửa bát", "rửa chén", "rửa đĩa", "sink", "washing dishes", "washing the dishes", "wash dishes",
                       "doing the dishes", "doing dishes"],
             "gated": []},
    "island": {"words": ["đảo bếp", "bàn đảo", "quầy bếp", "quầy giữa", "kitchen island", "island"],
               "gated": ["quầy", "counter"]},
    "dining": {"words": ["bàn ăn", "ghế ăn", "dining table", "dining area", "dining chairs"],
               "gated": ["bàn", "ghế", "table", "chair", "chairs", "stool", "stools"]},
    "living": {"words": ["phòng khách", "ghế sofa", "sofa", "ghế bành", "living room", "living area", "lounge",
                         "couch"],
               "gated": []},
    "side_counter": {"words": ["quầy bên trái", "góc pha đồ uống", "máy pha cà phê", "ấm đun nước", "ấm siêu tốc",
                               "coffee maker", "coffee machine", "kettle", "side counter"],
                     "gated": []},
    "foreground": {"words": ["gần camera", "sát camera", "trước camera", "gần máy quay", "tiền cảnh",
                             "near the camera", "close to the camera", "in the foreground", "foreground"],
                   "gated": []},
    "walkway": {"words": ["lối đi", "walkway", "aisle"], "gated": []},
    # tủ lạnh đen Samsung ở góc phải, gần camera
    "fridge": {"words": ["tủ lạnh", "tủ đông", "fridge", "refrigerator", "freezer"], "gated": []},
}
# Giới từ vị trí đứng ngay trước từ "gated"
ZONE_GATE = (r"(?:ở|tại|cạnh|gần|bên|bên\s+cạnh|quanh|trước|sau|đứng|ngồi|đứng\s+ở|ngồi\s+ở|ngồi\s+tại|vào|"
             r"at|near|by|beside|around|next\s+to|in\s+front\s+of|behind)")

# ---------------------------------------------------------------- màu quần áo
# tên màu chuẩn: black white gray red orange yellow green blue purple pink brown
COLOR_SYNONYMS: dict[str, set[str]] = {
    "xanh dương": {"blue"}, "xanh lam": {"blue"}, "xanh da trời": {"blue"}, "xanh nước biển": {"blue"},
    "xanh biển": {"blue"}, "xanh navy": {"blue"}, "xanh than": {"blue"},
    "xanh lá cây": {"green"}, "xanh lá": {"green"}, "xanh lục": {"green"}, "xanh rêu": {"green"},
    "xanh": {"blue", "green"},
    "đen": {"black"}, "trắng": {"white"}, "xám": {"gray"}, "ghi": {"gray"}, "bạc": {"gray"},
    "đỏ": {"red"}, "đỏ đô": {"red"}, "cam": {"orange"}, "vàng": {"yellow"}, "tím": {"purple"},
    "hồng": {"pink"}, "nâu": {"brown"}, "be": {"brown", "white"}, "kem": {"brown", "white"}, "màu kem": {"brown", "white"},
    "black": {"black"}, "white": {"white"}, "gray": {"gray"}, "grey": {"gray"}, "silver": {"gray"},
    "red": {"red"}, "maroon": {"red"}, "orange": {"orange"}, "yellow": {"yellow"}, "green": {"green"},
    "blue": {"blue"}, "navy": {"blue"}, "purple": {"purple"}, "violet": {"purple"}, "pink": {"pink"},
    "brown": {"brown"}, "beige": {"brown", "white"}, "cream": {"brown", "white"},
}

for _cls, _words in list(OBJECT_SYNONYMS_VISION.items()) + list(OBJECT_SYNONYMS_EXTRA.items()):
    OBJECT_SYNONYMS.setdefault(_cls, [])
    OBJECT_SYNONYMS[_cls] = OBJECT_SYNONYMS[_cls] + [w for w in _words if w not in OBJECT_SYNONYMS[_cls]]
