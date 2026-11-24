"""Dịch câu truy vấn tiếng Việt -> tiếng Anh trước khi so với ảnh (Cross-lingual IR: dịch truy vấn).

Vì sao: CLIP (kể cả bản XLM-R đa ngôn ngữ) học chủ yếu từ chú thích tiếng Anh, còn 5000 mô tả ảnh cũng là tiếng Anh.
Đo trên đề thi 23/11: câu tiếng Việt dài làm CLIP xếp ảnh đúng ở hạng 900–3000.

Cách làm:
  1. Bảng thuật ngữ của căn bếp (GLOSSARY): chèn từ tiếng Anh vào sau cụm tiếng Việt khó, vd "thớt" -> "thớt (cutting board)".
     Model dịch giữ nguyên chữ trong ngoặc nên không dịch sai "thớt" thành "noodle basket", "khăn caro" thành "carolyn".
  2. Dịch từng câu bằng VietAI/envit5-translation (T5, chạy CPU ~1 giây/câu).
  3. Cache theo từng câu: đề thi cho thêm 1 câu gợi ý mỗi 30 giây -> mỗi lần chỉ phải dịch câu MỚI.

Không có model (chưa tải / tắt trong config) -> trả về None, hệ thống chạy như cũ với câu tiếng Việt.
"""
from __future__ import annotations

import logging
import re
import threading

log = logging.getLogger(__name__)

# (mẫu tiếng Việt, từ tiếng Anh chèn vào). Thứ tự: cụm dài trước.
GLOSSARY: list[tuple[str, str]] = [
    (r"áo (?:thể thao|jersey(?: thể thao)?|đấu|bóng đá)", "sports jersey"),
    (r"camera (?:trên|ở) (?:trán|đầu)|máy quay (?:trên|ở) (?:trán|đầu)|đèn (?:trên|ở) (?:trán|đầu)", "head-mounted camera"),
    (r"mũ lưỡi trai|nón lưỡi trai|nón kết", "baseball cap"),
    (r"(?:khăn|vải|chăn) (?:kẻ )?(?:caro|kẻ ô|ca rô)", "plaid blanket"),
    (r"(?:áo )?(?:sơ mi )?(?:kẻ )?(?:caro|ca rô|kẻ ô)", "plaid"),
    (r"áo len sọc ngang", "horizontal striped sweater"),
    (r"sọc ngang", "horizontal stripes"),
    (r"áo len", "sweater"),
    (r"áo hoodie|áo nỉ có mũ", "hoodie"),
    (r"áo khoác", "jacket"),
    (r"áo sơ mi", "shirt"),
    (r"áo thun|áo phông", "t-shirt"),
    (r"tạp dề", "apron"),
    (r"thớt", "cutting board"),
    (r"đảo bếp", "kitchen island"),
    (r"bếp nấu|bếp ga|bếp từ", "stove"),
    (r"bồn rửa|chậu rửa", "sink"),
    (r"vòi nước", "faucet"),
    (r"phòng khách", "living room"),
    (r"thùng rác", "trash bins"),
    (r"tủ lạnh", "fridge"),
    (r"máy pha cà phê", "coffee machine"),
    (r"(?<!bánh )(?:màu )?kem(?! đánh răng)", "cream-colored"),
    (r"tóc xoăn|tóc quăn", "curly hair"),
    (r"đầu hói|hói đầu", "bald"),
    (r"đeo kính", "wearing glasses"),
    (r"quay lưng(?: về phía| lại)?(?: camera)?", "back to the camera"),
]
_RE_GLOSS = [(re.compile(r"(?<!\w)(?:" + p + r")(?!\w)", re.I), en) for p, en in GLOSSARY]
# lỗi dịch hay gặp của model -> sửa sau khi dịch
_POST = [(re.compile(r"\bcarolyn\b|\bcaro\b", re.I), "plaid"), (re.compile(r"\bchopping block\b", re.I), "cutting board"),
         (re.compile(r"\bback room\b", re.I), "living room")]


def add_glossary(vi: str) -> str:
    """'đang cầm thớt đỏ' -> 'đang cầm thớt (cutting board) đỏ'. Mỗi đoạn chỉ chèn 1 lần, không chèn chồng."""
    taken: list[tuple[int, int]] = []
    ins: list[tuple[int, str]] = []
    for pat, en in _RE_GLOSS:
        for m in pat.finditer(vi):
            if any(m.start() < b and a < m.end() for a, b in taken):
                continue
            taken.append(m.span())
            ins.append((m.end(), f" ({en})"))
    for pos, s in sorted(ins, reverse=True):
        vi = vi[:pos] + s + vi[pos:]
    return vi


class QueryTranslator:
    """Dịch VI -> EN có cache. translate(list câu) -> list câu tiếng Anh (câu đã là tiếng Anh giữ nguyên)."""

    def __init__(self, model: str = "VietAI/envit5-translation", max_new_tokens: int = 96, threads: int | None = None,
                 quantize: bool = True):
        self.model_name, self.max_new_tokens = model, max_new_tokens
        self.threads, self.quantize = threads, quantize
        self._tok = self._model = None
        self._cache: dict[str, str] = {}
        self._lock = threading.Lock()
        self.failed: str | None = None

    @classmethod
    def from_config(cls, cfg: dict) -> "QueryTranslator | None":
        t = (cfg.get("translation") or {})
        if not t.get("enabled", False):
            return None
        return cls(t.get("model", "VietAI/envit5-translation"), int(t.get("max_new_tokens", 96)),
                   t.get("threads"), bool(t.get("quantize", True)))

    def _load(self) -> bool:
        if self._model is not None or self.failed:
            return self._model is not None
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            if self.threads:
                torch.set_num_threads(int(self.threads))
            self._tok = AutoTokenizer.from_pretrained(self.model_name)
            m = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).eval()
            if self.quantize:            # int8 cho các lớp Linear: nhanh ~1.5–2 lần trên CPU, chất lượng gần như không đổi
                m = torch.quantization.quantize_dynamic(m, {torch.nn.Linear}, dtype=torch.qint8)
            self._model = m
            return True
        except Exception as e:  # noqa: BLE001 — thiếu gói/model: chạy tiếp bằng tiếng Việt
            self.failed = f"Không tải được model dịch {self.model_name}: {e}"
            log.warning(self.failed)
            return False

    def translate(self, sentences: list[str]) -> list[str] | None:
        from .parser import VI_CHARS
        todo = [s for s in dict.fromkeys(sentences) if s not in self._cache and VI_CHARS.search(s)]
        if todo:
            with self._lock:
                todo = [s for s in todo if s not in self._cache]
                if todo and not self._load():
                    return None
                if todo:
                    import torch
                    batch = self._tok(["vi: " + add_glossary(s) for s in todo], return_tensors="pt", padding=True)
                    with torch.no_grad():
                        out = self._model.generate(**batch, max_new_tokens=self.max_new_tokens, num_beams=1)
                    for s, en in zip(todo, self._tok.batch_decode(out, skip_special_tokens=True)):
                        en = re.sub(r"^\s*en\s*:\s*", "", en).strip()
                        for pat, rep in _POST:
                            en = pat.sub(rep, en)
                        self._cache[s] = en
        return [self._cache.get(s, s) for s in sentences]

    def warmup(self) -> None:
        """Gọi khi mở giao diện để lần tìm đầu tiên không phải chờ tải model."""
        self.translate(["một người đứng trong bếp"])
