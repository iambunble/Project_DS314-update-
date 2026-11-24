from __future__ import annotations

from .base import ImageTextEncoder, TextEncoder


def text_encoder_name(cfg: dict) -> str:
    c = cfg["encoders"]["text"]
    return "tfidf" if c["backend"] == "tfidf" else c["model"]


def clip_encoder_name(cfg: dict) -> str:
    """Tên dùng để kiểm tra chỉ mục đã build có khớp model trong config không (không cần nạp model)."""
    c = cfg["encoders"]["clip"]
    if c.get("backend", "sentence_transformers") == "open_clip":
        return f"open_clip:{c['model']}:{c.get('pretrained', '')}"
    return f"{c['image_model']}|{c['text_model']}"


def make_text_encoder(cfg: dict) -> TextEncoder:
    c = cfg["encoders"]["text"]
    if c["backend"] == "tfidf":
        from .text_encoders import TfidfTextEncoder
        return TfidfTextEncoder()
    if c["backend"] == "sentence_transformers":
        from .text_encoders import SentenceTransformerTextEncoder
        return SentenceTransformerTextEncoder(c["model"], c.get("query_prefix", ""), c.get("passage_prefix", ""),
                                              c.get("batch_size", 64), c.get("device"))
    raise ValueError(f"Unknown text backend: {c['backend']}")


def make_clip_encoder(cfg: dict, load_image_model: bool = True) -> ImageTextEncoder:
    c = cfg["encoders"]["clip"]
    if c.get("backend", "sentence_transformers") == "open_clip":
        from .clip_encoder import OpenClipEncoder
        return OpenClipEncoder(c["model"], c.get("pretrained", ""), c.get("batch_size", 32), c.get("device"))
    from .clip_encoder import SentenceTransformerClip
    return SentenceTransformerClip(c["image_model"], c["text_model"], c.get("batch_size", 32), c.get("device"),
                                   load_image_model=load_image_model)
