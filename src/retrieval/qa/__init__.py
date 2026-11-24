"""Hỏi đáp offline trên bộ ảnh (không cần mạng)."""
from .answer import Answer, QAEngine, classify, strip_question

__all__ = ["Answer", "QAEngine", "classify", "strip_question"]
