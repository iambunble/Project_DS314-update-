from __future__ import annotations

import logging
import pickle

import numpy as np

from .base import TextEncoder, l2norm

log = logging.getLogger(__name__)


class SentenceTransformerTextEncoder(TextEncoder):
    """Embedding đa ngôn ngữ (mặc định multilingual-e5-base): gõ tiếng Việt vẫn khớp mô tả tiếng Anh."""

    def __init__(self, model: str, query_prefix: str = "", passage_prefix: str = "", batch_size: int = 64,
                 device: str | None = None):
        from sentence_transformers import SentenceTransformer
        self.name = model
        self.model = SentenceTransformer(model, device=device)
        self.qp, self.pp, self.bs = query_prefix, passage_prefix, batch_size

    def encode_passages(self, texts):
        emb = self.model.encode([self.pp + t for t in texts], batch_size=self.bs, show_progress_bar=True,
                                normalize_embeddings=True, convert_to_numpy=True)
        return l2norm(emb)

    def encode_query(self, text):
        emb = self.model.encode([self.qp + text], normalize_embeddings=True, convert_to_numpy=True)
        return l2norm(emb)[0]


class TfidfTextEncoder(TextEncoder):
    """Phương án offline, không cần GPU/tải mô hình — chỉ tốt với truy vấn tiếng Anh. Dùng để test nhanh."""

    def __init__(self, **_):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.name = "tfidf"
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1, stop_words="english")

    def fit(self, texts):
        self.vec.fit(texts)

    def encode_passages(self, texts):
        return l2norm(self.vec.transform(texts).toarray())

    def encode_query(self, text):
        return l2norm(self.vec.transform([text]).toarray())[0]

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.vec, f)

    def load(self, path):
        with open(path, "rb") as f:
            self.vec = pickle.load(f)
