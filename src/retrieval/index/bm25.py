"""BM25 trên mô tả + tên đồ vật: bắt chính xác từ khoá hiếm (pineapple, Samsung, red tablecloth...)
mà embedding đôi khi bỏ qua. Build trong <1s với 5000 dòng nên không cần lưu ra đĩa."""
from __future__ import annotations

import re

import numpy as np
from rank_bm25 import BM25Okapi

from ..data.catalog import Catalog

_STOP = set("a an the is are was were be of in on at to and or with for by from this that it its there".split())


def tokenize(text: str) -> list[str]:
    toks = re.findall(r"[a-z0-9]+", text.lower())
    out = []
    for t in toks:
        if t in _STOP:
            continue
        if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]  # stem thô: chairs -> chair
        out.append(t)
    return out


class BM25Index:
    def __init__(self, docs: list[str]):
        self.bm25 = BM25Okapi([tokenize(d) for d in docs])

    @classmethod
    def from_catalog(cls, catalog: Catalog, detections: list[list[dict]] | None = None,
                     count_conf: float = 0.35, clean: bool = False) -> "BM25Index":
        """Văn bản mỗi ảnh = mô tả (bỏ câu nền nếu clean) + tên đồ vật (CSV) + đồ vật YOLO + "<màu> shirt"."""
        from ..scene.text_clean import content_text
        docs = []
        for i, (d, c) in enumerate(zip(catalog.df["description"].fillna(""), catalog.df["classes"])):
            if clean:
                d = content_text(d)
            extra = list(c)
            if detections is not None:
                for det in detections[i]:
                    if det["s"] >= count_conf:
                        extra.append(det["c"])
                        extra += [f"{col} shirt" for col in det.get("col", [])[:1]]
            docs.append(f"{d} {' '.join(extra)}")
        return cls(docs)

    def scores(self, query: str) -> np.ndarray:
        toks = tokenize(query)
        if not toks:
            return np.zeros(len(self.bm25.doc_len), np.float32)
        return self.bm25.get_scores(toks).astype(np.float32)
