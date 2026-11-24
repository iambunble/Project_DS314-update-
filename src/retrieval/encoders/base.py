"""Giao diện chung cho bộ mã hoá. Muốn thử mô hình mới (BLIP-2, SigLIP, bge-m3...) chỉ cần viết
1 class mới theo giao diện này rồi đăng ký trong factory.py — engine không phải sửa."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


def l2norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(n, 1e-12, None)


class TextEncoder(ABC):
    name: str

    @abstractmethod
    def encode_passages(self, texts: list[str]) -> np.ndarray: ...

    @abstractmethod
    def encode_query(self, text: str) -> np.ndarray: ...

    def fit(self, texts: list[str]) -> None:  # chỉ encoder thống kê (tfidf) cần
        pass

    def save(self, path) -> None:
        pass

    def load(self, path) -> None:
        pass


class ImageTextEncoder(ABC):
    """Không gian chung ảnh–chữ (CLIP)."""
    name: str

    @abstractmethod
    def encode_images(self, paths: list[str]) -> np.ndarray: ...

    def encode_pil(self, images: list) -> np.ndarray:
        """Mã hoá ảnh PIL đã nạp sẵn (dùng cho ảnh cắt: từng người, vùng đảo bếp...)."""
        raise NotImplementedError

    @abstractmethod
    def encode_query(self, text: str) -> np.ndarray: ...
