from __future__ import annotations

import logging
import time

import numpy as np

from .base import ImageTextEncoder, l2norm

log = logging.getLogger(__name__)


class SentenceTransformerClip(ImageTextEncoder):
    """CLIP ViT-B/32 cho ảnh + text encoder đa ngôn ngữ (đã distill vào cùng không gian) cho câu truy vấn."""

    def __init__(self, image_model: str = "clip-ViT-B-32",
                 text_model: str = "sentence-transformers/clip-ViT-B-32-multilingual-v1",
                 batch_size: int = 32, device: str | None = None, load_image_model: bool = True):
        from sentence_transformers import SentenceTransformer
        self.name = f"{image_model}|{text_model}"
        self.bs = batch_size
        self.device = device
        self._image_model_name = image_model
        self._img = SentenceTransformer(image_model, device=device) if load_image_model else None
        self._txt = SentenceTransformer(text_model, device=device)

    @property
    def img_model(self):
        if self._img is None:  # chỉ nạp khi cần (vd tìm ảnh tương tự bằng ảnh tải lên)
            from sentence_transformers import SentenceTransformer
            self._img = SentenceTransformer(self._image_model_name, device=self.device)
        return self._img

    def encode_images(self, paths):
        from PIL import Image
        out = []
        for i in range(0, len(paths), self.bs):
            batch = []
            for p in paths[i:i + self.bs]:
                with Image.open(p) as im:
                    batch.append(im.convert("RGB"))
            out.append(self.img_model.encode(batch, convert_to_numpy=True, normalize_embeddings=True))
            if (i // self.bs) % 20 == 0:
                log.info("CLIP: %d/%d ảnh", min(i + self.bs, len(paths)), len(paths))
        return l2norm(np.concatenate(out)) if out else np.zeros((0, 512), np.float32)

    def encode_pil(self, images):
        if not images:
            return np.zeros((0, 512), np.float32)
        return l2norm(self.img_model.encode(list(images), batch_size=self.bs, convert_to_numpy=True,
                                            normalize_embeddings=True))

    def encode_query(self, text):
        return l2norm(self._txt.encode([text], convert_to_numpy=True, normalize_embeddings=True))[0]


class OpenClipEncoder(ImageTextEncoder):
    """open_clip — mặc định xlm-roberta-base-ViT-B-32 (LAION-5B): bộ mã hoá chữ XLM-R hiểu tiếng Việt
    tốt hơn nhiều so với bản rút gọn của sentence-transformers. Đổi model trong config là dùng được
    các model mạnh hơn (vd ViT-B-16-SigLIP-i18n-256 nếu có GPU)."""

    def __init__(self, model: str = "xlm-roberta-base-ViT-B-32", pretrained: str = "laion5b_s13b_b90k",
                 batch_size: int = 32, device: str | None = None, **_):
        import open_clip
        import torch
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.name = f"open_clip:{model}:{pretrained}"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model, pretrained=pretrained or None, device=self.device)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model)
        self.bs = batch_size

    def encode_images(self, paths):
        from PIL import Image
        out = []
        t0 = time.time()
        with self.torch.no_grad():
            for i in range(0, len(paths), self.bs):
                ims = []
                for p in paths[i:i + self.bs]:
                    with Image.open(p) as im:
                        ims.append(self.preprocess(im.convert("RGB")))
                x = self.torch.stack(ims).to(self.device)
                out.append(self.model.encode_image(x).float().cpu().numpy())
                done = min(i + self.bs, len(paths))
                if (i // self.bs) % 10 == 0 or done == len(paths):
                    el = time.time() - t0
                    log.info("CLIP: %d/%d ảnh (%.0fs, còn ~%.0fs)", done, len(paths), el,
                             el / done * (len(paths) - done))
        return l2norm(np.concatenate(out)) if out else np.zeros((0, 512), np.float32)

    def encode_pil(self, images):
        if not images:
            return np.zeros((0, 512), np.float32)
        out = []
        with self.torch.no_grad():
            for i in range(0, len(images), self.bs):
                x = self.torch.stack([self.preprocess(im.convert("RGB")) for im in images[i:i + self.bs]])
                out.append(self.model.encode_image(x.to(self.device)).float().cpu().numpy())
        return l2norm(np.concatenate(out))

    def encode_query(self, text):
        with self.torch.no_grad():
            tok = self.tokenizer([text]).to(self.device)
            return l2norm(self.model.encode_text(tok).float().cpu().numpy())[0]
