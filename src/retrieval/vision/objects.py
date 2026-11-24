"""[PA2] Nhận diện đồ vật NGOÀI khả năng của YOLO COCO bằng YOLOE (detector "gõ chữ" — open-vocabulary).

Vì sao cần: YOLO11 chỉ biết 80 lớp COCO — không có nồi, chảo, thớt, đĩa; còn dao / điện thoại thì có lớp nhưng quá
nhỏ khi nhìn từ camera trên cao nên gần như không bắt được (0 lần trên 5000 ảnh).

Cách làm:
  - YOLOE nhận tên lớp bằng chữ ("cooking pot", "cutting board"...). Embedding của các tên này được TÍNH SẴN
    (weights/yoloe_text_emb.pt) -> máy chạy build không cần gói `clip` hay model chữ MobileCLIP (254 MB).
    Đổi danh sách lớp trong config mà không khớp file tính sẵn -> YOLOE tự tính (cần
    `pip install git+https://github.com/ultralytics/CLIP.git`, tải thêm model chữ lần đầu).
  - Chỉ chạy trên VÙNG LÀM VIỆC (đảo bếp + bếp nấu + bồn rửa) cắt ra rồi phóng to (imgsz 960) -> vật nhỏ to ra,
    bắt được nhiều hơn hẳn so với chạy cả ảnh 640 (thử trên 48 ảnh: thớt 3 -> 23, dao 1 -> 7, nồi 30 -> 55).

Kết quả: artifacts/objects.json = {"model", "classes", "dets": [[{"c", "s", "b"}, ...] mỗi ảnh]}
  c = tên lớp chuẩn (knife, pot, frying pan, cutting board, plate, cup, cell phone), s = độ tin cậy,
  b = [x1, y1, x2, y2] tỉ lệ 0..1 theo CẢ ẢNH (đã đổi toạ độ từ ảnh cắt về).
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

log = logging.getLogger(__name__)

# tên chuẩn (dùng trong hệ thống, trùng tên lớp COCO nếu có) -> câu gõ cho YOLOE
DEFAULT_CLASSES: dict[str, str] = {
    "knife": "knife",
    "pot": "cooking pot",
    "frying pan": "frying pan",
    "cutting board": "cutting board",
    "plate": "plate",
    "cup": "cup",
    "cell phone": "mobile phone",
}


def objects_name(cfg: dict) -> str:
    """Chuỗi định danh bước build (đổi model / lớp / vùng cắt -> build lại)."""
    oc = (cfg.get("objects") or {}).get("yoloe") or {}
    classes = oc.get("classes") or DEFAULT_CLASSES
    crop = ",".join(f"{v:.3f}" for v in oc.get("crop", [0.20, 0.08, 0.80, 0.66]))
    return (f"{Path(oc.get('model', 'yoloe-26s-seg.pt')).name}@{oc.get('imgsz', 960)}|crop={crop}|"
            f"conf={oc.get('conf', 0.15)}|" + ";".join(f"{k}={v}" for k, v in classes.items()))


class YoloeObjectDetector:
    def __init__(self, model: str, classes: dict[str, str], text_emb: str | None = None, imgsz: int = 960,
                 conf: float = 0.15, iou: float = 0.5, crop=(0.20, 0.08, 0.80, 0.66), device: str | None = None,
                 batch: int = 8):
        from ultralytics import YOLOE
        self.model = YOLOE(model)
        self.names = list(classes)                 # tên chuẩn
        prompts = [classes[k] for k in self.names]  # câu gõ cho YOLOE
        pe = None
        if text_emb and Path(text_emb).exists():
            import torch
            saved = torch.load(text_emb, map_location="cpu")
            if list(saved.get("prompts", [])) == prompts:
                pe = saved["pe"]
            else:
                log.warning("[objects] %s không khớp danh sách lớp trong config -> YOLOE tự tính embedding chữ "
                            "(cần gói clip + tải model chữ lần đầu)", text_emb)
        self.model.set_classes(prompts, pe) if pe is not None else self.model.set_classes(prompts)
        self.kw = dict(conf=conf, iou=iou, imgsz=imgsz, verbose=False)
        if device:
            self.kw["device"] = device
        self.crop = tuple(float(v) for v in crop)
        self.batch = batch

    def run(self, paths: list[str | None]) -> list[list[dict]]:
        from PIL import Image
        out: list[list[dict]] = [[] for _ in paths]
        todo = [(i, p) for i, p in enumerate(paths) if p]
        cx1, cy1, cx2, cy2 = self.crop
        t0 = time.time()
        for k in range(0, len(todo), self.batch):
            chunk = todo[k:k + self.batch]
            ims, sizes = [], []
            for _, p in chunk:
                with Image.open(p) as im:
                    im = im.convert("RGB")
                    W, H = im.size
                    sizes.append((W, H))
                    ims.append(im.crop((int(cx1 * W), int(cy1 * H), int(cx2 * W), int(cy2 * H))))
            results = self.model.predict(ims, **self.kw)
            for (i, _), (W, H), r in zip(chunk, sizes, results):
                out[i] = self._convert(r, W, H)
            done = k + len(chunk)
            if (k // self.batch) % 25 == 0 or done == len(todo):
                el = time.time() - t0
                log.info("YOLOE: %d/%d ảnh (%.0fs, còn ~%.0fs)", done, len(todo), el, el / done * (len(todo) - done))
        return out

    def _convert(self, r, W: int, H: int) -> list[dict]:
        if r.boxes is None or len(r.boxes) == 0:
            return []
        ox, oy = self.crop[0] * W, self.crop[1] * H
        dets = []
        for b, c, s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int),
                           r.boxes.conf.cpu().numpy()):
            dets.append({"c": self.names[int(c)], "s": round(float(s), 3),
                         "b": [round(float((b[0] + ox) / W), 4), round(float((b[1] + oy) / H), 4),
                               round(float((b[2] + ox) / W), 4), round(float((b[3] + oy) / H), 4)]})
        return dets
