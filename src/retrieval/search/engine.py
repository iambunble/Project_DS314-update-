"""Bộ máy tìm kiếm lai (hybrid), đã chỉnh theo đặc điểm bộ ảnh: 1 camera cố định quay 1 căn bếp cả ngày.

  query ─► QueryParser ─► (text, bộ lọc cứng, đồ vật, màu áo, VÙNG trong bếp)
                │
                ├─ lọc cứng metadata (giờ, ngày, địa điểm)  -> mask ứng viên
                ├─ điểm ngữ nghĩa (chuẩn hoá min-max trên tập ứng viên rồi cộng có trọng số):
                │     text   : mô tả đã bỏ câu nền -> 0.5·cả đoạn + 0.5·câu khớp nhất
                │     clip   : CLIP toàn ảnh
                │     region : CLIP trên vùng cắt (đảo bếp / bếp nấu / phòng khách) — vùng khớp nhất
                │     person : CLIP trên từng người — người khớp nhất (chỉ khi câu hỏi nói về người)
                │     bm25   : từ khoá (mô tả đã bỏ câu nền + nhãn YOLO)
                ├─ + thưởng đồ vật/số lượng (bỏ qua nội thất cố định: sofa, lò, bàn ăn... ảnh nào cũng có)
                ├─ + thưởng màu áo
                ├─ + thưởng VÙNG: người đứng đúng chỗ được nhắc (điểm chân trong khung vùng, xem scene/layout.py)
                ├─ + [nâng cấp] đồ vật YOLOE + CLIP theo vùng, tư thế (pose), đang cầm gì (hands)
                │     — search/vision_extras.py, mỗi phần bật/tắt riêng trong config
                └─ gộp khung hình liền nhau (< dedup_seconds) -> top-k đa dạng hơn
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field

import numpy as np

from ..config import load_config
from ..data.catalog import Catalog, load_catalog
from ..index.store import IndexBundle, load_index
from ..query.parser import RE_ACTION, RE_GARMENT, RE_SENT_SEP, CountConstraint, ParsedQuery, QueryParser
from ..query.translate import QueryTranslator
from ..scene.layout import SceneLayout, load_layout
from ..vision.colors import COLORS
from ..vision.outfits import OutfitBook
from ..vision.people import PeopleBook
from .vision_extras import VisionExtras

log = logging.getLogger(__name__)


@dataclass
class SearchResult:
    idx: int
    imageID: str
    score: float
    components: dict
    record: dict
    group: list[str] = field(default_factory=list)   # các khung hình liền kề đã gộp vào kết quả này


@dataclass
class SearchResponse:
    query: str
    parsed: dict
    results: list[SearchResult]
    n_candidates: int
    elapsed_ms: float
    notes: list[str] = field(default_factory=list)
    scores: np.ndarray | None = field(default=None, repr=False)   # điểm cuối mọi ảnh (-inf = ngoài bộ lọc)


@dataclass
class ManualFilters:
    """Bộ lọc người dùng chọn tay trên giao diện — ghi đè/ bổ sung phần parser hiểu được."""
    hour_from: int | None = None
    hour_to: int | None = None          # gồm cả giờ này
    days: list[int] | None = None
    owners: list[str] | None = None
    objects: dict[str, CountConstraint] | None = None
    zones: list[str] | None = None       # chỉ lấy ảnh có người ở (một trong) các vùng này
    dedup_seconds: int | None = None     # ghi đè search.dedup_seconds (0 = không gộp)
    seconds: tuple[int, int] | None = None   # chỉ lấy ảnh trong [a, b) giây trong ngày (giao với giờ trong câu)


class SearchEngine:
    def __init__(self, cfg: dict, catalog: Catalog, index: IndexBundle, text_encoder=None, clip_encoder=None):
        self.cfg, self.catalog, self.index = cfg, catalog, index
        self.text_encoder, self.clip_encoder = text_encoder, clip_encoder
        s = cfg["search"]
        self.sc = s
        self.layout: SceneLayout | None = load_layout(cfg)
        n = len(catalog)
        self.extras = VisionExtras(cfg, n, index, index.detections, clip_encoder,
                                   cfg.get("vision", {}).get("count_conf", 0.35), self.layout)
        if index.detections is not None and self.extras.pose_colors \
                and (cfg.get("colors") or {}).get("source", "box") == "pose":
            # màu áo theo vai–hông (pose) thay cho ô thân áo cố định; người pose không thấy rõ vai giữ màu cũ
            index.detections = [[({**d, "col": self.extras.pose_colors[(i, k)]}
                                  if (i, k) in self.extras.pose_colors else d) for k, d in enumerate(ds)]
                                for i, ds in enumerate(index.detections)]
        self._build_object_tables()
        self._build_zone_tables()
        self._build_person_rows()
        self._build_person_tables()
        # sổ tay trang phục (data/outfits.yaml): câu tả quần áo -> nhóm bộ đồ đã ghi nhãn
        from pathlib import Path
        self.outfits, warn = OutfitBook.load(cfg, Path(cfg["paths"]["index_dir"]), text_encoder, len(self.person_img))
        if warn:
            log.warning(warn)
            index.warnings.append(warn)
        # người đã biết (data/people.yaml): "thầy Linh" -> ưu tiên ảnh có đúng người đó
        self.people_book, warn = PeopleBook.load(cfg, Path(cfg["paths"]["index_dir"]), len(self.person_img))
        if warn:
            log.warning(warn)
            index.warnings.append(warn)
        self.parser = QueryParser(
            known_hours=set(catalog.df["hour"].astype(int)), known_owners=set(catalog.owners),
            known_classes=set(self.classes), approx_hours=s.get("approx_hours", 1),
            minute_window=s.get("minute_window", 15), zones=self.layout.zones if self.layout else None,
            holdable=self._holdable(),
            postures=set((cfg.get("pose") or {}).get("postures", ["standing", "bending"])) if self.extras.posture else False,
            people=self.people_book.people if self.people_book else None)
        df = catalog.df
        self.sec_of_day = df["sec_of_day"].to_numpy().astype(np.int64)
        self.day_num = df["day_num"].to_numpy()
        self.owner_arr = df["owner"].to_numpy()
        self.timestamp = df["timestamp"].to_numpy()
        if index.sent_doc is not None:
            if (np.diff(index.sent_doc) < 0).any():
                raise RuntimeError("sent_doc.npy không theo thứ tự ảnh — hãy build lại bước text")
            self._sent_u, self._sent_starts = np.unique(index.sent_doc, return_index=True)
        self.has_image = np.array([isinstance(p, str) for p in df["image_path"]]) if n else np.zeros(0, bool)
        self._clip_centered = None
        # dịch truy vấn VI -> EN (config translation.enabled) + cache embedding câu: đề thi cho thêm 1 câu gợi ý mỗi
        # 30 giây -> các câu cũ không phải dịch / mã hoá lại, mỗi lần tìm chỉ tốn cho câu mới
        self.translator = QueryTranslator.from_config(cfg)
        self.use_translation = True                   # công tắc ở giao diện (tắt = nhanh hơn, tìm bằng câu tiếng Việt)
        self._qcache: dict[tuple[str, str], np.ndarray] = {}
        self._compose: dict[str, list] = {}          # cả đoạn -> các câu (xem _query_vec)
        self._build_entity_tables()

    # ------------------------------------------------------------------ bảng tra dựng sẵn
    def _build_object_tables(self) -> None:
        """Bảng đếm đồ vật [N, C] + bảng màu áo [N, 11] + danh sách nội thất cố định."""
        cat, dets = self.catalog, self.index.detections
        vc = self.cfg.get("vision", {})
        n = len(cat)
        self.count_conf = vc.get("count_conf", 0.35)
        self.person_colors: list[list[list[str]]] = [[] for _ in range(n)]
        self.color_matrix = np.zeros((n, len(COLORS)), np.float32)
        if dets is None:
            self.classes, self.class_index, self.count_matrix = cat.all_classes, cat.class_index, cat.count_matrix
            self.object_source = "csv"
        else:
            from ..vision.detector import count_matrix, summarize
            cc = self.count_conf
            self.classes = sorted({d["c"] for ds in dets for d in ds} | set(cat.all_classes) | set(self.extras.classes))
            self.class_index = {c: j for j, c in enumerate(self.classes)}
            yolo = count_matrix(dets, self.classes, cc)
            for c in self.extras.obj_classes:              # YOLOE: lấy số lớn hơn giữa YOLO COCO và YOLOE
                yolo[:, self.class_index[c]] = np.maximum(yolo[:, self.class_index[c]], self.extras.counts(c))
            csv = np.zeros_like(yolo)
            for c, j in cat.class_index.items():
                csv[:, self.class_index[c]] = cat.count_matrix[:, j]
            src = vc.get("object_source", "yolo")
            no_img = np.array([not isinstance(p, str) for p in cat.df["image_path"]])
            if src == "max":
                m = np.maximum(yolo, csv)
            else:  # yolo; ảnh không có file -> dùng CSV
                m = yolo.copy()
                m[no_img] = csv[no_img]
            self.count_matrix, self.object_source = m, src
            ci = {c: k for k, c in enumerate(COLORS)}
            for i, ds in enumerate(dets):
                _, cols = summarize(ds, cc)
                self.person_colors[i] = cols
                for col in cols:
                    if col:
                        self.color_matrix[i, ci[col[0]]] = 1.0
                        for sec in col[1:]:
                            self.color_matrix[i, ci[sec]] = max(self.color_matrix[i, ci[sec]], 0.6)
        # nội thất cố định: khai báo trong config + lớp nào xuất hiện ở >= fixed_object_rate số ảnh
        sc = self.cfg.get("scene") or {}
        self.presence_rate = {c: float((self.count_matrix[:, j] > 0).mean()) if n else 0.0
                              for c, j in self.class_index.items()}
        self.fixed_classes: set[str] = set()
        if sc.get("enabled", True):
            rate = sc.get("fixed_object_rate", 1.01)
            self.fixed_classes = ({c for c in sc.get("fixed_objects", []) if c in self.class_index}
                                  | {c for c, r in self.presence_rate.items() if r >= rate}) - {"person"}
            # đồ vật của bước nâng cấp (nồi, thớt...) có mặt thường xuyên vẫn được cộng điểm theo độ tin cậy
            self.fixed_classes -= set(self.extras.classes)

    def _build_zone_tables(self) -> None:
        n = len(self.catalog)
        if self.layout is None or self.index.detections is None:
            self.zone_names, self.zone_matrix = [], np.zeros((n, 0), np.int16)
            return
        self.zone_names = self.layout.names
        self.zone_matrix = self.layout.zone_count_matrix(self.index.detections, self.count_conf)

    def _build_person_rows(self) -> None:
        """Bảng theo TỪNG NGƯỜI: thuộc ảnh nào, ở vùng nào, mặc màu gì (1.0 = màu chính, 0.6 = màu phụ).
        Dùng để chấm các điều kiện phải đúng trên CÙNG 1 người ("người áo xanh Ở bồn rửa", "hai người áo đen")."""
        ci = {c: k for k, c in enumerate(COLORS)}
        img, zon, col, det, cen = [], [], [], [], []
        for i, ds in enumerate(self.index.detections or []):
            for k, d in enumerate(ds):
                if d["c"] != "person" or d["s"] < self.count_conf:
                    continue
                zs = set(self.layout.zones_of_box(d["b"])) if self.zone_names else set()
                img.append(i)
                det.append(k)
                cen.append(((d["b"][0] + d["b"][2]) / 2, (d["b"][1] + d["b"][3]) / 2))
                zon.append([z in zs for z in self.zone_names])
                row = np.zeros(len(COLORS), np.float32)
                for r, c in enumerate(d.get("col") or []):
                    row[ci[c]] = 1.0 if r == 0 else 0.6
                col.append(row)
        self._pp_img = np.array(img, np.int64)
        self._pp_det = np.array(det, np.int64)
        self._pp_zone = np.array(zon, bool).reshape(-1, len(self.zone_names))
        self._pp_col = np.array(col, np.float32).reshape(-1, len(COLORS))
        self._pp_cen = np.array(cen, np.float32).reshape(-1, 2)

    # vùng cắt sẵn cho CLIP (scene.regions) -> các vùng trong bếp mà nó bao phủ
    REGION_ZONES = {"island": {"island", "sink", "dining"}, "stove": {"stove"}, "living": {"living"}}

    def _build_entity_tables(self) -> None:
        """Cho bước rerank ghép câu–người: mỗi người (hàng của person_emb) đứng ở vùng nào; mỗi vùng cắt phủ vùng nào."""
        zn = self.zone_names
        self._pe_zone = np.zeros((len(self.person_img), len(zn)), bool)
        if zn and len(self.person_img) and self.index.detections is not None:
            for p, (i, k) in enumerate(self.index.person_ref):
                zs = set(self.layout.zones_of_box(self.index.detections[int(i)][int(k)]["b"]))
                self._pe_zone[p] = [z in zs for z in zn]
        names = list(self.index.region_names or (self.cfg.get("scene") or {}).get("regions", {}).keys())
        self._region_zone = [self.REGION_ZONES.get(r, {r}) for r in names]

    def _build_person_tables(self) -> None:
        """person_emb[p] thuộc ảnh person_img[p]; _pid[(ảnh, detection)] -> p."""
        pr = self.index.person_ref
        if self.index.person_emb is None or pr is None or not len(pr):
            self.person_img = np.zeros(0, np.int64)
            self._pid = {}
            return
        self.person_img = pr[:, 0].astype(np.int64)
        order = np.argsort(self.person_img, kind="stable")
        if (order != np.arange(len(order))).any():          # build theo thứ tự ảnh nên luôn đã sắp xếp
            raise RuntimeError("person_ref.npy không theo thứ tự ảnh — hãy build lại bước crops")
        self._person_u, self._person_starts = np.unique(self.person_img, return_index=True)
        self._pid = {(int(i), int(k)): p for p, (i, k) in enumerate(pr)}

    def _holdable(self) -> set[str]:
        if not len(self.extras.h_img):
            return set()
        from .vision_extras import DEFAULT_HAND_PROMPTS
        return set(DEFAULT_HAND_PROMPTS) | set(((self.cfg.get("hands") or {}).get("prompts") or {}))

    def people(self, idx: int) -> list[dict]:
        """Danh sách người trong ảnh: khung, màu áo, vùng, mã người (để tìm người này ở ảnh khác)."""
        out = []
        if self.index.detections is None:
            return out
        for k, d in enumerate(self.index.detections[idx]):
            if d["c"] != "person" or d["s"] < self.count_conf:
                continue
            zones = self.layout.zones_of_box(d["b"]) if self.layout else []
            out.append({"det": k, "box": d["b"], "conf": d["s"], "colors": d.get("col", []),
                        "zones": zones, "zone_labels": [self.layout.label(z) for z in zones] if zones else [],
                        "pid": self._pid.get((idx, k)),
                        **({"posture": self.extras.posture.get((idx, k), "unknown")} if self.extras.posture else {})})
        return out

    def record(self, idx: int) -> dict:
        rec = self.catalog.record(idx)
        if self.index.detections is not None:
            row = self.count_matrix[idx]
            rec["detected"] = {self.classes[j]: int(row[j]) for j in np.flatnonzero(row)}
            rec["clothing"] = ["/".join(c) for c in self.person_colors[idx]]
            rec["people"] = self.people(idx)
            if len(self.extras.h_img):
                rec["hands"] = self.extras.hands_of(idx)
            if self.zone_names:
                zr = self.zone_matrix[idx]
                rec["zone_counts"] = {self.zone_names[j]: int(zr[j]) for j in np.flatnonzero(zr)}
        return rec

    # ------------------------------------------------------------------ khởi tạo
    @classmethod
    def from_config(cls, cfg_path=None, overrides: dict | None = None) -> "SearchEngine":
        from ..encoders.factory import clip_encoder_name, make_clip_encoder, make_text_encoder
        cfg = load_config(cfg_path, overrides)
        catalog = load_catalog(cfg)
        index = load_index(cfg, catalog)
        text_enc = clip_enc = None
        if index.text_emb is not None:
            text_enc = make_text_encoder(cfg)
            if text_enc.name == "tfidf":
                text_enc.load(cfg["paths"]["index_dir"] / "tfidf.pkl")
            elif text_enc.name != index.manifest.get("text_model"):
                raise RuntimeError("Model text trong config khác model đã build index — hãy build lại.")
        if index.clip_emb is not None and cfg["encoders"]["clip"]["enabled"]:
            if clip_encoder_name(cfg) != index.manifest.get("clip_model"):
                raise RuntimeError(
                    f"Model CLIP trong config ({clip_encoder_name(cfg)}) khác model đã build "
                    f"({index.manifest.get('clip_model')}). Chạy: python scripts/build_index.py")
            clip_enc = make_clip_encoder(cfg, load_image_model=False)
        return cls(cfg, catalog, index, text_enc, clip_enc)

    # ------------------------------------------------------------------ API chính
    def search(self, query: str, top_k: int | None = None, manual: ManualFilters | None = None) -> SearchResponse:
        t0 = time.perf_counter()
        top_k = top_k or self.sc.get("top_k", 20)
        pq = self.parser.parse(query)
        if manual:
            for p in [pq, *pq.alternatives]:
                self._apply_manual(p, manual)
        final, mask, comps, notes = self.score(pq)
        if manual and manual.zones and self.zone_names:
            js = [self.zone_names.index(z) for z in manual.zones if z in self.zone_names]
            mask &= (self.zone_matrix[:, js] > 0).any(axis=1)
        cand = np.flatnonzero(mask)
        order = cand[np.lexsort((self.timestamp[cand], -final[cand]))]
        gap = self.sc.get("dedup_seconds", 0) if manual is None or manual.dedup_seconds is None \
            else manual.dedup_seconds
        if pq.alternatives and "alt" in comps:
            # mỗi vế câu gộp khung hình riêng rồi xếp xen kẽ -> vế nào cũng có mặt trong top-k
            grp = comps["alt"][order].astype(int)
            lists = [self._dedup(order[grp == g], top_k, gap) for g in np.unique(grp)]
            picked = [lst[k] for k in range(top_k) for lst in lists if k < len(lst)][:top_k]
        else:
            picked = self._dedup(order, top_k, gap)
        results = [SearchResult(int(i), self.catalog.df.at[i, "imageID"], float(final[i]),
                                {k: round(float(v[i]), 4) for k, v in comps.items()}, self.record(int(i)),
                                [self.catalog.df.at[j, "imageID"] for j in grp])
                   for i, grp in picked]
        if gap and any(r.group for r in results):
            notes.append(f"Đã gộp các khung hình cách nhau < {gap}s (camera đứng yên nên gần như trùng nhau).")
        return SearchResponse(query, pq.describe(), results, int(mask.sum()),
                              (time.perf_counter() - t0) * 1000, notes, np.where(mask, final, -np.inf))

    def event_frames(self, idx: int, seconds: int = 90, scores: np.ndarray | None = None,
                     members=()) -> list[tuple[int, float]]:
        """MỌI khung trong ±seconds quanh ảnh idx (cùng 1 'sự kiện'), mở rộng để phủ hết các khung đã gộp vào nhóm
        (members), xếp theo điểm của lần tìm gần nhất (nếu có), không thì theo thời gian. Đo trên đề 23/11: hệ thống
        hay tìm đúng sự kiện nhưng chọn nhầm khung bên trong -> người dùng lướt các khung này để nộp đúng khung."""
        ts = self.timestamp[[idx, *members]]
        lo, hi = min(ts.min(), self.timestamp[idx] - seconds), max(ts.max(), self.timestamp[idx] + seconds)
        near = np.flatnonzero((self.timestamp >= lo) & (self.timestamp <= hi))
        if scores is not None:
            near = near[np.argsort(-np.nan_to_num(scores[near], neginf=-1e9), kind="stable")]
        else:
            near = near[np.argsort(self.timestamp[near], kind="stable")]
        return [(int(i), float(scores[i]) if scores is not None else 0.0) for i in near]

    def _dedup(self, order: np.ndarray, top_k: int, gap: int) -> list[tuple[int, list[int]]]:
        """Duyệt theo điểm giảm dần; ảnh cách 1 ảnh đã chọn < gap giây thì gộp vào ảnh đó thay vì trả riêng."""
        if not gap:
            return [(int(i), []) for i in order[:top_k]]
        picked: list[tuple[int, list[int]]] = []
        ts_sel = np.empty(0, np.int64)
        for i in order:
            t = self.timestamp[i]
            if ts_sel.size:
                d = np.abs(ts_sel - t)
                j = int(d.argmin())
                if d[j] < gap:
                    picked[j][1].append(int(i))
                    continue
            if len(picked) >= top_k:
                break
            picked.append((int(i), []))
            ts_sel = np.append(ts_sel, t)
        return picked

    def score(self, pq: ParsedQuery):
        """Trả về (điểm cuối [N], mask ứng viên [N], các thành phần điểm, ghi chú). Dùng chung cho
        tìm kiếm thường, chuỗi sự kiện và các tính năng sau này."""
        if pq.alternatives:
            return self._score_alternatives(pq)
        notes: list[str] = []
        mask = self.filter_mask(pq)
        if not mask.any():
            notes.append("Không có ảnh nào thoả bộ lọc thời gian/ngày/địa điểm.")
        weights = dict(self.sc["weights"])
        tr = self.translate_query(pq) if pq.text else {}
        if pq.is_vietnamese and not tr:          # BM25 chỉ có ích với chữ tiếng Anh (mô tả ảnh là tiếng Anh)
            weights["bm25"] = self.sc.get("bm25_weight_vietnamese", weights.get("bm25", 0))
        if tr:
            notes.append("Đã dịch sang tiếng Anh để so với ảnh: " + " ".join(tr.get(s, s) for s in (pq.sentences or [pq.text])))

        lq = self.sc.get("long_query") or {}
        long_query = lq.get("enabled", True) and len(pq.sentences) >= lq.get("min_sentences", 2)
        self._compose = ({pq.text: [(s, tr.get(s)) for s in pq.sentences]}
                         if long_query and lq.get("compose_paragraph", True) else {})
        comps = self._semantic_comps(pq.text, weights, pq.about_people, tr.get(pq.text)) if pq.text else {}
        semantic, normed = self._combine(comps, weights, mask)

        # [đề thi nhiều câu] mỗi câu chấm riêng rồi gộp: ảnh phải khớp TẤT CẢ các câu, không chỉ câu nổi bật nhất.
        # Vector của cả đoạn dài bị "loãng" — câu chi tiết ("áo len có vài sọc ngang") bị chìm.
        # first_weight: câu gợi ý đầu tiên (BTC đưa câu quan trọng nhất trước) nặng hơn các câu sau.
        if long_query and comps:
            sent = self._sentence_score(pq, weights, mask, lq.get("fusion", "weighted"), tr,
                                        float(lq.get("first_weight", 1.0)))
            beta = float(lq.get("sentence_weight", 0.5))
            semantic = (1 - beta) * semantic + beta * sent
            normed["sentences"] = sent
            notes.append(f"Đề có {len(pq.sentences)} câu -> chấm từng câu rồi gộp "
                         f"(ảnh phải khớp tất cả các câu).")

        bonus = self._object_bonus(pq, notes)
        cbonus = self._color_bonus(pq)
        zbonus = self._zone_bonus(pq, notes)
        hbonus = self._holding_bonus(pq, notes)
        pbonus = self._posture_bonus(pq, notes)
        nbonus = self._named_bonus(pq, notes)      # không giảm theo câu dài: tên người là chi tiết rất chắc
        fbonus = self._frame_bonus(pq, notes)
        if long_query:
            # điều kiện đọc được từ câu dài dễ sai hơn (đếm người, vùng, màu của người khác...) -> giảm sức nặng
            k = float(lq.get("bonus_scale", 0.5))
            bonus, cbonus, zbonus, hbonus, pbonus = (x * k for x in (bonus, cbonus, zbonus, hbonus, pbonus))
        # bonus_weight: hệ số chung cho MỌI điểm thưởng (màu áo, vùng, số người, đang cầm, tư thế, tên, vị trí khung).
        # Đo 13 đề, bỏ-1-đề: 0.5 được chọn ở 12/13 lần (điểm thưởng đọc từ câu hay sai, lấn át điểm mô tả).
        g = float(self.sc.get("bonus_weight", 1.0))
        bonus, cbonus, zbonus, hbonus, pbonus, nbonus, fbonus = (
            x * g for x in (bonus, cbonus, zbonus, hbonus, pbonus, nbonus, fbonus))
        final = semantic + bonus + cbonus + zbonus + hbonus + pbonus + nbonus + fbonus
        if pq.objects and self.sc.get("strict_objects"):
            mask &= self._objects_ok(pq)
        out_comps = {**normed, "object_bonus": bonus}
        if pq.frame_pos:
            out_comps["frame_bonus"] = fbonus
        ent = self._entity_rerank(pq, final, mask, tr, notes)
        if ent is not None:
            final = final + ent
            out_comps["entity_rerank"] = ent
        if pq.colors or pq.colors_none or pq.colors_except:
            out_comps["color_bonus"] = cbonus
            if self.index.detections is None:
                notes.append("Chưa build bước nhận diện (detect) nên chưa lọc được theo màu áo.")
        if pq.zones:
            out_comps["zone_bonus"] = zbonus
        if pq.holding:
            out_comps["holding_bonus"] = hbonus
        if pq.postures:
            out_comps["posture_bonus"] = pbonus
        if pq.people:
            out_comps["named_bonus"] = nbonus
        if not pq.text and not pq.objects and not pq.zones:
            notes.append("Truy vấn chỉ có điều kiện metadata -> sắp xếp theo thời gian.")
        return final, mask, out_comps, notes

    # ------------------------------------------------------------------ mã hoá câu (có cache) + dịch
    def _enc(self, kind: str, text: str) -> np.ndarray:
        key = (kind, text)
        if key not in self._qcache:
            enc = self.text_encoder if kind == "text" else self.clip_encoder
            if len(self._qcache) > 2000:
                self._qcache.clear()
            self._qcache[key] = np.asarray(enc.encode_query(text), np.float32)
        return self._qcache[key]

    def _query_vec(self, kind: str, text: str, text_en: str | None) -> np.ndarray:
        """Vector truy vấn theo translation.<kind>_query: vi | en | both (trung bình 2 vector rồi chuẩn hoá).
        Cả đoạn nhiều câu (self._compose): trung bình vector TỪNG CÂU — các câu cũ đã có trong cache nên mỗi lần BTC
        thêm 1 câu gợi ý chỉ phải mã hoá câu mới (mã hoá lại cả đoạn dài ~0.4 giây/lần/model trên CPU)."""
        parts = self._compose.get(text)
        if parts:
            v = np.sum([self._query_vec(kind, s, tr) for s, tr in parts], axis=0)
            return v / max(float(np.linalg.norm(v)), 1e-6)
        mode = ((self.cfg.get("translation") or {}).get(f"{kind}_query") or ("en" if kind == "clip" else "both"))
        if not text_en or text_en == text or mode == "vi":
            return self._enc(kind, text)
        if mode == "en":
            return self._enc(kind, text_en)
        v = self._enc(kind, text) + self._enc(kind, text_en)
        return v / max(float(np.linalg.norm(v)), 1e-6)

    # CLIP khớp cụm ngắn tốt hơn nhiều so với câu dài: "a man in a blue sports jersey" -> đúng người z=2.3, cả câu
    # "... wearing a head-mounted camera holding a red cutting board in the kitchen island" -> z=0.2 (đề 23/11).
    RE_PHRASE_CUT = re.compile(r"\s*[,;]\s*(?:and\s+)?|\s+(?:while|who|which)\s+|\s+(?=(?:wearing|holding|standing|"
                               r"sitting|carrying|looking|using|leaning|walking|bending|reaching|with)\b)")
    RE_SUBJECT = re.compile(r"^(?:a|an|the|one|another)?\s*(?:(?:young|old|tall|short|bald)\s+)?"
                            r"(?P<s>man|woman|person|girl|boy|guy|lady|men|women|people|someone|individual)\b")

    @classmethod
    def phrases(cls, en: str) -> list[str]:
        """Tách câu tiếng Anh thành cụm thuộc tính, mỗi cụm có chủ ngữ: 'a man in a blue jersey holding a red board'
        -> ['a man in a blue jersey', 'a man holding a red board']."""
        parts = [p.strip(" .") for p in cls.RE_PHRASE_CUT.split(en.strip()) if p and p.strip(" .")]
        if len(parts) < 2:
            return parts
        m = cls.RE_SUBJECT.search(parts[0].lower())
        subj = f"a {m['s']}" if m else "a person"
        out = [parts[0]] + [p if cls.RE_SUBJECT.search(p.lower()) or len(p.split()) > 8 else f"{subj} {p}"
                            for p in parts[1:]]
        return [p for p in out if len(p.split()) >= 3 and not re.search(r"\b(?:was|is|were|are)$", p)]

    def _phrase_z(self, kind: str, text: str, text_en: str | None) -> np.ndarray | None:
        """z-score CLIP (trung bình theo cụm thuộc tính) cho mọi người (kind=person) hoặc mọi vùng cắt (kind=region).
        Có cache vì rerank và điểm 'person' dùng chung."""
        key = (kind + "_z", text + "\x00" + (text_en or ""))
        if key in self._qcache:
            return self._qcache[key]
        src = text_en or text
        use = (self.sc.get("rerank") or {}).get("phrases", False) or self.sc.get("phrase_match", False)
        ph = [p for s in re.split(r"(?<=\.)\s+", src) for p in self.phrases(s)] if text_en and use else []
        emb = self.index.person_emb if kind == "person" else self.index.region_emb
        vecs = [self._enc("clip", p) for p in ph] if len(ph) >= 2 else []
        vecs.append(self._query_vec("clip", text, text_en))   # + cả câu (giữ quan hệ giữa các cụm)
        zs = []
        for v in vecs:
            s = emb @ v
            zs.append((s - s.mean(axis=0)) / (s.std(axis=0) + 1e-6))
        z = np.mean(zs, axis=0).astype(np.float32)
        self._qcache[key] = z
        return z

    def translate_query(self, pq: ParsedQuery) -> dict[str, str]:
        """{câu tiếng Việt: câu tiếng Anh} cho các câu nội dung + cả đoạn. Rỗng nếu tắt dịch / câu đã là tiếng Anh."""
        if self.translator is None or not self.use_translation or not pq.is_vietnamese or not pq.text:
            return {}
        sents = pq.sentences or [pq.text]
        en = self.translator.translate(sents)
        if en is None:
            return {}
        out = dict(zip(sents, en))
        out.setdefault(pq.text, " ".join(e if e.endswith(".") else e + "." for e in en))
        return out

    def _semantic_comps(self, text: str, weights: dict, about_people: bool,
                        text_en: str | None = None) -> dict[str, np.ndarray]:
        """Điểm thô của 1 đoạn chữ: mô tả (e5), CLIP toàn ảnh / vùng cắt / từng người, BM25.
        text_en = bản dịch tiếng Anh (nếu có): CLIP và BM25 dùng tiếng Anh, e5 dùng cả hai (translation.*_query)."""
        n = len(self.catalog)
        comps: dict[str, np.ndarray] = {}
        idx = self.index
        if self.text_encoder is not None and idx.text_emb is not None:
            q = self._query_vec("text", text, text_en)
            doc = idx.text_emb @ q
            if idx.sent_emb is not None:
                best = np.full(n, -1.0, np.float32)
                best[self._sent_u] = np.maximum.reduceat(idx.sent_emb @ q, self._sent_starts)
                comps["text"] = 0.5 * doc + 0.5 * best
            else:
                comps["text"] = doc
        if self.clip_encoder is not None and idx.clip_emb is not None:
            qc = self._query_vec("clip", text, text_en)
            c = idx.clip_emb @ qc
            if idx.clip_mask is not None and (~idx.clip_mask).any():
                c[~idx.clip_mask] = np.nan
            comps["clip"] = c
            # phrase_match: CLIP vùng cắt / từng người chấm theo từng cụm thuộc tính của câu tiếng Anh (xem phrases())
            phr = bool(text_en) and self.sc.get("phrase_match", False)
            if idx.region_emb is not None and weights.get("region"):
                r = (self._phrase_z("region", text, text_en) if phr else idx.region_emb @ qc).max(axis=1)
                r[~self.has_image] = np.nan
                comps["region"] = r
            if idx.person_emb is not None and len(self.person_img) and weights.get("person") and about_people:
                p = np.full(n, np.nan, np.float32)    # ảnh không có người -> điểm thấp nhất
                ps = self._phrase_z("person", text, text_en) if phr else idx.person_emb @ qc
                p[self._person_u] = np.maximum.reduceat(ps, self._person_starts)
                comps["person"] = p
        if self.outfits is not None and about_people and weights.get("outfit") and RE_GARMENT.search(text.lower()):
            comps["outfit"] = self._outfit_comp(text)
        comps["bm25"] = idx.bm25.scores(text_en or text)
        return comps

    def _outfit_comp(self, text: str) -> np.ndarray:
        """Sổ tay trang phục: mỗi câu tả quần áo -> điểm người theo nhãn bộ đồ -> ảnh lấy người khớp nhất;
        nhiều câu (nhiều người) thì lấy trung bình -> ảnh có đủ các bộ đồ được tả sẽ cao nhất."""
        n = len(self.catalog)
        parts = [s.strip() for s in RE_SENT_SEP.split(text) if s.strip() and RE_GARMENT.search(s.lower())] or [text]
        out = np.zeros(n, np.float32)
        for s in parts:
            p = self.outfits.person_scores(self._enc("text", s))
            img = np.zeros(n, np.float32)
            img[self._person_u] = np.maximum.reduceat(p, self._person_starts)
            out += img
        return out / len(parts)

    # câu tả quần áo / ngoại hình 1 người -> sổ tay trang phục + CLIP trên từng người là tín hiệu tốt nhất
    PERSON_HEAVY = {"text": 0.25, "clip": 0.10, "region": 0.05, "person": 0.55, "bm25": 0.05, "outfit": 0.40}

    def _sentence_score(self, pq: ParsedQuery, weights: dict, mask: np.ndarray, fusion: str,
                        tr: dict | None = None, first_weight: float = 1.0) -> np.ndarray:
        """Điểm gộp các câu của đề (câu đầu nặng first_weight, các câu sau 1). Mỗi câu gộp các thành phần theo `fusion`:
          weighted : cộng có trọng số như cả đoạn (mặc định)
          max      : lấy thành phần cao nhất (mô tả / CLIP / từng người) — câu nào dùng tín hiệu hợp với câu đó
          mixed    : ½ weighted (câu tả quần áo thì nặng về CLIP từng người) + ½ max
        Đo trên eval/queries.kis.jsonl:  python eval/evaluate.py --kis"""
        out = []
        tr = tr or {}
        for s in pq.sentences:
            comps = self._semantic_comps(s, weights, pq.about_people, tr.get(s))
            if fusion == "weighted":
                out.append(self._combine(comps, weights, mask)[0])
                continue
            normed = {k: _minmax(v, mask) for k, v in comps.items()}
            best = np.nanmax(np.stack([normed[k] for k in ("text", "clip", "person") if k in normed]), axis=0)
            if fusion == "max":
                out.append(best)
            else:
                w = {**weights, **self.PERSON_HEAVY} if RE_GARMENT.search(s) and "person" in comps else weights
                out.append(0.5 * self._combine(comps, w, mask)[0] + 0.5 * best)
        w = np.ones(len(out), np.float32)
        w[0] = first_weight
        return np.tensordot(w / w.sum(), np.stack(out), axes=1)

    def _frame_bonus(self, pq: ParsedQuery, notes: list[str] | None = None) -> np.ndarray:
        """'đứng ở góc dưới bên trái khung hình': có người mà tâm khung nằm trong vùng đó -> thưởng (zone_bonus)."""
        n = len(self.catalog)
        if not pq.frame_pos or not len(self._pp_img):
            return np.zeros(n, np.float32)
        total = np.zeros(n, np.float32)
        x, y = self._pp_cen[:, 0], self._pp_cen[:, 1]
        for x1, y1, x2, y2 in pq.frame_pos:
            inside = ((x >= x1) & (x <= x2) & (y >= y1) & (y <= y2)).astype(np.float32)
            hit = np.zeros(n, np.float32)
            np.maximum.at(hit, self._pp_img, inside)
            total += hit
        if notes is not None:
            notes.append("Ưu tiên ảnh có người ở vị trí được tả trong khung hình.")
        return self.sc.get("zone_bonus", 0.3) * total / len(pq.frame_pos)

    # ------------------------------------------------------------------ rerank: ghép từng câu với từng người / vùng
    RE_PRONOUN = re.compile(
        r"^(?:anh|cô|chị|ông|bà|cậu|em)\s+(?:ấy|ta)|^(?:người|cô gái|chàng trai)\s+(?:này|đó|ấy)|^(?:he|she)\b|^(?:his|her)\b")
    RE_PERSONISH = re.compile(
        r"(?<!\w)(?:người|đàn\s+ông|phụ\s+nữ|cô\s+gái|chàng|anh|chị|cô|ông|bà|man|woman|person|guy|girl|lady|he|she|someone)(?!\w)")

    def entities(self, pq: ParsedQuery) -> list[str]:
        """Gom câu của đề thành các 'thực thể': câu bắt đầu bằng 'anh ấy / cô ấy / he' nói tiếp về người ở câu trước."""
        out: list[list[str]] = []
        for s in pq.sentences:
            if out and self.RE_PRONOUN.search(s.strip().lower()):
                out[-1].append(s)
            else:
                out.append([s])
        return [". ".join(g) for g in out]

    def _person_range(self, i: int) -> tuple[int, int]:
        """Các hàng person_emb thuộc ảnh i: [a, b) (rỗng nếu ảnh không có người)."""
        k = int(np.searchsorted(self._person_u, i))
        if k >= len(self._person_u) or self._person_u[k] != i:
            return 0, 0
        end = int(self._person_starts[k + 1]) if k + 1 < len(self._person_starts) else len(self.person_img)
        return int(self._person_starts[k]), end

    def _entity_rerank(self, pq: ParsedQuery, final: np.ndarray, mask: np.ndarray, tr: dict,
                       notes: list[str] | None = None) -> np.ndarray | None:
        """Đề tả NHIỀU người/vật trong cùng 1 ảnh ("người rửa tay ở bồn" + "người áo trắng ngồi sofa"): với top-N ảnh,
        ghép mỗi thực thể với 1 người khác nhau (CLIP từng người) hoặc 1 vùng cắt (đảo bếp / bếp nấu / phòng khách —
        dùng khi YOLO không thấy người đó) bằng thuật toán Hungarian, điểm = trung bình độ khớp (z-score). Ảnh chỉ có
        người rửa tay mà phòng khách trống sẽ bị điểm thấp ở thực thể thứ 2. Chỉ chấm lại top-N nên rất nhanh."""
        rc = self.sc.get("rerank") or {}
        if not rc.get("enabled", False) or self.clip_encoder is None or self.index.person_emb is None:
            return None
        ents = self.entities(pq)
        if len(ents) < int(rc.get("min_entities", 2)):
            return None
        from scipy.optimize import linear_sum_assignment
        n = len(self.catalog)
        cand = np.flatnonzero(mask)
        top = cand[np.argsort(-final[cand], kind="stable")[: int(rc.get("top_n", 300))]]
        if len(top) < 2:
            return None
        pe, re_ = self.index.person_emb, self.index.region_emb
        zpen, rpen = float(rc.get("zone_penalty", 1.0)), float(rc.get("region_penalty", 0.5))
        rows = []                                    # mỗi thực thể: (z người [P], z vùng [N, R], vùng nhắc, là người?)
        for e in ents:
            sub = self.parser.parse(e, _split=False)
            en = ". ".join(tr.get(s, s) for s in e.split(". ")) if tr else None
            if ". " in e:                             # thực thể nhiều câu: vector = trung bình các câu (đã cache)
                self._compose[e] = [(s, tr.get(s)) for s in e.split(". ")]
            zp = self._phrase_z("person", e, en)
            ow = float(rc.get("outfit_weight", 0.0))
            if ow and self.outfits is not None and RE_GARMENT.search(e):
                # nhãn trang phục do người ghi (data/outfits.yaml) — chắc hơn CLIP trên ảnh cắt nhỏ
                os_ = self.outfits.person_scores(self._enc("text", e))
                if os_.std() > 1e-6:
                    zp = (zp + ow * (os_ - os_.mean()) / os_.std()) / (1 + ow)
            zr = self._phrase_z("region", e, en) if re_ is not None else None
            zones = {z for z, c in sub.zones.items() if not (c.op == "==" and c.n == 0)}
            person = bool(RE_GARMENT.search(e) or self.RE_PERSONISH.search(e) or RE_ACTION.search(e))
            rows.append((zp, zr, zones, person))
        zn = self.zone_names
        score = np.zeros(len(top), np.float32)
        for t, i in enumerate(top):
            a, b = self._person_range(int(i))
            R = 0 if rows[0][1] is None else rows[0][1].shape[1]
            M = np.full((len(rows), (b - a) + 2 * R), -3.0, np.float32)      # mỗi vùng cắt dùng được 2 lần
            for e, (zp, zr, zones, person) in enumerate(rows):
                zi = [zn.index(z) for z in zones if z in zn]
                if b > a and person:
                    v = zp[a:b].copy()
                    if zi:
                        v -= zpen * (~self._pe_zone[a:b][:, zi].any(axis=1))
                    M[e, : b - a] = v
                for r in range(R):
                    if zones and not (zones & self._region_zone[r]):
                        continue                         # câu nói "ở phòng khách" -> chỉ so với vùng phòng khách
                    v = zr[i, r] - (rpen if person and not zones else 0.0)
                    M[e, b - a + 2 * r] = M[e, b - a + 2 * r + 1] = v
            ri, ci = linear_sum_assignment(-M)
            got = np.full(len(rows), -3.0, np.float32)
            got[ri] = M[ri, ci]
            score[t] = np.clip(got, -3, 3).mean()
        lo, hi = float(score.min()), float(score.max())
        out = np.zeros(n, np.float32)
        if hi - lo > 1e-6:
            out[top] = float(rc.get("weight", 0.3)) * (score - lo) / (hi - lo)
        if notes is not None:
            notes.append(f"Rerank top-{len(top)}: ghép {len(ents)} người/vật được tả với từng người / vùng trong ảnh.")
        return out

    @staticmethod
    def _combine(comps: dict, weights: dict, mask: np.ndarray) -> tuple[np.ndarray, dict]:
        """Chuẩn hoá min-max từng thành phần trên tập ứng viên rồi cộng có trọng số -> (điểm [N], các thành phần)."""
        n = len(mask)
        semantic = np.zeros(n, np.float32)
        wsum = sum(weights.get(k, 0) for k in comps)
        normed = {}
        for k, v in comps.items():
            nv = _minmax(v, mask)
            normed[k] = nv
            if wsum > 0:
                semantic += (weights.get(k, 0) / wsum) * nv
        return semantic, normed

    def _score_alternatives(self, pq: ParsedQuery):
        """Câu nhiều vế thời gian ("buổi trưa có ba người, buổi tối không ai"): chấm riêng từng vế với bộ lọc
        của vế đó, mỗi ảnh lấy điểm của vế nó khớp nhất."""
        n = len(self.catalog)
        res = [self.score(a) for a in pq.alternatives]
        best = np.full(n, -np.inf, np.float32)
        which = np.zeros(n, np.int64)
        mask = np.zeros(n, bool)
        for k, (f, m, _, _) in enumerate(res):
            f2 = np.where(m, f, -np.inf)
            upd = f2 > best
            best[upd], which[upd] = f2[upd], k
            mask |= m
        final = np.where(mask, best, 0).astype(np.float32)
        comps = {}
        for key in {k for _, _, c, _ in res for k in c}:
            arr = np.zeros(n, np.float32)
            for k, (_, _, c, _) in enumerate(res):
                if key in c:
                    sel = which == k
                    arr[sel] = c[key][sel]
            comps[key] = arr
        comps["alt"] = which.astype(np.float32)          # vế nào cho điểm cao nhất (search() xếp xen kẽ theo vế)
        notes = [f"Câu có {len(res)} vế với thời gian khác nhau -> tìm riêng từng vế, kết quả xếp xen kẽ."]
        for _, _, _, nt in res:
            notes += [x for x in nt if x not in notes]
        return final, mask, comps, notes

    # ------------------------------------------------------------------ lọc & thưởng
    def filter_mask(self, pq: ParsedQuery) -> np.ndarray:
        m = np.ones(len(self.catalog), bool)
        if pq.intervals:
            t = np.zeros_like(m)
            for a, b in pq.intervals:
                t |= (self.sec_of_day >= a) & (self.sec_of_day < b)
            m &= t
        if pq.days:
            m &= np.isin(self.day_num, list(pq.days))
        if pq.owners:
            m &= np.isin(self.owner_arr, list(pq.owners))
        return m

    def _counts(self, cls: str) -> np.ndarray:
        j = self.class_index.get(cls)
        return self.count_matrix[:, j] if j is not None else np.zeros(len(self.catalog), np.int16)

    def _color_bonus(self, pq: ParsedQuery) -> np.ndarray:
        """Điểm màu áo (trung bình các điều kiện màu trong câu):
          - "áo xanh"              : có người mặc xanh (màu phụ 0.6)
          - "hai người áo đen"     : đủ 2 người mặc đen = 1, chỉ 1 người = 0.5
          - "người áo xanh ở bồn rửa": người mặc xanh phải đứng ở bồn rửa (ở chỗ khác = 0.5)
          - "không ai mặc áo đỏ"   : không có người nào mặc đỏ (màu chính)
          - "người không mặc áo đen": có ít nhất 1 người màu chính khác đen"""
        n = len(self.catalog)
        if not (pq.colors or pq.colors_none or pq.colors_except) or self.index.detections is None:
            return np.zeros(n, np.float32)
        zj_all = [self.zone_names.index(z) for z, c in pq.zones.items()
                  if z in self.zone_names and not (c.op == "==" and c.n == 0)] if self.zone_names else []
        P = len(self._pp_img)
        total = np.zeros(n, np.float32)
        parts = 0
        counts = pq.color_counts or [1] * len(pq.colors)
        for k, (group, need) in enumerate(zip(pq.colors, counts)):
            cols = [COLORS.index(c) for c in group if c in COLORS]
            if not cols:
                continue
            s = self.color_matrix[:, cols].max(axis=1)
            if len(pq.color_zones) == len(pq.colors):
                # chỉ ghép màu với vùng nhắc CÙNG CÂU: "thầy Linh mặc áo xanh. Những người khác rửa chén."
                # -> người áo xanh không cần đứng ở bồn rửa
                zj = [j for j in (self.zone_names.index(z) for z in pq.color_zones[k] if z in self.zone_names)
                      if j in zj_all]
            else:
                zj = zj_all
            if P and zj:
                pc = self._pp_col[:, cols].max(axis=1) * self._pp_zone[:, zj].any(axis=1)
                there = np.zeros(n, np.float32)
                np.maximum.at(there, self._pp_img, pc)
                s = np.maximum(there, 0.5 * s)            # đúng màu nhưng ở chỗ khác: nửa điểm
            if P and need > 1:
                cnt = np.zeros(n, np.float32)
                np.add.at(cnt, self._pp_img, (self._pp_col[:, cols].max(axis=1) >= 1.0).astype(np.float32))
                s = s * np.where(cnt >= need, 1.0, 0.5)
            total += s
            parts += 1
        for group in pq.colors_none:
            cols = [COLORS.index(c) for c in group if c in COLORS]
            if cols:
                total += (self.color_matrix[:, cols].max(axis=1) < 1.0).astype(np.float32)
                parts += 1
        for group in pq.colors_except:
            cols = [COLORS.index(c) for c in group if c in COLORS]
            if cols and P:
                ok = ((self._pp_col.max(axis=1) >= 1.0) & (self._pp_col[:, cols].max(axis=1) < 1.0)).astype(np.float32)
                other = np.zeros(n, np.float32)
                np.maximum.at(other, self._pp_img, ok)
                total += other
                parts += 1
        return self.sc.get("color_bonus", 0.25) * total / max(1, parts)

    def _zone_bonus(self, pq: ParsedQuery, notes: list[str] | None = None) -> np.ndarray:
        n = len(self.catalog)
        if not pq.zones:
            return np.zeros(n, np.float32)
        if not self.zone_names:
            if notes is not None:
                notes.append("Chưa có detections.json nên chưa xét được người đứng ở vùng nào.")
            return np.zeros(n, np.float32)
        total = np.zeros(n, np.float32)
        used = 0
        for z, cons in pq.zones.items():
            if z not in self.zone_names:
                continue
            v = self.zone_matrix[:, self.zone_names.index(z)]
            if cons.op == "==" and cons.n == 0:
                s = (v == 0).astype(np.float32)
            else:
                target = max(1, cons.n)
                s = np.where(v >= target, 1.0, np.where(v >= 1, 0.5, 0.0)).astype(np.float32)
            total += pq.zone_weights.get(z, 1.0) * s      # "như đang rửa chén": phỏng đoán -> nửa điểm
            used += 1
        if notes is not None and used:
            lbl = ", ".join(f"{self.layout.label(z)} ({c})" for z, c in pq.zones.items())
            notes.append(f"Ưu tiên ảnh có người ở: {lbl}.")
        return self.sc.get("zone_bonus", 0.25) * total / max(1, used)

    def _object_bonus(self, pq: ParsedQuery, notes: list[str] | None = None) -> np.ndarray:
        n = len(self.catalog)
        if not pq.objects:
            return np.zeros(n, np.float32)
        tol = self.sc.get("count_tolerance", 1)
        total = np.zeros(n, np.float32)
        used, skipped = 0, []
        for cls, cons in pq.objects.items():
            if cls in self.fixed_classes:
                skipped.append(cls)
                continue
            v = self._counts(cls)
            ok = np.vectorize(cons.ok)(v).astype(np.float32)
            presence_only = cons.op == ">=" and cons.n == 1
            if cls in self.extras.classes and (presence_only or (cons.op == "==" and cons.n == 0)):
                p, used_src = self.extras.presence(cls, self._coco_counts(cls))
                if p is not None:
                    ok = p if presence_only else 1 - p
                    if notes is not None:
                        notes.append(f"{cls}: điểm có mặt từ " + " + ".join(used_src) + ".")
            w = self.sc.get("object_bonus", 0.15) if presence_only else self.sc.get("count_bonus", 0.15)
            if cons.op == "==" and cons.n > 0 and tol:
                near = (np.abs(v - cons.n) <= tol).astype(np.float32) * 0.5
                ok = np.maximum(ok, near)
            total += w * ok
            used += 1
        if skipped and notes is not None:
            notes.append("Bỏ qua đồ vật cố định của căn bếp (ảnh nào cũng có): " + ", ".join(
                f"{c} ({self.presence_rate.get(c, 0):.0%} ảnh)" for c in skipped))
        if not used:
            return total
        return total / used * min(used, 2)

    def _coco_counts(self, cls: str) -> np.ndarray | None:
        """Số đếm của YOLO COCO (detections.json) cho lớp này — dao/cốc/điện thoại COCO cũng có."""
        if self.index.detections is None:
            return None
        if not hasattr(self, "_coco_cache"):
            self._coco_cache: dict[str, np.ndarray] = {}
        if cls not in self._coco_cache:
            from ..vision.detector import count_matrix
            self._coco_cache[cls] = count_matrix(self.index.detections, [cls], self.count_conf)[:, 0]
        return self._coco_cache[cls]

    def _holding_bonus(self, pq: ParsedQuery, notes: list[str] | None = None) -> np.ndarray:
        """'người cầm dao': điểm = bàn tay giống cầm vật đó nhất trong ảnh (YOLOE ở bàn tay / CLIP bàn tay)."""
        n = len(self.catalog)
        if not pq.holding:
            return np.zeros(n, np.float32)
        total = np.zeros(n, np.float32)
        wsum = 0.0
        for cls, w in pq.holding.items():
            s, used = self.extras.holding_score(cls)
            if s is None:
                continue
            total += w * s
            wsum += w
            if notes is not None:
                notes.append(f"Đang cầm {cls}: dựa vào " + " + ".join(used) + ("" if w >= 1 else " (ngầm hiểu từ hành động)") + ".")
        if not wsum:
            return total
        # 1 vật cầm rõ ràng -> đủ trọng số · ngầm định (w < 1) -> giảm theo w · nhiều vật -> trung bình
        return self.sc.get("holding_bonus", 0.25) * total / max(1.0, wsum)

    def _named_bonus(self, pq: ParsedQuery, notes: list[str] | None = None) -> np.ndarray:
        """Đề nhắc tên người đã biết -> thưởng ảnh có người đó (người khớp nhất trong ảnh, [0,1] × named_bonus).
        Nhắc nhiều người -> trung bình (ảnh có đủ cả những người được nhắc cao nhất)."""
        n = len(self.catalog)
        w = float(self.sc.get("named_bonus", 0.3))
        names = [p for p in pq.people if self.people_book and p in self.people_book.sims]
        if not names or not w:
            return np.zeros(n, np.float32)
        out = np.zeros(n, np.float32)
        for name in names:
            img = np.zeros(n, np.float32)
            img[self._person_u] = np.maximum.reduceat(self.people_book.person_bonus(name), self._person_starts)
            out += img
        if notes is not None:
            desc = [self.people_book.people[p].get("description") or p for p in names]
            notes.append("Ưu tiên ảnh có người được nhắc tên: " + "; ".join(f"{p} ({d})" for p, d in zip(names, desc)))
        return w * out / len(names)

    def _posture_bonus(self, pq: ParsedQuery, notes: list[str] | None = None) -> np.ndarray:
        n = len(self.catalog)
        if not pq.postures or not self.extras.posture:
            return np.zeros(n, np.float32)
        rows = list(zip(self._pp_img.tolist(), self._pp_det.tolist()))
        total = np.zeros(n, np.float32)
        for p, cons in pq.postures.items():
            if cons.op == "==" and cons.n == 0:
                total += self.extras.posture_zero(p, rows)
            else:
                total += self.extras.posture_score(p, cons.n, rows)
        if notes is not None:
            vi = {"sitting": "ngồi", "standing": "đứng", "bending": "cúi"}
            notes.append("Ưu tiên tư thế (theo khung xương): " + ", ".join(f"{vi.get(p, p)} {c}" for p, c in pq.postures.items()) + ".")
        return self.sc.get("posture_bonus", 0.15) * total / len(pq.postures)

    def _objects_ok(self, pq: ParsedQuery) -> np.ndarray:
        m = np.ones(len(self.catalog), bool)
        for cls, cons in pq.objects.items():
            if cls not in self.fixed_classes:
                m &= np.vectorize(cons.ok)(self._counts(cls))
        return m

    def _apply_manual(self, pq: ParsedQuery, mf: ManualFilters) -> None:
        if mf.hour_from is not None or mf.hour_to is not None:
            a = (mf.hour_from if mf.hour_from is not None else 0) * 3600
            b = ((mf.hour_to if mf.hour_to is not None else 23) + 1) * 3600
            pq.intervals = [(a, b)]
        if mf.seconds is not None:
            from ..qa.timeline import intersect
            pq.intervals = intersect(pq.intervals, mf.seconds)
        if mf.days:
            pq.days = set(mf.days)
        if mf.owners:
            pq.owners = set(mf.owners)
        if mf.objects:
            pq.objects.update(mf.objects)

    # ------------------------------------------------------------------ tiện ích cho features/
    def clip_centered(self) -> np.ndarray | None:
        """CLIP ảnh đã trừ trung bình (= "căn bếp trống trung bình") rồi chuẩn hoá lại: so ảnh–ảnh theo
        phần KHÁC nhau (người, đồ vật) thay vì nền giống hệt nhau."""
        if self._clip_centered is None and self.index.clip_emb is not None:
            e = self.index.clip_emb
            ok = self.index.clip_mask if self.index.clip_mask is not None else np.ones(len(e), bool)
            c = e - e[ok].mean(axis=0, keepdims=True)
            c /= np.clip(np.linalg.norm(c, axis=1, keepdims=True), 1e-6, None)
            c[~ok] = 0
            self._clip_centered = c.astype(np.float32)
        return self._clip_centered


def _minmax(v: np.ndarray, mask: np.ndarray) -> np.ndarray:
    v = v.astype(np.float32).copy()
    sel = v[mask & ~np.isnan(v)]
    if sel.size == 0:
        return np.zeros_like(v)
    lo, hi = float(sel.min()), float(sel.max())
    v = np.nan_to_num(v, nan=lo)
    if hi - lo < 1e-9:
        return np.zeros_like(v)
    return np.clip((v - lo) / (hi - lo), 0, 1)
