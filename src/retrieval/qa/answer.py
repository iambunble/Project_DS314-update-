"""Hỏi đáp offline (không cần mạng / model ngôn ngữ): câu hỏi -> loại câu hỏi -> tìm ảnh bằng engine -> soạn câu trả
lời từ dữ liệu đã có của các ảnh đó (giờ, số người / đồ vật từ YOLO, vị trí trong bếp, màu áo, nhãn bộ đồ, người
đã biết). Mỗi câu trả lời kèm ảnh làm bằng chứng và đáp án ngắn (để sau này nộp đáp án chữ lên DRES).

Loại câu hỏi (nhận bằng luật, xem QUESTION_TYPES):
  count : "có bao nhiêu người lúc 9 giờ", "mấy cái cốc trên đảo bếp khi thầy Linh rửa bát"
  when  : "lúc mấy giờ thầy Linh mở tủ lạnh", "khi nào có người ngồi sofa"
  wear  : "thầy Linh mặc áo gì", "người đang nấu ở bếp mặc áo màu gì"
  where : "thầy Linh thường đứng ở đâu", "lúc 12 giờ người áo đỏ ở đâu"
  who   : "ai đang nấu ở bếp lúc 18 giờ"
  yesno : "có ai ngồi sofa lúc 8 giờ không", "thầy Linh có rửa bát không"
  find  : còn lại -> tìm ảnh như bình thường, tóm tắt ảnh khớp nhất
Câu hỏi theo thời gian (gộp nhiều ảnh thành từng "lần", xem timeline.py):
  times   : "thầy Linh mở tủ lạnh mấy lần", "buổi sáng có mấy lần có người mở tủ lạnh"
  duration: "thầy Linh ở trong bếp bao lâu", "buổi chiều bếp trống bao lâu"
  extreme : "lúc nào đông người nhất", "buổi sáng giờ nào vắng nhất"
  when + "lần đầu / lần cuối", "sau khi X thì ...", "... trước khi X", count + "người khác nhau" trong 1 khoảng giờ
Không hiểu được thì nói rõ là không chắc, không bịa: mọi con số đều đọc từ dữ liệu của ảnh bằng chứng.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from . import timeline as tl

# thứ tự quan trọng: loại đứng trước được ưu tiên ("ai mặc áo gì" -> wear, "có bao nhiêu" -> count)
QUESTION_TYPES = [
    ("times", tl.TIMES),
    ("duration", tl.DURATION),
    ("extreme", tl.EXTREME),
    ("count", re.compile(r"(?<!\w)(?:bao\s+nhiêu|mấy)\s+(?!giờ|h\b|phút|lần)(?P<what>[^\s?,.]+(?:\s+[^\s?,.]+)?)"
                         r"|(?<!\w)(?:đếm|số\s+lượng|số\s+người)(?!\w)")),
    ("when", re.compile(r"(?<!\w)(?:(?:lúc|vào)\s+mấy\s+giờ|mấy\s+giờ|khi\s+nào|lúc\s+nào|bao\s+giờ|thời\s+điểm\s+nào|"
                        r"giờ\s+nào|khoảng\s+mấy\s+giờ)(?!\w)")),
    ("wear", re.compile(r"(?<!\w)(?:mặc\s+(?:áo|quần|đồ|trang\s+phục)?\s*(?:màu\s+)?(?:gì|nào|như\s+thế\s+nào|ra\s+sao)|"
                        r"(?:áo|quần|trang\s+phục)\s+(?:màu\s+)?(?:gì|nào)|mặc\s+gì)(?!\w)")),
    ("yesno", re.compile(r"^\s*(?:có|liệu|phải)\s.*(?<!\w)(?:không|chưa)\s*\??\s*$")),
    ("where", re.compile(r"(?<!\w)(?:ở\s+đâu|chỗ\s+nào|vị\s+trí\s+nào|khu\s+vực\s+nào|đứng\s+đâu|ngồi\s+đâu)(?!\w)")),
    ("who", re.compile(r"(?<!\w)(?:ai|người\s+nào|những\s+ai)(?!\w)")),
    ("yesno", re.compile(r"(?<!\w)(?:không|chưa|phải\s+không|đúng\s+không|hay\s+không)\s*\??\s*$")),
]

# hỏi về cả buổi thu dữ liệu ("có bao nhiêu người trong buổi thu dữ liệu này", "tổng cộng có mấy người tham gia")
# -> đếm số người khác nhau, không phải số người trong 1 ảnh
_SESSION = re.compile(
    r"(?<!\w)(?:buổi\s+(?:thu|quay|ghi|chụp)|thu\s+(?:dữ\s+liệu|thập|hình|video)|bộ\s+(?:dữ\s+liệu|ảnh|dữ\s+kiện)|"
    r"cả\s+(?:buổi|ngày|video|bộ)|toàn\s+bộ|tổng\s+cộng|tổng\s+số|tất\s+cả|khác\s+nhau|tham\s+gia|có\s+mặt|"
    r"trong\s+video|dataset)(?!\w)", re.IGNORECASE)

# cụm hỏi bỏ đi trước khi đưa phần còn lại cho engine tìm ảnh
_STRIP = re.compile(
    r"(?<!\w)(?:có\s+)?(?:bao\s+nhiêu|mấy)(?!\s+giờ)|(?:lúc|vào|khoảng)\s+mấy\s+giờ|mấy\s+giờ|khi\s+nào|lúc\s+nào|"
    r"bao\s+giờ|thời\s+điểm\s+nào|giờ\s+nào|(?:đang\s+)?mặc\s+(?:áo|quần|đồ|trang\s+phục)?\s*(?:màu\s+)?"
    r"(?:gì|nào|như\s+thế\s+nào|ra\s+sao)|(?:áo|quần|trang\s+phục)\s+(?:màu\s+)?(?:gì|nào)|ở\s+đâu|chỗ\s+nào|"
    r"vị\s+trí\s+nào|khu\s+vực\s+nào|(?:phải|đúng|hay)\s+không|người\s+nào|những\s+ai|(?<!\w)ai(?!\w)|"
    r"(?:thường|hay)(?=\s)|có\s+(?=.*\bkhông\s*\??\s*$)|không\s*\??\s*$|chưa\s*\??\s*$|hãy\s+cho\s+biết|"
    r"cho\s+(?:tôi|mình)\s+biết|(?<!\w)là(?!\w)|\?", re.IGNORECASE)


@dataclass
class Answer:
    question: str
    kind: str                              # count | when | wear | where | who | yesno | find | times | duration | extreme
    text: str                              # câu trả lời đầy đủ
    short: str = ""                        # đáp án ngắn (số, giờ, màu...) — để nộp đáp án chữ sau này
    evidence: list[str] = field(default_factory=list)   # imageID làm bằng chứng, quan trọng nhất trước
    confidence: str = "thấp"               # cao | trung bình | thấp
    details: list[str] = field(default_factory=list)    # cách hệ thống tìm ra câu trả lời

    def as_dict(self) -> dict:
        return {"question": self.question, "kind": self.kind, "text": self.text, "short": self.short,
                "evidence": self.evidence, "confidence": self.confidence, "details": self.details}


def classify(question: str) -> tuple[str, re.Match | None]:
    q = question.lower().strip()
    for kind, rx in QUESTION_TYPES:
        m = rx.search(q)
        if m:
            return kind, m
    return "find", None


def strip_question(question: str) -> str:
    """Bỏ cụm hỏi, giữ phần mô tả sự việc để tìm ảnh: "lúc mấy giờ thầy Linh mở tủ lạnh?" -> "thầy Linh mở tủ lạnh"."""
    s = _STRIP.sub(" ", question.strip())
    return re.sub(r"\s+", " ", s).strip(" ,.")


def is_session_question(question: str) -> bool:
    """Câu hỏi về cả buổi thu dữ liệu (không phải 1 thời điểm / 1 ảnh)."""
    return bool(_SESSION.search(question))


def load_participants(cfg: dict) -> dict | None:
    """data/participants.yaml (đường dẫn đổi được ở config: qa.participants) -> dict, thiếu file -> None."""
    import yaml
    from pathlib import Path
    from ..config import PROJECT_ROOT
    p = Path((cfg.get("qa") or {}).get("participants", "data/participants.yaml"))
    p = p if p.is_absolute() else PROJECT_ROOT / p
    if not p.exists():
        return None
    y = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    y["participants"] = [x for x in (y.get("participants") or []) if isinstance(x, dict)]
    return y


def hhmm(sec: int, with_sec: bool = False) -> str:
    sec = int(sec)
    return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}" + (f":{sec % 60:02d}" if with_sec else "")


class QAEngine:
    """Trả lời câu hỏi bằng dữ liệu của SearchEngine. Không gọi dịch vụ ngoài."""

    def __init__(self, eng, top_k: int = 5):
        self.eng, self.top_k = eng, top_k
        self.qc = eng.cfg.get("qa") or {}
        self._window: tuple[int, int, int] | None = None     # (ngày, giây đầu, giây cuối) cho "sau khi / trước khi"
        self._oids = None

    # ------------------------------------------------------------------ public
    def ask(self, question: str) -> Answer:
        rel = tl.split_relative(question)
        if rel:
            try:
                ans = self._relative(question, *rel)
            except Exception as e:
                ans = Answer(question, "when", f"Chưa trả lời được câu này ({type(e).__name__}: {e}).")
            if ans is not None:
                return ans
        kind, m = classify(question)
        rest = strip_question(question)
        fn = getattr(self, f"_{kind}")
        try:
            ans = fn(question, rest, m)
        except Exception as e:                                   # không để 1 câu lạ làm sập giao diện
            ans = Answer(question, kind, f"Chưa trả lời được câu này ({type(e).__name__}: {e}).", confidence="thấp")
        ans.details.insert(0, f"Loại câu hỏi: {kind}. Phần dùng để tìm ảnh: \"{rest or '(trống)'}\".")
        return ans

    # ------------------------------------------------------------------ tiện ích
    def _time(self, idx: int) -> str:
        return hhmm(self.eng.sec_of_day[idx], with_sec=True)

    def _named(self, text: str) -> list[str]:
        pb = self.eng.people_book
        return [n for n in self.eng.parser.parse(text).people if pb and n in pb.sims] if text else []

    def _residual(self, pq, named: list[str]) -> list[str]:
        """Chữ còn lại sau khi bỏ cụm thay tên ("người đàn ông") và từ chung chung -> có tả sự việc cụ thể không."""
        t = pq.text
        for n in named:
            t = t.replace(self.eng.people_book.people[n].get("as", "người"), " ")
        return [w for w in t.split() if w not in _GENERIC]

    def display(self, name: str) -> str:
        p = self.eng.people_book.people.get(name, {}) if self.eng.people_book else {}
        return p.get("display") or (p.get("names") or [name])[0]

    def _person_prob(self, name: str, pid) -> float:
        pb = self.eng.people_book
        return float(pb.sims[name][pid]) if pb and pid is not None and name in pb.sims else 0.0

    def _outfit(self, pid) -> str | None:
        ob = self.eng.outfits
        if ob is None or pid is None:
            return None
        j = int(ob.person_label[pid])
        return ob.label_texts[j] if j >= 0 else None

    def describe_person(self, p: dict, named: list[str] | None = None) -> str:
        """1 người trong ảnh -> "thầy Linh (áo len xanh dương) ở bếp nấu"."""
        pb = self.eng.people_book
        who = None
        for name in (named or (list(pb.sims) if pb else [])):
            if self._person_prob(name, p.get("pid")) >= (pb.hi if pb else 1):
                who = self.display(name)
                break
        look = self._outfit(p.get("pid")) or (f"áo {'/'.join(_vi_color(c) for c in p['colors'])}" if p.get("colors")
                                              else "không rõ trang phục")
        where = f" ở {', '.join(p['zone_labels'])}" if p.get("zone_labels") else ""
        if who:
            return f"{who} ({look}){where}"
        return f"{look}{where}" if re.match(r"(?:đàn ông|cô gái|phụ nữ|người)", look) else f"người {look}{where}"

    def _manual(self):
        from ..search.engine import ManualFilters
        d, a, b = self._window
        return ManualFilters(seconds=(a, b), days=[d])

    def _search(self, text: str, k: int | None = None):
        if not text:
            return None
        return self.eng.search(text, k or self.top_k, manual=self._manual() if self._window else None)

    def _parse(self, text: str):
        """parse + áp cửa sổ thời gian (nếu đang trả lời phần sau của câu "sau khi X thì ...")."""
        pq = self.eng.parser.parse(text)
        if self._window:
            mf = self._manual()
            for p in [pq, *pq.alternatives]:
                self.eng._apply_manual(p, mf)
        return pq

    # ------------------------------------------------------------------ dòng thời gian
    def _abs(self) -> np.ndarray:
        """Thời điểm tuyệt đối (giây) để sắp xếp / đo khoảng cách: ngày * 86400 + giây trong ngày."""
        e = self.eng
        if getattr(self, "_abs_t", None) is None:
            self._abs_t = e.day_num.astype(np.int64) * 86400 + e.sec_of_day.astype(np.int64)
            t = np.sort(self._abs_t)
            d = np.diff(t)
            d = d[(d > 0) & (d < 600)]
            self._step = float(np.median(d)) if len(d) else 10.0
        return self._abs_t

    def _presence(self, name: str) -> np.ndarray:
        """[N] xác suất cao nhất trong ảnh là người <name> (0 nếu không có ai)."""
        e, pb = self.eng, self.eng.people_book
        out = np.zeros(len(e.catalog), np.float32)
        ref = np.asarray(e.index.person_ref)
        if pb and name in pb.sims and len(ref):
            np.maximum.at(out, ref[:, 0].astype(int), pb.sims[name])
        return out

    def _specific(self, pq, named: list[str]) -> list[str]:
        """Từ tả sự việc cụ thể (sau khi bỏ tên người, từ chung chung, từ chỉ có mặt / thời gian)."""
        return [w for w in self._residual(pq, named) if w.lower() not in _PRESENCE_WORDS]

    def _timeline(self, text: str, mode: str = "auto"):
        """-> (thứ tự ảnh theo thời gian, thời điểm, ảnh nào khớp, điểm, mô tả đang đếm cái gì, đoạn).
        mode: auto = tự chọn (bếp trống / người đã biết có mặt / sự việc), event = luôn coi là sự việc."""
        e = self.eng
        pq = self._parse(text) if text else self._parse("")
        named = self._named(text)
        tm = e.filter_mask(pq)
        if not tm.any():
            return None
        t_all = self._abs()
        order = np.flatnonzero(tm)
        order = order[np.argsort(t_all[order], kind="stable")]
        t = t_all[order]
        miss, gap = int(self.qc.get("episode_miss", 1)), float(self.qc.get("episode_gap", 120))
        if mode == "auto" and tl.EMPTY.search(text or ""):
            hit = e._counts("person")[order] == 0
            score = hit.astype(np.float32)
            what = "bếp không có ai"
        elif mode == "auto" and named and not self._specific(pq, named):
            pres = self._presence(named[0])
            hit = pres[order] >= e.people_book.hi
            score = pres[order]
            what = f"{self.display(named[0])} có mặt"
            miss = int(self.qc.get("presence_miss", 5))
            gap = float(self.qc.get("presence_gap", 180))
        else:
            final, mask = e.score(pq)[:2]
            ok = mask.copy()
            if named:
                ok &= self._presence(named[0]) > e.people_book.lo
            if not ok.any():
                return None
            thr = tl.hit_threshold(final[ok], float(self.qc.get("event_rel", 0.8)))
            hit = (ok & (final >= thr))[order]
            score = final[order]
            what = pq.text.strip() or text
        eps = tl.episodes(t, hit, miss, gap)
        return order, t, hit, score, what, eps

    def _span(self, order, ep) -> str:
        a, b = self._time(int(order[ep[0]])), self._time(int(order[ep[1]]))
        return a if ep[0] == ep[1] or a[:5] == b[:5] else f"{a[:5]}–{b[:5]}"

    def _ep_best(self, order, score, hit, ep) -> int:
        """Ảnh đại diện của 1 đoạn: ảnh khớp có điểm cao nhất."""
        sl = slice(ep[0], ep[1] + 1)
        cand = np.flatnonzero(hit[sl]) + ep[0]
        return int(order[cand[np.argmax(score[cand])]])

    def _scope(self, order) -> str:
        e = self.eng
        a, b = e.sec_of_day[order].min(), e.sec_of_day[order].max()
        full = len(order) == len(e.catalog)
        return "" if full else f" trong khoảng {hhmm(a)}–{hhmm(b)}"

    def _persons_in(self, idx: int, pq, named: list[str]) -> list[dict]:
        """Người trong ảnh khớp câu hỏi: đúng người được nhắc tên > đúng vùng được nhắc > tất cả."""
        ps = self.eng.people(idx)
        if named:
            hit = [p for p in ps if any(self._person_prob(n, p.get("pid")) >= 0.5 for n in named)]
            if hit:
                return sorted(hit, key=lambda p: -max(self._person_prob(n, p.get("pid")) for n in named))
        want = {z for z, c in pq.zones.items() if not (c.op == "==" and c.n == 0)}
        if want:
            return [p for p in ps if set(p["zones"]) & want]      # rỗng = không ai ở đúng chỗ được hỏi
        return ps

    # ------------------------------------------------------------------ các loại câu hỏi
    def _count(self, q: str, rest: str, m) -> Answer:
        eng = self.eng
        what = (m.groupdict().get("what") or "người") if m and m.groupdict() else "người"
        cls, cname = "person", "người"
        hit = eng.parser.re_obj.search(what)
        if hit:
            cls, cname = eng.parser.syn2cls.get(hit.group(0), "person"), hit.group(0)
        if cls not in eng.classes:
            return Answer(q, "count", f"Hệ thống chưa nhận diện được loại \"{what}\" nên không đếm được.")
        cname = "người" if cls == "person" else cname
        counts = eng._counts(cls)
        pq = self._parse(rest) if rest else None
        if cls == "person" and is_session_question(q) and not (pq and (pq.intervals or pq.zones)):
            return self._count_participants(q, counts)
        if cls == "person" and is_session_question(q) and pq and pq.intervals:
            mask = eng.filter_mask(pq)
            if not mask.any():
                return Answer(q, "count", "Không có ảnh nào trong khoảng thời gian đó.")
            return self._count_distinct(q, mask)
        if pq is not None and pq.intervals and len(pq.text.split()) > 2 and \
                self._specific(pq, self._named(rest)) and any(w not in _COUNT_NOUNS for w in self._specific(pq, [])):
            return self._count_event(q, rest, cls, cname, pq)
        # chỉ có điều kiện thời gian / vùng (không tả sự việc) -> thống kê trên mọi ảnh trong khoảng đó
        if pq is None or not pq.text.strip() or (pq.intervals and len(pq.text.split()) <= 2):
            mask = eng.filter_mask(pq) if pq is not None else np.ones(len(counts), bool)
            if not mask.any():
                return Answer(q, "count", "Không có ảnh nào trong khoảng thời gian đó.")
            c = counts[mask]
            idx = np.flatnonzero(mask)
            top = idx[np.argsort(-c, kind="stable")[:3]]
            span = (f"từ {hhmm(eng.sec_of_day[idx].min())} đến {hhmm(eng.sec_of_day[idx].max())}"
                    if len(idx) > 1 else f"lúc {self._time(int(idx[0]))}")
            med = int(np.median(c))
            text = (f"Trong {len(idx)} ảnh {span}: thường có khoảng {med} {cname} (trung vị), "
                    f"ít nhất {int(c.min())}, nhiều nhất {int(c.max())} (lúc {self._time(int(top[0]))}).")
            return Answer(q, "count", text, short=str(med), evidence=[eng.catalog.ids[i] for i in top],
                          confidence="trung bình",
                          details=[f"Đếm bằng YOLO trên {len(idx)} ảnh; số trên ảnh đông người có thể thiếu 1–2."])
        resp = self._search(rest)
        if not resp or not resp.results:
            return Answer(q, "count", "Không tìm thấy ảnh khớp mô tả để đếm.")
        r0 = resp.results[0]
        n0 = int(counts[r0.idx])
        others = [(int(counts[r.idx]), r) for r in resp.results[1:]]
        same = sum(1 for n, _ in others if n == n0)
        text = f"Có {n0} {cname} trong ảnh khớp nhất ({r0.imageID}, lúc {self._time(r0.idx)})."
        if others:
            text += " Các ảnh khớp tiếp theo: " + ", ".join(f"{n} ({self._time(r.idx)})" for n, r in others) + "."
        conf = "cao" if same >= len(others) * 0.6 and others else "trung bình" if same else "thấp"
        return Answer(q, "count", text, short=str(n0), evidence=[r.imageID for r in resp.results], confidence=conf,
                      details=["Đếm bằng YOLO trên ảnh tìm được; ảnh khớp nhất có thể chưa phải đúng khoảnh khắc."])

    def _count_participants(self, q: str, counts: np.ndarray) -> Answer:
        """Số người khác nhau trong cả buổi: đọc danh sách người ghi (data/participants.yaml). YOLO chỉ đếm được
        từng ảnh, nhóm trang phục thì chia / lẫn người -> không tự đếm; thiếu danh sách thì chỉ nói cận dưới."""
        eng = self.eng
        busiest = int(np.argmax(counts)) if len(counts) else 0
        peak = int(counts[busiest]) if len(counts) else 0
        roster = load_participants(eng.cfg)
        if not roster or not roster["participants"]:
            return Answer(q, "count", f"Không chắc — chưa có danh sách người tham gia (data/participants.yaml). Chỉ biết "
                          f"ít nhất {peak} người (ảnh đông nhất lúc {self._time(busiest)}).", short=f">={peak}",
                          evidence=[eng.catalog.ids[busiest]] if len(counts) else [], confidence="thấp",
                          details=["Số người trong 1 ảnh không phải số người của cả buổi."])
        ps = roster["participants"]
        n, unc, reviewed = len(ps), int(roster.get("uncertain", 0) or 0), bool(roster.get("reviewed"))
        text = f"Khoảng {n} người tham gia buổi thu dữ liệu" if not reviewed else f"Có {n} người tham gia buổi thu dữ liệu"
        text += f" (có thể sai ± {unc})." if unc and not reviewed else "."
        text += f" Lúc đông nhất có {peak} người cùng lúc ({self._time(busiest)})."
        return Answer(q, "count", text, short=str(n), evidence=self._participant_evidence(ps) or [eng.catalog.ids[busiest]],
                      confidence="cao" if reviewed else "trung bình",
                      details=[f"Đếm theo danh sách người ghi trong data/participants.yaml ({'đã' if reviewed else 'chưa'} "
                               "kiểm lại), không phải số người trong 1 ảnh."]
                              + [f"{i + 1}. {p.get('description') or p.get('name')}" for i, p in enumerate(ps)])

    def _participant_evidence(self, ps: list[dict]) -> list[str]:
        """1 ảnh / người: người gần tâm nhóm trang phục nhất trong các nhóm của người đó."""
        from pathlib import Path
        d = Path(self.eng.cfg["paths"]["index_dir"])
        try:
            ids, conf = np.load(d / "outfit_ids.npy"), np.load(d / "outfit_conf.npy")
            ref = self.eng.index.person_ref
        except Exception:
            return []
        if len(ids) != len(ref) or len(conf) != len(ids):
            return []
        out = []
        for p in ps:
            sel = np.flatnonzero(np.isin(ids, p.get("groups") or []))
            if len(sel):
                out.append(self.eng.catalog.ids[int(ref[sel[np.argmax(conf[sel])]][0])])
        return list(dict.fromkeys(out))

    def _when(self, q: str, rest: str, m) -> Answer:
        first, last = tl.FIRST.search(q.lower()), tl.LAST.search(q.lower())
        if first or last:
            return self._when_order(q, last=bool(last and not first))
        resp = self._search(rest)
        if not resp or not resp.results:
            return Answer(q, "when", "Không tìm thấy ảnh khớp sự việc này.")
        rs = resp.results
        t0 = self._time(rs[0].idx)
        near = [r for r in rs[1:] if abs(int(self.eng.sec_of_day[r.idx]) - int(self.eng.sec_of_day[rs[0].idx])) <= 300]
        text = f"Khoảng {t0} (ảnh {rs[0].imageID})."
        if len(rs) > 1:
            text += " Các thời điểm khớp tiếp theo: " + ", ".join(self._time(r.idx) for r in rs[1:]) + "."
        conf = "trung bình" if near else "thấp"
        if rs[0].score - (rs[1].score if len(rs) > 1 else 0) > 0.05:
            conf = "cao" if near else "trung bình"
        return Answer(q, "when", text, short=hhmm(self.eng.sec_of_day[rs[0].idx], with_sec=True),
                      evidence=[r.imageID for r in rs], confidence=conf,
                      details=["Giờ đọc từ tên ảnh (giờ_giây). Nhiều ảnh khớp gần nhau -> tin cậy hơn."])

    def _wear(self, q: str, rest: str, m) -> Answer:
        eng = self.eng
        named = self._named(rest or q)
        pq = eng.parser.parse(rest) if rest else eng.parser.parse(q)
        # người đã biết, không kèm sự việc cụ thể -> tổng hợp trên mọi ảnh có người đó
        if named and not self._residual(pq, named) and not pq.intervals and not pq.zones:
            name = named[0]
            pb = eng.people_book
            pids = np.flatnonzero(pb.sims[name] >= pb.hi)
            if len(pids):
                # không dùng nhãn bộ đồ ở đây: nhóm chứa người này có thể lẫn người khác -> mô tả trong
                # data/people.yaml (người ghi) là chính, màu đo được để đối chiếu
                cols = Counter(c for p in pids for c in self._colors_of_pid(int(p))[:1])
                desc = pb.people[name].get("description")
                label = self.display(name)
                parts = []
                if desc:
                    parts.append(desc)
                if cols:
                    parts.append("màu áo hệ thống đo được: " + ", ".join(
                        f"{_vi_color(c)} ({n * 100 // sum(cols.values())}%)" for c, n in cols.most_common(2)))
                imgs = list(dict.fromkeys(int(eng.index.person_ref[p][0]) for p in pids))[:self.top_k]
                return Answer(q, "wear", f"{_cap(label)}: " + "; ".join(parts) + ".",
                              short=desc or "/".join(_vi_color(c) for c, _ in cols.most_common(1)),
                              evidence=[eng.catalog.ids[i] for i in imgs], confidence="cao",
                              details=[f"Tổng hợp trên {len(pids)} lượt xuất hiện chắc chắn của {label}."])
        resp = self._search(rest)
        if not resp or not resp.results:
            return Answer(q, "wear", "Không tìm thấy ảnh khớp mô tả.")
        r0 = resp.results[0]
        ps = self._persons_in(r0.idx, pq, named)
        if not ps:
            return Answer(q, "wear", f"Không chắc — ảnh khớp nhất ({r0.imageID}, lúc {self._time(r0.idx)}) không có ai "
                          "đúng như mô tả.", evidence=[r.imageID for r in resp.results])
        descs = [self.describe_person(p, named) for p in ps[:3]]
        main = self._outfit(ps[0].get("pid")) or "/".join(_vi_color(c) for c in ps[0].get("colors", [])) or "không rõ"
        text = f"Trong ảnh khớp nhất ({r0.imageID}, lúc {self._time(r0.idx)}): " + "; ".join(descs) + "."
        return Answer(q, "wear", text, short=main, evidence=[r.imageID for r in resp.results],
                      confidence="trung bình" if len(ps) == 1 or named or pq.zones else "thấp",
                      details=["Trang phục lấy từ sổ tay trang phục (nếu nhóm có nhãn), không thì từ màu áo đo được."])

    def _colors_of_pid(self, pid: int) -> list[str]:
        i, k = self.eng.index.person_ref[pid]
        return self.eng.index.detections[int(i)][int(k)].get("col", [])

    def _where(self, q: str, rest: str, m) -> Answer:
        eng = self.eng
        named = self._named(rest or q)
        pq = eng.parser.parse(rest) if rest else eng.parser.parse(q)
        if named and not self._residual(pq, named) and not pq.zones:
            name = named[0]
            pb = eng.people_book
            mask = eng.filter_mask(pq)
            pids = [int(p) for p in np.flatnonzero(pb.sims[name] >= pb.hi) if mask[int(eng.index.person_ref[p][0])]]
            if pids:
                zc = Counter()
                for p in pids:
                    i, k = eng.index.person_ref[p]
                    zs = eng.layout.zones_of_box(eng.index.detections[int(i)][int(k)]["b"]) if eng.layout else []
                    zc.update(zs or ["(ngoài các vùng đã đặt tên)"])
                tot = sum(zc.values())
                label = self.display(name)
                top = [(eng.layout.label(z) if eng.layout and not z.startswith("(") else z, n) for z, n in zc.most_common(3)]
                text = (f"{_cap(label)} hay ở: " + ", ".join(f"{z} ({n * 100 // tot}%)" for z, n in top)
                        + f" — trên {len(pids)} lượt xuất hiện" + (" trong khoảng thời gian hỏi." if pq.intervals else "."))
                imgs = list(dict.fromkeys(int(eng.index.person_ref[p][0]) for p in pids))[:self.top_k]
                return Answer(q, "where", text, short=top[0][0], evidence=[eng.catalog.ids[i] for i in imgs],
                              confidence="cao" if top[0][1] / tot > 0.4 else "trung bình")
        resp = self._search(rest)
        if not resp or not resp.results:
            return Answer(q, "where", "Không tìm thấy ảnh khớp mô tả.")
        r0 = resp.results[0]
        ps = self._persons_in(r0.idx, pq, named)
        zones = [z for p in ps[:1] for z in p.get("zone_labels", [])]
        text = (f"Trong ảnh khớp nhất ({r0.imageID}, lúc {self._time(r0.idx)}): "
                + ("; ".join(self.describe_person(p, named) for p in ps[:3]) if ps else "không thấy người") + ".")
        return Answer(q, "where", text, short=", ".join(zones) or "không rõ", evidence=[r.imageID for r in resp.results],
                      confidence="trung bình" if zones else "thấp")

    def _who(self, q: str, rest: str, m) -> Answer:
        resp = self._search(rest)
        if not resp or not resp.results:
            return Answer(q, "who", "Không tìm thấy ảnh khớp mô tả.")
        pq = self.eng.parser.parse(rest)
        r0 = resp.results[0]
        ps = self._persons_in(r0.idx, pq, [])
        if not ps:
            where = ", ".join(self.eng.layout.label(z) for z in pq.zones) if pq.zones and self.eng.layout else ""
            return Answer(q, "who", f"Không chắc — ảnh khớp nhất ({r0.imageID}, lúc {self._time(r0.idx)}) không có ai "
                          + (f"ở {where}." if where else "nhận rõ."), evidence=[r.imageID for r in resp.results])
        descs = [self.describe_person(p) for p in ps[:4]]
        focused = len(ps) < len(self.eng.people(r0.idx))
        text = (f"Trong ảnh khớp nhất ({r0.imageID}, lúc {self._time(r0.idx)})"
                + (" — người ở đúng vị trí được hỏi" if focused else "") + ": " + "; ".join(descs) + ".")
        return Answer(q, "who", text, short=descs[0], evidence=[r.imageID for r in resp.results],
                      confidence="trung bình" if focused or len(ps) == 1 else "thấp",
                      details=["Chỉ nhận ra tên với người đã có trong data/people.yaml; người khác tả bằng trang phục."])

    def _yesno(self, q: str, rest: str, m) -> Answer:
        eng = self.eng
        resp = self._search(rest)
        if not resp or not resp.results:
            return Answer(q, "yesno", "Không — không có ảnh nào thoả các điều kiện (thời gian / địa điểm).",
                          short="không", confidence="trung bình")
        pq = eng.parser.parse(rest)
        r0 = resp.results[0]
        checks, ok = [], True
        if pq.objects:
            good = bool(eng._objects_ok(pq)[r0.idx])
            checks.append(f"số người / đồ vật {'khớp' if good else 'không khớp'}")
            ok &= good
        for name in self._named(rest):
            prob = max((self._person_prob(name, p.get("pid")) for p in eng.people(r0.idx)), default=0)
            good = prob >= 0.5
            checks.append(f"{self.display(name)} {'có' if good else 'không thấy'} trong ảnh")
            ok &= good
        want = {z for z, c in pq.zones.items() if not (c.op == "==" and c.n == 0)}
        if want:
            zs = {z for p in eng.people(r0.idx) for z in p["zones"]}
            good = bool(zs & want)
            checks.append(f"vị trí {'khớp' if good else 'không khớp'}")
            ok &= good
        where = f"ảnh {r0.imageID} lúc {self._time(r0.idx)}"
        if ok and checks:
            return Answer(q, "yesno", f"Có — ví dụ {where} ({', '.join(checks)}).", short="có",
                          evidence=[r.imageID for r in resp.results], confidence="trung bình", details=checks)
        if checks:
            return Answer(q, "yesno", f"Có lẽ không — ảnh khớp nhất ({where}) không thoả: "
                          + ", ".join(c for c in checks if "không" in c) + ".", short="không",
                          evidence=[r.imageID for r in resp.results], confidence="thấp", details=checks)
        return Answer(q, "yesno", f"Có thể có — ảnh khớp nhất là {where}. Hệ thống không kiểm chứng được hết điều kiện "
                      "trong câu, hãy xem ảnh để chắc chắn.", short="có thể",
                      evidence=[r.imageID for r in resp.results], confidence="thấp")

    def _find(self, q: str, rest: str, m) -> Answer:
        resp = self._search(rest or q)
        if not resp or not resp.results:
            return Answer(q, "find", "Không tìm thấy ảnh khớp mô tả.")
        r0 = resp.results[0]
        ps = self.eng.people(r0.idx)
        text = (f"Ảnh khớp nhất: {r0.imageID} lúc {self._time(r0.idx)}, có {len(ps)} người"
                + (": " + "; ".join(self.describe_person(p) for p in ps[:4]) if ps else "") + ".")
        return Answer(q, "find", text, short=r0.imageID, evidence=[r.imageID for r in resp.results],
                      confidence="trung bình")

    # ------------------------------------------------------------------ câu hỏi theo thời gian
    def _event_text(self, q: str) -> str:
        return strip_question(tl.strip_time_words(q))

    def _times(self, q: str, rest: str, m) -> Answer:
        ev = self._event_text(q)
        tlr = self._timeline(ev)
        if tlr is None:
            return Answer(q, "times", "Không có ảnh nào trong khoảng thời gian đó.", short="0")
        order, t, hit, score, what, eps = tlr
        e = self.eng
        if not eps:
            return Answer(q, "times", f"Không thấy lần nào ({what}){self._scope(order)}.", short="0",
                          confidence="thấp")
        n = len(eps)
        shown = ", ".join(self._span(order, ep) for ep in eps[:12]) + (", …" if n > 12 else "")
        best = [self._ep_best(order, score, hit, ep) for ep in eps]
        pos = {int(i): k for k, i in enumerate(order)}
        rank = np.argsort([-score[pos[i]] for i in best], kind="stable")
        text = f"Khoảng {n} lần ({what}){self._scope(order)}: {shown}."
        return Answer(q, "times", text, short=str(n), evidence=[e.catalog.ids[best[j]] for j in rank[:10]],
                      confidence="trung bình" if n <= 8 else "thấp",
                      details=[f"Gộp các ảnh khớp liền nhau (cách ≤ {int(self.qc.get('episode_gap', 120))} giây) "
                               "thành 1 lần; ảnh nhận sai có thể làm thừa / thiếu 1–2 lần.",
                               f"Số ảnh khớp: {int(hit.sum())} / {len(order)}."])

    def _duration(self, q: str, rest: str, m) -> Answer:
        ev = self._event_text(q)
        tlr = self._timeline(ev)
        if tlr is None:
            return Answer(q, "duration", "Không có ảnh nào trong khoảng thời gian đó.", short="0")
        order, t, hit, score, what, eps = tlr
        self._abs()
        if not eps:
            return Answer(q, "duration", f"Không thấy lúc nào ({what}){self._scope(order)}.", short="0 phút",
                          confidence="thấp")
        lens = [tl.episode_seconds(t, ep, self._step) for ep in eps]
        total = sum(lens)
        top = sorted(np.argsort(lens)[::-1][:5])
        parts = ", ".join(f"{self._span(order, eps[j])} ({tl.fmt_duration(lens[j])})" for j in top)
        text = (f"Tổng cộng khoảng {tl.fmt_duration(total)} ({what}){self._scope(order)}, chia thành {len(eps)} "
                f"đoạn. " + ("Các đoạn dài nhất: " if len(eps) > 5 else "Các đoạn: ") + parts + ".")
        if len(eps) > 1:
            text += (f" Lần đầu {self._time(int(order[eps[0][0]]))[:5]}, lần cuối "
                     f"{self._time(int(order[eps[-1][1]]))[:5]}.")
        ev_imgs = [self._ep_best(order, score, hit, eps[j]) for j in np.argsort(lens)[::-1][:self.top_k]]
        return Answer(q, "duration", text, short=tl.fmt_duration(total),
                      evidence=[self.eng.catalog.ids[i] for i in ev_imgs],
                      confidence="trung bình" if len(eps) <= 10 else "thấp",
                      details=[f"Ảnh chụp cách nhau ~{int(self._step)} giây; mỗi đoạn tính từ ảnh đầu đến ảnh cuối + "
                               "1 bước chụp, nên là ước lượng (sai ± vài chục giây mỗi đoạn)."])

    def _extreme(self, q: str, rest: str, m) -> Answer:
        e = self.eng
        want_max = bool(m and m.group("max"))
        pq = self._parse(self._event_text(tl.EXTREME.sub(" ", q.lower())))
        mask = e.filter_mask(pq)
        if not mask.any():
            return Answer(q, "extreme", "Không có ảnh nào trong khoảng thời gian đó.")
        idx = np.flatnonzero(mask)
        c = e._counts("person")[idx].astype(float)
        sec = e.sec_of_day[idx]
        bins = sec // 600
        stats = [(b, c[bins == b].mean(), int((bins == b).sum())) for b in np.unique(bins)]
        stats = [s for s in stats if s[2] >= 3] or stats
        b, avg, _ = (max if want_max else min)(stats, key=lambda s: s[1])
        win = f"{hhmm(b * 600)}–{hhmm(b * 600 + 600)}"
        if want_max:
            i = int(idx[np.argmax(c)])
            text = (f"Đông nhất lúc {self._time(i)} với {int(c.max())} người (ảnh {e.catalog.ids[i]}). Tính trung bình "
                    f"theo 10 phút thì đông nhất khoảng {win} (trung bình {avg:.1f} người).")
            ev = [e.catalog.ids[i]] + [e.catalog.ids[int(j)] for j in idx[bins == b][np.argsort(-c[bins == b])][:4]]
            return Answer(q, "extreme", text, short=self._time(i), evidence=list(dict.fromkeys(ev)),
                          confidence="cao", details=["Đếm người bằng YOLO trên từng ảnh."])
        tlr = self._timeline(self._event_text(q) + " bếp trống")
        longest = ""
        if tlr and tlr[5]:
            order, t, _, _, _, eps = tlr
            self._abs()
            j = int(np.argmax([tl.episode_seconds(t, ep, self._step) for ep in eps]))
            longest = (f" Đoạn trống dài nhất: {self._span(order, eps[j])} "
                       f"({tl.fmt_duration(tl.episode_seconds(t, eps[j], self._step))}).")
        sel = idx[bins == b]
        text = (f"Vắng nhất khoảng {win} (trung bình {avg:.1f} người). Có {int((c == 0).sum())} / {len(idx)} ảnh "
                f"không có ai.{longest}")
        return Answer(q, "extreme", text, short=win, evidence=[e.catalog.ids[int(j)] for j in sel[:self.top_k]],
                      confidence="trung bình", details=["Đếm người bằng YOLO trên từng ảnh, trung bình theo 10 phút."])

    def _when_order(self, q: str, last: bool) -> Answer:
        ev = self._event_text(q)
        tlr = self._timeline(ev)
        if tlr is None or not tlr[5]:
            return Answer(q, "when", "Không thấy lần nào khớp sự việc này.", confidence="thấp")
        order, t, hit, score, what, eps = tlr
        ep = eps[-1] if last else eps[0]
        i = self._ep_best(order, score, hit, ep)
        start = int(order[ep[1] if last else ep[0]])
        word = "Lần cuối" if last else "Lần đầu"
        text = f"{word}: khoảng {self._span(order, ep)} (ảnh {self.eng.catalog.ids[i]}) — {what}."
        if len(eps) > 1:
            other = eps[-4:-1][::-1] if last else eps[1:4]
            text += f" Tổng cộng thấy {len(eps)} lần; " + ("các lần trước đó: " if last else "các lần sau: ") + \
                ", ".join(self._span(order, x) for x in other) + "."
        return Answer(q, "when", text, short=hhmm(self.eng.sec_of_day[start], with_sec=True),
                      evidence=[self.eng.catalog.ids[i]] + [self.eng.catalog.ids[self._ep_best(order, score, hit, x)]
                                                             for x in (eps[1:4] if not last else eps[-4:-1])],
                      confidence="trung bình",
                      details=["Lấy đoạn sớm / muộn nhất trong các đoạn ảnh khớp; 1 ảnh nhận nhầm có thể làm sai "
                               "lần đầu / lần cuối — xem ảnh bằng chứng."])

    def _relative(self, q: str, rel: str, x: str, y: str) -> Answer | None:
        """"sau khi X thì Y" / "Y trước khi X": tìm lần X khớp nhất, rồi trả lời Y trong 30 phút sau / trước đó."""
        e = self.eng
        xs = strip_question(tl.strip_time_words(x))
        tlr = self._timeline(xs, mode="event")
        if tlr is None or not tlr[5]:
            return Answer(q, "when", f"Không tìm thấy lúc \"{xs}\".", confidence="thấp")
        order, t, hit, score, what, eps = tlr
        top = int(np.argmax(np.where(hit, score, -np.inf)))
        ep = next(p for p in eps if p[0] <= top <= p[1])
        a, b = int(order[ep[0]]), int(order[ep[1]])
        W = int(self.qc.get("relative_window", 1800))
        day = int(e.day_num[a])
        win = (int(e.sec_of_day[b]) + 1, int(e.sec_of_day[b]) + W) if rel == "sau" else \
            (int(e.sec_of_day[a]) - W, int(e.sec_of_day[a]))
        anchor = f"\"{xs}\" khớp nhất lúc {self._span(order, ep)} (ảnh {e.catalog.ids[int(order[top])]})."
        rest_y = strip_question(y)
        kind, m = classify(y)
        named_y = self._named(rest_y)
        if rest_y and self._specific(self.eng.parser.parse(rest_y), named_y):
            self._window = (day, *win)
            try:
                ans = getattr(self, f"_{kind}")(y, rest_y, m)
            finally:
                self._window = None
            ans.question = q
            ans.text = f"{anchor} {'Sau' if rel == 'sau' else 'Trước'} đó: {ans.text}"
            ans.details.insert(0, f"Câu có \"{rel} khi\": chỉ xét {W // 60} phút {rel} lúc đó.")
            return ans
        # Y chung chung ("mọi người làm gì") -> tả các ảnh ngay sau / trước đó
        same = np.flatnonzero(e.day_num == day)
        offs = (60, 300, 900) if rel == "sau" else (-60, -300, -900)
        ref = int(e.sec_of_day[b] if rel == "sau" else e.sec_of_day[a])
        lines, ev = [], []
        for o in offs:
            i = int(same[np.argmin(np.abs(e.sec_of_day[same] - (ref + o)))])
            if i in ev:
                continue
            ps = e.people(i)
            desc = "; ".join(self.describe_person(p) for p in ps[:4]) if ps else "không có ai"
            lines.append(f"{self._time(i)[:5]} ({len(ps)} người): {desc}")
            ev.append(i)
        text = f"{anchor} {'Sau' if rel == 'sau' else 'Trước'} đó — " + " · ".join(lines) + "."
        return Answer(q, "find", text, short=self._time(ev[0]) if ev else "",
                      evidence=[e.catalog.ids[i] for i in ev], confidence="thấp",
                      details=[f"Tả người và vị trí trong các ảnh ~1, 5, 15 phút {rel} lúc đó; hệ thống không hiểu "
                               "hành động nên hãy xem ảnh để biết họ đang làm gì."])

    def _outfit_ids(self):
        if self._oids is None:
            from pathlib import Path
            try:
                self._oids = np.load(Path(self.eng.cfg["paths"]["index_dir"]) / "outfit_ids.npy").astype(int)
            except Exception:
                self._oids = np.empty(0, int)
        return self._oids

    def _count_distinct(self, q: str, mask: np.ndarray) -> Answer:
        """Số người khác nhau trong 1 khoảng giờ: người trong danh sách tham gia có ít nhất vài lượt xuất hiện
        (theo nhóm trang phục / nhận diện người đã biết) trong các ảnh của khoảng đó."""
        e = self.eng
        idx = np.flatnonzero(mask)
        counts = e._counts("person")
        peak_i = int(idx[np.argmax(counts[idx])])
        peak = int(counts[peak_i])
        span = self._scope(idx)
        roster = load_participants(e.cfg)
        ids = self._outfit_ids()
        ref = np.asarray(e.index.person_ref)
        if not roster or not roster["participants"] or len(ids) != len(ref):
            return Answer(q, "count", f"Ít nhất {peak} người khác nhau{span} (lúc đông nhất {self._time(peak_i)}). "
                          "Chưa có danh sách người tham gia nên không đếm chính xác hơn được.", short=f">={peak}",
                          evidence=[e.catalog.ids[peak_i]], confidence="thấp")
        in_mask = mask[ref[:, 0].astype(int)]
        need = int(self.qc.get("distinct_min_hits", 3))
        pb = e.people_book
        present = []
        for p in roster["participants"]:
            h = in_mask & np.isin(ids, p.get("groups") or [])
            if p.get("name") and pb and p["name"] in pb.sims:
                h |= in_mask & (pb.sims[p["name"]] >= pb.hi)
            if h.sum() >= need:
                first = int(ref[np.flatnonzero(h)][np.argmin(self._abs()[ref[np.flatnonzero(h), 0].astype(int)]), 0])
                present.append((p, int(h.sum()), first))
        n = max(len(present), peak)
        text = f"Khoảng {n} người khác nhau{span}."
        if present:
            text += " Nhận ra: " + "; ".join(
                f"{(p.get('description') or p.get('name') or '?').split(' — ')[0]} (từ {self._time(f)[:5]})"
                for p, _, f in sorted(present, key=lambda x: self._abs()[x[2]])) + "."
        text += f" Lúc đông nhất có {peak} người cùng lúc ({self._time(peak_i)})."
        if len(present) < peak:
            text += f" Chỉ nhận ra {len(present)} người theo trang phục, nên lấy số lúc đông nhất làm cận dưới."
        return Answer(q, "count", text, short=str(n),
                      evidence=[e.catalog.ids[peak_i]] + [e.catalog.ids[f] for _, _, f in present][:self.top_k - 1],
                      confidence="trung bình" if len(present) >= peak else "thấp",
                      details=[f"Người được tính khi có ≥ {need} lượt xuất hiện trong khoảng giờ (theo nhóm trang phục "
                               "trong data/participants.yaml); người đổi áo / nhóm lẫn có thể làm lệch ± 1–2."])

    def _count_event(self, q: str, rest: str, cls: str, cname: str, pq) -> Answer:
        """"từ 9h đến 11h có bao nhiêu người nấu ăn": đếm trên mọi lần xảy ra sự việc trong khoảng, không chỉ 1 ảnh."""
        e = self.eng
        tlr = self._timeline(rest, mode="event")
        if tlr is None or not tlr[5]:
            return Answer(q, "count", "Không thấy lúc nào khớp sự việc trong khoảng thời gian đó.", confidence="thấp")
        order, t, hit, score, what, eps = tlr
        counts = e._counts(cls)
        want = [z for z, c in pq.zones.items() if not (c.op == "==" and c.n == 0)]
        zj = [e.zone_names.index(z) for z in want if e.zone_names and z in e.zone_names]
        per = []
        for ep in eps:
            i = self._ep_best(order, score, hit, ep)
            n = int(e.zone_matrix[i, zj].sum()) if (zj and cls == "person") else int(counts[i])
            per.append((n, i, ep))
        ns = [p[0] for p in per]
        med = int(np.median(ns))
        where = f" ở {', '.join(e.layout.label(z) for z in want)}" if zj and e.layout else ""
        shown = ", ".join(f"{n} ({self._span(order, ep)})" for n, _, ep in per[:8]) + (", …" if len(per) > 8 else "")
        text = (f"Thấy {len(eps)} lần{self._scope(order)} khớp \"{what}\"; mỗi lần thường có {med} {cname}{where} "
                f"(nhiều nhất {max(ns)}). Từng lần: {shown}.")
        return Answer(q, "count", text, short=str(med),
                      evidence=[e.catalog.ids[i] for _, i, _ in sorted(per, key=lambda p: -p[0])[:self.top_k]],
                      confidence="trung bình" if len(eps) <= 8 else "thấp",
                      details=["Đếm bằng YOLO trên ảnh đại diện của từng lần (số người trong cả ảnh, hoặc trong vùng "
                               "được nhắc)."])


_VI_COLORS = {"black": "đen", "white": "trắng", "gray": "xám", "red": "đỏ", "orange": "cam", "yellow": "vàng",
              "green": "xanh lá", "blue": "xanh dương", "purple": "tím", "pink": "hồng", "brown": "nâu"}


def _vi_color(c: str) -> str:
    return _VI_COLORS.get(c, c)


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


# từ không làm câu thành "sự việc cụ thể" (hỏi chung về 1 người: "thầy Linh thường đứng ở đâu")
_GENERIC = {"đứng", "ngồi", "hay", "thường", "đang", "có", "trong", "ảnh", "bếp", "phòng", "xuất", "hiện", "là",
            "người", "mặc", "áo", "đồ", "nhất", "lúc", "khi", "được", "thấy", "nhìn"}
# thêm cho câu hỏi theo thời gian: "thầy Linh vào bếp lần đầu", "sau khi ... thì mọi người làm gì"
_PRESENCE_WORDS = _GENERIC | {"vào", "ra", "đến", "tới", "mặt", "ở", "lại", "đi", "mọi", "làm", "gì", "việc", "sao",
                              "thế", "nào", "những", "các", "cả", "tất", "nhau", "khác", "trong", "từ", "giờ", "h",
                              "sáng", "trưa", "chiều", "tối", "buổi", "và", "thì", "đó", "này", "cùng", "lúc"}
# danh từ đếm (không phải sự việc): "bao nhiêu cái cốc lúc 9h" vẫn là thống kê theo giờ
_COUNT_NOUNS = {"cái", "chiếc", "con", "cốc", "ly", "đĩa", "dao", "nồi", "chảo", "thớt", "điện", "thoại"}
