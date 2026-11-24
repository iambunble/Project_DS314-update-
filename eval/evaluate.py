"""Đánh giá hệ thống — dùng để so sánh khi đổi model / trọng số.

    python eval/evaluate.py                          # self-retrieval + file queries mặc định
    python eval/evaluate.py --queries eval/my_queries.jsonl -k 10

1) Self-retrieval: lấy câu đầu (có nội dung) của mô tả 1 ảnh làm truy vấn, xem ảnh đó (hoặc khung hình liền kề ±60s,
   vì ảnh liên tiếp gần như giống hệt) có nằm trong top-K không. Đo nhanh chất lượng phần ngữ nghĩa.
2) File truy vấn có nhãn (jsonl), mỗi dòng:
     {"query": "...", "relevant": ["day1_Kitchen_12_2702", ...]}   -> Recall@K, MRR
     {"query": "...", "expect": {"hours": [12], "objects": {"person": "==3"}}} -> % kết quả đúng metadata
     {"query": "...", "expect": {"zones": {"stove": ">=1"}}}  -> có người ở vùng (theo YOLO + scene.zones)

So sánh trước/sau nâng cấp:  python eval/evaluate.py --compare
   chạy 2 lần: cấu hình hiện tại vs tắt các phần "scene" (không vùng, không crops, không gộp khung hình).

So sánh BẬT/TẮT từng phần nâng cấp nhận diện (đồ vật, pose, bàn tay):  python eval/evaluate.py --ablation
   chạy 6 cấu hình (tắt hết = bản cũ, + PA1 CLIP vùng, + YOLOE, + pose, + hands, tất cả) với
   3) truy vấn NHÃN YẾU (eval/queries.objects.jsonl): {"query": "...", "weak": "<regex trên mô tả>"}
      -> P@K = tỉ lệ kết quả top-K có mô tả khớp regex, so với tỉ lệ ngẫu nhiên. Mô tả gốc hay bỏ sót đồ vật nên
         số tuyệt đối thấp, chỉ dùng để SO SÁNH giữa các cấu hình.
   4) màu áo: trong các ảnh mô tả có nói màu áo ("blue shirt"...), bao nhiêu % có ít nhất 1 người được nhận đúng màu.
"""
import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from retrieval.scene.text_clean import content_sentences  # noqa: E402
from retrieval.search.engine import ManualFilters, SearchEngine  # noqa: E402


def self_retrieval(eng, n=300, k=10, seed=0):
    df = eng.catalog.df
    rng = random.Random(seed)
    rows = rng.sample(range(len(df)), min(n, len(df)))
    hit_exact = hit_near = 0
    rr, lat = [], []
    for i in rows:
        # câu đầu tiên CÓ NỘI DUNG (câu chỉ tả nội thất cố định thì ảnh nào cũng khớp -> không đo được gì)
        q = content_sentences(df.at[i, "description"])[0]
        t = time.perf_counter()
        res = eng.search(q, k, ManualFilters(dedup_seconds=0))
        lat.append((time.perf_counter() - t) * 1000)
        ids = [r.idx for r in res.results]
        if i in ids:
            hit_exact += 1
            rr.append(1 / (ids.index(i) + 1))
        else:
            rr.append(0)
        if any(abs(int(eng.timestamp[j]) - int(eng.timestamp[i])) <= 60 for j in ids):
            hit_near += 1
    print(f"[self-retrieval] n={len(rows)}  Recall@{k}={hit_exact / len(rows):.3f}  "
          f"Recall@{k}(±60s)={hit_near / len(rows):.3f}  MRR={np.mean(rr):.3f}  "
          f"latency p50={np.percentile(lat, 50):.1f}ms p95={np.percentile(lat, 95):.1f}ms")


def labeled(eng, path, k=10, dedup_off=False):
    lines = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    rec, mrr, meta = [], [], []
    mf = ManualFilters(dedup_seconds=0) if dedup_off else None
    for item in lines:
        res = eng.search(item["query"], k, mf)
        ids = [r.imageID for r in res.results]
        if "relevant" in item:
            rel = set(item["relevant"])
            rec.append(len(rel & set(ids)) / max(1, len(rel)))
            mrr.append(next((1 / (p + 1) for p, x in enumerate(ids) if x in rel), 0))
        if "expect" in item:
            ok = [_meets(r.record, item["expect"]) for r in res.results]
            meta.append(np.mean(ok) if ok else 0)
            flag = "" if all(ok) else "  <-- có kết quả sai metadata"
            print(f"  {np.mean(ok) if ok else 0:5.2f}  {item['query']}{flag}   hiểu={res.parsed}")
    if rec:
        print(f"[labeled] Recall@{k}={np.mean(rec):.3f}  MRR={np.mean(mrr):.3f}  (n={len(rec)})")
    if meta:
        print(f"[metadata] tỉ lệ kết quả đúng điều kiện trong top-{k}: {np.mean(meta):.3f}  (n={len(meta)})")


def kis_compare(path):
    """Đề thi DRES (đoạn nhiều câu) có đáp án -> hạng của ảnh đúng trên CẢ 5000 ảnh, với từng cách chấm câu dài.
    ±30s: tính cả khung hình liền kề (gần như trùng ảnh đúng). Thêm đề + đáp án mỗi buổi thi vào file này."""
    items = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    eng = SearchEngine.from_config()
    base = dict(eng.sc.get("long_query") or {})
    variants = [("tắt (như bản cũ, chỉ có sửa parser)", {**base, "enabled": False})] + \
               [(f"fusion={f}", {**base, "enabled": True, "fusion": f}) for f in ("weighted", "max", "mixed")]
    print(f"{len(items)} đề, cấu hình hiện tại: {base}")
    for name, lq in variants:
        eng.sc["long_query"] = lq
        ranks, near = [], []
        for it in items:
            final, mask, _, _ = eng.score(eng.parser.parse(it["query"]))
            f = np.where(mask, final, -np.inf)
            t = [eng.catalog.index_of(x) for x in it["relevant"]]
            ranks.append(min(int((f > f[i]).sum()) + 1 for i in t))
            nb = [j for i in t for j in np.flatnonzero(np.abs(eng.timestamp - eng.timestamp[i]) <= 30)]
            near.append(min(int((f > f[j]).sum()) + 1 for j in nb))
        r, nr = np.array(ranks), np.array(near)
        print(f"  {name:38s} MRR={np.mean(1 / r):.3f}  R@1={np.mean(r <= 1):.2f}  R@10={np.mean(r <= 10):.2f}  "
              f"R@50={np.mean(r <= 50):.2f}  R@10(±30s)={np.mean(nr <= 10):.2f}  hạng={ranks}")
    eng.sc["long_query"] = base


def kis_hints(path, gaps=(0, 30, 90, 180), overrides=None):
    """Mô phỏng buổi thi: mỗi 30 giây BTC cho thêm 1 câu gợi ý -> chấm lần lượt câu 1, câu 1+2, ... và in hạng ảnh đúng.
    Hạng theo từng cách gộp khung (gap giây): ảnh đúng nằm trong 1 nhóm đã trả về = trúng (người dùng mở nhóm ra nộp).
      python eval/evaluate.py --hints"""
    items = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    eng = SearchEngine.from_config(overrides=overrides)
    split = lambda q: [s.strip() for s in re.split(r"(?<=[.?!])\s+", q) if s.strip()]
    allr = {g: [] for g in gaps}
    last = {g: [] for g in gaps}
    byh = {g: [] for g in gaps}            # hạng theo từng đề: [hạng sau gợi ý 1, 1+2, ...]
    lat = []
    print(f"{len(items)} đề. Mỗi ô: hạng ảnh đúng sau câu 1, 1+2, ...  (gộp {gaps[1] if len(gaps) > 1 else 0}s)")
    for it in items:
        hs = split(it["query"])
        target = eng.catalog.index_of(it["relevant"][0])
        rows = {g: [] for g in gaps}
        for h in range(1, len(hs) + 1):
            t = time.perf_counter()
            final, mask, _, _ = eng.score(eng.parser.parse(" ".join(hs[:h])))
            lat.append((time.perf_counter() - t) * 1000)
            f = np.where(mask, final, -np.inf)
            order = np.argsort(-f, kind="stable")
            for g in gaps:
                r = _group_rank(eng, order, target, g)
                rows[g].append(r)
                allr[g].append(r)
        for g in gaps:
            last[g].append(rows[g][-1])
            byh[g].append(rows[g])
        show = gaps[1] if len(gaps) > 1 else gaps[0]
        print(f"  {(it.get('task') or it['relevant'][0][13:]):8s} chính xác {rows[gaps[0]]}   gộp {show}s {rows[show]}")
    print(f"độ trễ chấm điểm p50={np.percentile(lat, 50):.0f}ms p95={np.percentile(lat, 95):.0f}ms (lần đầu có dịch câu mới)")
    for g in gaps:
        a, b = np.array(allr[g]), np.array(last[g])
        print(f"  gộp {g:4d}s  mọi lượt gợi ý: MRR={np.mean(1 / a):.3f} H@10={np.mean(a <= 10):.2f} H@20={np.mean(a <= 20):.2f}"
              f" H@50={np.mean(a <= 50):.2f}  |  đủ gợi ý: H@10={np.mean(b <= 10):.2f} H@20={np.mean(b <= 20):.2f}")
    print("\nMỤC TIÊU: ảnh đúng nằm trong top 5/10/15 càng sớm càng tốt (người chọn bằng mắt nên top 1 không quan trọng)")
    for g in gaps:
        print(hint_table(byh[g], f"gộp {g}s" if g else "đúng khung"))


def hint_table(rows, name, ks=(5, 10, 15), max_hint=3):
    """rows: mỗi đề 1 list hạng theo lượt gợi ý. -> bảng Hit@k sau gợi ý 1, 1+2, ..., max_hint và khi đủ gợi ý.
    Đề ít câu hơn max_hint thì lượt sau dùng hạng của lượt cuối (đã đủ gợi ý)."""
    cols = [(f"gợi ý {h}" if h == 1 else f"1..{h}", lambda r, h=h: r[min(h, len(r)) - 1]) for h in range(1, max_hint + 1)]
    cols.append(("đủ", lambda r: r[-1]))
    out = [f"  {name:10s} " + " ".join(f"{c:>18s}" for c, _ in cols)]
    for k in ks:
        cells = []
        for _, pick in cols:
            hit = sum(pick(r) <= k for r in rows)
            cells.append(f"{hit:>2d}/{len(rows)} ({hit / len(rows):4.0%})")
        out.append(f"  top {k:<6d} " + " ".join(f"{c:>18s}" for c in cells))
    return "\n".join(out)


def _group_rank(eng, order, target, gap, limit=5000):
    """Hạng (theo nhóm) của nhóm chứa ảnh target khi gộp các khung cách nhau < gap giây giống SearchEngine._dedup."""
    if not gap:
        return int(np.flatnonzero(order == target)[0]) + 1
    ts = eng.timestamp
    sel = []
    for i in order[:limit]:
        near = [k for k, j in enumerate(sel) if abs(ts[j] - ts[i]) < gap]
        if near:
            if i == target:
                return near[0] + 1
            continue
        sel.append(i)
        if i == target:
            return len(sel)
    return limit


COLOR_RE = re.compile(r"\b(black|white|gr[ae]y|red|orange|yellow|green|blue|purple|pink|brown)\s+(?:shirt|t-shirt|top|"
                      r"jacket|hoodie|sweater|dress|clothing|attire|outfit|clothes|apron|sweatshirt|coat|jersey)")


def weak_labeled(eng, path, k=20):
    """Nhãn yếu: kết quả 'đúng' nếu mô tả gốc khớp regex. Trả về {query: (P@K, tỉ lệ ngẫu nhiên)}."""
    lines = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    desc = eng.catalog.df["description"].fillna("").str.lower()
    out = {}
    for item in lines:
        rx = re.compile(item["weak"], re.I)
        hit = desc.map(lambda t: bool(rx.search(t))).to_numpy()
        res = eng.search(item["query"], k, ManualFilters(dedup_seconds=0))
        ids = [r.idx for r in res.results]
        out[item["query"]] = (float(np.mean([hit[i] for i in ids])) if ids else 0.0, float(hit.mean()))
    return out


def color_agreement(eng):
    """% ảnh mô tả nói màu áo X mà hệ thống thấy có người mặc X (màu chính hoặc phụ). -> (tổng, theo màu)"""
    tot = hit = 0
    per: dict[str, list[int]] = {}
    for i, d in enumerate(eng.catalog.df["description"].fillna("")):
        want = {c.replace("grey", "gray") for c in COLOR_RE.findall(d.lower())}
        if not want:
            continue
        have = {c for p in eng.person_colors[i] for c in p}
        for c in want:
            ok = int(c in have)
            tot += 1; hit += ok
            per.setdefault(c, [0, 0]); per[c][0] += ok; per[c][1] += 1
    return (hit / tot if tot else 0.0), {c: (a / b, b) for c, (a, b) in sorted(per.items(), key=lambda x: -x[1][1])}


ABLATION_OFF = {"objects": {"clip_tags": {"enabled": False}, "yoloe": {"enabled": False}},
                "pose": {"enabled": False}, "hands": {"enabled": False}, "colors": {"source": "box"}}
ABLATIONS = [
    ("CŨ — tắt hết nâng cấp", ABLATION_OFF),
    ("+ PA1 CLIP theo vùng", {**ABLATION_OFF, "objects": {"clip_tags": {"enabled": True}, "yoloe": {"enabled": False}}}),
    ("+ PA2 YOLOE đồ vật", {**ABLATION_OFF, "objects": {"clip_tags": {"enabled": False}, "yoloe": {"enabled": True}}}),
    ("+ PA2 pose (màu áo, tư thế)", {**ABLATION_OFF, "pose": {"enabled": True}, "colors": {"source": "pose"}}),
    ("+ PA2 pose + bàn tay", {**ABLATION_OFF, "pose": {"enabled": True}, "hands": {"enabled": True},
                              "colors": {"source": "pose"}}),
    ("TẤT CẢ (PA1 + PA2)", {}),
]


def ablation(k=20, queries=None, n_self=0):
    queries = queries or str(ROOT / "eval" / "queries.objects.jsonl")
    table, colors = {}, {}
    for name, ov in ABLATIONS:
        print(f"\n========== {name} ==========")
        eng = SearchEngine.from_config(overrides=ov or None)
        print("  nguồn đang bật:", eng.extras.sources() or "không có")
        table[name] = weak_labeled(eng, queries, k)
        colors[name] = color_agreement(eng)
        if n_self:
            self_retrieval(eng, n_self, 10)
    names = [n for n, _ in ABLATIONS]
    short = ["cũ", "PA1", "YOLOE", "pose", "pose+tay", "tất cả"]
    print(f"\n===== P@{k} theo nhãn yếu (mô tả gốc khớp regex) — cao hơn = tốt hơn =====")
    print(f"{'truy vấn':44s} {'ngẫu nhiên':>10s} " + " ".join(f"{s:>8s}" for s in short))
    for q in table[names[0]]:
        base = table[names[0]][q][1]
        print(f"{q[:44]:44s} {base:10.2f} " + " ".join(f"{table[n][q][0]:8.2f}" for n in names))
    print("\n===== Màu áo: % màu trong mô tả được nhận đúng =====")
    for n, s in zip(names, short):
        tot, per = colors[n]
        print(f"  {s:8s} {tot:6.1%}   " + "  ".join(f"{c} {a:.0%} (n={b})" for c, (a, b) in list(per.items())[:5]))


def _meets(rec, exp):
    if "hours" in exp and rec["hour"] not in exp["hours"]:
        return False
    for z, c in exp.get("zones", {}).items():
        v = rec.get("zone_counts", {}).get(z, 0)
        op, n = c[:2], int(c[2:])
        if not {"==": v == n, ">=": v >= n, "<=": v <= n}[op]:
            return False
    for cls, c in exp.get("objects", {}).items():
        v = rec.get("detected", rec["objects"]).get(cls, 0)
        op, n = c[:2], int(c[2:])
        if not {"==": v == n, ">=": v >= n, "<=": v <= n}[op]:
            return False
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default=str(ROOT / "eval" / "queries.example.jsonl"))
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("-n", type=int, default=300)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--compare", action="store_true", help="so với cấu hình tắt phần 'scene'")
    ap.add_argument("--ablation", action="store_true", help="bật/tắt từng phần nâng cấp nhận diện")
    ap.add_argument("--hints", nargs="?", const=str(ROOT / "eval" / "queries.kis.jsonl"),
                    help="mô phỏng buổi thi: thêm từng câu gợi ý, in hạng ảnh đúng (chính xác và theo nhóm khung)")
    ap.add_argument("--kis", nargs="?", const=str(ROOT / "eval" / "queries.kis.jsonl"),
                    help="đề thi nhiều câu: so các cách chấm câu dài (search.long_query.fusion)")
    a = ap.parse_args()
    if a.ablation:
        ablation(k=20)
        sys.exit(0)
    if a.hints:
        kis_hints(a.hints)
        sys.exit(0)
    if a.kis:
        kis_compare(a.kis)
        sys.exit(0)
    ov = ({"encoders": {"text": {"backend": "tfidf"}, "clip": {"enabled": False}}}
          if a.offline else {})
    runs = [("HIỆN TẠI", ov, False)]
    if a.compare:
        base = {**ov, "scene": {"enabled": False},
                "search": {"weights": {"region": 0, "person": 0}, "zone_bonus": 0, "dedup_seconds": 0}}
        runs.insert(0, ("KHÔNG CÓ SCENE (gần như bản cũ)", base, True))
    for name, o, off in runs:
        print(f"\n========== {name} ==========")
        eng = SearchEngine.from_config(overrides=o or None)
        self_retrieval(eng, a.n, a.k)
        if Path(a.queries).exists():
            labeled(eng, a.queries, a.k, dedup_off=off)
