"""Tìm kiếm từ dòng lệnh.

    python scripts/search.py "ba người đang nấu ăn lúc 12 giờ trưa"
    python scripts/search.py --seq "một người cắt dứa rồi mọi người ngồi ăn"
    python scripts/search.py --similar day1_Kitchen_12_2702 --cond "buổi chiều"
    python scripts/search.py --person day1_Kitchen_12_2702:0   # người thứ 0 trong ảnh -> ảnh khác có người đó
    python scripts/search.py "người rửa bát" --no-dedup        # không gộp khung hình liền kề
    python scripts/search.py -i         # chế độ gõ liên tục
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retrieval.features.similar import similar_person, similar_to_id  # noqa: E402
from retrieval.features.temporal import search_sequence  # noqa: E402
from retrieval.search.engine import ManualFilters, SearchEngine  # noqa: E402


def show(resp):
    print(f"\nHiểu truy vấn: {resp.parsed}")
    print(f"{resp.n_candidates} ứng viên sau lọc | {resp.elapsed_ms:.1f} ms")
    for n in resp.notes:
        print("  *", n)
    for r, res in enumerate(resp.results, 1):
        rec = res.record
        grp = f"  (+{len(res.group)} khung liền kề)" if res.group else ""
        print(f"{r:>2}. {res.imageID:<24} {rec['time']}  score={res.score:.3f}{grp}  {res.components}")
        where = ", ".join(f"{z}:{c}" for z, c in rec.get("zone_counts", {}).items())
        print(f"    YOLO={rec.get('detected', rec['objects'])}  vùng=[{where}]")
        print(f"    {rec['description'][:110]}...")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?")
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--seq", action="store_true", help="tìm chuỗi sự kiện")
    ap.add_argument("--similar", help="imageID ảnh gốc")
    ap.add_argument("--cond", default="", help="điều kiện kèm theo khi tìm ảnh tương tự")
    ap.add_argument("--person", help="imageID:số_thứ_tự_người  -> tìm người đó ở ảnh khác")
    ap.add_argument("--no-dedup", action="store_true", help="không gộp các khung hình liền kề")
    ap.add_argument("-i", "--interactive", action="store_true")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    eng = SearchEngine.from_config(args.config)

    mf = ManualFilters(dedup_seconds=0) if args.no_dedup else None
    if args.person:
        iid, _, k = args.person.rpartition(":")
        ppl = eng.people(eng.catalog.index_of(iid))
        if not ppl or int(k) >= len(ppl) or ppl[int(k)]["pid"] is None:
            print("Ảnh này không có người số", k, "(hoặc chưa build bước crops)."); return
        for r in similar_person(eng, ppl[int(k)]["pid"], args.k, args.cond):
            print(r["imageID"], r["record"]["time"], r["score"], r.get("match", {}).get("box"))
        return
    if args.similar:
        for r in similar_to_id(eng, args.similar, args.k, args.cond):
            print(r["imageID"], r["record"]["time"], r["score"], r["record"]["description"][:90])
        return
    if args.seq:
        for s in search_sequence(eng, args.query, args.k):
            print(f"\nchuỗi score={s.score:.3f}")
            for st in s.steps:
                print(f"   [{st['time']}] {st['imageID']}  ← {st['event']}")
        return
    if args.interactive:
        while True:
            try:
                q = input("\nTruy vấn> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if q:
                show(eng.search(q, args.k, mf))
        return
    show(eng.search(args.query, args.k, mf))


if __name__ == "__main__":
    main()
