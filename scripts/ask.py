"""Hỏi đáp offline trên 5000 ảnh (không cần mạng). Xem src/retrieval/qa/answer.py.

    python scripts/ask.py "Lúc mấy giờ thầy Linh mở tủ lạnh?"
    python scripts/ask.py -i          # hỏi liên tục, gõ trống để thoát
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

from retrieval.qa import QAEngine  # noqa: E402
from retrieval.search.engine import SearchEngine  # noqa: E402


def show(ans, verbose: bool) -> None:
    print(f"\n{ans.text}")
    print(f"  đáp án ngắn: {ans.short or '-'}   | tin cậy: {ans.confidence}   | loại: {ans.kind}")
    print(f"  ảnh bằng chứng: {', '.join(ans.evidence[:5]) or '-'}")
    if verbose:
        for d in ans.details:
            print("   ·", d)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question", nargs="?")
    ap.add_argument("-i", "--interactive", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true", help="in cách hệ thống tìm ra câu trả lời")
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    qa = QAEngine(SearchEngine.from_config())
    if a.question:
        show(qa.ask(a.question), a.verbose)
    if a.interactive or not a.question:
        while True:
            try:
                q = input("\nHỏi: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q:
                break
            show(qa.ask(q), a.verbose)


if __name__ == "__main__":
    main()
