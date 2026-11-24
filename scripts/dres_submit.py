"""Nộp bài DRES từ dòng lệnh (khi không dùng giao diện).

    python scripts/dres_submit.py check                 # checklist trước giờ thi: đăng nhập, buổi thi, task
    python scripts/dres_submit.py list                  # các buổi thi được tham gia (lấy Evaluation ID)
    python scripts/dres_submit.py submit 10_1528.jpg    # task KIS: nộp 1 ảnh (tên file, imageID hoặc đường dẫn)
    python scripts/dres_submit.py answer 3              # task QA : nộp 1 câu trả lời bằng chữ
    python scripts/dres_submit.py answer "áo đỏ"        #           (có dấu cách thì đặt trong nháy)
    python scripts/dres_submit.py -i                    # gõ liên tục, Enter để nộp (tự chọn ảnh/chữ theo task)

Tài khoản đọc từ .env (DRES_USERNAME / DRES_PASSWORD) hoặc --user / --password.
Nộp ngay, không hỏi lại (điểm tính cả thời gian nộp); --confirm để hỏi [y/N] trước mỗi lần nộp.
Lưu ý: mỗi lần nộp SAI trong lúc task đang chạy đều bị trừ điểm.
"""
from __future__ import annotations

import argparse
import getpass
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _s in (sys.stdout, sys.stderr):          # console Windows (cp1258) không in được tiếng Việt
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
sys.path.insert(0, str(ROOT / "src"))
from retrieval.config import load_config  # noqa: E402
from retrieval.submit import DresClient, DresError, clean_answer, load_dres_settings, task_kind, to_item_name  # noqa: E402


def task_label(t: dict | None) -> str:
    return (t or {}).get("name") or (t or {}).get("templateId") or ""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", choices=["check", "list", "submit", "answer"], default="check")
    ap.add_argument("images", nargs="*", help="ảnh cần nộp (submit) hoặc câu trả lời (answer, mỗi đối số 1 đáp án)")
    ap.add_argument("-i", "--interactive", action="store_true", help="nhập tên ảnh liên tục")
    ap.add_argument("--user")
    ap.add_argument("--password")
    ap.add_argument("--evaluation", help="Evaluation ID (mặc định: buổi thi đang ACTIVE)")
    ap.add_argument("-y", "--yes", action="store_true", help="(mặc định đã nộp ngay, giữ để lệnh cũ vẫn chạy)")
    ap.add_argument("--confirm", action="store_true", help="hỏi [y/N] trước mỗi lần nộp (mặc định: nộp ngay vì "
                    "điểm tính cả thời gian nộp)")
    a = ap.parse_args()

    s = load_dres_settings(load_config())
    cli = DresClient.from_settings(s, ROOT, username=a.user, password=a.password, evaluation_id=a.evaluation)
    if not cli.username:
        cli.username = input("DRES username: ").strip()
    if not cli.password:
        cli.password = getpass.getpass("DRES password: ")

    try:
        cli.login()
        print(f"[1] Đăng nhập OK: {cli.user.get('username')} ({cli.user.get('role')})")
        evs = cli.evaluations()
        if a.cmd == "list" or a.cmd == "check":
            print(f"[2] {len(evs)} buổi thi:")
            for e in evs:
                print(f"    {e.get('id')}  {e.get('status', ''):<10} {e.get('name', '')}")
        ev = cli.resolve_evaluation()
        print(f"[3] Evaluation ID: {ev}")
        task = cli.current_task(ev)
        kind = task_kind(task)
        print(f"[4] Task đang chạy: {task_label(task) or '(không có)'}" + (f" — loại {kind}" if kind else ""))
        print(f"[5] Đổi tên: 10_1528.jpg -> {to_item_name('10_1528.jpg', cli.item_prefix)}")
    except DresError as e:
        sys.exit(f"LỖI: {e}")
    if a.cmd == "check" and not a.interactive:
        print("Muốn thử kết nối nộp bài: nộp khi CHƯA có task chạy, DRES trả 400 'wrong state' là đúng.")
        return

    def do(value: str, as_text: bool | None = None):
        t = cli.current_task(ev)
        kind = task_kind(t)
        if as_text is None:                       # chế độ -i: theo loại task đang chạy
            as_text = kind == "QA"
        if as_text:
            item = clean_answer(value)
            if kind == "KIS":
                print("  CẢNH BÁO: task đang chạy là KIS (cần tên ảnh) — nộp chữ sẽ bị từ chối.")
            prompt = f'Nộp đáp án "{item}"? [y/N] '
        else:
            item = to_item_name(value, cli.item_prefix)
            if kind == "QA":
                print("  CẢNH BÁO: task đang chạy là QA (cần đáp án chữ) — nộp ảnh sẽ bị từ chối.")
            prompt = f"Nộp {item}? [y/N] "
        if not item:
            print("  bỏ qua (rỗng)")
            return
        if a.confirm and not a.yes and input(prompt).strip().lower() != "y":
            print("  bỏ qua")
            return
        r = (cli.submit_text if as_text else cli.submit)(item, ev, task=task_label(t))
        print(f"  [{r.http_status}] {r.status}: {r.message}")
        if r.status == "ERROR":
            print("  phản hồi gốc:", r.raw[:300])

    for value in a.images:
        do(value, as_text=a.cmd == "answer")
    if a.interactive:
        print("Gõ tên ảnh (task KIS) hoặc đáp án (task QA) — Enter trống để thoát:")
        while (value := input("> ").strip()):
            do(value, as_text=True if a.cmd == "answer" else False if a.cmd == "submit" else None)


if __name__ == "__main__":
    main()
