"""Phần nộp bài DRES của giao diện: đăng nhập ở thanh bên, nút "Nộp" dưới mỗi ảnh (task KIS), ô nộp đáp án chữ
(task QA — ở thanh bên và trang Hỏi đáp), hộp báo kết quả, lịch sử nộp.

Client nằm trong st.session_state (mỗi trình duyệt 1 tài khoản). Tài khoản mặc định đọc từ .env.
Điểm tính cả thời gian nộp -> bấm Nộp là gửi ngay, không hỏi xác nhận. Được nộp nhiều lần, kể cả nộp lại
ảnh / đáp án đã nộp trong task -> không chặn.
"""
from __future__ import annotations

import time
from pathlib import Path

import streamlit as st

import ui
from retrieval.submit import DresClient, DresError, clean_answer, load_dres_settings, task_kind, to_item_name

KEY = "_dres"
TASK_TTL = 10          # giây — làm mới thông tin task đang chạy
STATUS_COLOR = {"CORRECT": ui.MINT, "WRONG": ui.TOMATO, "INDETERMINATE": ui.PINE, "SKIPPED": ui.MUTED}
STATUS_VI = {"CORRECT": "ĐÚNG", "WRONG": "SAI", "INDETERMINATE": "CHỜ CHẤM", "SKIPPED": "ĐÃ NỘP",
             "NO_TASK": "CHƯA CÓ TASK", "BAD_NAME": "SAI TÊN", "UNAUTHORIZED": "HẾT PHIÊN",
             "NO_EVALUATION": "SAI BUỔI THI", "REJECTED": "BỊ TỪ CHỐI", "WRONG_TYPE": "NHẦM LOẠI BÀI",
             "DUPLICATE": "TRÙNG", "ERROR": "LỖI"}

CSS = f"""
<style>
.dres-badge {{ display:inline-block; padding:2px 8px; border-radius:999px; font-size:.72rem; font-weight:600;
              color:#fff; letter-spacing:.02em; }}
.dres-row {{ display:flex; gap:8px; align-items:center; font-size:.8rem; margin:3px 0; flex-wrap:wrap; }}
.dres-row code {{ font-size:.75rem; }}
.dres-bar {{ background:#fff; border:1px solid var(--line); border-left:5px solid var(--cobalt); border-radius:12px;
            padding:8px 14px; margin:0 0 12px; font-size:.9rem; display:flex; gap:14px; flex-wrap:wrap;
            align-items:center; }}
.dres-bar.solved {{ border-left-color: {ui.MINT}; }}
</style>
"""


# ------------------------------------------------------------------ trạng thái
def client(cfg: dict, root: Path) -> DresClient:
    if KEY not in st.session_state:
        st.session_state[KEY] = DresClient.from_settings(load_dres_settings(cfg), root)
    return st.session_state[KEY]


def _task(cli: DresClient, fresh: bool = False) -> dict | None:
    """Task đang chạy, cache TASK_TTL giây để không gọi DRES ở mỗi lần Streamlit chạy lại."""
    cache = st.session_state.get("_dres_task")
    if not fresh and cache and time.time() - cache[0] < TASK_TTL:
        return cache[1]
    t = cli.current_task() if cli.logged_in else None
    st.session_state["_dres_task"] = (time.time(), t)
    return t


def task_name(t: dict | None) -> str:
    return (t or {}).get("name") or (t or {}).get("templateId") or ""


def badge(status: str) -> str:
    return (f'<span class="dres-badge" style="background:{STATUS_COLOR.get(status, ui.INK)}">'
            f'{ui.esc(STATUS_VI.get(status, status))}</span>')


# ------------------------------------------------------------------ thanh bên
def sidebar(cli: DresClient, valid_ids: set[str]) -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    with st.expander("Nộp bài DRES", expanded=not cli.logged_in, icon=":material/send:"):
        if not cli.logged_in:
            _login_form(cli)
            return
        st.markdown(f"Đăng nhập: **{ui.esc(cli.user.get('username', cli.username))}**")
        try:
            evs = cli.evaluations()
        except DresError as e:
            st.error(str(e))
            evs = []
        if evs:
            ids = [e["id"] for e in evs]
            label = {e["id"]: f"{e.get('name', e['id'])} · {e.get('status', '')}" for e in evs}
            default = cli.evaluation_id if cli.evaluation_id in ids else next(
                (e["id"] for e in evs if str(e.get("status")).upper() == "ACTIVE"), ids[0])
            cli.evaluation_id = st.selectbox("Buổi thi", ids, index=ids.index(default), format_func=label.get)
        else:
            st.caption("Tài khoản chưa có buổi thi nào — liên hệ BTC.")
        t = _task(cli)
        kind = task_kind(t)
        st.caption(f"Task đang chạy: **{ui.esc(task_name(t)) or 'không có'}**" + (f" ({kind})" if kind else ""))
        c1, c2 = st.columns(2)
        if c1.button("Làm mới", key="dres_refresh", width="stretch"):
            _task(cli, fresh=True)
            st.rerun()
        if c2.button("Đăng xuất", key="dres_logout", width="stretch"):
            cli.session_id, cli.user = None, {}
            st.rerun()

        def quick_image():
            quick = st.text_input("Nộp theo mã ảnh (KIS)", key="dres_quick",
                                  placeholder="day1_Kitchen_10_1528 hoặc 10_1528.jpg")
            if quick and st.button("Nộp mã này", key="dres_quick_btn", type="primary", width="stretch"):
                item = to_item_name(quick, cli.item_prefix)
                if item not in valid_ids:
                    st.warning(f"{item} không có trong metadata — DRES sẽ báo 'Could not find media item'.")
                else:
                    confirm(cli, item, click=_click())

        def quick_text():
            ans = st.text_input("Nộp đáp án chữ (QA)", key="dres_quick_text", placeholder="vd: 3, red, 09:15")
            if clean_answer(ans) and st.button("Nộp đáp án này", key="dres_quick_text_btn", type="primary",
                                               width="stretch"):
                confirm_text(cli, ans, click=_click())

        # ô của loại task đang chạy lên trước
        for f in ((quick_text, quick_image) if kind == "QA" else (quick_image, quick_text)):
            f()
        history(cli)


def _login_form(cli: DresClient) -> None:
    with st.form("dres_login", border=False):
        u = st.text_input("Username", value=cli.username, placeholder="UIA_TenCuaBan")
        p = st.text_input("Password", value=cli.password, type="password")
        ev = st.text_input("Evaluation ID (tuỳ chọn)", value=cli.evaluation_id,
                           help="Bỏ trống = tự lấy buổi thi đang ACTIVE")
        sid = st.text_input("hoặc Session ID", help="Chép từ trang Profile của DRES nếu không muốn nhập mật khẩu")
        if st.form_submit_button("Đăng nhập", type="primary", width="stretch"):
            cli.username, cli.password, cli.evaluation_id = u.strip(), p, ev.strip()
            try:
                if sid.strip():
                    cli.set_session(sid)
                    cli.evaluations()          # kiểm tra Session ID có dùng được không
                    cli.user = {"username": u or "(session)"}
                else:
                    cli.login()
                st.rerun()
            except DresError as e:
                cli.session_id = None
                st.error(str(e))
    st.caption("Tài khoản do BTC gửi riêng. Có thể đặt sẵn trong file .env (xem .env.example).")


def history(cli: DresClient, n: int = 8) -> None:
    if not cli.history:
        return
    st.markdown("**Đã nộp**")
    for h in reversed(cli.history[-n:]):
        d = h.as_dict()
        label = f"“{ui.esc(h.item_name)}”" if h.kind == "text" else f"<code>{ui.esc(h.item_name)}</code>"
        st.markdown(f'<div class="dres-row">{badge(h.status)}{label}'
                    f'<span style="color:var(--muted)">{d["time"]}</span></div>', unsafe_allow_html=True)


# ------------------------------------------------------------------ trang chính
def status_bar(cli: DresClient) -> None:
    """Dải trạng thái phía trên kết quả: task đang chạy, số lần nộp, đã đúng chưa."""
    if not cli.logged_in:
        return
    t = _task(cli)
    name = task_name(t)
    mine = [h for h in cli.history if h.task == name and h.evaluation_id == cli.evaluation_id and h.ok]
    solved = cli.solved(name)
    kind = task_kind(t)
    parts = [f"<b>DRES</b> · task: <b>{ui.esc(name) or 'chưa có task chạy'}</b>" + (f" ({kind})" if kind else "")]
    if name:
        what = "đáp án" if kind == "QA" else "bài"
        parts.append(f"đã nộp {len(mine)} {what}, {sum(h.status == 'WRONG' for h in mine)} sai")
        if cli.pending(name):
            parts.append(f"{len(cli.pending(name))} đang chờ trọng tài chấm")
        if kind == "QA":
            parts.append("nộp đáp án chữ ở trang <b>Hỏi đáp</b> hoặc thanh bên")
    if solved:
        parts.append(badge("CORRECT") + " team đã có điểm cho task này")
    st.markdown(f'<div class="dres-bar{" solved" if solved else ""}">' + " · ".join(parts) + "</div>",
                unsafe_allow_html=True)


def submit_button(cli: DresClient, image_id: str, key: str) -> None:
    """Nút "Nộp" dưới mỗi ảnh (chỉ hiện khi đã đăng nhập DRES). Bấm là nộp ngay, không hỏi lại."""
    if cli.logged_in and st.button("Nộp", key=f"dres_{key}", type="primary", width="stretch",
                                   help=f"Nộp {image_id} lên DRES ngay"):
        confirm(cli, image_id, click=_click())


def _click() -> int:
    """Số thứ tự lần bấm Nộp: mỗi lần bấm là 1 lần gửi mới (được nộp lại cùng ảnh / đáp án), còn các lần Streamlit
    chạy lại bên trong hộp kết quả thì không gửi lại."""
    st.session_state["_dres_click"] = st.session_state.get("_dres_click", 0) + 1
    return st.session_state["_dres_click"]


def _submit_now(cli: DresClient, value: str, as_text: bool):
    """Nộp ngay (điểm tính cả thời gian nộp -> không có bước xác nhận). Task lấy từ cache để khỏi chờ thêm 1 lần gọi
    DRES. Mỗi lần bấm là 1 lần gửi, kể cả ảnh / đáp án đã nộp. Lỗi -> (None, thông báo)."""
    name = task_name(_task(cli))
    try:
        res = cli.submit_text(value, task=name) if as_text else cli.submit(value, task=name)
    except (DresError, ValueError) as e:
        return None, str(e)
    label = f"“{res.item_name}”" if as_text else res.item_name
    st.toast(f"{STATUS_VI.get(res.status, res.status)}: {label}")
    return res, ""


def _show_result(res, err: str, as_text: bool, kind: str | None) -> None:
    if res is None:
        st.error(err)
        return
    label = f"“{ui.esc(res.item_name)}”" if as_text else f"<code>{ui.esc(res.item_name)}</code>"
    st.markdown(f'<div class="dres-row">{badge(res.status)}{label}</div>', unsafe_allow_html=True)
    ok = res.status == "CORRECT"
    wait = res.status in ("INDETERMINATE", "SKIPPED")
    (st.success if ok else st.info if wait else st.error)(res.message)
    if res.status == "INDETERMINATE" and as_text:
        st.caption("Điểm tính theo thời điểm nộp, không phải lúc trọng tài chấm — không cần nộp thêm để thử.")
    if res.status in ("WRONG_TYPE", "REJECTED") and kind:
        st.caption(f"Task đang chạy là {kind}: " + ("cần nộp đáp án chữ." if kind == "QA" else "cần nộp tên ảnh."))
    if res.status == "ERROR" and res.raw:
        st.code(res.raw[:500])


@st.dialog("Nộp bài lên DRES")
def confirm(cli: DresClient, image_id: str, image_path: str | None = None, click: int = 0) -> None:
    """Nộp ảnh ngay khi mở (giữ tên cũ cho các chỗ gọi), hộp thoại chỉ để báo kết quả."""
    item = to_item_name(image_id, cli.item_prefix)
    key = f"_dres_res_img_{click}_{item}"
    if key not in st.session_state:
        st.session_state[key] = _submit_now(cli, item, as_text=False)
    _show_result(*st.session_state[key], as_text=False, kind=task_kind(_task(cli)))
    if st.button("Đóng", key="dres_close_dlg", width="stretch"):
        st.session_state.pop(key, None)
        st.rerun()             # chạy lại cả trang để thanh bên / dải trạng thái cập nhật


def inline_confirm(cli: DresClient, image_id: str, key: str = "dres_inline_open", label: str | None = None) -> None:
    """Dùng trong cửa sổ "Chi tiết khung hình" (Streamlit không cho mở dialog lồng nhau). Bấm là nộp ngay.
    key/label riêng cho từng nút khi có nhiều nút (các khung liền kề trong cùng sự kiện)."""
    if not cli.logged_in:
        if label is None:
            st.caption("Đăng nhập DRES ở thanh bên để nộp ảnh này.")
        return
    item = to_item_name(image_id, cli.item_prefix)
    done = st.session_state.get("_dres_inline")
    if done and done[0] == item:
        _show_result(*done[1], as_text=False, kind=task_kind(_task(cli)))
    label = label or ("Nộp lại ảnh này" if done and done[0] == item else "Nộp ảnh này lên DRES")
    if st.button(label, key=key, type="primary", width="stretch"):
        st.session_state["_dres_inline"] = (item, _submit_now(cli, item, as_text=False))
        st.rerun(scope="fragment")


# ------------------------------------------------------------------ task QA: đáp án chữ
def answer_box(cli: DresClient, suggestion: str, key: str) -> None:
    """Ô nộp đáp án chữ dưới câu trả lời ở trang Hỏi đáp (điền sẵn đáp án ngắn, sửa được trước khi nộp)."""
    if not cli.logged_in:
        st.caption("Đăng nhập DRES ở thanh bên để nộp đáp án này cho task QA.")
        return
    c1, c2 = st.columns([4, 1], vertical_alignment="bottom")
    ans = c1.text_input("Đáp án nộp lên DRES (task QA)", value=suggestion, key=f"dres_ans_{key}",
                        help="Ngắn gọn, chỉ ghi đáp án, không giải thích (vd: 3, red). Sửa lại nếu cần.")
    if c2.button("Nộp đáp án", key=f"dres_ans_btn_{key}", type="primary", width="stretch",
                 disabled=not clean_answer(ans)):
        confirm_text(cli, ans, click=_click())


@st.dialog("Nộp đáp án QA lên DRES")
def confirm_text(cli: DresClient, answer: str, click: int = 0) -> None:
    """Nộp đáp án chữ ngay khi mở, hộp thoại chỉ để báo kết quả."""
    text = clean_answer(answer)
    key = f"_dres_res_text_{click}_{text}"
    if key not in st.session_state:
        st.session_state[key] = _submit_now(cli, text, as_text=True)
    _show_result(*st.session_state[key], as_text=True, kind=task_kind(_task(cli)))
    if st.button("Đóng", key="dres_text_close", width="stretch"):
        st.session_state.pop(key, None)
        st.rerun()
