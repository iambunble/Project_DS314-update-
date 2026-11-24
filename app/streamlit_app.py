"""Giao diện web:  streamlit run app/streamlit_app.py

Màu sắc, chip, dải thời gian... nằm ở app/ui.py; theme gốc ở .streamlit/config.toml.
Nộp bài DRES (đăng nhập, nút "Nộp", lịch sử) nằm ở app/dres_ui.py.
"""
import sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))
import dres_ui  # noqa: E402
import ui  # noqa: E402
from retrieval.features.similar import similar_person, similar_to_id  # noqa: E402
from retrieval.features.temporal import search_sequence  # noqa: E402
from retrieval.qa import QAEngine  # noqa: E402
from retrieval.search.engine import ManualFilters, SearchEngine  # noqa: E402

st.set_page_config(page_title="Tìm khung hình trong bếp", page_icon="🍍", layout="wide")
st.markdown(ui.CSS, unsafe_allow_html=True)

PAGES = ["Tìm theo mô tả", "Hỏi đáp", "Chuỗi sự kiện", "Ảnh tương tự", "Theo người"]
QA_EXAMPLES = ["Có bao nhiêu người lúc 9 giờ?", "Lúc mấy giờ thầy Linh mở tủ lạnh?", "Thầy Linh mặc áo gì?",
               "Thầy Linh thường đứng ở đâu?", "Ai đang nấu ở bếp lúc 18 giờ?", "Có ai ngồi ở sofa lúc 8 giờ không?",
               "Buổi sáng có mấy lần có người mở tủ lạnh?", "Buổi chiều bếp trống bao lâu?",
               "Lúc nào trong bếp đông người nhất?", "Lần đầu tiên có người nấu ăn là lúc mấy giờ?",
               "Sau khi cắt dứa thì mọi người làm gì?"]
EXAMPLES = ["hai người đứng ở bếp nấu lúc 12 giờ trưa", "người áo xanh dương đang rửa bát",
            "có người ngồi trên sofa buổi tối", "không có ai ở bếp nấu vào buổi chiều",
            "ba người trở lên quanh đảo bếp", "a man in a striped sweater"]
SEQ_EXAMPLES = ["một người cắt dứa -> mọi người ngồi ăn ở bàn", "bếp không có ai rồi có người vào nấu ăn"]
COLS = 4


# ============================================================ dữ liệu
@st.cache_resource(show_spinner="Đang nạp mô hình và chỉ mục...")
def get_engine():
    e = SearchEngine.from_config()
    if e.translator is not None:
        e.translator.warmup()        # tải model dịch ngay (~20 giây lần đầu) để lần tìm đầu tiên không phải chờ
        if e.translator.failed:
            e.index.warnings.append(e.translator.failed + " -> đang tìm bằng câu tiếng Việt (chưa dịch).")
    return e


eng = get_engine()
df = eng.catalog.df
HAS_PERSON_EMB = eng.index.person_emb is not None
dres = dres_ui.client(eng.cfg, ROOT)


@st.cache_resource
def day_activity():
    """Số người trung bình theo từng 10 phút trong ngày -> nền của dải thời gian."""
    people = eng._counts("person").astype(float)
    bins = (eng.sec_of_day // 600).astype(int)
    out = []
    for b in np.unique(bins):
        out.append((int(b * 600 + 300), float(people[bins == b].mean())))
    return out


def zone_label(z):
    return eng.layout.label(z) if eng.layout else z


def timecode(rec) -> str:
    return rec["time"].split(" ")[-1]


# ============================================================ vẽ ảnh
def _font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:            # Pillow cũ
        return ImageFont.load_default()


@st.cache_data(max_entries=512, show_spinner=False)
def boxed(path: str, hl: tuple, others: tuple) -> Image.Image:
    """Ảnh thu nhỏ: người khớp câu hỏi khoanh vàng dày, người khác viền trắng mảnh."""
    im = Image.open(path).convert("RGB")
    im.thumbnail((720, 720))
    W, H = im.size
    d = ImageDraw.Draw(im)
    for x1, y1, x2, y2 in others:
        d.rectangle([x1 * W, y1 * H, x2 * W, y2 * H], outline=(255, 255, 255), width=1)
    for x1, y1, x2, y2 in hl:
        d.rectangle([x1 * W - 1, y1 * H - 1, x2 * W + 1, y2 * H + 1], outline=(26, 28, 61), width=6)
        d.rectangle([x1 * W, y1 * H, x2 * W, y2 * H], outline=(255, 197, 49), width=4)
    return im


@st.cache_data(max_entries=128, show_spinner=False)
def numbered(path: str, boxes: tuple) -> Image.Image:
    """Ảnh lớn trong cửa sổ chi tiết: mỗi người 1 khung cobalt có số thứ tự (khớp danh sách bên cạnh)."""
    im = Image.open(path).convert("RGB")
    W, H = im.size
    d = ImageDraw.Draw(im)
    f = _font(max(18, W // 50))
    for k, (x1, y1, x2, y2) in enumerate(boxes, 1):
        d.rectangle([x1 * W, y1 * H, x2 * W, y2 * H], outline=(46, 68, 245), width=max(3, W // 320))
        tw = d.textlength(str(k), font=f)
        d.rectangle([x1 * W, y1 * H, x1 * W + tw + 14, y1 * H + f.size + 10], fill=(46, 68, 245))
        d.text((x1 * W + 7, y1 * H + 4), str(k), fill=(255, 255, 255), font=f)
    return im


@st.cache_data(max_entries=1024, show_spinner=False)
def person_crop(path: str, box: tuple) -> Image.Image:
    im = Image.open(path).convert("RGB")
    W, H = im.size
    x1, y1, x2, y2 = box
    return im.crop((int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H)))


# ============================================================ điều hướng
def go(page: str, **state):
    """Ghi lại yêu cầu chuyển trang; được áp dụng ở đầu lần chạy sau (trước khi tạo widget)."""
    st.session_state["_pending"] = {"page": page, **state}


pending = st.session_state.pop("_pending", None)
if pending:
    for k, v in pending.items():
        st.session_state[k] = v
st.session_state.setdefault("page", PAGES[0])


def use_example(src_key: str, dst_key: str):
    if st.session_state.get(src_key):
        st.session_state[dst_key] = st.session_state[src_key]
    st.session_state[src_key] = None


# ============================================================ cửa sổ chi tiết
@st.dialog("Chi tiết khung hình", width="large")
def detail(image_id: str, group_ids: tuple = ()):
    rec = eng.record(eng.catalog.index_of(image_id))
    people = rec.get("people", [])
    left, right = st.columns([3, 2], gap="large")
    with left:
        if rec.get("image_path"):
            st.image(numbered(rec["image_path"], tuple(tuple(p["box"]) for p in people)), width="stretch")
        st.markdown(ui.tile(None, timecode(rec), None, len(people),
                            sorted({z for p in people for z in p["zone_labels"]}),
                            [p["colors"][0] for p in people if p["colors"]], 0, False), unsafe_allow_html=True)
        st.caption(rec["imageID"])
        st.write(rec["description"])
        if st.button("Tìm ảnh tương tự", type="primary", key="dlg_sim"):
            go("Ảnh tương tự", sim_id=image_id)
            st.rerun()
        dres_ui.inline_confirm(dres, image_id)
    with right:
        st.markdown("**Người trong ảnh**" if people else "**Không thấy người nào trong ảnh**")
        for k, p in enumerate(people, 1):
            c1, c2 = st.columns([1, 3])
            with c1:
                if rec.get("image_path"):
                    st.image(person_crop(rec["image_path"], tuple(p["box"])), width=96)
            with c2:
                shirts = "/".join(ui.SHIRT_VI.get(c, c) for c in p["colors"]) or "không rõ"
                st.markdown(f'<div class="tags"><span class="tag people">{k}</span>'
                            f'<span class="tag">{ui.shirt_swatch(p["colors"])}áo {ui.esc(shirts)}</span>'
                            + "".join(f'<span class="tag zone">{ui.esc(z)}</span>' for z in p["zone_labels"])
                            + (f'<span class="tag">{ui.esc(ui.POSTURE_VI.get(p["posture"], p["posture"]))}</span>'
                               if p.get("posture") and p["posture"] != "unknown" else "")
                            + "".join(f'<span class="tag">cầm {ui.esc(ui.OBJECT_VI.get(c, c))}</span>'
                                      for h in rec.get("hands", []) if h["det"] == p["det"] for c in h["holding"])
                            + "</div>", unsafe_allow_html=True)
                if p["pid"] is not None and st.button("Tìm người này", key=f"dlg_p_{p['det']}"):
                    go("Theo người", person_ref=(image_id, p["pid"], tuple(p["box"])))
                    st.rerun()
        if rec.get("detected"):
            st.markdown("**Nhận diện được**")
            st.markdown('<div class="tags">' + "".join(
                f'<span class="tag">{ui.esc(ui.OBJECT_VI.get(c, c))} × {n}</span>' for c, n in rec["detected"].items())
                + "</div>",
                unsafe_allow_html=True)
    event_strip(image_id, group_ids)


EVENT_SECONDS = 90


def event_strip(image_id: str, group_ids: tuple = ()):
    """MỌI khung trong ±90 giây (mở rộng để phủ hết nhóm đã gộp), xếp theo điểm của lần tìm gần nhất. Hệ thống hay
    tìm đúng sự kiện nhưng chọn nhầm khung bên trong (đề 23/11: đáp án lệch 63 giây bị chấm sai) -> lướt ở đây để
    nộp đúng khung. Không cắt bớt: khung đáp án có điểm thấp vẫn phải hiện."""
    idx = eng.catalog.index_of(image_id)
    members = {eng.catalog.index_of(g) for g in group_ids}
    scores = st.session_state.get("_last_scores")
    frames = eng.event_frames(idx, EVENT_SECONDS, scores, sorted(members))
    if len(frames) <= 1:
        return
    by = "theo điểm khớp với mô tả" if scores is not None else "theo thời gian"
    span = f"{timecode(eng.catalog.record(min(frames, key=lambda f: eng.timestamp[f[0]])[0]))}–"            f"{timecode(eng.catalog.record(max(frames, key=lambda f: eng.timestamp[f[0]])[0]))}"
    st.markdown(f"**Tất cả {len(frames)} khung cùng sự kiện ({span}), xếp {by}** — so kỹ chi tiết rồi nộp đúng "
                f"khung. ★ = điểm cao nhất · ● = thuộc nhóm của ô vừa bấm")
    cols = st.columns(6, gap="small")
    for k, (i, _) in enumerate(frames):
        rec = eng.catalog.record(i)
        with cols[k % 6]:
            if rec.get("image_path"):
                st.image(rec["image_path"], width="stretch")
            mark = ("★ " if k == 0 and scores is not None else "") + ("● " if i in members or i == idx else "")
            here = " (đang xem)" if i == idx else ""
            st.caption(f"{mark}#{k + 1} · {timecode(rec)}{here}")
            dres_ui.inline_confirm(dres, rec["imageID"], key=f"dres_ev_{rec['imageID']}", label="Nộp")


# ============================================================ lưới kết quả
def grid(items: list[dict], key: str):
    """items: {rec, rank, frac, top, group, hl, event}"""
    if not items:
        st.markdown(ui.empty("Không có khung hình nào khớp. Thử bỏ bớt điều kiện giờ hoặc vị trí."),
                    unsafe_allow_html=True)
        return
    for row in range(0, len(items), COLS):
        cols = st.columns(COLS, gap="medium")
        for col, it in zip(cols, items[row:row + COLS]):
            rec = it["rec"]
            people = rec.get("people", [])
            hl = {tuple(b) for b in it.get("hl", [])}
            with col:
                if rec.get("image_path"):
                    st.image(boxed(rec["image_path"], tuple(hl),
                                   tuple(tuple(p["box"]) for p in people if tuple(p["box"]) not in hl)),
                             width="stretch")
                else:
                    st.markdown(ui.empty("Không tìm thấy file ảnh"), unsafe_allow_html=True)
                zs = sorted({z for p in people for z in p["zone_labels"]},
                            key=lambda z: (not any(z in p["zone_labels"] for p in people if tuple(p["box"]) in hl), z))
                st.markdown(ui.tile(it.get("rank"), timecode(rec), it.get("frac"), len(people), zs,
                                    [p["colors"][0] for p in people if p["colors"]][:6],
                                    it.get("group", 0), it.get("top", False), it.get("event"),
                                    group_span(rec, it.get("group_ids", ()))),
                            unsafe_allow_html=True)
                b1, b2 = st.columns([3, 2]) if dres.logged_in else (st.container(), None)
                if b1.button("Xem chi tiết", key=f"{key}_{rec['imageID']}_{row}", width="stretch"):
                    detail(rec["imageID"], tuple(it.get("group_ids", ())))
                if b2 is not None:
                    with b2:
                        dres_ui.submit_button(dres, rec["imageID"], f"{key}_{rec['imageID']}_{row}")


def ranked(results, hl_fn=lambda rec: [], score=lambda r: r["score"]) -> list[dict]:
    best = max([score(r) for r in results] + [1e-9])
    return [{"rec": r["record"], "rank": k, "frac": score(r) / best, "top": k <= 3,
             "group": len(r.get("group", [])), "group_ids": list(r.get("group", [])), "hl": hl_fn(r)}
            for k, r in enumerate(results, 1)]


def group_span(rec: dict, group_ids):
    """'12:04:48–12:06:10': nhóm khung đã gộp phủ những phút nào (ảnh đại diện + các khung trong nhóm)."""
    if not group_ids:
        return None
    idx = [eng.catalog.index_of(rec["imageID"])] + [eng.catalog.index_of(g) for g in group_ids]
    lo, hi = min(idx, key=lambda i: eng.timestamp[i]), max(idx, key=lambda i: eng.timestamp[i])
    return f"{timecode(eng.catalog.record(lo))}–{timecode(eng.catalog.record(hi))}"


def day(hits_recs, intervals=(), caption=""):
    hits = []
    best = max([s for _, s in hits_recs] + [1e-9])
    for k, (rec, s) in enumerate(hits_recs):
        sec = int(df.at[eng.catalog.index_of(rec["imageID"]), "sec_of_day"])
        hits.append((sec, s / best, k < 3))
    st.markdown(ui.daybar(day_activity(), list(intervals), hits, caption), unsafe_allow_html=True)


# ============================================================ thanh bên
with st.sidebar:
    dres_ui.sidebar(dres, set(df["imageID"]))
    st.markdown("### Bộ lọc")
    hours = sorted(df["hour"].unique())
    use_hours = st.toggle("Chỉ trong khung giờ", value=False)
    h = st.select_slider("Khung giờ", options=hours, value=(hours[0], hours[-1]), disabled=not use_hours,
                         format_func=lambda x: f"{x}h")
    zone_opts = {zone_label(z): z for z in eng.zone_names}
    zones = st.multiselect("Có người ở", list(zone_opts), placeholder="Vùng bất kỳ")
    top_k = st.slider("Số kết quả", 8, 96, 20, step=4)
    if eng.translator is not None:
        eng.use_translation = st.toggle(
            "Dịch câu sang tiếng Anh", value=True,
            help="CLIP và mô tả ảnh đều là tiếng Anh nên dịch giúp tìm đúng hơn (đề 23/11: câu dài tiếng Việt làm CLIP "
                 "xếp ảnh đúng ở hạng 900–3000). Mỗi câu MỚI tốn ~1–3 giây trên CPU; câu đã dịch được nhớ lại.")
    dedup = st.toggle("Gộp khung hình liền kề", value=bool(eng.sc.get("dedup_seconds")),
                      help="Ảnh cách nhau khoảng 10 giây nên nhiều ảnh gần như trùng. Gộp lại để thấy nhiều "
                           "thời điểm khác nhau hơn.")
    if len(eng.catalog.owners) > 1:
        owners = st.multiselect("Địa điểm", eng.catalog.owners)
    else:
        owners = []
    days = sorted(df["day_num"].unique())
    pick_days = st.multiselect("Ngày", days) if len(days) > 1 else []
    st.caption("Bộ lọc ở đây được ưu tiên hơn phần giờ/địa điểm hệ thống đọc được từ câu.")
    zones_img = ROOT / "docs" / "zones.jpg"
    if zones_img.exists():
        with st.expander("Các vùng trong bếp"):
            st.image(str(zones_img), width="stretch")
            st.caption("Người được tính là ở một vùng khi chân họ nằm trong khung đó.")
    for w in eng.index.warnings:
        st.warning(w)
    _src = {"clip_tags": "CLIP theo vùng", "yoloe": "YOLOE đồ vật", "pose": "tư thế + màu áo theo pose",
            "hands": "đang cầm gì"}
    if eng.extras.sources():
        st.caption("Nhận diện bổ sung đang bật: " + ", ".join(_src.get(x, x) for x in eng.extras.sources()) + ".")

manual = ManualFilters(hour_from=h[0] if use_hours else None, hour_to=h[1] if use_hours else None,
                       owners=owners or None, days=pick_days or None,
                       zones=[zone_opts[z] for z in zones] or None, dedup_seconds=None if dedup else 0)

# ============================================================ đầu trang + chế độ
n_people = int(eng._counts("person").sum())
st.markdown(ui.hero(len(df), n_people, f"{min(hours)}:00", f"{max(hours)}:59"), unsafe_allow_html=True)
page = st.segmented_control("Chế độ", PAGES, key="page", label_visibility="collapsed") or PAGES[0]
dres_ui.status_bar(dres)

# ------------------------------------------------------------ 1. tìm theo mô tả
if page == "Tìm theo mô tả":
    q = st.text_input("Mô tả khung hình", key="q", label_visibility="collapsed",
                      placeholder="Ví dụ: hai người đứng ở bếp nấu lúc 12 giờ trưa")
    st.pills("Câu mẫu", EXAMPLES, key="ex", on_change=use_example, args=("ex", "q"),
             label_visibility="collapsed")
    if q:
        resp = eng.search(q, top_k, manual)
        st.session_state["_last_scores"] = resp.scores      # để cửa sổ chi tiết xếp các khung cùng sự kiện
        pq = eng.parser.parse(q)
        parts = pq.alternatives or [pq]
        want_z = {z for p in parts for z in p.zones if not (p.zones[z].op == "==" and p.zones[z].n == 0)}
        want_c = {c for p in parts for g in p.colors for c in g}
        st.markdown(ui.chips(resp.parsed, zone_label), unsafe_allow_html=True)
        st.markdown(f'<div class="meta">{resp.n_candidates} ảnh thoả bộ lọc, tìm trong '
                    f'{resp.elapsed_ms:.0f} ms</div>', unsafe_allow_html=True)
        if resp.notes:
            st.markdown(ui.note("<br>".join(ui.esc(n) for n in resp.notes), raw=True), unsafe_allow_html=True)

        def hl(r):
            return [pp["box"] for pp in r["record"].get("people", [])
                    if set(pp["zones"]) & want_z or (pp["colors"][:1] and pp["colors"][0] in want_c)]
        res = [{"record": r.record, "score": r.score, "group": r.group} for r in resp.results]
        day([(r["record"], r["score"]) for r in res], pq.intervals or
            ([(h[0] * 3600, (h[1] + 1) * 3600)] if use_hours else []),
            f"{len(res)} kết quả trong ngày")
        grid(ranked(res, hl), "s")
    else:
        day([], caption="")
        st.markdown(ui.empty(
            "<b>Gõ điều bạn nhớ về khung hình</b>, hoặc bấm một câu mẫu phía trên.<br>"
            "Hệ thống hiểu giờ (<i>lúc 3 rưỡi chiều, buổi tối</i>), số người (<i>ba người trở lên</i>), "
            "màu áo (<i>áo xanh, không ai mặc áo đỏ</i>) và chỗ đứng (<i>ở bếp nấu, trên sofa, gần camera</i>). "
            "Gõ tiếng Việt có dấu hoặc tiếng Anh."), unsafe_allow_html=True)

# ------------------------------------------------------------ 2. chuỗi sự kiện
elif page == "Chuỗi sự kiện":
    sq = st.text_input("Chuỗi sự kiện", key="sq", label_visibility="collapsed",
                       placeholder="Các sự kiện theo thứ tự, ngăn bằng 'rồi', 'sau đó' hoặc ->")
    st.pills("Câu mẫu", SEQ_EXAMPLES, key="sex", on_change=use_example, args=("sex", "sq"),
             label_visibility="collapsed")
    if sq:
        try:
            chains = search_sequence(eng, sq, top_n=5)
        except ValueError:
            chains = None
            st.markdown(ui.note("Cần ít nhất 2 sự kiện, ngăn bằng 'rồi', 'sau đó', '->' hoặc 'then'."),
                        unsafe_allow_html=True)
        if chains is not None and not chains:
            st.markdown(ui.empty("Không tìm được chuỗi nào theo đúng thứ tự trong vòng 1 giờ."),
                        unsafe_allow_html=True)
        for n, s in enumerate(chains or [], 1):
            st.markdown(f"#### Chuỗi {n}")
            cols = st.columns(max(len(s.steps), 2), gap="medium")
            for k, (col, stp) in enumerate(zip(cols, s.steps), 1):
                rec = stp["record"]
                with col:
                    if rec.get("image_path"):
                        st.image(boxed(rec["image_path"], (), tuple(tuple(p["box"]) for p in rec.get("people", []))),
                                 width="stretch")
                    st.markdown(ui.tile(None, timecode(rec), None, len(rec.get("people", [])),
                                        sorted({z for p in rec.get("people", []) for z in p["zone_labels"]}), [], 0,
                                        False, event=f"Bước {k}: {stp['event']}"), unsafe_allow_html=True)
                    if st.button("Xem chi tiết", key=f"seq_{n}_{k}", width="stretch"):
                        detail(rec["imageID"])
                    dres_ui.submit_button(dres, rec["imageID"], f"seq_{n}_{k}")
            st.divider()

# ------------------------------------------------------------ 3. ảnh tương tự
elif page == "Ảnh tương tự":
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        sid = st.text_input("Mã ảnh gốc", key="sim_id", placeholder="vd: day1_Kitchen_12_2702")
    with c2:
        cond = st.text_input("Điều kiện thêm", key="sim_cond", placeholder="vd: buổi chiều, có 2 người")
    with c3:
        skip = st.toggle("Bỏ khung ±60 giây", value=True)
    if not sid:
        st.markdown(ui.empty("Mở <b>Xem chi tiết</b> của một ảnh rồi bấm <b>Tìm ảnh tương tự</b>, "
                             "hoặc dán mã ảnh vào ô trên."), unsafe_allow_html=True)
    elif sid not in set(df["imageID"]):
        st.markdown(ui.note(f"Không có ảnh mã {sid}. Mã có dạng day1_Kitchen_<giờ>_<frame>."),
                    unsafe_allow_html=True)
    else:
        ref = eng.record(eng.catalog.index_of(sid))
        a, b = st.columns([1, 2], gap="large")
        with a:
            st.markdown("**Ảnh gốc**")
            if ref.get("image_path"):
                st.image(numbered(ref["image_path"], tuple(tuple(p["box"]) for p in ref.get("people", []))),
                         width="stretch")
            st.markdown(ui.tile(None, timecode(ref), None, len(ref.get("people", [])),
                                sorted({z for p in ref.get("people", []) for z in p["zone_labels"]}),
                                [p["colors"][0] for p in ref.get("people", []) if p["colors"]], 0, False),
                        unsafe_allow_html=True)
        with b:
            res = similar_to_id(eng, sid, top_k, cond, exclude_near_seconds=60 if skip else 0)
            day([(r["record"], r["score"]) for r in res], caption="các khung giống ảnh gốc nằm ở đâu trong ngày")
        grid(ranked(res), "sim")

# ------------------------------------------------------------ hỏi đáp (offline)
elif page == "Hỏi đáp":
    qq = st.text_input("Câu hỏi", key="qa_q", label_visibility="collapsed",
                       placeholder="Ví dụ: Lúc mấy giờ thầy Linh mở tủ lạnh?")
    st.pills("Câu hỏi mẫu", QA_EXAMPLES, key="qa_ex", on_change=use_example, args=("qa_ex", "qa_q"),
             label_visibility="collapsed")
    if qq:
        ans = QAEngine(eng).ask(qq)
        badge = {"cao": "🟢 tin cậy cao", "trung bình": "🟡 tin cậy trung bình", "thấp": "🔴 tin cậy thấp"}
        with st.container(border=True):
            st.markdown(f"**{ui.esc(ans.text)}**")
            st.caption(f"{badge.get(ans.confidence, ans.confidence)} · loại câu hỏi: {ans.kind}")
            if ans.short:
                st.code(ans.short, language=None)
            dres_ui.answer_box(dres, ans.short, key=str(abs(hash(qq))))
        with st.expander("Hệ thống tìm ra câu trả lời thế nào"):
            for d in ans.details:
                st.markdown(f"- {ui.esc(d)}")
        recs = [eng.record(eng.catalog.index_of(i)) for i in ans.evidence]
        if recs:
            st.markdown(f'<div class="meta">{len(recs)} ảnh làm bằng chứng (quan trọng nhất trước)</div>',
                        unsafe_allow_html=True)
            grid([{"rec": r, "rank": k, "top": k == 1} for k, r in enumerate(recs, 1)], "qa")
    else:
        st.markdown(ui.empty(
            "<b>Hỏi bằng tiếng Việt</b> về những gì xảy ra trong bếp. Hệ thống trả lời từ dữ liệu đã có "
            "(giờ, số người / đồ vật, chỗ đứng, trang phục, người đã biết như thầy Linh) và đưa ảnh làm bằng chứng.<br>"
            "Hiểu các dạng: <i>bao nhiêu / mấy</i>, <i>lúc mấy giờ / khi nào</i>, <i>mặc áo gì</i>, "
            "<i>ở đâu</i>, <i>ai</i>, <i>có … không</i>. Chạy offline, không cần mạng."), unsafe_allow_html=True)

# ------------------------------------------------------------ 4. theo người
else:
    ref = st.session_state.get("person_ref")
    if not HAS_PERSON_EMB:
        st.markdown(ui.note("Chưa build CLIP cho từng người. Chạy: python scripts/build_index.py --stages crops"),
                    unsafe_allow_html=True)
    elif not ref:
        st.markdown(ui.empty("Mở <b>Xem chi tiết</b> của một ảnh có người, rồi bấm <b>Tìm người này</b> bên cạnh "
                             "người bạn muốn theo dõi. Hệ thống so ngoại hình (quần áo, dáng người), "
                             "không nhận diện khuôn mặt."), unsafe_allow_html=True)
    else:
        iid, pid, box = ref
        rec = eng.record(eng.catalog.index_of(iid))
        a, b = st.columns([1, 5], gap="large")
        with a:
            if rec.get("image_path"):
                st.image(person_crop(rec["image_path"], box), width="stretch")
            st.markdown(f'<div class="tile"><span class="tc">{timecode(rec)}</span></div>', unsafe_allow_html=True)
        with b:
            c1, c2 = st.columns([3, 2])
            with c1:
                pcond = st.text_input("Điều kiện thêm", key="pcond", placeholder="vd: buổi tối, ở bếp nấu")
            with c2:
                near = st.slider("Bỏ các khung trong ± giây", 0, 600, 60, step=30)
            res = similar_person(eng, pid, top_k, pcond, exclude_near_seconds=near)
            day([(r["record"], r["score"]) for r in res], caption="người này xuất hiện lúc nào")
        grid(ranked(res, lambda r: [r["match"]["box"]] if r.get("match") else []), "per")
