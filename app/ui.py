"""Phần trình bày của giao diện: CSS, chip "hệ thống hiểu", dải thời gian trong ngày, nhãn dưới mỗi ảnh.

Bảng màu lấy từ chính căn bếp trong ảnh, và MỖI MÀU GẮN VỚI 1 LOẠI ĐIỀU KIỆN (giữ nguyên ở chip, dải thời gian,
khung khoanh người):
    vàng dứa  #FFC531  -> thời gian (khung giờ đang lọc)
    đỏ gạch   #FF4A3D  -> số người, kết quả hàng đầu
    bạc hà    #16C49B  -> vùng trong bếp
    cobalt    #2E44F5  -> thao tác chính, các kết quả còn lại
    áo        -> đúng màu áo
"""
from __future__ import annotations

import html

PAPER, INK, MUTED, LINE = "#F4F6FD", "#1A1C3D", "#5D6185", "#D9DDF2"
COBALT, TOMATO, PINE, MINT = "#2E44F5", "#FF4A3D", "#FFC531", "#16C49B"

SHIRT_HEX = {"black": "#26283F", "white": "#FFFFFF", "gray": "#9AA0B5", "red": "#E5322D", "orange": "#FF8A1F",
             "yellow": "#FFD23F", "green": "#2DBE60", "blue": "#2E6BFF", "purple": "#8E4CE0", "pink": "#FF7AB6",
             "brown": "#8B5A2B"}
SHIRT_VI = {"black": "đen", "white": "trắng", "gray": "xám", "red": "đỏ", "orange": "cam", "yellow": "vàng",
            "green": "xanh lá", "blue": "xanh dương", "purple": "tím", "pink": "hồng", "brown": "nâu"}

OBJECT_VI = {"knife": "dao", "pot": "nồi", "frying pan": "chảo", "cutting board": "thớt", "plate": "đĩa", "cup": "cốc",
             "cell phone": "điện thoại"}
POSTURE_VI = {"sitting": "ngồi", "standing": "đứng", "bending": "cúi", "unknown": "không rõ tư thế"}

DAY_START, DAY_END = 8 * 3600, 21 * 3600       # dữ liệu 8:00–20:59

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;800&display=swap');
:root {{ --ink:{INK}; --paper:{PAPER}; --muted:{MUTED}; --line:{LINE};
        --cobalt:{COBALT}; --tomato:{TOMATO}; --pine:{PINE}; --mint:{MINT}; }}
.stApp, .stApp [data-testid="stMarkdownContainer"], .stApp [data-testid="stMarkdownContainer"] *,
.stApp label, .stApp input, .stApp textarea, .stApp button, .stApp button p, .stApp [data-baseweb="select"] *,
.stApp h1, .stApp h2, .stApp h3, .stApp h4, [data-testid="stDialog"] * {{
    font-family:'Be Vietnam Pro', system-ui, sans-serif; }}
[data-testid="stIconMaterial"] {{ font-family: "Material Symbols Rounded" !important; }}
.stApp {{ background: var(--paper); color: var(--ink); }}
[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 1.2rem; max-width: 1400px; }}

/* ---------- đầu trang */
.hero {{ background: var(--cobalt); color:#fff; border-radius: 22px; padding: 22px 28px 20px;
         position: relative; overflow: hidden; margin-bottom: 14px; }}
.hero::after {{ content:""; position:absolute; right:-40px; top:-40px; width:220px; height:220px;
               background: repeating-linear-gradient(0deg, var(--tomato) 0 22px, #ff7a6f 22px 25px);
               transform: rotate(8deg); border-radius: 26px; opacity:.95; }}
.hero::before {{ content:""; position:absolute; right:150px; bottom:-60px; width:120px; height:120px;
                background: var(--pine); border-radius:50%; }}
.hero h1 {{ font-size: clamp(1.6rem, 3vw, 2.4rem); font-weight:800; line-height:1.1; margin:0 0 6px;
           letter-spacing:-.02em; color:#fff; position:relative; z-index:1; max-width: 70%; }}
.hero p {{ margin:0; font-size:1rem; opacity:.92; position:relative; z-index:1; max-width: 60ch; color:#fff; }}
@media (max-width: 760px) {{
  .hero {{ padding: 18px 18px 16px; }}
  .hero::after {{ width:90px; height:90px; right:-34px; top:-34px; }}
  .hero::before {{ display:none; }}
  .hero h1, .hero p {{ max-width: 100%; }}
  .hero h1 {{ padding-right: 40px; }}
}}

/* ---------- ô tìm kiếm */
[data-testid="stTextInput"] [data-baseweb="input"] {{ border:2px solid var(--ink); border-radius:14px;
    background:#fff; }}
[data-testid="stTextInput"] [data-baseweb="input"]:focus-within {{ border-color: var(--cobalt);
    box-shadow: 0 0 0 4px var(--pine); }}
.st-key-q input, .st-key-sq input {{ font-size:1.15rem; padding:14px 16px; }}

/* ---------- chọn chế độ */
[data-testid="stBaseButton-segmented_control"], [data-testid="stBaseButton-segmented_controlActive"] {{
    font-weight:600; padding: 8px 18px; }}
[data-testid="stBaseButton-segmented_controlActive"] {{ background: var(--ink) !important; color:#fff !important;
    border-color: var(--ink) !important; }}

/* ---------- câu mẫu */
[data-testid="stBaseButton-pills"] {{ background:#fff; border:1.5px solid var(--line); }}
[data-testid="stBaseButton-pills"]:hover {{ border-color: var(--cobalt); color: var(--cobalt); }}

/* ---------- nút */
.stButton button {{ border-radius: 10px; font-weight:600; }}
[class*="st-key-s_"] button, [class*="st-key-sim_"] button, [class*="st-key-per_"] button,
[class*="st-key-seq_"] button {{ min-height: 34px; padding: 3px 10px; font-size: .88rem; border-width:1.5px; }}
.stButton button[kind="secondary"] {{ border:1.5px solid var(--ink); background:#fff; color:var(--ink); }}
.stButton button[kind="secondary"]:hover {{ background: var(--pine); border-color: var(--ink); color: var(--ink); }}
button:focus-visible {{ outline: 3px solid var(--pine) !important; outline-offset: 2px; }}

/* ---------- ảnh */
[data-testid="stImage"] img {{ border-radius: 14px; }}
[data-testid="stSidebar"] {{ background: #fff; border-right: 2px solid var(--line); }}

/* ---------- chip "hệ thống hiểu" */
.understood {{ display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin: 4px 0 2px; }}
.understood .lead {{ font-weight:600; color: var(--muted); margin-right:4px; }}
.chip {{ display:inline-flex; align-items:center; gap:7px; padding:6px 12px; border-radius:999px;
        font-weight:600; font-size:.92rem; border:2px solid var(--ink); background:#fff; color:var(--ink); }}
.chip.time {{ background: var(--pine); }}
.chip.count {{ background: var(--tomato); color:#fff; border-color: var(--tomato); }}
.chip.zone {{ background: var(--mint); color:#06261d; border-color: var(--mint); }}
.chip.text {{ background:#fff; font-weight:500; }}
.chip.part {{ background: var(--ink); color:#fff; }}
.chip .sw {{ width:14px; height:14px; border-radius:50%; border:1.5px solid var(--ink); display:inline-block; }}
.meta {{ color: var(--muted); font-size:.9rem; margin: 2px 0 10px; }}
.note {{ background:#fff; border-left: 5px solid var(--cobalt); border-radius: 8px; padding: 8px 12px;
        margin: 6px 0; font-size:.93rem; }}

/* ---------- dải thời gian */
.daybar {{ background:#fff; border:2px solid var(--ink); border-radius:18px; padding: 12px 16px 6px; margin: 10px 0 16px; }}
.daybar .legend {{ display:flex; flex-wrap:wrap; gap:16px; font-size:.84rem; color: var(--muted); margin-top:2px; }}
.daybar .legend i {{ display:inline-block; width:12px; height:12px; border-radius:3px; margin-right:6px;
                    vertical-align:-1px; }}
.daybar svg {{ width:100%; height:96px; display:block; }}
.daybar .hours {{ position:relative; height:18px; font-size:.78rem; color: var(--muted); margin-top:2px; }}
.daybar .hours span {{ position:absolute; transform:translateX(-50%); font-variant-numeric: tabular-nums; }}
.daybar .hours span:first-child {{ transform:none; }}
@media (max-width: 760px) {{ .daybar .hours span:nth-child(even) {{ display:none; }} }}

/* ---------- nhãn dưới ảnh */
.tile {{ margin: 6px 2px 4px; }}
.tile .row {{ display:flex; align-items:baseline; justify-content:space-between; gap:8px; }}
.tile .tc {{ font-weight:800; font-size:1.25rem; letter-spacing:-.01em; font-variant-numeric: tabular-nums; }}
.tile .rank {{ font-weight:800; font-size:.85rem; color:#fff; background: var(--cobalt); border-radius:7px;
              padding:1px 8px; }}
.tile .rank.top {{ background: var(--tomato); }}
.tile .bar {{ height:6px; background: var(--line); border-radius:6px; margin:6px 0 8px; overflow:hidden; }}
.tile .bar span {{ display:block; height:100%; background: var(--cobalt); border-radius:6px; }}
.tile .bar.top span {{ background: var(--tomato); }}
.tags {{ display:flex; flex-wrap:wrap; gap:5px; }}
.tag {{ font-size:.78rem; font-weight:600; padding:2px 8px; border-radius:6px; background: var(--paper);
       border:1px solid var(--line); color: var(--ink); }}
.tag.zone {{ background:#DDF7EF; border-color:#9FE6D2; }}
.tag.people {{ background:#FFE3E0; border-color:#FFB8B1; }}
.tag.group {{ background:#FFF1C9; border-color:#FFD978; }}
.tag .sw {{ width:9px; height:9px; border-radius:50%; display:inline-block; border:1px solid var(--ink);
           margin-right:4px; vertical-align:0; }}
.event {{ font-weight:600; color: var(--cobalt); font-size:.95rem; margin-bottom:4px; }}
.empty {{ background:#fff; border:2px dashed var(--line); border-radius:18px; padding:26px; color:var(--muted); }}
.empty b {{ color: var(--ink); }}
@media (prefers-reduced-motion: reduce) {{ * {{ transition:none !important; animation:none !important; }} }}
</style>
"""


def esc(s) -> str:
    return html.escape(str(s))


def _n(x: int) -> str:
    return f"{x:,}".replace(",", ".")


def hero(n_images: int, n_people: int, first: str, last: str) -> str:
    return (f'<div class="hero"><h1>Tìm khung hình trong căn bếp</h1>'
            f'<p>{_n(n_images)} ảnh từ một camera cố định, {first} đến {last}. Gõ điều bạn nhớ về khung hình: '
            f'ai, mặc gì, đứng ở đâu, lúc mấy giờ. Hệ thống đã nhận ra {_n(n_people)} lượt người trong ảnh.</p></div>')


def shirt_swatch(names, cls="sw") -> str:
    out = []
    for c in names:
        out.append(f'<span class="{cls}" style="background:{SHIRT_HEX.get(c, "#ccc")}"></span>')
    return "".join(out)


def _count_vi(s: str) -> str:
    if ".." in s:
        a, b = s.split("..")
        return f"{a}–{b}"
    op, n = s[:2], s[2:]
    return {"==": "", ">=": "từ ", "<=": "tối đa "}[op] + n


def _person_chip(s: str) -> str:
    if s == "==0":
        return "không có người"
    return f"{_count_vi(s)} người" if not s.startswith(">=") else f"{_count_vi(s)} người trở lên".replace("từ ", "")


def chips(parsed: dict, zone_label) -> str:
    """dict từ ParsedQuery.describe() -> HTML các chip theo màu ý nghĩa."""
    out = []
    if parsed.get("alternatives"):
        for k, a in enumerate(parsed["alternatives"], 1):
            out.append(f'<span class="chip part">Vế {k}</span>' + _chips_one(a, zone_label, with_text=False))
        return '<div class="understood"><span class="lead">Hệ thống hiểu</span>' + "".join(out) + "</div>"
    return '<div class="understood"><span class="lead">Hệ thống hiểu</span>' + _chips_one(parsed, zone_label) + "</div>"


def _chips_one(p: dict, zone_label, with_text: bool = True) -> str:
    out = []
    for t in p.get("time", []):
        out.append(f'<span class="chip time">{esc(t)}</span>')
    if p.get("days"):
        out.append(f'<span class="chip time">ngày {", ".join(map(str, p["days"]))}</span>')
    for cls, cons in p.get("objects", {}).items():
        if cls == "person":
            out.append(f'<span class="chip count">{esc(_person_chip(cons))}</span>')
        else:
            out.append(f'<span class="chip">{esc(OBJECT_VI.get(cls, cls))} {esc(_count_vi(cons))}</span>')
    for c in p.get("clothing", []):
        names = [w for w in c.replace("không ai ", "").replace("có người không mặc ", "").split(" ")[0].split("/")]
        label = "/".join(SHIRT_VI.get(n, n) for n in names)
        if c.startswith("không ai"):
            label = f"không ai mặc {label}"
        elif c.startswith("có người không mặc"):
            label = f"có người không mặc {label}"
        else:
            label = "áo " + label + (f" ×{c.split('×')[1]}" if "×" in c else "")
        out.append(f'<span class="chip">{shirt_swatch(names)}{esc(label)}</span>')
    for z, cons in p.get("zones", {}).items():
        lbl = "không ai ở " + zone_label(z) if cons == "==0" else f"{zone_label(z)} ({_person_chip(cons)})"
        out.append(f'<span class="chip zone">{esc(lbl)}</span>')
    for cls, w in p.get("holding", {}).items():
        lbl = f"cầm {OBJECT_VI.get(cls, cls)}" + ("" if w >= 1 else " (ngầm hiểu)")
        out.append(f'<span class="chip count">{esc(lbl)}</span>')
    for name, cons in p.get("postures", {}).items():
        lbl = f"không ai {POSTURE_VI.get(name, name)}" if cons == "==0" else \
            f"{POSTURE_VI.get(name, name)} ({_person_chip(cons)})"
        out.append(f'<span class="chip count">{esc(lbl)}</span>')
    if with_text and p.get("text"):
        out.append(f'<span class="chip text">“{esc(p["text"])}”</span>')
    return "".join(out)


def _x(sec: float) -> float:
    return (min(max(sec, DAY_START), DAY_END) - DAY_START) / (DAY_END - DAY_START) * 1000


def daybar(activity: list[tuple[int, float]], intervals: list[tuple[int, int]], hits: list[tuple[int, float, bool]],
           caption: str) -> str:
    """activity: [(giây, số người trung bình)], intervals: khung giờ đang lọc, hits: [(giây, điểm 0..1, top?)]."""
    W, H, base = 1000, 96, 92
    ns = 'vector-effect="non-scaling-stroke"'
    peak = max([a for _, a in activity] + [1e-6])
    parts = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{esc(caption)}" preserveAspectRatio="none">']
    for a, b in intervals:                                       # khung giờ đang lọc
        x1, x2 = _x(a), _x(b)
        if x2 > x1:
            parts.append(f'<rect x="{x1:.1f}" y="4" width="{x2 - x1:.1f}" height="{base - 4}" '
                         f'fill="{PINE}" opacity=".55"/>')
    if activity:                                                 # mức đông người trong ngày
        pts = " ".join(f"{_x(t):.1f},{base - a / peak * 46:.1f}" for t, a in activity)
        parts.append(f'<polygon points="0,{base} {pts} {W},{base}" fill="{LINE}"/>')
    for t, s, top in sorted(hits, key=lambda h: h[2]):           # kết quả: đỏ = top 3
        x = _x(t)
        y = base - 16 - s * 64
        col = TOMATO if top else COBALT
        parts.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{base}" y2="{y:.1f}" stroke="{col}" '
                     f'stroke-width="{3 if top else 2}" stroke-linecap="round" {ns}/>')
        parts.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{y:.1f}" y2="{y:.1f}" stroke="{col}" '
                     f'stroke-width="{12 if top else 8}" stroke-linecap="round" {ns}/>')
    parts.append(f'<line x1="0" x2="{W}" y1="{base}" y2="{base}" stroke="{INK}" stroke-width="2" {ns}/>')
    for h in range(8, 22):
        x = _x(h * 3600)
        parts.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{base - 4}" y2="{base}" stroke="{INK}" '
                     f'stroke-width="1.5" {ns}/>')
    parts.append("</svg>")
    hours = "".join(f'<span style="left:{_x(h * 3600) / 10:.2f}%">{h}h</span>' for h in range(8, 21))
    parts.append(f'<div class="hours">{hours}</div>')
    items = [f'<span><i style="background:{LINE}"></i>bếp đông người lúc nào</span>']
    if intervals:
        items.append(f'<span><i style="background:{PINE}"></i>khung giờ đang lọc</span>')
    if hits:
        items += [f'<span><i style="background:{TOMATO}"></i>3 kết quả đầu</span>',
                  f'<span><i style="background:{COBALT}"></i>kết quả khác</span>']
    if caption:
        items.append(f'<span>{esc(caption)}</span>')
    legend = f'<div class="legend">{"".join(items)}</div>'
    return f'<div class="daybar">{"".join(parts)}{legend}</div>'


def tile(rank: int | None, timecode: str, score_frac: float, n_people: int, zones: list[str], shirts: list[str],
         group: int, top: bool, event: str | None = None, group_span: str | None = None) -> str:
    rank_html = f'<span class="rank{" top" if top else ""}">#{rank}</span>' if rank else ""
    tags = [f'<span class="tag people">{n_people} người</span>' if n_people else
            '<span class="tag">không có người</span>']
    tags += [f'<span class="tag zone">{esc(z)}</span>' for z in zones[:3]]
    if len(zones) > 3:
        tags.append(f'<span class="tag zone" title="{esc(", ".join(zones[3:]))}">+{len(zones) - 3} vùng</span>')
    if shirts:
        tags.append('<span class="tag">' + "".join(
            f'<span class="sw" style="background:{SHIRT_HEX.get(c, "#ccc")}" title="{SHIRT_VI.get(c, c)}"></span>'
            for c in shirts) + "áo</span>")
    if group:
        span = f" · {esc(group_span)}" if group_span else ""
        tags.append(f'<span class="tag group" title="Mở Xem chi tiết để thấy và nộp từng khung trong nhóm">'
                    f'+{group} khung liền kề{span}</span>')
    ev = f'<div class="event">{esc(event)}</div>' if event else ""
    bar = (f'<div class="bar{" top" if top else ""}"><span style="width:{max(4, min(100, score_frac * 100)):.0f}%">'
           f'</span></div>') if score_frac is not None else '<div style="height:8px"></div>'
    return (f'<div class="tile">{ev}<div class="row"><span class="tc">{esc(timecode)}</span>{rank_html}</div>'
            f'{bar}<div class="tags">{"".join(tags)}</div></div>')


def note(text: str, raw: bool = False) -> str:
    return f'<div class="note">{text if raw else esc(text)}</div>'


def empty(text_html: str) -> str:
    return f'<div class="empty">{text_html}</div>'
