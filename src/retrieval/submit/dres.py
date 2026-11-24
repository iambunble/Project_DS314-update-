"""Nộp bài lên DRES (cuộc thi Castle5000) — theo tài liệu "Hướng dẫn sử dụng DRES".

Luồng:  login() -> lấy Session ID  ·  active_evaluations() -> Evaluation ID đang ACTIVE
        ·  submit(tên ảnh)       — task KIS (mục 7)
        ·  submit_text(đáp án)   — task QA  (mục 8)

    POST {url}/submit/{evaluation_id}?session={session_id}
    KIS: {"answerSets": [{"answers": [{"mediaItemName": "day1_Kitchen_10_1528", "mediaItemCollectionName": "castle5000"}]}]}
    QA : {"answerSets": [{"answers": [{"text": "3"}]}]}          # KHÔNG gửi kèm tên ảnh

Task QA do trọng tài chấm: DRES thường trả 202 INDETERMINATE (đang chờ chấm). Đáp án chỉ khác 1 ký tự ("3" và
"3 người") là 2 lần nộp khác nhau — mỗi lần bị chấm sai đều bị trừ điểm.

Tài khoản KHÔNG để trong config.yaml (file này có trong git). Đặt bằng biến môi trường hoặc file .env (đã gitignore):
    DRES_USERNAME=UIA_TenCuaBan
    DRES_PASSWORD=...
    DRES_EVALUATION_ID=...      # tuỳ chọn; bỏ trống = tự lấy buổi thi đang ACTIVE

Được nộp nhiều lần trong 1 task (kể cả nộp lại đúng ảnh / đáp án cũ) -> mặc định client gửi hết lên DRES, không chặn.
Lịch sử nộp vẫn được nhớ để hiện trên giao diện; skip_repeat=True nếu muốn bỏ qua ảnh / đáp án đã nộp trong task.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

DEFAULT_URL = "https://dres.2hbread.com/api/v2"
DEFAULT_COLLECTION = "castle5000"
DEFAULT_PREFIX = "day1_Kitchen_"


class DresError(RuntimeError):
    """Lỗi khi nói chuyện với DRES (đăng nhập sai, không có buổi thi ACTIVE, mất mạng...)."""


# ------------------------------------------------------------------ tên ảnh
def to_item_name(name: str, prefix: str = DEFAULT_PREFIX) -> str:
    """Tên file / đường dẫn / imageID -> tên ảnh DRES.  '10_1528.jpg' -> 'day1_Kitchen_10_1528'.

    Nộp '10_1528' hoặc '10_1528.jpg' sẽ bị DRES báo "Could not find media item" (mục 2 tài liệu)."""
    stem = re.split(r"[\\/]", str(name).strip())[-1]
    if re.search(r"\.(jpe?g|png|webp|bmp)$", stem, re.I):
        stem = stem.rsplit(".", 1)[0]
    if not stem:
        raise ValueError("Tên ảnh rỗng")
    if re.match(r"^day\d+_", stem):          # đã là imageID (day1_Kitchen_..., hoặc ngày/địa điểm khác)
        return stem
    return prefix + stem


def clean_answer(text: str) -> str:
    """Đáp án QA: cắt dấu cách đầu/cuối, gộp dấu cách liền nhau (mục 8.4: không thêm dấu cách thừa)."""
    return re.sub(r"\s+", " ", str(text or "")).strip()


def task_kind(task: dict | None) -> str:
    """'QA' | 'KIS' | '' — dựa vào taskType ('LSC Question & Answer Text' / 'LSC Known-Item Search'),
    nếu không có thì đoán theo tên task (QA01 / KIS01)."""
    if not task:
        return ""
    tt = str(task.get("taskType") or task.get("taskGroup") or "")
    if "question" in tt.lower():
        return "QA"
    if tt:
        return "KIS"
    name = str(task.get("name") or "").upper()
    return "QA" if name.startswith("QA") else "KIS" if name.startswith("KIS") else ""


# ------------------------------------------------------------------ cấu hình
def _read_dotenv(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def load_dres_settings(cfg: dict | None = None, dotenv: Path | None = None) -> dict[str, Any]:
    """Gộp phần `dres:` trong config.yaml + biến môi trường + file .env (biến môi trường thắng .env)."""
    d = dict((cfg or {}).get("dres") or {})
    if dotenv is None:
        dotenv = Path(__file__).resolve().parents[3] / ".env"
    env = {**_read_dotenv(dotenv), **{k: v for k, v in os.environ.items() if k.startswith("DRES_")}}
    return {
        "url": (env.get("DRES_URL") or d.get("url") or DEFAULT_URL).rstrip("/"),
        "collection": env.get("DRES_COLLECTION") or d.get("collection") or DEFAULT_COLLECTION,
        "item_prefix": d.get("item_prefix") or DEFAULT_PREFIX,
        "username": env.get("DRES_USERNAME") or "",
        "password": env.get("DRES_PASSWORD") or "",
        "evaluation_id": env.get("DRES_EVALUATION_ID") or d.get("evaluation_id") or "",
        "timeout": float(d.get("timeout", 10)),
        "log_file": d.get("log_file", "artifacts/dres_submissions.jsonl"),
    }


# ------------------------------------------------------------------ kết quả
# (mã HTTP, đoạn chữ trong phản hồi) -> (trạng thái, lời khuyên) — bảng 7.3 + 8.6 của tài liệu
_HINTS = [
    (400, "wrong state", "NO_TASK", "Hiện không có task nào đang chạy (chưa bắt đầu hoặc đã hết giờ)."),
    (412, "textual information", "WRONG_TYPE",
     "Task đang chạy là QA nhưng bài nộp không có đáp án chữ (đã gửi tên ảnh?). Nộp câu trả lời bằng chữ."),
    (412, "duplicate", "DUPLICATE", "Đáp án / ảnh này team đã nộp rồi — không bị tính thêm."),
    (None, "could not find media item", "BAD_NAME",
     "Sai tên ảnh hoặc sai collection. Tên phải có dạng day1_Kitchen_HH_frame, collection castle5000."),
    (401, "", "UNAUTHORIZED", "Session ID sai hoặc đã hết hạn — cần đăng nhập lại."),
    (404, "", "NO_EVALUATION", "Sai Evaluation ID, hoặc tài khoản không thuộc buổi thi này."),
    (412, "", "REJECTED", "Bị từ chối: đã nộp bài này rồi, team đã nộp đúng cho task này, hoặc gửi nhầm loại bài "
                          "(chữ vào task KIS / ảnh vào task QA)."),
]
_STATUS_VI = {"CORRECT": "ĐÚNG — team đã có điểm cho task này, không cần nộp thêm.",
              "WRONG": "SAI — bị trừ điểm phạt, có thể tìm tiếp và nộp lại.",
              "INDETERMINATE": "Đã nhận, đang chờ trọng tài chấm."}


@dataclass
class SubmitResult:
    item_name: str              # tên ảnh (KIS) hoặc đáp án chữ (QA)
    http_status: int
    status: str                 # CORRECT | WRONG | INDETERMINATE | NO_TASK | BAD_NAME | UNAUTHORIZED | ... | ERROR
    message: str                # lời giải thích tiếng Việt
    raw: str = ""               # nội dung phản hồi gốc (gửi BTC khi gặp lỗi lạ)
    evaluation_id: str = ""
    task: str = ""
    at: float = field(default_factory=time.time)
    kind: str = "image"         # image (KIS) | text (QA)

    @property
    def ok(self) -> bool:
        return self.status in ("CORRECT", "WRONG", "INDETERMINATE")

    def as_dict(self) -> dict:
        return {"kind": self.kind, "item": self.item_name, "http": self.http_status, "status": self.status, "message": self.message,
                "raw": self.raw[:500], "evaluation_id": self.evaluation_id, "task": self.task,
                "time": time.strftime("%H:%M:%S", time.localtime(self.at))}


def interpret(http_status: int, text: str) -> tuple[str, str]:
    """Đọc phản hồi DRES -> (trạng thái, lời giải thích)."""
    body: Any = None
    try:
        body = json.loads(text) if text else None
    except ValueError:
        pass
    if isinstance(body, dict) and body.get("submission") in _STATUS_VI:
        return body["submission"], _STATUS_VI[body["submission"]]
    low = (text or "").lower()
    for code, needle, status, msg in _HINTS:
        if (code is None or code == http_status) and needle in low:
            return status, msg
    desc = body.get("description") if isinstance(body, dict) else None
    if 200 <= http_status < 300:
        return "OK", desc or "DRES đã nhận bài."
    return "ERROR", f"Lỗi {http_status}: {desc or (text or '')[:200]}. Chụp lại mã + nội dung này gửi BTC."


# ------------------------------------------------------------------ client
class DresClient:
    """Client DRES dùng chung cho giao diện, API và dòng lệnh. An toàn khi gọi từ nhiều luồng (Streamlit)."""

    def __init__(self, url: str = DEFAULT_URL, username: str = "", password: str = "",
                 collection: str = DEFAULT_COLLECTION, evaluation_id: str = "", item_prefix: str = DEFAULT_PREFIX,
                 timeout: float = 10, log_file: str | Path | None = None, http: requests.Session | None = None):
        self.url = url.rstrip("/")
        self.username, self.password = username.strip(), password
        self.collection, self.item_prefix, self.timeout = collection, item_prefix, timeout
        self.evaluation_id = evaluation_id.strip()
        self.session_id: str | None = None
        self.user: dict = {}
        self.http = http or requests.Session()
        self.log_file = Path(log_file) if log_file else None
        self.history: list[SubmitResult] = []
        self._lock = threading.Lock()

    @classmethod
    def from_settings(cls, s: dict, root: Path | None = None, **over) -> "DresClient":
        log = s.get("log_file")
        if log and root and not Path(log).is_absolute():
            log = root / log
        kw = dict(url=s["url"], username=s["username"], password=s["password"], collection=s["collection"],
                  evaluation_id=s["evaluation_id"], item_prefix=s["item_prefix"], timeout=s["timeout"], log_file=log)
        kw.update({k: v for k, v in over.items() if v is not None})
        return cls(**kw)

    # ---------------------------------------------------------- đăng nhập
    @property
    def logged_in(self) -> bool:
        return bool(self.session_id)

    def login(self) -> str:
        """POST /login -> sessionId. Mật khẩu phân biệt hoa/thường, username thì không."""
        if not self.username or not self.password:
            raise DresError("Chưa có tài khoản DRES (đặt DRES_USERNAME / DRES_PASSWORD trong .env hoặc nhập trên giao diện).")
        try:
            r = self.http.post(f"{self.url}/login", json={"username": self.username, "password": self.password},
                               timeout=self.timeout)
        except requests.RequestException as e:
            raise DresError(f"Không kết nối được DRES ({self.url}): {e}") from e
        if r.status_code != 200:
            raise DresError(f"Đăng nhập thất bại ({r.status_code}). Kiểm tra lại mật khẩu (phân biệt hoa/thường) "
                            f"và dấu cách thừa khi sao chép. Phản hồi: {r.text[:200]}")
        data = r.json()
        self.session_id, self.user = data["sessionId"], data
        return self.session_id

    def set_session(self, session_id: str) -> None:
        """Dùng Session ID chép từ trang Profile (https://dres.2hbread.com/user) thay vì đăng nhập."""
        self.session_id = session_id.strip() or None

    def _get(self, path: str, retry: bool = True) -> Any:
        if not self.session_id:
            self.login()
        try:
            r = self.http.get(f"{self.url}{path}", params={"session": self.session_id}, timeout=self.timeout)
        except requests.RequestException as e:
            raise DresError(f"Không kết nối được DRES: {e}") from e
        if r.status_code == 401 and retry and self.password:
            self.login()
            return self._get(path, retry=False)
        if r.status_code != 200:
            raise DresError(f"{path}: lỗi {r.status_code} — {interpret(r.status_code, r.text)[1]}")
        return r.json()

    # ---------------------------------------------------------- buổi thi
    def evaluations(self) -> list[dict]:
        """GET /client/evaluation/list — các buổi thi tài khoản được tham gia."""
        return list(self._get("/client/evaluation/list") or [])

    def active_evaluations(self) -> list[dict]:
        return [e for e in self.evaluations() if str(e.get("status", "")).upper() == "ACTIVE"]

    def resolve_evaluation(self) -> str:
        """Evaluation ID đặt sẵn, nếu không thì buổi thi ACTIVE đầu tiên."""
        if self.evaluation_id:
            return self.evaluation_id
        active = self.active_evaluations()
        if not active:
            raise DresError("Chưa có buổi thi nào đang ACTIVE.")
        self.evaluation_id = active[0]["id"]
        return self.evaluation_id

    def current_task(self, evaluation_id: str | None = None) -> dict | None:
        """GET /client/evaluation/currentTask/{id} — task đang chạy (tên, thời lượng). None nếu không có / lỗi."""
        try:
            return self._get(f"/client/evaluation/currentTask/{evaluation_id or self.resolve_evaluation()}")
        except DresError:
            return None

    # ---------------------------------------------------------- nộp bài
    def already_submitted(self, item_name: str, task: str = "", kind: str = "image") -> SubmitResult | None:
        """Ảnh / đáp án này đã nộp (và DRES đã nhận) trong cùng task chưa? Đáp án so sau khi cắt dấu cách."""
        if kind == "text":
            item_name = clean_answer(item_name)
        for h in reversed(self.history):
            if (h.kind == kind and h.item_name == item_name and h.ok and h.task == task
                    and h.evaluation_id == self.evaluation_id):
                return h
        return None

    def pending(self, task: str = "") -> list[SubmitResult]:
        """Các đáp án QA trong task này đang chờ trọng tài chấm."""
        return [h for h in self.history if h.kind == "text" and h.status == "INDETERMINATE" and h.task == task
                and h.evaluation_id == self.evaluation_id]

    def solved(self, task: str = "") -> bool:
        return any(h.status == "CORRECT" and h.task == task and h.evaluation_id == self.evaluation_id
                   for h in self.history)

    def submit(self, name: str, evaluation_id: str | None = None, task: str = "",
               skip_repeat: bool = False) -> SubmitResult:
        """Nộp MỘT ảnh. `name` có thể là imageID, tên file '10_1528.jpg' hoặc đường dẫn.

        Tự đăng nhập lại 1 lần khi gặp 401. Ảnh đã nộp trong task vẫn được gửi lại, trừ khi skip_repeat=True."""
        item = to_item_name(name, self.item_prefix)
        with self._lock:
            if skip_repeat:
                prev = self.already_submitted(item, task)
                if prev:
                    return SubmitResult(item, 0, "SKIPPED",
                                        f"Đã nộp ảnh này lúc {prev.as_dict()['time']} ({prev.status}) — không nộp lại.",
                                        evaluation_id=prev.evaluation_id, task=task)
            ev = evaluation_id or self.resolve_evaluation()
            answer = {"mediaItemName": item, "mediaItemCollectionName": self.collection}
            return self._send(ev, item, answer, task, "image")

    def submit_text(self, answer: str, evaluation_id: str | None = None, task: str = "",
                    skip_repeat: bool = False) -> SubmitResult:
        """Task QA: nộp MỘT câu trả lời bằng chữ (mục 8). Body chỉ có trường `text`, không kèm tên ảnh.

        Đáp án được cắt dấu cách thừa. Đáp án y hệt đã nộp vẫn được gửi lại (DRES tự xử lý), trừ khi skip_repeat=True."""
        text = clean_answer(answer)
        if not text:
            raise ValueError("Câu trả lời rỗng")
        with self._lock:
            if skip_repeat:
                prev = self.already_submitted(text, task, kind="text")
                if prev:
                    return SubmitResult(text, 0, "SKIPPED",
                                        f"Đã nộp đáp án này lúc {prev.as_dict()['time']} ({prev.status}) — không nộp lại.",
                                        evaluation_id=prev.evaluation_id, task=task, kind="text")
            ev = evaluation_id or self.resolve_evaluation()
            return self._send(ev, text, {"text": text}, task, "text")

    def _send(self, ev: str, item: str, answer: dict, task: str, kind: str) -> SubmitResult:
        """Gửi 1 đáp án (đang giữ khoá), tự đăng nhập lại 1 lần khi 401, ghi lịch sử + log."""
        res = self._post_submit(ev, item, answer, task, kind)
        if res.status == "UNAUTHORIZED" and self.password:
            self.login()
            res = self._post_submit(ev, item, answer, task, kind)
        self.history.append(res)
        self._log(res)
        return res

    def _post_submit(self, ev: str, item: str, answer: dict, task: str, kind: str) -> SubmitResult:
        if not self.session_id:
            self.login()
        body = {"answerSets": [{"answers": [answer]}]}
        try:
            r = self.http.post(f"{self.url}/submit/{ev}", params={"session": self.session_id}, json=body,
                               timeout=self.timeout)
        except requests.RequestException as e:
            return SubmitResult(item, 0, "ERROR", f"Không kết nối được DRES: {e}", evaluation_id=ev, task=task,
                                kind=kind)
        status, msg = interpret(r.status_code, r.text)
        return SubmitResult(item, r.status_code, status, msg, raw=r.text, evaluation_id=ev, task=task, kind=kind)

    def _log(self, res: SubmitResult) -> None:
        if not self.log_file:
            return
        try:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps({**res.as_dict(), "user": self.username}, ensure_ascii=False) + "\n")
        except OSError:
            pass
