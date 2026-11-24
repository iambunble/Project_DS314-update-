"""REST API (để nối với web/app khác):  uvicorn app.api:app --reload
   GET /search?q=...&k=20&dedup=30   GET /sequence?q=...   GET /similar/{image_id}?cond=...   GET /image/{image_id}
   GET /people/{image_id}  (danh sách người + pid)        GET /person/{pid}?cond=...  (tìm người đó ở ảnh khác)
   Nộp bài DRES (tài khoản lấy từ .env hoặc POST /dres/login):
   POST /dres/login {"username","password"}   GET /dres/status   GET /dres/history
   POST /dres/submit/{image_id}  (task KIS)   ·   POST /dres/answer {"text": "3"}  (task QA)"""
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from retrieval.features.similar import similar_person, similar_to_id  # noqa: E402
from retrieval.features.temporal import search_sequence  # noqa: E402
from retrieval.qa import QAEngine  # noqa: E402
from retrieval.search.engine import ManualFilters, SearchEngine  # noqa: E402
from retrieval.submit import DresClient, DresError, load_dres_settings, task_kind  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
app = FastAPI(title="Image Retrieval")
eng = SearchEngine.from_config()
dres = DresClient.from_settings(load_dres_settings(eng.cfg), ROOT)


@app.get("/search")
def search(q: str, k: int = 20, dedup: int | None = None):
    r = eng.search(q, k, ManualFilters(dedup_seconds=dedup) if dedup is not None else None)
    return {"parsed": r.parsed, "n_candidates": r.n_candidates, "elapsed_ms": r.elapsed_ms, "notes": r.notes,
            "results": [{"imageID": x.imageID, "score": x.score, "components": x.components, "group": x.group,
                         **x.record} for x in r.results]}


@app.get("/ask")
def ask(q: str):
    """Hỏi đáp offline: câu trả lời + đáp án ngắn + ảnh bằng chứng."""
    return QAEngine(eng).ask(q).as_dict()


@app.get("/people/{image_id}")
def people(image_id: str):
    try:
        return eng.people(eng.catalog.index_of(image_id))
    except KeyError:
        raise HTTPException(404, "imageID không tồn tại")


@app.get("/person/{pid}")
def person(pid: int, k: int = 20, cond: str = "", near: int = 60):
    if eng.index.person_emb is None or not 0 <= pid < len(eng.index.person_emb):
        raise HTTPException(404, "pid không tồn tại (hoặc chưa build bước crops)")
    return similar_person(eng, pid, k, cond, exclude_near_seconds=near)


@app.get("/sequence")
def sequence(q: str, k: int = 10):
    return [{"score": s.score, "steps": [{k2: v for k2, v in st.items() if k2 != "record"} for st in s.steps]}
            for s in search_sequence(eng, q, k)]


@app.get("/similar/{image_id}")
def similar(image_id: str, k: int = 20, cond: str = ""):
    try:
        return similar_to_id(eng, image_id, k, cond, exclude_near_seconds=60)
    except KeyError:
        raise HTTPException(404, "imageID không tồn tại")


@app.get("/image/{image_id}")
def image(image_id: str):
    try:
        rec = eng.record(eng.catalog.index_of(image_id))
    except KeyError:
        raise HTTPException(404)
    if not rec["image_path"]:
        raise HTTPException(404, "không có file ảnh")
    return FileResponse(rec["image_path"])


# ============================================================ nộp bài DRES
class Login(BaseModel):
    username: str = ""
    password: str = ""
    evaluation_id: str = ""


@app.post("/dres/login")
def dres_login(body: Login):
    if body.username:
        dres.username, dres.password = body.username.strip(), body.password
    if body.evaluation_id:
        dres.evaluation_id = body.evaluation_id.strip()
    try:
        dres.login()
    except DresError as e:
        raise HTTPException(401, str(e))
    return {"username": dres.user.get("username"), "role": dres.user.get("role")}


@app.get("/dres/status")
def dres_status():
    try:
        evs = dres.evaluations()
        ev = dres.resolve_evaluation()
    except DresError as e:
        raise HTTPException(503, str(e))
    task = dres.current_task(ev)
    return {"evaluations": evs, "evaluation_id": ev, "task": task, "task_kind": task_kind(task)}


@app.post("/dres/submit/{image_id}")
def dres_submit(image_id: str, skip_repeat: bool = False):
    """Nộp 1 ảnh, gửi ngay (kể cả ảnh đã nộp trong task). skip_repeat=true: bỏ qua ảnh đã nộp (SKIPPED)."""
    try:
        ev = dres.resolve_evaluation()
        task = dres.current_task(ev) or {}
        r = dres.submit(image_id, ev, task=task.get("name") or task.get("templateId") or "", skip_repeat=skip_repeat)
    except (DresError, ValueError) as e:
        raise HTTPException(503, str(e))
    return r.as_dict()


class Answer(BaseModel):
    text: str
    skip_repeat: bool = False


@app.post("/dres/answer")
def dres_answer(body: Answer):
    """Task QA: nộp 1 câu trả lời bằng chữ. Thường trả INDETERMINATE (chờ trọng tài chấm).
    Đáp án đã nộp trong task vẫn được gửi lại, trừ khi skip_repeat=true."""
    try:
        ev = dres.resolve_evaluation()
        task = dres.current_task(ev) or {}
        r = dres.submit_text(body.text, ev, task=task.get("name") or task.get("templateId") or "", skip_repeat=body.skip_repeat)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except DresError as e:
        raise HTTPException(503, str(e))
    return r.as_dict()


@app.get("/dres/history")
def dres_history():
    return [h.as_dict() for h in reversed(dres.history)]
