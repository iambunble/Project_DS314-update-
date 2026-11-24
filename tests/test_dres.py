"""Test client nộp bài DRES — không gọi mạng: dùng phiên HTTP giả trả lời như DRES."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from retrieval.submit import DresClient, DresError, clean_answer, load_dres_settings, task_kind, to_item_name  # noqa: E402
from retrieval.submit.dres import interpret  # noqa: E402


class Resp:
    def __init__(self, code, body):
        self.status_code = code
        self.text = body if isinstance(body, str) else json.dumps(body)

    def json(self):
        return json.loads(self.text)


class FakeDres:
    """Giả lập DRES: 1 tài khoản, 1 buổi thi ACTIVE, ảnh đúng là day1_Kitchen_12_2702."""

    def __init__(self, task_running=True, qa=False):
        self.qa = qa                      # task đang chạy là QA (đáp án chữ, trọng tài chấm)
        self.texts = set()
        self.sessions = {"s1"}
        self.next_session = 1
        self.task_running = task_running
        self.calls = []

    def post(self, url, json=None, params=None, timeout=None):
        self.calls.append(("POST", url, json, params))
        if url.endswith("/login"):
            if json["password"] != "pw":
                return Resp(401, {"status": False, "description": "Invalid credentials"})
            self.next_session += 1
            sid = f"s{self.next_session}"
            self.sessions.add(sid)
            return Resp(200, {"id": "u", "username": json["username"].lower(), "role": "PARTICIPANT",
                              "sessionId": sid})
        if "/submit/" in url:
            if params["session"] not in self.sessions:
                return Resp(401, {"status": False, "description": "Unauthorized"})
            if not url.endswith("/ev1"):
                return Resp(404, {"status": False, "description": "Evaluation ev9 could not be found."})
            if not self.task_running:
                return Resp(400, {"status": False, "description": "Run manager is in wrong state and cannot accept"})
            ans = json["answerSets"][0]["answers"][0]
            if self.qa:
                if "text" not in ans or "mediaItemName" in ans:
                    return Resp(412, {"status": False, "description": "Submission does not include textual information"})
                if ans["text"] in self.texts:
                    return Resp(412, {"status": False, "description": "Duplicate submission received."})
                self.texts.add(ans["text"])
                return Resp(202, {"status": True, "submission": "INDETERMINATE", "description": "..."})
            if "text" in ans:
                return Resp(412, {"status": False, "description": "Submission rejected by filter"})
            if ans["mediaItemCollectionName"] != "castle5000" or not ans["mediaItemName"].startswith("day1_"):
                return Resp(400, {"status": False, "description": "Could not find media item with name x"})
            ok = ans["mediaItemName"] == "day1_Kitchen_12_2702"
            return Resp(200, {"status": True, "submission": "CORRECT" if ok else "WRONG", "description": "..."})
        raise AssertionError(url)

    def get(self, url, params=None, timeout=None):
        self.calls.append(("GET", url, None, params))
        if params["session"] not in self.sessions:
            return Resp(401, {"status": False, "description": "Unauthorized"})
        if url.endswith("/client/evaluation/list"):
            return Resp(200, [{"id": "ev0", "name": "Luyện tập", "status": "ENDED"},
                              {"id": "ev1", "name": "Vòng 1", "status": "ACTIVE"}])
        if "/currentTask/" in url:
            if self.qa:
                return Resp(200, {"templateId": "t2", "name": "QA01", "taskType": "LSC Question & Answer Text"})
            return Resp(200, {"templateId": "t1", "name": "task-01", "duration": 300, "taskType": "LSC Known-Item Search"})
        raise AssertionError(url)


def make(fake=None, **kw):
    fake = fake or FakeDres()
    return DresClient(username="UIA_Test", password="pw", http=fake, **kw), fake


def test_to_item_name():
    assert to_item_name("10_1528.jpg") == "day1_Kitchen_10_1528"
    assert to_item_name("08_102") == "day1_Kitchen_08_102"
    assert to_item_name("D:\\data\\Batch1\\20_3499.JPG") == "day1_Kitchen_20_3499"
    assert to_item_name("data/images/Batch2/09_706.jpg") == "day1_Kitchen_09_706"
    assert to_item_name("day1_Kitchen_12_2702") == "day1_Kitchen_12_2702"
    assert to_item_name(" day1_Kitchen_12_2702.jpg ") == "day1_Kitchen_12_2702"
    with pytest.raises(ValueError):
        to_item_name("")


def test_interpret_table():
    assert interpret(200, '{"submission":"CORRECT"}')[0] == "CORRECT"
    assert interpret(200, '{"submission":"WRONG"}')[0] == "WRONG"
    assert interpret(202, '{"submission":"INDETERMINATE"}')[0] == "INDETERMINATE"
    assert interpret(400, '{"description":"Run manager is in wrong state"}')[0] == "NO_TASK"
    assert interpret(500, "Could not find media item x")[0] == "BAD_NAME"
    assert interpret(401, "Unauthorized")[0] == "UNAUTHORIZED"
    assert interpret(404, '{"description":"Evaluation x could not be found"}')[0] == "NO_EVALUATION"
    assert interpret(412, "Submission rejected by filter")[0] == "REJECTED"
    assert interpret(418, "teapot")[0] == "ERROR"


def test_login_and_active_evaluation():
    cli, _ = make()
    assert cli.login().startswith("s")
    assert cli.user["role"] == "PARTICIPANT"
    assert cli.resolve_evaluation() == "ev1"
    assert cli.current_task()["name"] == "task-01"


def test_wrong_password():
    cli, _ = make()
    cli.password = "PW"
    with pytest.raises(DresError, match="Đăng nhập thất bại"):
        cli.login()


def test_submit_body_and_results(tmp_path):
    log = tmp_path / "log.jsonl"
    cli, fake = make(log_file=log)
    r = cli.submit("10_1528.jpg", task="task-01")
    assert (r.status, r.http_status, r.ok) == ("WRONG", 200, True)
    method, url, body, params = fake.calls[-1]
    assert url == "https://dres.2hbread.com/api/v2/submit/ev1" and params == {"session": cli.session_id}
    assert body == {"answerSets": [{"answers": [{"mediaItemName": "day1_Kitchen_10_1528",
                                                 "mediaItemCollectionName": "castle5000"}]}]}
    assert cli.submit("day1_Kitchen_12_2702", task="task-01").status == "CORRECT"
    assert cli.solved("task-01") and not cli.solved("task-02")
    lines = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines()]
    assert [x["status"] for x in lines] == ["WRONG", "CORRECT"] and "password" not in lines[0]


def test_resubmit_same_task_is_sent():
    cli, fake = make()
    cli.submit("10_1528.jpg", task="task-01")
    n = len(fake.calls)
    assert cli.submit("10_1528", task="task-01").status == "WRONG" and len(fake.calls) == n + 1   # được nộp lại
    skipped = cli.submit("10_1528", task="task-01", skip_repeat=True)
    assert skipped.status == "SKIPPED" and len(fake.calls) == n + 1                            # chỉ khi yêu cầu


def test_relogin_on_401():
    cli, fake = make()
    cli.login()
    fake.sessions.clear()                     # DRES khởi động lại -> session cũ hết hạn
    r = cli.submit("12_2702.jpg", task="t")
    assert r.status == "CORRECT" and cli.session_id in fake.sessions


def test_no_task_running_is_not_recorded_as_submitted():
    cli, _ = make(FakeDres(task_running=False))
    r = cli.submit("10_1528.jpg")
    assert r.status == "NO_TASK" and not r.ok
    assert cli.already_submitted("day1_Kitchen_10_1528") is None


def test_wrong_evaluation_id():
    cli, _ = make(evaluation_id="ev9")
    assert cli.submit("10_1528.jpg").status == "NO_EVALUATION"


def test_no_active_evaluation():
    fake = FakeDres()
    fake.get = lambda url, params=None, timeout=None: Resp(200, [{"id": "ev0", "status": "ENDED"}])
    cli, _ = make(fake)
    with pytest.raises(DresError, match="ACTIVE"):
        cli.resolve_evaluation()


def test_settings_from_dotenv(tmp_path, monkeypatch):
    for k in ("DRES_USERNAME", "DRES_PASSWORD", "DRES_EVALUATION_ID", "DRES_URL", "DRES_COLLECTION"):
        monkeypatch.delenv(k, raising=False)
    env = tmp_path / ".env"
    env.write_text('# comment\nDRES_USERNAME=UIA_A\nDRES_PASSWORD="p w"\n', encoding="utf-8")
    s = load_dres_settings({"dres": {"collection": "castle5000", "timeout": 5}}, dotenv=env)
    assert (s["username"], s["password"], s["collection"], s["timeout"]) == ("UIA_A", "p w", "castle5000", 5.0)
    monkeypatch.setenv("DRES_USERNAME", "UIA_B")              # biến môi trường thắng .env
    assert load_dres_settings({}, dotenv=env)["username"] == "UIA_B"


# ------------------------------------------------------------------ task QA (đáp án chữ)
def test_task_kind():
    assert task_kind({"taskType": "LSC Question & Answer Text"}) == "QA"
    assert task_kind({"taskType": "LSC Known-Item Search", "name": "QA01"}) == "KIS"
    assert task_kind({"name": "QA03"}) == "QA" and task_kind({"name": "KIS02"}) == "KIS"
    assert task_kind(None) == "" and task_kind({"name": "x"}) == ""
    cli, _ = make(FakeDres(qa=True))
    assert task_kind(cli.current_task()) == "QA"


def test_clean_answer():
    assert clean_answer("  3 ") == "3"
    assert clean_answer("áo\t  đỏ\n") == "áo đỏ"
    assert clean_answer(None) == ""


def test_submit_text_body_and_pending(tmp_path):
    log = tmp_path / "log.jsonl"
    cli, fake = make(FakeDres(qa=True), log_file=log)
    r = cli.submit_text("  3 ", task="QA01")
    assert (r.status, r.http_status, r.ok, r.kind, r.item_name) == ("INDETERMINATE", 202, True, "text", "3")
    method, url, body, params = fake.calls[-1]
    assert url.endswith("/submit/ev1") and params == {"session": cli.session_id}
    assert body == {"answerSets": [{"answers": [{"text": "3"}]}]}          # không kèm tên ảnh
    assert [h.item_name for h in cli.pending("QA01")] == ["3"] and not cli.solved("QA01")
    line = json.loads(log.read_text(encoding="utf-8").splitlines()[0])
    assert line["kind"] == "text" and line["item"] == "3"


def test_resubmit_same_text_is_sent():
    cli, fake = make(FakeDres(qa=True))
    cli.submit_text("3", task="QA01")
    n = len(fake.calls)
    assert cli.submit_text(" 3", task="QA01").status == "DUPLICATE" and len(fake.calls) == n + 1  # gửi lại, DRES trả 412
    assert cli.submit_text("3 người", task="QA01").status == "INDETERMINATE"   # khác 1 ký tự = bài khác
    assert cli.submit_text("3", task="QA01", skip_repeat=True).status == "SKIPPED"
    # ảnh và chữ trùng tên không chặn nhau
    assert cli.already_submitted("3", "QA01", kind="image") is None


def test_empty_text_rejected():
    cli, _ = make(FakeDres(qa=True))
    with pytest.raises(ValueError):
        cli.submit_text("   ")


def test_wrong_type_submissions():
    cli, _ = make(FakeDres(qa=True))
    assert cli.submit("10_1528.jpg", task="QA01").status == "WRONG_TYPE"      # ảnh vào task QA
    cli2, _ = make()
    assert cli2.submit_text("3", task="task-01").status == "REJECTED"         # chữ vào task KIS


def test_interpret_qa_rows():
    assert interpret(202, '{"submission":"INDETERMINATE"}')[0] == "INDETERMINATE"
    assert interpret(412, "Submission does not include textual information")[0] == "WRONG_TYPE"
    assert interpret(412, "Duplicate submission received.")[0] == "DUPLICATE"
