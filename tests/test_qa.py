"""Hỏi đáp offline: nhận loại câu hỏi và tách phần sự việc để tìm ảnh."""
import pytest

from retrieval.qa.answer import classify, hhmm, strip_question


@pytest.mark.parametrize("q,kind", [
    ("Có bao nhiêu người lúc 9 giờ?", "count"),
    ("Lúc 12 giờ trưa có mấy người trong bếp?", "count"),
    ("Có bao nhiêu cái cốc trên bàn?", "count"),
    ("Lúc mấy giờ thầy Linh mở tủ lạnh?", "when"),
    ("Khi nào có người ngồi trên sofa?", "when"),
    ("Thầy Linh mặc áo gì?", "wear"),
    ("Người đang nấu ở bếp mặc áo màu gì?", "wear"),
    ("Ai mặc áo đỏ?", "who"),
    ("Thầy Linh thường đứng ở đâu?", "where"),
    ("Ai đang rửa bát ở bồn rửa?", "who"),
    ("Có ai ngồi ở sofa lúc 8 giờ không?", "yesno"),
    ("Thầy Linh có ở bếp nấu không?", "yesno"),
    ("một người đàn ông mở tủ lạnh", "find"),
])
def test_classify(q, kind):
    assert classify(q)[0] == kind


def test_mấy_giờ_is_not_a_count():
    assert classify("Mấy giờ thì có người nấu ăn?")[0] == "when"


def test_count_captures_object():
    m = classify("Có bao nhiêu cái cốc trên bàn?")[1]
    assert "cốc" in m.group("what")


@pytest.mark.parametrize("q,rest", [
    ("Lúc mấy giờ thầy Linh mở tủ lạnh?", "thầy Linh mở tủ lạnh"),
    ("Có bao nhiêu người lúc 9 giờ?", "người lúc 9 giờ"),
    ("Thầy Linh mặc áo gì?", "Thầy Linh"),
    ("Có ai ngồi ở sofa lúc 8 giờ không?", "ngồi ở sofa lúc 8 giờ"),
    ("Thầy Linh thường đứng ở đâu?", "Thầy Linh đứng"),
])
def test_strip_question(q, rest):
    assert strip_question(q) == rest


def test_hhmm():
    assert hhmm(9 * 3600 + 26 * 60 + 13) == "09:26"
    assert hhmm(9 * 3600 + 26 * 60 + 13, with_sec=True) == "09:26:13"


@pytest.mark.parametrize("q,session", [
    ("Có bao nhiêu người trong buổi thu dữ liệu này", True),
    ("Có bao nhiêu người trong buổi thu dữ liệu này?", True),
    ("Tổng cộng có mấy người tham gia?", True),
    ("Trong bộ dữ liệu có bao nhiêu người khác nhau?", True),
    ("Có bao nhiêu người lúc 9 giờ?", False),
    ("Lúc 12 giờ trưa có mấy người trong bếp?", False),
    ("Buổi sáng có bao nhiêu người?", False),
])
def test_session_question(q, session):
    from retrieval.qa.answer import is_session_question
    assert classify(q)[0] == "count"
    assert is_session_question(q) is session


def test_participants_file_loads():
    from retrieval.qa.answer import load_participants
    y = load_participants({})
    assert y and len(y["participants"]) >= 10
    groups = [g for p in y["participants"] for g in p.get("groups", [])]
    assert len(groups) == len(set(groups)), "1 nhóm trang phục chỉ thuộc 1 người"
