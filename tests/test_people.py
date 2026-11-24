"""Người đã biết (thầy Linh): nhận tên trong đề, khớp khung tham chiếu, điểm phân biệt pos/neg."""
import numpy as np

from retrieval.query.parser import QueryParser
from retrieval.vision import people as pp

PEOPLE = {"linh": {"names": ["thầy linh", "thầy", "linh"], "as": "người đàn ông",
                   "not_before": ["tinh", "hoạt"]}}


def _parser():
    return QueryParser(people=PEOPLE)


def test_name_detected_and_replaced():
    for q in ["Ảnh có thầy Linh đứng ở bếp", "thầy đang rửa chén", "Linh mặc áo len xanh"]:
        pq = _parser().parse(q)
        assert pq.people == ["linh"], q
        assert "linh" not in pq.text and "thầy" not in pq.text
        assert "người đàn ông" in pq.text
        assert pq.about_people


def test_compound_words_are_not_names():
    for q in ["trên bàn có đồ đạc linh tinh", "một người di chuyển linh hoạt"]:
        pq = _parser().parse(q)
        assert pq.people == [], q
        assert "linh" in pq.text


def test_no_people_config_keeps_old_behaviour():
    pq = QueryParser().parse("Thầy Linh cũng có trong bức ảnh đó mặc áo len màu xanh")
    assert pq.people == [] and "người đàn ông" in pq.text


def test_people_kept_on_time_alternatives():
    pq = _parser().parse("buổi sáng thầy Linh nấu ăn, buổi tối thầy ngồi sofa")
    if pq.alternatives:                                   # câu nhiều vế thời gian -> mỗi vế giữ tên
        assert all(a.people == ["linh"] for a in pq.alternatives)
    assert pq.people == ["linh"]


def test_resolve_refs_by_iou():
    ids = ["day1_Kitchen_08_1", "day1_Kitchen_08_2"]
    dets = [[{"c": "person", "b": [0.1, 0.1, 0.3, 0.6]}, {"c": "person", "b": [0.5, 0.2, 0.7, 0.9]}],
            [{"c": "person", "b": [0.2, 0.2, 0.4, 0.8]}]]
    ref = np.array([[0, 0], [0, 1], [1, 0]])
    rows, missing = pp.resolve_refs(["day1_Kitchen_08_1:0.51,0.21,0.70,0.88", "day1_Kitchen_08_2:0.2,0.2,0.4,0.8",
                                     "day1_Kitchen_08_1:0.9,0.9,1.0,1.0", "day1_Kitchen_09_9:0,0,1,1"], ids, dets, ref)
    assert rows == [1, 2] and missing == 2


def test_classifier_separates_people():
    rng = np.random.default_rng(0)
    target, other = rng.normal(size=16), rng.normal(size=16)
    X = np.stack([target + 0.3 * rng.normal(size=16) for _ in range(30)]
                 + [other + 0.3 * rng.normal(size=16) for _ in range(30)]).astype(np.float32)
    pos, neg = list(range(12)), list(range(30, 42))
    prob, cv = pp.classify(X, pos, neg)
    assert cv is not None and cv > 0.9
    assert (prob[12:30] > 0.5).all() and (prob[42:] < 0.5).all()   # người chưa gán nhãn được chia đúng
    assert (prob[pos] == 1).all() and (prob[neg] == 0).all()       # đã gán nhãn giữ nguyên nhãn


def test_color_hist_blue_vs_grey():
    blue = np.zeros((10, 10, 3), np.uint8); blue[..., 2] = 160; blue[..., 0] = 40; blue[..., 1] = 70
    grey = np.full((10, 10, 3), 128, np.uint8)
    hb, hg = pp.color_hist(blue), pp.color_hist(grey)
    assert abs(hb.sum() - 3.0) < 1e-5                      # sắc độ + bão hoà + sáng, mỗi nhóm cộng 1
    assert hb[7] == 1 and hb[:12].sum() == 1               # 210–240°: xanh dương
    assert hg[:12].sum() == 0 and hg[12:16].sum() == 1     # xám: chỉ có mức xám
    assert pp.color_hist(np.zeros((0, 0, 3), np.uint8)).sum() == 0


def test_person_bonus_clipped():
    book = pp.PeopleBook(PEOPLE, {"linh": np.array([0.1, 0.5, 0.7, 0.9, 1.0], np.float32)}, lo=0.5, hi=0.9)
    b = book.person_bonus("linh")
    assert np.allclose(b, [0, 0, 0.5, 1, 1], atol=1e-5)
