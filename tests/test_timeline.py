"""Câu hỏi theo thời gian: nhận loại câu, gộp ảnh khớp thành từng "lần", tính thời lượng (không cần model)."""
import numpy as np
import pytest

from retrieval.qa import timeline as tl
from retrieval.qa.answer import classify


@pytest.mark.parametrize("q,kind", [
    ("Thầy Linh mở tủ lạnh bao nhiêu lần?", "times"),
    ("Có mấy lần có người mở tủ lạnh trong buổi sáng?", "times"),
    ("Thầy Linh ở trong bếp bao lâu?", "duration"),
    ("Bếp trống trong bao lâu vào buổi chiều?", "duration"),
    ("Thầy Linh nấu ăn mấy phút?", "duration"),
    ("Lúc nào trong bếp đông người nhất?", "extreme"),
    ("Buổi sáng giờ nào vắng nhất?", "extreme"),
    ("Lần đầu tiên có người nấu ăn là lúc mấy giờ?", "when"),
    # các câu cũ không đổi loại
    ("Có bao nhiêu người lúc 9 giờ?", "count"),
    ("Mấy giờ thì có người nấu ăn?", "when"),
    ("Từ 9h đến 11h có bao nhiêu người khác nhau vào bếp?", "count"),
])
def test_classify_temporal(q, kind):
    assert classify(q)[0] == kind


def test_extreme_direction():
    assert tl.EXTREME.search("lúc nào đông người nhất").group("max")
    assert tl.EXTREME.search("giờ nào vắng nhất").group("min")
    assert tl.EXTREME.search("khi nào ít người nhất").group("min")


@pytest.mark.parametrize("q,expect", [
    ("Sau khi cắt dứa thì mọi người làm gì?", ("sau", "cắt dứa", "mọi người làm gì?")),
    ("Trước khi nấu ăn, thầy Linh làm gì?", ("trước", "nấu ăn", "thầy Linh làm gì?")),
    ("Ai rửa bát sau khi ăn xong?", ("sau", "ăn xong", "Ai rửa bát")),
    ("Thầy Linh mở tủ lạnh lúc mấy giờ?", None),
])
def test_split_relative(q, expect):
    assert tl.split_relative(q) == expect


def test_strip_time_words():
    assert tl.strip_time_words("Thầy Linh mở tủ lạnh bao nhiêu lần?") == "Thầy Linh mở tủ lạnh"
    assert tl.strip_time_words("Lần đầu tiên có người nấu ăn là lúc mấy giờ?") == "có người nấu ăn là"


def test_episodes_merges_close_hits():
    t = np.arange(0, 200, 10)                       # 20 ảnh, cách nhau 10 giây
    hit = np.zeros(20, bool)
    hit[[2, 3, 5, 6]] = True                        # 1 ảnh không khớp xen giữa -> vẫn 1 lần
    hit[[12, 13]] = True                            # cách xa -> lần thứ 2
    assert tl.episodes(t, hit, max_miss=1) == [(2, 6), (12, 13)]


def test_episodes_split_on_misses_and_gaps():
    t = np.array([0, 10, 20, 30, 40, 400, 410])
    hit = np.array([1, 0, 0, 1, 1, 1, 1], bool)
    # 2 ảnh trượt liên tiếp > max_miss=1 -> tách; khoảng trống 360 giây > max_gap -> tách
    assert tl.episodes(t, hit, max_miss=1, max_gap=120) == [(0, 0), (3, 4), (5, 6)]
    assert tl.episodes(t, hit, max_miss=2, max_gap=500) == [(0, 6)]


def test_episodes_empty():
    assert tl.episodes(np.array([0, 10]), np.array([False, False])) == []
    assert tl.episodes(np.array([], int), np.array([], bool)) == []


def test_episode_seconds_and_format():
    t = np.array([0, 10, 70])
    assert tl.episode_seconds(t, (0, 2), step=10) == 80
    assert tl.episode_seconds(t, (1, 1), step=10) == 10
    assert tl.fmt_duration(45) == "45 giây"
    assert tl.fmt_duration(600) == "10 phút"
    assert tl.fmt_duration(3900) == "1 giờ 5 phút"
    assert tl.fmt_duration(7200) == "2 giờ"


def test_hit_threshold_is_relative():
    s = np.array([0.1, 0.2, 0.2, 0.3, 1.1])
    thr = tl.hit_threshold(s, 0.8)
    assert 0.2 < thr < 1.1 and (s >= thr).sum() == 1
    assert (s * 10 >= tl.hit_threshold(s * 10, 0.8)).sum() == 1     # không phụ thuộc thang điểm


def test_intersect():
    assert tl.intersect([], (100, 200)) == [(100, 200)]
    assert tl.intersect([(0, 150), (180, 300)], (100, 200)) == [(100, 150), (180, 200)]
    assert tl.intersect([(0, 50)], (100, 200)) == [(0, 0)]
