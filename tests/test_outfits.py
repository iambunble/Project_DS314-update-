"""Sổ tay trang phục: nạp nhãn, chấm điểm câu tả quần áo, khung thân trên."""
import numpy as np
import yaml

from retrieval.vision import outfits as ob


class FakeText:
    """Mỗi từ khoá -> 1 chiều; câu/nhãn = tổng các chiều của từ khoá có trong đó."""
    KEYS = ["sọc", "caro", "xám", "trắng", "xanh"]

    def _v(self, t):
        v = np.array([1.0 if k in t else 0.0 for k in self.KEYS] + [0.3])
        return v / np.linalg.norm(v)

    def encode_passages(self, texts):
        return np.stack([self._v(t) for t in texts])

    def encode_query(self, t):
        return self._v(t)


def _setup(tmp_path, labels, ids, fp=None, conf=None):
    out = tmp_path / "artifacts"
    out.mkdir()
    np.save(out / ob.IDS_FILE, np.array(ids, np.int16))
    if conf is not None:
        np.save(out / ob.CONF_FILE, np.array(conf, np.float16))
    k = max(ids) + 1
    lab = tmp_path / "outfits.yaml"
    lab.write_text(yaml.safe_dump({"k": k, "fingerprint": fp if fp is not None else ob.fingerprint(np.array(ids), k),
                                   "labels": labels}, allow_unicode=True), encoding="utf-8")
    cfg = {"outfits": {"enabled": True, "labels": str(lab), "drop_far_pct": 0}}
    return cfg, out


def test_person_scores_pick_matching_outfit(tmp_path):
    ids = [0, 0, 1, 2, 2]
    cfg, out = _setup(tmp_path, {0: "áo len sọc", 1: "áo sơ mi caro", 2: ""}, ids)
    book, warn = ob.OutfitBook.load(cfg, out, FakeText(), len(ids))
    assert warn is None
    p = book.person_scores(FakeText().encode_query("người mặc áo len sọc ngang"))
    assert p[0] == p[1] > 0                 # cả nhóm "sọc" được điểm
    assert p[2] == 0                        # nhóm "caro" không khớp
    assert p[3] == p[4] == 0                # nhóm lẫn (nhãn rỗng) -> 0
    assert book.best(FakeText().encode_query("áo caro"), 1)[0][0] == "áo sơ mi caro"


def test_fingerprint_mismatch_disables(tmp_path):
    ids = [0, 0, 1]
    cfg, out = _setup(tmp_path, {0: "áo sọc", 1: "áo caro"}, ids, fp=[1, 2])
    book, warn = ob.OutfitBook.load(cfg, out, FakeText(), len(ids))
    assert book is None and "ghi nhãn lại" in warn


def test_person_count_mismatch_disables(tmp_path):
    cfg, out = _setup(tmp_path, {0: "áo sọc"}, [0, 0])
    book, warn = ob.OutfitBook.load(cfg, out, FakeText(), 5)
    assert book is None and "build" in warn


def test_far_members_lose_label(tmp_path):
    ids = [0] * 10
    conf = np.linspace(0.5, 0.95, 10)
    cfg, out = _setup(tmp_path, {0: "áo sọc"}, ids, conf=conf)
    cfg["outfits"]["drop_far_pct"] = 20
    book, _ = ob.OutfitBook.load(cfg, out, FakeText(), len(ids))
    assert (book.person_label == -1).sum() == 2 and book.person_label[0] == -1


def test_disabled_or_missing(tmp_path):
    assert ob.OutfitBook.load({"outfits": {"enabled": False}}, tmp_path, FakeText(), 3) == (None, None)
    book, warn = ob.OutfitBook.load({"outfits": {"enabled": True, "labels": str(tmp_path / "x.yaml")}},
                                    tmp_path, FakeText(), 3)
    assert book is None and "outfits.py build" in warn


def test_upper_box_uses_shoulders_and_hips():
    kp = np.zeros((17, 3))
    kp[5] = kp[6] = [0.5, 0.30, 0.9]        # vai
    kp[11] = kp[12] = [0.5, 0.50, 0.9]      # hông
    x1, y1, x2, y2 = ob.upper_box([0.4, 0.2, 0.6, 0.9], kp)
    assert abs(y1 - (0.30 - 0.024)) < 1e-6 and abs(y2 - 0.52) < 1e-6
    # không có khung xương -> theo tỉ lệ khung
    assert ob.upper_box([0.0, 0.0, 1.0, 1.0])[1:4:2] == [0.12, 0.58]


def test_template_roundtrip(tmp_path):
    ids = np.array([0, 1, 1, 2])
    f = tmp_path / "t.yaml"
    ob.write_template(f, ids, 3)
    y = yaml.safe_load(f.read_text(encoding="utf-8"))
    assert y["fingerprint"] == [1, 2, 1] and set(y["labels"]) == {0, 1, 2}
