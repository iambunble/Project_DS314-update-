"""Cho mọi file test import được gói retrieval (src/) kể cả khi chạy riêng 1 file."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _no_translation_model(monkeypatch, request):
    """Test không tải model dịch (~1 GB, chậm): engine tìm bằng câu tiếng Việt như khi tắt translation.
    Riêng test_translate.py tự kiểm phần dịch (không cần model)."""
    if request.node.fspath.basename == "test_translate.py":
        return
    from retrieval.query import translate
    monkeypatch.setattr(translate.QueryTranslator, "from_config", classmethod(lambda cls, cfg: None))
