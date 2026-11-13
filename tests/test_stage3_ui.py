import copy
from pathlib import Path

import pytest

st = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from agatha.database.ingest import prepare_import
from stage3_helpers import scene_files


class FakeRepository:
    def __init__(self, bundle):
        self.bundle = bundle
        row = bundle.incidents[0]
        self.row = {**row, "duration_s": row["end_s"] - row["start_s"],
                    "context_name": bundle.segment["context_name"], "split": "training"}
        self.jobs = []
        self.filters = None
        self.human = []
        self.empty = False

    def search(self, filters, **kwargs):
        self.filters = filters
        return ([], 0) if self.empty or "CONFIRMED" not in filters.statuses else ([self.row], 1)

    def detail(self, iid):
        return {**self.row, "dataset_version": "v2.0.1", "labels": [self.bundle.labels[0]],
                "evidence": self.bundle.evidence[:3], "objects": self.bundle.objects[:1],
                "assets": [{"kind": "source_video", "bucket": "agatha-incidents", "object_path": "synthetic.mp4",
                            "metadata": {"start_s": 0.0, "end_s": 1.5, "frame_count": 8, "max_gap_s": 0.5}}],
                "jobs": self.jobs}

    def enqueue(self, iid):
        if not self.jobs:
            self.jobs.append({"id": "synthetic-job", "status": "QUEUED", "start_s": 0.4, "end_s": 0.8,
                              "error": None, "output_clip_path": None})
        return self.jobs[0]

    def scenes(self):
        return [self.bundle.segment]

    def comparison(self, sid):
        return {"human_labels": self.human, "runs": [{**self.bundle.run, "reports": [self.row["raw_report"]]}]}


class FakeStorage:
    def __init__(self):
        self.calls = 0
    def signed_url(self, asset):
        self.calls += 1
        return "https://synthetic.invalid/plain.mp4?temporary=1"


@pytest.fixture
def app(tmp_path):
    repo = FakeRepository(prepare_import(scene_files(tmp_path)))
    storage = FakeStorage()
    at = AppTest.from_string("""
import streamlit as st
from agatha.database.explorer import render_app
render_app(st.session_state.repository, st.session_state.storage)
""")
    at.session_state.repository = repo
    at.session_state.storage = storage
    at.run(timeout=15)
    assert not at.exception
    return at, repo, storage


def test_explorer_order_plain_video_reasoning_and_queue_button(app):
    at, repo, storage = app
    assert at.get("video")
    # Verify the content order in the actual rendered tree, not source text.
    nodes = list(at.main)
    kinds = [(n.type, str(getattr(n, "value", ""))) for n in nodes]
    video = next(i for i, n in enumerate(kinds) if n[0] == "video")
    title = next(i for i, n in enumerate(kinds) if n == ("subheader", "Sudden cut in"))
    labels = next(i for i, n in enumerate(kinds) if "SUDDEN_CUT_IN (confirmed)" in n[1])
    reasoning = next(i for i, n in enumerate(kinds) if n == ("subheader", "VLM reasoning"))
    assert video < title < labels < reasoning
    next(b for b in at.button if b.label == "Run Simulation").click().run()
    assert not at.exception and len(repo.jobs) == 1
    assert "QUEUED" in at.success[0].value
    next(b for b in at.button if b.label == "Run Simulation").click().run()
    assert len(repo.jobs) == 1
    before = storage.calls
    next(b for b in at.button if b.label == "Refresh video link").click().run()
    assert storage.calls > before


def test_filters_and_invalid_duration(app):
    at, repo, _ = app
    assert repo.filters.statuses == ["CONFIRMED"]
    next(w for w in at.checkbox if w.label == "Include rejected").check().run()
    next(w for w in at.radio if w.label.startswith("Combine")).set_value("OR").run()
    next(w for w in at.multiselect if w.label == "Subcategories").set_value(["SUDDEN_CUT_IN"]).run()
    assert repo.filters.statuses == ["CONFIRMED", "REJECTED"]
    assert repo.filters.combine == "OR" and repo.filters.labels == ["SUDDEN_CUT_IN"]
    next(w for w in at.checkbox if w.label == "Filter duration").check().run()
    next(w for w in at.number_input if w.label == "Minimum seconds").set_value(25.0).run()
    assert at.error[0].value == "Minimum duration must not exceed maximum duration."
    assert not at.exception


def test_zero_report_scenes_keep_human_comparison_accessible(app):
    at, repo, _ = app
    repo.empty = True
    repo.human = [{"scene": "random15_1", "raw_labels": {"scene": "random15_1", "reviewed": True,
                    "no_incident": False, "incidents": [{"labels": ["ROAD_DEBRIS"], "start_s": 0.0, "end_s": 1.0}]}}]
    repo.bundle.run["complete"] = False
    at.run()
    assert not at.exception
    assert any("did not assess" in w.value for w in at.warning)
    assert any(w.label == "Review scene" for w in at.selectbox)
    assert any("Andrew only" in w.value for w in at.text)


def test_app_without_credentials_shows_setup_not_traceback(monkeypatch):
    monkeypatch.delenv("AGATHA_DATABASE_URL", raising=False)
    original = st.get_option
    monkeypatch.setattr(st, "get_option", lambda key: "127.0.0.1" if key == "server.address" else original(key))
    at = AppTest.from_file(str(Path(__file__).parents[1] / "apps/streamlit_app/app.py")).run()
    assert not at.exception
    assert "Supabase is not configured yet" in at.info[0].value


def test_app_refuses_public_binding(monkeypatch):
    original = st.get_option
    monkeypatch.setattr(st, "get_option", lambda key: "0.0.0.0" if key == "server.address" else original(key))
    at = AppTest.from_file(str(Path(__file__).parents[1] / "apps/streamlit_app/app.py")).run()
    assert not at.exception
    assert "binds to localhost" in at.error[0].value
