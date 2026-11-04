import json

import numpy as np
import pytest

from scenario_miner.data.synthetic import generate
from scenario_miner.evaluate import evaluate, paired_f1_difference, write_report
from scenario_miner.pipeline import run_labeling, select_samples
from scenario_miner.prompts import STRATEGIES, ImagePart, TextPart, build_request, parse_response
from scenario_miner.store import connect, ingest, make_splits
from scenario_miner.taxonomy import label_names, load_taxonomy
from scenario_miner.vlm import Backend, HeuristicBackend, VLMResult

TAXONOMY = load_taxonomy()
LABELS = label_names(TAXONOMY)


class FakeVLM(Backend):
    """Answers from the synthetic ground truth, flipping one label, wrapped in a code fence."""

    name = "fake"

    def __init__(self, flip: str = "night", fence: bool = True):
        super().__init__("fake-1")
        self.calls, self.flip, self.fence = 0, flip, fence

    def generate(self, req, temperature):
        self.calls += 1
        truth = dict(req.sample.gt)
        truth["night"] = req.sample.context.get("time_of_day") == "Night"
        out = {"labels": {n: {"present": bool(truth.get(n, False)) ^ (n == self.flip), "confidence": 0.8}
                          for n in req.labels}}
        text = json.dumps(out)
        return VLMResult(f"```json\n{text}\n```" if self.fence else text, 0.01, 1000, 50)


@pytest.fixture(scope="module")
def db(tmp_path_factory):
    root = tmp_path_factory.mktemp("synth")
    generate(root / "extracted", n_segments=10, n_samples=4, seed=1)
    con = connect(root / "miner.duckdb")
    assert ingest(con, root / "extracted") == 40
    make_splits(con, fewshot_segments=2, per_segment=2, min_gap_s=3.0)
    return con


def test_ingest_derives_ground_truth(db):
    rows = dict(db.execute("SELECT label, count(*) FROM ground_truth GROUP BY label").fetchall())
    assert set(rows) == set(LABELS)
    assert all(n == 40 for n in rows.values())
    sources = {r[0] for r in db.execute("SELECT DISTINCT source FROM ground_truth").fetchall()}
    assert sources == {"derived", "stats", "synthetic"}
    # The SQL rule agrees with a direct recount from the objects table.
    n_sql = db.execute("SELECT count(*) FROM ground_truth WHERE label='crowded_scene' AND present").fetchone()[0]
    n_py = db.execute("""SELECT count(*) FROM (SELECT sample_id FROM objects WHERE in_camera AND type='pedestrian'
                         AND range_m <= 30 GROUP BY sample_id HAVING count(*) >= 5)""").fetchone()[0]
    assert n_sql == n_py


def test_splits_are_segment_disjoint(db):
    rows = db.execute("SELECT split, count(DISTINCT segment_id) FROM splits JOIN samples USING (sample_id) GROUP BY split").fetchall()
    assert dict(rows) == {"fewshot": 2, "eval": 8}
    overlap = db.execute("""SELECT count(*) FROM (SELECT segment_id FROM splits JOIN samples USING (sample_id)
                            GROUP BY segment_id HAVING count(DISTINCT split) > 1)""").fetchone()[0]
    assert overlap == 0


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_build_request_parts(db, strategy):
    sample = select_samples(db, "eval", limit=1)[0]
    fewshot = [(sample, {n: False for n in LABELS})] * 2
    req = build_request(sample, strategy, TAXONOMY, LABELS, fewshot)
    n_images = sum(isinstance(p, ImagePart) for p in req.parts)
    expected = {"few_shot": 3, "map_overlay": 2, "map_overlay_context": 2}.get(strategy, 1)
    assert n_images == expected
    text = " ".join(p.text for p in req.parts if isinstance(p, TextPart))
    assert ("3D context from LiDAR" in text) == (strategy in ("map_context", "map_overlay_context"))
    assert ("observations" in req.schema.model_fields) == (strategy == "cot")
    assert len(req.cache_key) == 64


def test_parse_response_is_lenient():
    labels = ["a", "b", "c"]
    p = parse_response('```json\n{"labels": {"a": {"present": "true", "confidence": 1.4}, "b": false}}\n```', labels)
    assert p.decisions == {"a": (True, 1.0), "b": (False, 0.0), "c": (False, 0.0)}
    assert p.error == "missing labels: c"
    assert parse_response('{"a": true, "b": {"present": false, "confidence": 0.2}, "c": 1}', labels).error is None
    bad = parse_response("I cannot help with that.", labels)
    assert bad.error.startswith("unparseable") and bad.decisions["a"] == (False, 0.0)


def test_heuristic_run_and_report(db):
    samples = select_samples(db, "eval")
    run = run_labeling(db, HeuristicBackend(), "zero_shot", TAXONOMY, samples, log=lambda *_: None)
    labels = {r[0] for r in db.execute("SELECT DISTINCT label FROM predictions WHERE run_id = ?", [run]).fetchall()}
    assert labels == set(HeuristicBackend.LABELS)
    results = evaluate(db, [run], n_boot=200)
    assert {r.label for r in results} == labels
    for r in results:
        lo, hi = r.ci["f1"]
        if np.isfinite(r.f1) and np.isfinite(lo):
            assert lo - 1e-9 <= r.f1 <= hi + 1e-9
    assert "Per label" in write_report(db, results, n_boot=200)


def test_vlm_run_cache_votes_and_fewshot(db):
    samples = select_samples(db, "eval")
    fake = FakeVLM(flip="night")
    run = run_labeling(db, fake, "few_shot", TAXONOMY, samples, votes=3, fewshot_k=2, log=lambda *_: None)
    assert fake.calls == 3 * len(samples)
    # Rerunning the identical experiment is served from the cache.
    run_labeling(db, fake, "few_shot", TAXONOMY, samples, votes=3, fewshot_k=2, run_id="again", log=lambda *_: None)
    assert fake.calls == 3 * len(samples)

    # 3 identical votes -> self-consistency confidence is exactly 0 or 1.
    confs = {r[0] for r in db.execute("SELECT DISTINCT confidence FROM predictions WHERE run_id = ?", [run]).fetchall()}
    assert confs <= {0.0, 1.0}
    res = {r.label: r for r in evaluate(db, [run], n_boot=100)}
    assert res["construction_zone"].f1 == 1.0
    assert res["night"].recall == 0.0 or np.isnan(res["night"].precision) or res["night"].precision == 0.0
    errors = db.execute("SELECT count(*) FROM responses WHERE run_id = ? AND error IS NOT NULL", [run]).fetchone()[0]
    assert errors == 0  # code fences parsed fine

    d = paired_f1_difference(db, run, "again", "construction_zone", n_boot=100)
    assert d["delta_f1"] == 0.0
