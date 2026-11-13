import copy
import json

import pytest

from agatha.database.cli import import_scene, main
from agatha.database.ingest import prepare_import, read_json, result_directories
from stage3_helpers import scene_files, write_json, write_results


def test_import_preserves_reports_assessments_exact_frames_and_raw_text(tmp_path):
    folder = scene_files(tmp_path)
    bundle = prepare_import(folder)
    assert len(bundle.incidents) == len(bundle.labels) == len(bundle.objects) == 4
    assert len(bundle.evidence) == 12
    assert bundle.incidents[1]["subcategories"] == []
    assert bundle.labels[1]["status"] == "REJECTED"
    assert bundle.objects[0]["object_classes"] == ["vehicle"]
    assert bundle.evidence[2]["frame_timestamp_micros"] == 1550000000400000
    assert bundle.run["archive"]["responses"]["pass2.json"]["text"].endswith("\n")
    assert bundle.run["complete"]
    assert bundle.incidents[3]["duplicate_of"] == bundle.incidents[0]["id"]
    assert bundle.incidents[0]["raw_report"] == read_json(folder / "stage2/incidents.json")[0]


def test_identity_is_independent_of_copy_path_and_preserves_model_prompt_revisions(tmp_path):
    folder = scene_files(tmp_path)
    first = prepare_import(folder)
    raw = read_json(folder / "stage2/incidents.json")
    copy_dir = folder / "stage2/by_model/copy"
    write_results(copy_dir, raw)
    assert prepare_import(folder, copy_dir) == first
    for field in ("model_id", "prompt_version", "pipeline_version", "vlm_reasoning"):
        changed = copy.deepcopy(raw)
        changed[0][field] += "-new"
        write_results(copy_dir, changed)
        second = prepare_import(folder, copy_dir)
        assert first.incidents[0]["id"] != second.incidents[0]["id"]
    assert result_directories(folder) == [folder / "stage2", copy_dir]


@pytest.mark.parametrize("change", [
    {"segment_id": "wrong"}, {"preprocessing_version": "wrong"}, {"schema_version": "999"},
    {"start_s": 2.0}, {"end_s": -1.0}, {"start_s": float("nan")},
    {"object_ids": ["missing"]}, {"object_ids": ["synthetic-car", "synthetic-car"]},
    {"subcategories": []}, {"categories": []}, {"verification_status": "REJECTED"},
    {"duplicate_of": "INC-004"}, {"source_frames": []},
    {"source_frames": [{"camera": "FRONT", "timestamp_s": 0.4, "frame_timestamp_micros": 1550000000400001}]},
    {"source_frames": [{"camera": "SIDE_LEFT", "timestamp_s": 0.4, "frame_timestamp_micros": 1550000000400000}]},
    {"source_frames": [{"camera": "FRONT", "timestamp_s": 1.2, "frame_timestamp_micros": 1550000000400000}]},
])
def test_invalid_report_is_rejected_before_any_write(tmp_path, change):
    folder = scene_files(tmp_path)
    raw = read_json(folder / "stage2/incidents.json")
    raw[0].update(change)
    write_json(folder / "stage2/incidents.json", raw)
    with pytest.raises(ValueError):
        prepare_import(folder)


def test_missing_response_and_missing_accepted_report_are_errors(tmp_path):
    folder = scene_files(tmp_path)
    (folder / "stage2/responses/pass2.json").unlink()
    with pytest.raises(ValueError, match="missing raw response"):
        prepare_import(folder)
    folder = scene_files(tmp_path)
    write_json(folder / "stage2/incidents.json", [])
    with pytest.raises(ValueError, match="no report"):
        prepare_import(folder)


def test_negative_scene_and_failed_candidate_are_not_confused(tmp_path):
    folder = scene_files(tmp_path)
    write_results(folder / "stage2", [])
    bundle = prepare_import(folder)
    assert bundle.incidents == [] and bundle.run["complete"]
    candidates = read_json(folder / "stage2/candidates.json")
    candidates["merged"] = [{"incident_id": "INC-001", "outcome": "invalid", "response": "pass2"}]
    write_json(folder / "stage2/candidates.json", candidates)
    write_json(folder / "stage2/errors/pass2.json", {"errors": ["timestamp not in window"]})
    bundle = prepare_import(folder)
    assert not bundle.run["complete"]
    assert bundle.run["archive"]["errors"]["pass2.json"]["errors"] == ["timestamp not in window"]


def test_human_labels_are_validated_and_unreviewed_is_not_negative(tmp_path):
    folder = scene_files(tmp_path)
    path = tmp_path / "labels.json"
    labels = {"scene": "random15_1", "reviewed": False, "no_incident": None, "incidents": []}
    write_json(path, labels)
    first = prepare_import(folder, human_path=path, scene="random15_1")
    assert first.human["no_incident"] is None and not first.human["reviewed"]
    labels.update(reviewed=True, no_incident=True)
    write_json(path, labels)
    second = prepare_import(folder, human_path=path, scene="random15_1")
    assert second.human["id"] != first.human["id"]
    with pytest.raises(ValueError, match="another scene"):
        prepare_import(folder, human_path=path, scene="random15_2")
    labels.update(no_incident=False, incidents=[{"labels": ["HARD_BRAKING"], "start_s": 0.0, "end_s": 50.0}])
    write_json(path, labels)
    with pytest.raises(ValueError, match="duration"):
        prepare_import(folder, human_path=path, scene="random15_1")


def test_cli_validate_is_offline_and_list_failures_exit_nonzero(tmp_path, monkeypatch, capsys):
    folder = scene_files(tmp_path)
    monkeypatch.delenv("AGATHA_DATABASE_URL", raising=False)
    assert main(["validate", "--segment", str(folder)]) == 0
    listing = tmp_path / "random15.txt"
    listing.write_text(folder.name + "\nsynthetic_missing\n")
    assert main(["validate", "--list", str(listing), "--outputs", str(tmp_path)]) == 1
    assert "1 failed" in capsys.readouterr().out


def test_all_models_validate_before_database_mutation(tmp_path):
    folder = scene_files(tmp_path)
    other = folder / "stage2/by_model/bad"
    write_results(other, [])
    (other / "incidents.json").write_text("bad json")
    class NoWrites:
        def ingest(self, _):
            pytest.fail("must validate every model first")
    with pytest.raises(ValueError):
        import_scene(folder, result_directories(folder), repository=NoWrites())


def test_full_stage1_fake_stage2_contract(tmp_path):
    from agatha.config import load_config
    from agatha.run_stage1 import process_segment
    from agatha.store import load_segment
    from agatha.synthetic import write_synthetic_segment
    from agatha.vlm.model import FakeModel
    from agatha.vlm.run import load_vlm_config, run_segment
    from pathlib import Path

    project = Path(__file__).parents[1]
    root = tmp_path / "input"
    ref = write_synthetic_segment(root, n_frames=16, cameras=("FRONT",))
    config = load_config(project / "configs/stage1.toml", root=root, outputs=tmp_path / "output")
    directory, _ = process_segment(config, ref)
    segment = load_segment(directory)
    def answer(system, content, schema):
        context = json.loads(next(b["text"] for b in content if b["type"] == "text" and b["text"].startswith('{"window"')))
        alias = context["objects"][0]["alias"]
        if schema["title"] == "Pass1Answer":
            return json.dumps({"candidates": [{"labels": ["SUDDEN_CUT_IN"], "start_s": 0.4, "end_s": 0.8,
                                              "object_aliases": [alias], "evidence": ["synthetic"], "driving_impact": "test"}]})
        return json.dumps({"label_assessments": [{"label": "SUDDEN_CUT_IN", "status": "CONFIRMED",
                                                  "evidence": ["synthetic"], "contradicting_evidence": []}],
                           "source_frames": [{"camera": "FRONT", "t_s": 0.0}], "explanation": "Synthetic test",
                           "uncertainty_reasons": [], "start_s": 0.4, "end_s": 0.8,
                           "object_aliases": [alias], "status": "CONFIRMED", "driving_impact": "test"})
    counts = run_segment(segment, config, load_vlm_config(project / "configs/stage2.toml"), FakeModel(answer))
    assert counts["reports"] == 1
    bundle = prepare_import(directory)
    assert len(bundle.incidents) == 1 and bundle.run["complete"]


def test_saved_model_uses_shared_response_cache_without_unrelated_models(tmp_path):
    folder = scene_files(tmp_path)
    saved = folder / "stage2/by_model/saved"
    for name in ("incidents.json", "candidates.json"):
        write_json(saved / name, read_json(folder / "stage2" / name))
    write_json(folder / "stage2/responses/unrelated.json", {"model_id": "unrelated", "text": "unused"})
    bundle = prepare_import(folder, saved)
    assert bundle.run["model_ids"] == ["fake-model"]
    assert set(bundle.run["archive"]["responses"]) == {"pass1.json", "pass2.json"}
    assert bundle == prepare_import(folder)


def test_invalid_pass1_candidates_do_not_make_a_complete_negative_scene(tmp_path):
    folder = scene_files(tmp_path)
    write_results(folder / "stage2", [])
    candidates = read_json(folder / "stage2/candidates.json")
    candidates["candidates"] = [{"errors": ["span misses window"]}]
    write_json(folder / "stage2/candidates.json", candidates)
    assert not prepare_import(folder).run["complete"]
