import base64
import importlib.util
import inspect
import json
import os
import shutil
import stat
import subprocess
import sys
import typing
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from agatha.config import load_config
from agatha.report import PUBLIC_SEGMENT, context_label
from agatha.run_stage1 import process_segment
from agatha.scene_window import build_scene_window, default_windows
from agatha.store import load_segment
from agatha.synthetic import OBJECT_IDS, write_synthetic_segment
from agatha.vlm.model import FALLBACK_BETA, ClaudeModel, FakeModel, ModelFailure, NotSent
from agatha.vlm.prompts import PASS1_SYSTEM, PASS2_SYSTEM, PROMPT_VERSION, request_content, window_context
from agatha.vlm.run import (
    PIPELINE_VERSION, GuardRefused, Run, check_guard, incident_report, load_vlm_config, mark_duplicates,
    merge_candidates, request_key, run_segment,
)
from agatha.vlm.schema import (
    CATEGORIES, PASS1_SCHEMA, PASS2_SCHEMA, TAXONOMY, Candidate, IncidentReport, LabelAssessment, Pass1Answer,
    Pass2Answer, check_verdict, fit_candidate,
)

PROJECT = Path(__file__).parents[1]
STAGE1_CONFIG = PROJECT / "configs" / "stage1.toml"
STAGE2_CONFIG = PROJECT / "configs" / "stage2.toml"
PRIVATE_CONTEXT = "12345678901234567890_100_000_120_000"  # shaped like a real name, so it gets hashed
UNPROCESSED_CONTEXT = "98765432109876543210_100_000_120_000"  # real-shaped too, and with no outputs at all
CUT_IN_S = (8.0, 12.0)  # synthetic.py: the cut-in car enters the ego path from 8 to 12 s

SUBSET_KEYWORDS = {"type", "properties", "required", "additionalProperties", "items", "enum", "title", "$defs", "$ref"}
SUBSET_TYPES = {"object", "array", "string", "number", "boolean"}

# The SDK is an optional extra: tests that build its responses or errors skip without it.
needs_sdk = pytest.mark.skipif(importlib.util.find_spec("anthropic") is None, reason="needs the anthropic SDK")


@pytest.fixture(scope="module")
def processed(synthetic_root, tmp_path_factory):
    """(config, directory) of Stage 1's FRONT outputs for the synthetic segment, made once: a run takes seconds."""
    root, ref = synthetic_root
    config = load_config(STAGE1_CONFIG, root=root, outputs=tmp_path_factory.mktemp("outputs"))
    directory, _ = process_segment(config, ref)
    return config, directory


@pytest.fixture(scope="module")
def three_cameras(synthetic_root, tmp_path_factory):
    """(config, segment) of Stage 1's outputs with FRONT_LEFT and FRONT_RIGHT too, a change stage1.toml allows."""
    root, ref = synthetic_root
    config = load_config(STAGE1_CONFIG, root=root, outputs=tmp_path_factory.mktemp("three_cameras"),
                         cameras=["FRONT", "FRONT_LEFT", "FRONT_RIGHT"])
    directory, _ = process_segment(config, ref)
    return config, load_segment(directory)


@pytest.fixture
def config(processed):
    return processed[0]


@pytest.fixture
def segment(processed, tmp_path):
    """A copy of Stage 1's outputs, so each test's stage2/ starts empty."""
    copy = tmp_path / "synthetic_cut_in"
    shutil.copytree(processed[1], copy, ignore=shutil.ignore_patterns("clips", "qa"))
    return load_segment(copy)


@pytest.fixture
def vlm():
    return load_vlm_config(STAGE2_CONFIG)


def alias_of(segment, name: str) -> str:
    tracks = segment.tracks
    return tracks.loc[tracks["laser_object_id"] == OBJECT_IDS[name], "alias"].iloc[0]


def cut_in_alias(segment) -> str:
    return alias_of(segment, "cut_in")


def texts(content: list[dict]) -> list[str]:
    return [block["text"] for block in content if block["type"] == "text"]


def captions(content: list[dict]) -> list[tuple[str, float]]:
    """(camera, t_s) of each caption, as a model reads them."""
    found = []
    for block, following in zip(content, content[1:]):
        if block["type"] == "text" and following["type"] == "image":
            camera, time_text = block["text"].split(" t=")
            found.append((camera, float(time_text.removesuffix(" s"))))
    return found


def window_json(content: list[dict]) -> dict:
    return json.loads(next(text for text in texts(content) if text.startswith('{"window"')))


def hypothesis_of(content: list[dict]) -> dict:
    return json.loads(texts(content)[-1].split(": ", 1)[1])


def assessment(status: str, label="SUDDEN_CUT_IN") -> dict:
    return {"label": label, "evidence": ["it crosses into the path"], "contradicting_evidence": [], "status": status}


def verdict(hypothesis: dict, status="CONFIRMED", frame=None, changes=None) -> str:
    """A Pass 2 answer that confirms, rejects or doubts every hypothesized label alike."""
    answer = {"status": status, "uncertainty_reasons": [] if status != "UNCERTAIN" else ["the gap is hard to see"],
              "label_assessments": [assessment(status, label) for label in hypothesis["labels"]],
              "start_s": hypothesis["start_s"], "end_s": hypothesis["end_s"],
              "object_aliases": hypothesis["object_aliases"],
              "source_frames": [] if frame is None else [{"camera": frame[0], "t_s": frame[1]}],
              "explanation": "The car moves from the next lane into the ego's path.",
              "driving_impact": "Less clearance ahead of the ego."}
    return json.dumps({**answer, **(changes or {})})


def scripted(alias: str, statuses=("CONFIRMED",), pass1=None):
    """A fake model's answers: the cut-in in each window that overlaps it, then Pass 2's statuses in turn."""
    statuses = list(statuses)

    def answer(system, content, schema):
        if schema is PASS1_SCHEMA:
            if pass1 is not None:
                return pass1(content)
            window = window_json(content)["window"]
            start_s, end_s = max(CUT_IN_S[0], window["start_s"]), min(CUT_IN_S[1], window["end_s"])
            if start_s >= end_s:
                return json.dumps({"candidates": []})
            return json.dumps({"candidates": [{
                "labels": ["SUDDEN_CUT_IN"], "start_s": start_s, "end_s": end_s, "object_aliases": [alias],
                "evidence": ["lateral_m falls from -3.7 to 0"], "driving_impact": "Less clearance ahead."}]})
        shown = captions(content)
        return verdict(hypothesis_of(content), statuses.pop(0), frame=shown[len(shown) // 2])
    return answer


def stored(segment, kind: str) -> list[dict]:
    return [json.loads(path.read_text()) for path in sorted((segment.directory / "stage2" / kind).glob("*.json"))]


def results(segment) -> tuple[dict, list[dict]]:
    stage2 = segment.directory / "stage2"
    return json.loads((stage2 / "candidates.json").read_text()), json.loads((stage2 / "incidents.json").read_text())


def subset_problems(schema: dict) -> list[str]:
    """What in `schema` leaves strings, numbers, booleans, string enums, arrays and closed objects."""
    problems, nodes = [], [schema]
    while nodes:
        node = nodes.pop()
        problems += [f"keyword {key}" for key in sorted(set(node) - SUBSET_KEYWORDS)]
        if node.get("type", "object") not in SUBSET_TYPES:
            problems.append(f"type {node.get('type')}")
        if node.get("type") == "object" and (node.get("additionalProperties") is not False
                                             or set(node.get("required", ())) != set(node.get("properties", {}))):
            problems.append("an object that is open or has an optional property")
        if not all(isinstance(value, str) for value in node.get("enum", ())):
            problems.append("an enum that isn't all strings")
        if "$ref" in node and node["$ref"].removeprefix("#/$defs/") not in schema.get("$defs", {}):
            problems.append("a $ref with no definition")
        nodes += [*node.get("$defs", {}).values(), *node.get("properties", {}).values()]
        if "items" in node:
            nodes.append(node["items"])
    return problems


# --- schema and taxonomy ---


def test_the_taxonomy_is_fourteen_labels_in_three_categories():
    assert len(TAXONOMY) == 14
    assert CATEGORIES == ("DYNAMIC_HAZARD", "CONSTRUCTION_DISRUPTION", "ROAD_OBSTRUCTION")
    by_category = {category: [label for label, (owner, _) in TAXONOMY.items() if owner == category]
                   for category in CATEGORIES}
    assert [len(labels) for labels in by_category.values()] == [5, 5, 4]
    assert TAXONOMY["SUDDEN_CUT_IN"][0] == "DYNAMIC_HAZARD" and TAXONOMY["ROAD_DEBRIS"][0] == "ROAD_OBSTRUCTION"
    assert TAXONOMY["TEMPORARY_LANE_SHIFT"][0] == "CONSTRUCTION_DISRUPTION"
    for system in (PASS1_SYSTEM, PASS2_SYSTEM):  # the full taxonomy and the rule go to every request
        assert all(f"- {label} ({category}): {text}" in system for label, (category, text) in TAXONOMY.items())
        assert "Mere visibility of roadside construction does not qualify" in " ".join(system.split())


def test_the_pass2_prompt_states_the_rules_its_answers_are_checked_against():
    text = " ".join(PASS2_SYSTEM.split())
    assert ("The event's status is its labels' strongest: CONFIRMED if any label is, else UNCERTAIN if any is, "
            "else REJECTED.") in text
    assert "at least one for a CONFIRMED event" in text


def test_pass1_asks_for_recall_and_pass2_for_the_evidence_against_each_label():
    """Design 5.2 and 5.3: Pass 1 leaves the doubt to Pass 2, which tests the hypothesis rather than assume it."""
    assert "Recall matters more than precision here" in " ".join(PASS1_SYSTEM.split())
    assert "give the evidence for it and the evidence against it" in " ".join(PASS2_SYSTEM.split())


def test_the_output_schemas_keep_to_the_structured_outputs_subset():
    for schema in (PASS1_SCHEMA, PASS2_SCHEMA):
        assert subset_problems(schema) == []
    labels = PASS2_SCHEMA["$defs"]["LabelAssessment"]["properties"]
    assert labels["label"]["enum"] == list(TAXONOMY)
    assert labels["status"]["enum"] == PASS2_SCHEMA["properties"]["status"]["enum"] == ["CONFIRMED", "REJECTED",
                                                                                          "UNCERTAIN"]
    assert PASS1_SCHEMA["$defs"]["Candidate"]["properties"]["labels"]["items"]["enum"] == list(TAXONOMY)


def test_a_verdict_is_written_after_its_evidence():
    """Keys come out in schema order, so the evidence comes before each status."""
    event = list(PASS2_SCHEMA["properties"])
    assert event.index("label_assessments") < event.index("source_frames") < event.index("status")
    assert event.index("explanation") < event.index("status")
    label = list(PASS2_SCHEMA["$defs"]["LabelAssessment"]["properties"])
    assert label == ["label", "evidence", "contradicting_evidence", "status"]


def test_the_subset_check_finds_a_default_an_open_object_and_an_optional_field():
    class Defaulted(BaseModel):
        model_config = ConfigDict(extra="forbid")
        note: str = ""

    class Open(BaseModel):
        note: str

    class Optional_(BaseModel):
        model_config = ConfigDict(extra="forbid")
        count: int | None

    assert all(subset_problems(model.model_json_schema()) for model in (Defaulted, Open, Optional_))


GOOD_CANDIDATE = {"labels": ["SUDDEN_CUT_IN"], "start_s": 8, "end_s": 9.5, "object_aliases": ["V002"],
                  "evidence": ["it crosses"], "driving_impact": "Less clearance."}


@pytest.mark.parametrize("change", [
    {"extra": 1}, {"labels": ["CUT_IN"]}, {"start_s": "8"}, {"end_s": None}, {"object_aliases": "V002"},
], ids=["extra key", "label outside the taxonomy", "number as text", "null time", "string for a list"])
def test_answers_parse_strictly(change):
    assert Pass1Answer.model_validate_json(json.dumps({"candidates": [GOOD_CANDIDATE]})).candidates[0].end_s == 9.5
    with pytest.raises(ValidationError):
        Pass1Answer.model_validate_json(json.dumps({"candidates": [{**GOOD_CANDIDATE, **change}]}))


def test_an_answer_missing_a_field_or_not_json_fails_to_parse():
    for text in ['{"candidates": [{"labels": []}]}', "not json", "{}"]:
        with pytest.raises(ValidationError):
            Pass1Answer.model_validate_json(text)


@pytest.mark.parametrize("number", ["NaN", "Infinity", "-Infinity"])
def test_a_time_that_isnt_finite_fails_to_parse(number):
    text = json.dumps({"candidates": [GOOD_CANDIDATE]}).replace('"start_s": 8', f'"start_s": {number}')
    assert number in text  # inside the candidate, where the float is, not beside it
    with pytest.raises(ValidationError, match="finite number"):
        Pass1Answer.model_validate_json(text)


# --- referential checks ---


def pass1_window(segment, config, start_s=8.0, end_s=13.0, **kwargs):
    return build_scene_window(segment, config, start_s, end_s, 2.0, config.cameras, "pass1", **kwargs)


def candidate(**changes) -> Candidate:
    return Candidate(**{**GOOD_CANDIDATE, "start_s": 8.0, **changes})


@pytest.mark.parametrize("span", [(8.0, 9.5), (9.0, 9.0)], ids=["span", "one instant"])
def test_a_candidate_inside_its_window_passes_unchanged(segment, config, span):
    given = candidate(start_s=span[0], end_s=span[1], object_aliases=[cut_in_alias(segment)])
    assert fit_candidate(given, pass1_window(segment, config)) == (given, [], [])


def test_pass1_fits_a_candidate_to_its_window_rather_than_lose_it(segment, config):
    alias = cut_in_alias(segment)  # the window is [8, 13), and its JSON's track context runs from 6 to 15 s
    given = candidate(start_s=7.5, end_s=14.0, object_aliases=[alias, "V999"])
    fitted, changes, errors = fit_candidate(given, pass1_window(segment, config))
    assert errors == [] and (fitted.start_s, fitted.end_s, fitted.object_aliases) == (8.0, 13.0, [alias])
    assert changes == ["dropped alias V999, not in the window", "7.5 to 14.0 s: clipped to the window"]
    assert given.object_aliases == [alias, "V999"]  # the answer as given is left alone


@pytest.mark.parametrize("change, error", [
    ({"start_s": 13.5, "end_s": 14.0}, "13.5 to 14.0 s misses the window"),
    ({"start_s": 10.0, "end_s": 9.0}, "10.0 to 9.0 s misses the window"),
    ({"labels": []}, "no labels"),
])
def test_a_candidate_that_cant_be_fitted_says_why(segment, config, change, error):
    window = pass1_window(segment, config)
    _, _, errors = fit_candidate(candidate(**{"object_aliases": [cut_in_alias(segment)], **change}), window)
    assert errors == [error]


def test_an_omitted_object_is_a_valid_citation(segment, config):
    window = pass1_window(segment, config, max_objects=1)
    omitted = window.omitted_objects[0].alias
    assert omitted not in {track.alias for track in window.objects}
    assert fit_candidate(candidate(object_aliases=[omitted]), window)[1:] == ([], [])
    assert fit_candidate(candidate(object_aliases=["S001"]), window)[1] == ["dropped alias S001, not in the window"]


def pass2_case(segment, config, changes=None):
    """(window, hypothesis, answer) for a Pass 2 check, the answer citing the window's first frame."""
    alias = cut_in_alias(segment)
    window = build_scene_window(segment, config, 7.0, 13.0, 5.0, config.cameras, "full", keep_aliases=[alias])
    hypothesis = candidate(start_s=8.0, end_s=12.0, object_aliases=[alias])
    frame = window.frames[0]
    text = verdict(hypothesis.model_dump(), frame=(frame.camera, round(frame.t_s, 2)), changes=changes)
    return window, hypothesis, Pass2Answer.model_validate_json(text)


def test_a_verdict_inside_its_window_passes(segment, config):
    window, hypothesis, answer = pass2_case(segment, config)
    assert check_verdict(answer, window, hypothesis) == []


@pytest.mark.parametrize("change, error", [
    ({"source_frames": [{"camera": "FRONT", "t_s": 7.1}]}, "FRONT at 7.1 s is not a frame in the window"),
    ({"source_frames": [{"camera": "FRONT_LEFT", "t_s": 7.0}]}, "FRONT_LEFT at 7.0 s is not a frame in the window"),
    ({"source_frames": []}, "a CONFIRMED event cites no frame"),
    ({"start_s": 6.0}, "not a span inside the window"),
    ({"start_s": 11.0, "end_s": 9.0}, "11.0 to 9.0 s is not a span inside the window"),
    ({"object_aliases": ["P099"]}, "alias P099 is not in the window"),
    ({"label_assessments": []}, "SUDDEN_CUT_IN from the hypothesis is not assessed"),
    ({"label_assessments": [assessment("CONFIRMED")] * 2}, "SUDDEN_CUT_IN is assessed more than once"),
    ({"status": "REJECTED"}, "a REJECTED event whose labels make it CONFIRMED"),
    ({"status": "UNCERTAIN"}, "a UNCERTAIN event whose labels make it CONFIRMED"),
    ({"label_assessments": [assessment("REJECTED")]}, "a CONFIRMED event whose labels make it REJECTED"),
    ({"label_assessments": [assessment("UNCERTAIN")]}, "a CONFIRMED event whose labels make it UNCERTAIN"),
    ({"status": "REJECTED", "label_assessments": [assessment("UNCERTAIN")]},
     "a REJECTED event whose labels make it UNCERTAIN"),
    ({"status": "UNCERTAIN", "label_assessments": [assessment("UNCERTAIN")], "uncertainty_reasons": []},
     "an UNCERTAIN event gives no reason"),
])
def test_each_fault_in_a_verdict_is_named(segment, config, change, error):
    window, hypothesis, answer = pass2_case(segment, config, change)
    assert any(error in problem for problem in check_verdict(answer, window, hypothesis))


@pytest.mark.parametrize("statuses, event", [
    (["CONFIRMED", "REJECTED"], "CONFIRMED"), (["CONFIRMED", "UNCERTAIN"], "CONFIRMED"),
    (["UNCERTAIN", "REJECTED"], "UNCERTAIN"), (["REJECTED", "REJECTED"], "REJECTED"),
])
def test_an_event_takes_its_labels_strongest_status(segment, config, statuses, event):
    labels = [assessment(status, label) for status, label in zip(statuses, ["SUDDEN_CUT_IN", "HARD_BRAKING"])]
    reasons = ["the gap is hard to see"] if event == "UNCERTAIN" else []  # an UNCERTAIN event has to say why
    window, hypothesis, answer = pass2_case(segment, config, {"status": event, "label_assessments": labels,
                                                              "uncertainty_reasons": reasons})
    assert check_verdict(answer, window, hypothesis) == []


def test_every_label_of_the_hypothesis_must_be_assessed(segment, config):
    window, hypothesis, answer = pass2_case(segment, config)
    both = hypothesis.model_copy(update={"labels": ["SUDDEN_CUT_IN", "HARD_BRAKING"]})
    assert check_verdict(answer, window, both) == ["HARD_BRAKING from the hypothesis is not assessed"]


def test_a_frame_is_cited_by_its_caption_even_off_the_10_hz_grid(segment, config):
    window = pass1_window(segment, config)  # Stage 1 has rendered its images
    hypothesis = candidate(object_aliases=[cut_in_alias(segment)])
    # Real frame times drift off the 0.1 s grid. Past 5 ms, a caption with one decimal would cite no frame at all.
    shifted = window.model_copy(update={"frames": [frame.model_copy(update={"t_s": frame.t_s + 0.037})
                                                   for frame in window.frames]})
    cited = [{"camera": camera, "t_s": t_s} for camera, t_s in captions(request_content(shifted, segment.directory))]
    answer = Pass2Answer.model_validate_json(verdict(hypothesis.model_dump(), changes={"source_frames": cited}))
    assert len(cited) == len(window.frames) and check_verdict(answer, shifted, hypothesis) == []


def test_a_new_label_in_a_verdict_is_allowed_beside_the_hypothesis(segment, config):
    extra = LabelAssessment(label="HARD_BRAKING", status="UNCERTAIN", evidence=[], contradicting_evidence=["steady"])
    window, hypothesis, answer = pass2_case(segment, config)
    widened = answer.model_copy(update={"label_assessments": [*answer.label_assessments, extra]})
    assert check_verdict(widened, window, hypothesis) == []


def test_a_label_pass2_adds_can_confirm_an_event_whose_hypothesis_it_rejects(segment, config):
    """As the live run's INC-006: the cut-in rejected, and the pedestrian beside it confirmed."""
    relabeled = {"label_assessments": [assessment("REJECTED"), assessment("CONFIRMED", "HARD_BRAKING")]}
    window, hypothesis, answer = pass2_case(segment, config, relabeled)
    assert hypothesis.labels == ["SUDDEN_CUT_IN"] and check_verdict(answer, window, hypothesis) == []
    report = incident_report(answer, window, "fake-model", "INC-001")
    assert (report.verification_status, report.subcategories) == ("CONFIRMED", ["HARD_BRAKING"])


# --- merge ---


def original(id_, start_s, end_s, aliases=("V002",), labels=("SUDDEN_CUT_IN",), errors=()):
    """A Pass 1 candidate as candidates.json keeps it. Its window is named after its id's: W1-0 is W1's."""
    return {"id": id_, "window": f"{id_.split('-')[0]}.json", "labels": list(labels), "start_s": start_s,
            "end_s": end_s, "object_aliases": list(aliases), "evidence": [f"evidence {id_}"],
            "driving_impact": f"impact {id_}", "changes": [], "errors": list(errors)}


def test_overlapping_duplicates_merge_and_name_their_sources():
    candidates = [original("W1-0", 8.0, 9.0), original("W2-0", 8.5, 12.0, aliases=("V002", "V005"),
                                                       labels=("HARD_BRAKING", "SUDDEN_CUT_IN"))]
    [(sources, merged)] = merge_candidates(candidates)
    assert sources == ["W1-0", "W2-0"]
    assert merged.labels == ["SUDDEN_CUT_IN", "HARD_BRAKING"]  # taxonomy order
    assert (merged.start_s, merged.end_s) == (8.0, 12.0)
    assert merged.object_aliases == ["V002", "V005"]
    assert merged.evidence == ["evidence W1-0", "evidence W2-0"]
    assert merged.driving_impact == "impact W1-0; impact W2-0"
    assert candidates[0]["labels"] == ["SUDDEN_CUT_IN"]  # the originals stay as they were, for audit


def test_touching_spans_count_as_overlapping():
    assert len(merge_candidates([original("A", 8.0, 9.0), original("B", 9.0, 12.0)])) == 1


@pytest.mark.parametrize("other", [
    original("W2-0", 9.5, 12.0),                                          # no overlap in time
    original("W2-0", 8.5, 12.0, aliases=("V003",)),                       # no shared object
    original("W2-0", 8.5, 12.0, aliases=()),                              # only one names an object
    original("W2-0", 8.5, 12.0, labels=("STOPPED_VEHICLE_OBSTRUCTION",)),  # another category
    original("W1-1", 8.5, 12.0, labels=("HARD_BRAKING",)),                # the same window: the model split them
], ids=["time", "object", "one names none", "category", "same window"])
def test_merging_takes_two_windows_overlap_a_shared_object_and_a_shared_category(other):
    assert len(merge_candidates([original("W1-0", 8.0, 9.0), other])) == 2


def test_candidates_that_name_no_object_merge_on_a_shared_label():
    debris = [original(f"W{number}-0", 4.0, 4.9, aliases=(), labels=("ROAD_DEBRIS",)) for number in (0, 1)]
    [(sources, merged)] = merge_candidates(debris)
    assert sources == ["W0-0", "W1-0"] and merged.labels == ["ROAD_DEBRIS"] and merged.object_aliases == []
    stopped = original("W1-0", 4.0, 4.9, aliases=(), labels=("STOPPED_VEHICLE_OBSTRUCTION",))
    assert len(merge_candidates([debris[0], stopped])) == 2  # one category, but no object or label in common


def test_a_shared_category_is_enough_for_compatible_labels():
    pairs = merge_candidates([original("A", 8.0, 9.0), original("B", 8.5, 12.0, labels=("HARD_BRAKING",))])
    assert [merged.labels for _, merged in pairs] == [["SUDDEN_CUT_IN", "HARD_BRAKING"]]


def test_merging_is_transitive_and_skips_candidates_that_failed_their_checks():
    chain = [original("A", 8.0, 9.0), original("C", 10.0, 12.0), original("B", 8.5, 10.5)]
    [(sources, merged)] = merge_candidates(chain)
    assert sources == ["A", "C", "B"] and (merged.start_s, merged.end_s) == (8.0, 12.0)
    assert merge_candidates([original("A", 8.0, 9.0, errors=["no labels"])]) == []


def test_a_group_takes_one_candidate_from_each_window_at_most():
    """W1-0 overlaps both halves of W2's split: it joins the first, and W2-1 stays apart rather than join through it."""
    split = [original("W2-0", 8.0, 9.0), original("W2-1", 8.5, 10.0, labels=("HARD_BRAKING",))]
    both = original("W1-0", 8.0, 9.0, labels=("SUDDEN_CUT_IN", "HARD_BRAKING"))
    pairs = merge_candidates([*split, both])
    assert [sources for sources, _ in pairs] == [["W2-0", "W1-0"], ["W2-1"]]
    assert [(merged.start_s, merged.end_s) for _, merged in pairs] == [(8.0, 9.0), (8.5, 10.0)]


def test_merged_candidates_come_in_time_order():
    pairs = merge_candidates([original("A", 12.0, 14.0, aliases=("V001",)), original("B", 2.0, 3.0)])
    assert [sources for sources, _ in pairs] == [["B"], ["A"]]


# --- Pass 1 and Pass 2 on the synthetic segment ---


def test_the_cut_in_comes_through_both_passes_as_an_incident_report(segment, config, vlm):
    alias = cut_in_alias(segment)
    model = FakeModel(scripted(alias))
    counts = run_segment(segment, config, vlm, model)

    windows = default_windows(segment, config)
    assert [request["schema"] for request in model.requests] == [PASS1_SCHEMA] * len(windows) + [PASS2_SCHEMA]
    candidates, incidents = results(segment)
    assert [item["id"] for item in candidates["windows"]] == [f"W{number}" for number in range(len(windows))]
    assert {item["outcome"] for item in candidates["windows"]} == {"answered"}
    assert [item["id"] for item in candidates["candidates"]] == ["W1-0", "W2-0"]  # [4, 9) and [8, 13)
    [merged] = candidates["merged"]
    assert merged["sources"] == ["W1-0", "W2-0"] and merged["outcome"] == "CONFIRMED"
    assert (merged["start_s"], merged["end_s"]) == CUT_IN_S and merged["incident_id"] == "INC-001"
    [attempt] = merged["attempts"]
    assert attempt["outcome"] == "CONFIRMED" and attempt["fps"] == vlm.fps and merged["response"] == attempt["response"]
    # Each result names what it came from: every request's response and window are on disk.
    stage2 = segment.directory / "stage2"
    for item in [*candidates["windows"], attempt]:
        assert (stage2 / "responses" / f"{item['response']}.json").exists()
        assert (stage2 / "windows" / item["window"]).exists()

    [report] = incidents
    assert IncidentReport.model_validate(report).verification_status == "CONFIRMED"
    assert report["segment_id"] == "synthetic_cut_in" and report["incident_id"] == "INC-001"
    assert report["object_ids"] == [OBJECT_IDS["cut_in"]]
    assert report["subcategories"] == ["SUDDEN_CUT_IN"] and report["categories"] == ["DYNAMIC_HAZARD"]
    assert (report["start_s"], report["end_s"]) == CUT_IN_S
    assert report["vlm_reasoning"].startswith("The car moves") and report["driving_impact"]
    assert report["schema_version"] == "0.1" and report["pipeline_version"] == PIPELINE_VERSION
    assert report["model_id"] == "fake-model" and report["prompt_version"] == PROMPT_VERSION
    assert report["preprocessing_version"] == segment.meta["preprocessing_version"]
    [frame] = report["source_frames"]
    shown = segment.frames[segment.frames["frame_timestamp_micros"] == frame["frame_timestamp_micros"]]
    assert shown["camera_name"].tolist() == [frame["camera"]]
    assert round(shown["t_s"].iloc[0], 2) == frame["timestamp_s"]
    assert counts["pass1_sent"] == len(windows) and counts["pass2_sent"] == 1 and counts["reports"] == 1


def test_an_event_reported_past_its_windows_still_comes_through(segment, config, vlm):
    alias = cut_in_alias(segment)

    def spill(content):  # both windows that see the cut-in give its span from the track context, past their ends
        if window_json(content)["window"]["start_s"] in (4.0, 8.0):
            return json.dumps({"candidates": [{**GOOD_CANDIDATE, "start_s": 7.5, "end_s": 9.5,
                                               "object_aliases": [alias]}]})
        return json.dumps({"candidates": []})
    counts = run_segment(segment, config, vlm, FakeModel(scripted(alias, pass1=spill)))
    candidates, incidents = results(segment)
    assert [item["changes"] for item in candidates["candidates"]] == [["7.5 to 9.5 s: clipped to the window"]] * 2
    assert [(item["start_s"], item["end_s"]) for item in candidates["candidates"]] == [(7.5, 9.0), (8.0, 9.5)]
    [merged] = candidates["merged"]
    assert (merged["start_s"], merged["end_s"]) == (7.5, 9.5) and len(incidents) == 1 and counts["failing_checks"] == 0


def test_pass1_sends_each_window_frame_after_its_caption_then_the_tracks(segment, config, vlm):
    model = FakeModel(scripted(cut_in_alias(segment)))
    run_segment(segment, config, vlm, model)
    windows = default_windows(segment, config)
    for request, window in zip(model.requests, windows):
        content = request["content"]
        images = [block for block in content if block["type"] == "image"]
        assert [image["ref"] for image in images] == [frame.annotated_ref for frame in window.frames]
        assert all(image["ref"].endswith("__pass1.jpg") for image in images)
        assert images[0]["data"] == (segment.directory / images[0]["ref"]).read_bytes()
        assert captions(content) == [(frame.camera, round(frame.t_s, 2)) for frame in window.frames]
        context = window_json(content)
        assert context["window"] == {"start_s": round(window.window_start_s, 2), "end_s": round(window.window_end_s, 2),
                                     "cameras": ["FRONT"], "fps": 2.0}
        assert context["object_columns"] == ["t_s", "x_m", "y_m", "lateral_m", "front_gap_m", "v_rel_x_mps",
                                             "v_rel_y_mps"]
        assert [item["alias"] for item in context["objects"]] == [track.alias for track in window.objects]
        assert [len(item["rows"]) for item in context["objects"]] == [len(track.samples) for track in window.objects]
        assert [item["alias"] for item in context["omitted_objects"]] == [item.alias for item in window.omitted_objects]
        assert len(context["ego"]) == len(window.ego) and context["limits"] == window.limits.model_dump()
        assert request["system"] == PASS1_SYSTEM and request["schema"] == PASS1_SCHEMA
    assert any(cut_in_alias(segment) in [item["alias"] for item in window_json(request["content"])["objects"]]
               for request in model.requests)


def test_track_rows_are_rounded_and_flags_come_from_the_samples(segment, config):
    window = pass1_window(segment, config)
    context = window_json(request_content(window, segment.directory))
    rows = [value for item in context["objects"] for row in item["rows"] for value in row if value is not None]
    assert all(round(value, 2) == value for value in rows)
    by_alias = {item["alias"]: item for item in context["objects"]}
    for track in window.objects:
        flags = by_alias[track.alias]["flags"]
        assert ("stationary" in flags) == all(sample.stationary for sample in track.samples)
        assert ("present_at_segment_start" in flags) == track.truncated_start
        assert ("present_at_segment_end" in flags) == track.truncated_end
        assert ("not_directly_observed_at_times" in flags) == any(s.not_directly_observed for s in track.samples)
        assert ("past_driven_path_at_times" in flags) == (not all(s.on_driven_path for s in track.samples))
        assert by_alias[track.alias]["seen_s"] == [round(track.first_t_s, 2), round(track.last_t_s, 2)]
    assert any("stationary" in item["flags"] for item in context["objects"])  # the parked car


def test_each_flag_says_what_its_name_does(segment, config):
    window = pass1_window(segment, config)
    track = next(track for track in window.objects if len(track.samples) > 1)
    first, *rest = track.samples

    def flags(edited):
        return window_context(window.model_copy(update={"objects": [edited]}), {"t_s": "t_s"})["objects"][0]["flags"]
    every = track.model_copy(update={"truncated_start": False, "truncated_end": True, "samples": [
        first.model_copy(update={"stationary": True, "not_directly_observed": True, "on_driven_path": False,
                                 "type": "cyclist" if first.type != "cyclist" else "pedestrian"}),
        *[sample.model_copy(update={"stationary": True}) for sample in rest]]})
    assert flags(every) == ["stationary", "present_at_segment_end", "not_directly_observed_at_times",
                            "past_driven_path_at_times", "type_changes"]
    none = track.model_copy(update={"truncated_start": True, "truncated_end": False, "samples": [
        sample.model_copy(update={"stationary": False, "not_directly_observed": False, "on_driven_path": True,
                                  "type": first.type}) for sample in track.samples]})
    assert flags(none) == ["present_at_segment_start"]


def test_an_object_with_no_rows_in_its_window_is_not_called_stationary(segment, config):
    walker = alias_of(segment, "pedestrian")  # first seen near 11 s, so kept in 0 to 5 s with nothing to show
    window = pass1_window(segment, config, 0.0, 5.0, keep_aliases=[walker])
    [item] = [item for item in window_json(request_content(window, segment.directory))["objects"]
              if item["alias"] == walker]
    assert item["rows"] == [] and "stationary" not in item["flags"]


# JSON column -> ObjectSample field, written out here so a swapped or dropped field shows.
OBJECT_FIELDS = {"t_s": "t_s", "x_m": "pos_ego_x_m", "y_m": "pos_ego_y_m", "lateral_m": "lateral_offset_m",
                 "front_gap_m": "front_gap_m", "v_rel_x_mps": "v_rel_ego_x_mps", "v_rel_y_mps": "v_rel_ego_y_mps"}
PASS2_FIELDS = {**OBJECT_FIELDS, "path_clearance_m": "path_clearance_m", "a_x_mps2": "a_abs_ego_x_mps2",
                "a_y_mps2": "a_abs_ego_y_mps2"}
EGO_FIELDS = ("t_s", "speed_mps", "a_long_mps2", "a_lat_mps2", "yaw_rate_radps")


@pytest.mark.parametrize("pass2", [False, True], ids=["pass 1", "pass 2"])
def test_the_json_carries_each_samples_numbers_to_two_decimals(segment, config, pass2):
    window = pass1_window(segment, config)
    hypothesis = candidate(object_aliases=[cut_in_alias(segment)]) if pass2 else None
    context = window_json(request_content(window, segment.directory, hypothesis))
    fields = PASS2_FIELDS if pass2 else OBJECT_FIELDS

    def row(sample, names):
        return [None if getattr(sample, name) is None else round(getattr(sample, name), 2) for name in names]
    assert context["object_columns"] == list(fields) and context["ego_columns"] == list(EGO_FIELDS)
    assert [item["rows"] for item in context["objects"]] == [[row(sample, fields.values()) for sample in track.samples]
                                                             for track in window.objects]
    assert context["ego"] == [row(sample, EGO_FIELDS) for sample in window.ego]
    numbers = [value for item in context["objects"] for values in item["rows"] for value in values if value is not None]
    assert any(round(value, 1) != value for value in numbers)  # two decimals, as the docs say, not one


def test_pass2_sees_the_candidate_in_a_wider_window_at_5_fps_in_full_resolution(segment, config, vlm):
    model = FakeModel(scripted(cut_in_alias(segment)))
    run_segment(segment, config, vlm, model)
    content = model.requests[-1]["content"]
    context = window_json(content)
    margin = vlm.margin_s
    assert context["window"] == {"start_s": CUT_IN_S[0] - margin, "end_s": CUT_IN_S[1] + margin,
                                 "cameras": ["FRONT"], "fps": 5.0}
    images = [block for block in content if block["type"] == "image"]
    assert len(images) == 30 and all(image["ref"].endswith("__full.jpg") for image in images)
    assert all((segment.directory / image["ref"]).exists() for image in images)  # rendered for this request
    assert context["object_columns"][-3:] == ["path_clearance_m", "a_x_mps2", "a_y_mps2"]
    assert hypothesis_of(content)["labels"] == ["SUDDEN_CUT_IN"]
    assert model.requests[-1]["system"] == PASS2_SYSTEM


def test_pass2_keeps_the_candidates_objects_whatever_their_rank(segment, config, vlm):
    alias = cut_in_alias(segment)
    capped = replace(config, max_objects=0)
    model = FakeModel(scripted(alias))
    Run(segment, capped, vlm, model).pass2(candidate(start_s=8.0, end_s=12.0, object_aliases=[alias]))
    assert [item["alias"] for item in window_json(model.requests[0]["content"])["objects"]] == [alias]


def test_pass2_reaches_the_segments_last_frame_and_no_further(segment, config, vlm):
    alias, last_t_s = cut_in_alias(segment), segment.meta["last_t_s"]
    late = candidate(start_s=17.0, end_s=last_t_s - 0.1, object_aliases=[alias])  # its margin runs past the end

    def past_the_data(system, content, schema):
        return verdict(hypothesis_of(content), frame=captions(content)[-1], changes={"end_s": last_t_s + 0.9})
    model = FakeModel(past_the_data)
    [attempt], stands, accepted = Run(segment, config, vlm, model).pass2(late)
    [request] = model.requests
    assert captions(request["content"])[-1] == ("FRONT", round(last_t_s, 2))  # the last frame is in
    assert window_json(request["content"])["window"]["end_s"] == round(last_t_s, 2)  # and nothing after it
    assert stands["outcome"] == "invalid" and accepted is None  # an event past the data isn't inside the window


def test_a_long_window_takes_a_lower_rate_to_keep_to_max_images(segment, config, vlm):
    whole = candidate(start_s=0.0, end_s=segment.meta["last_t_s"], object_aliases=[cut_in_alias(segment)])
    for max_images in (40, vlm.max_images):  # at 5 FPS, the whole segment would be 101 images
        model = FakeModel(scripted(cut_in_alias(segment)))
        [attempt], stands, _ = Run(segment, config, replace(vlm, max_images=max_images), model).pass2(whole)
        [request] = model.requests
        images = sum(block["type"] == "image" for block in request["content"])
        assert max_images - 2 <= images <= max_images and attempt["fps"] < vlm.fps
        assert stands["outcome"] == "CONFIRMED"


def test_max_images_counts_every_cameras_frames(three_cameras, vlm):
    config, segment = three_cameras
    whole = candidate(start_s=0.0, end_s=segment.meta["last_t_s"], object_aliases=[cut_in_alias(segment)])
    model = FakeModel(scripted(cut_in_alias(segment)))
    Run(segment, config, vlm, model).pass2(whole)
    [request] = model.requests
    shown = captions(request["content"])
    assert vlm.max_images - 3 <= len(shown) <= vlm.max_images  # 33 frames from each camera, not 100
    assert {camera for camera, _ in shown} == {"FRONT", "FRONT_LEFT", "FRONT_RIGHT"}


def test_a_window_over_max_request_mb_is_built_again_at_a_rate_lowered_in_proportion(segment, config, vlm):
    alias = cut_in_alias(segment)
    hypothesis = candidate(start_s=8.0, end_s=12.0, object_aliases=[alias])

    def ask(budget_mb):
        model = FakeModel(scripted(alias))
        [attempt], _, _ = Run(segment, config, replace(vlm, max_request_mb=budget_mb), model).pass2(hypothesis)
        [request] = model.requests
        images = [block["data"] for block in request["content"] if block["type"] == "image"]
        return attempt["fps"], len(images), sum(map(len, images)) * 4 / 3 / 1e6  # base64, in MB
    fps, count, size_mb = ask(vlm.max_request_mb)
    assert (fps, count) == (5.0, 30) and size_mb < vlm.max_request_mb  # synthetic frames are small
    assert ask(size_mb / 2)[:2] == (2.5, 15)  # half the budget, half the rate
    fps, count, smaller_mb = ask(0.45 * size_mb)  # built once more only, so it can come out a little over
    assert (fps, count) == (2.25, 14) and 0.45 * size_mb < smaller_mb < 0.5 * size_mb


def test_a_window_with_no_frames_asks_nothing(segment, config, vlm):
    model = FakeModel(scripted(cut_in_alias(segment)))
    point = candidate(start_s=8.05, end_s=8.05, object_aliases=[cut_in_alias(segment)])  # between 5 FPS frames
    [attempt], stands, accepted = Run(segment, config, replace(vlm, margin_s=0.0), model).pass2(point)
    assert model.requests == [] and attempt["outcome"] == "no_frames" and accepted is None


def test_an_uncertain_verdict_is_retried_once_at_the_retry_rate(segment, config, vlm):
    model = FakeModel(scripted(cut_in_alias(segment), statuses=["UNCERTAIN", "CONFIRMED"]))
    counts = run_segment(segment, config, vlm, model)
    first, retry = model.requests[-2:]
    assert window_json(retry["content"])["window"]["fps"] == vlm.retry_fps == 10.0
    assert len(captions(retry["content"])) > 1.8 * len(captions(first["content"]))
    assert counts["pass2_retried"] == 1 and counts["pass2_sent"] == 2
    [merged] = results(segment)[0]["merged"]
    assert [(attempt["fps"], attempt["outcome"]) for attempt in merged["attempts"]] == [(5.0, "UNCERTAIN"),
                                                                                         (10.0, "CONFIRMED")]
    assert merged["response"] == merged["attempts"][1]["response"]  # the report's own answer
    assert [report["verification_status"] for report in results(segment)[1]] == ["CONFIRMED"]
    assert [record["step"] for record in stored(segment, "responses")].count("pass2") == 2  # both kept


@pytest.mark.parametrize("statuses, retry, kept", [
    (["UNCERTAIN", "UNCERTAIN"], True, "UNCERTAIN"),
    (["UNCERTAIN"], False, "UNCERTAIN"),
    (["REJECTED"], True, "REJECTED"),
])
def test_rejected_and_uncertain_verdicts_are_kept(segment, config, vlm, statuses, retry, kept):
    model = FakeModel(scripted(cut_in_alias(segment), statuses=statuses))
    counts = run_segment(segment, config, replace(vlm, retry_uncertain=retry), model)
    [report] = results(segment)[1]
    assert report["verification_status"] == kept and len(model.requests) == 5 + len(statuses)
    assert report["label_assessments"][0]["status"] == kept and counts[kept] == 1
    assert report["uncertainty_reasons"] == (["the gap is hard to see"] if kept == "UNCERTAIN" else [])
    # subcategories are what Pass 2 didn't reject; the assessments keep the rest
    assert report["subcategories"] == ([] if kept == "REJECTED" else ["SUDDEN_CUT_IN"])
    assert report["categories"] == ([] if kept == "REJECTED" else ["DYNAMIC_HAZARD"])


@pytest.mark.parametrize("retry, outcome", [
    (lambda content: ModelFailure("max_tokens"), "failed"),
    (lambda content: verdict(hypothesis_of(content), frame=captions(content)[0], changes={"object_aliases": ["V999"]}),
     "invalid"),  # it cites an object the window doesn't have
], ids=["failed", "invalid"])
def test_a_retry_that_fails_keeps_the_first_verdict(segment, config, vlm, retry, outcome):
    alias = cut_in_alias(segment)
    answers = scripted(alias, statuses=["UNCERTAIN"])
    calls = []

    def answer(system, content, schema):
        calls.append(schema)
        return retry(content) if calls.count(PASS2_SCHEMA) == 2 else answers(system, content, schema)
    run_segment(segment, config, vlm, FakeModel(answer))
    [merged] = results(segment)[0]["merged"]
    assert [attempt["outcome"] for attempt in merged["attempts"]] == ["UNCERTAIN", outcome]
    assert merged["outcome"] == "UNCERTAIN" and merged["response"] == merged["attempts"][0]["response"]
    assert [report["verification_status"] for report in results(segment)[1]] == ["UNCERTAIN"]


def test_an_uncertain_verdict_is_not_asked_again_at_a_rate_no_higher(segment, config, vlm):
    """The whole segment at 10 FPS is capped to the 4.95 FPS it had already: the retry would be the same request."""
    alias = cut_in_alias(segment)
    whole = candidate(start_s=0.0, end_s=segment.meta["last_t_s"], object_aliases=[alias])
    model = FakeModel(scripted(alias, statuses=["UNCERTAIN", "CONFIRMED"]))
    run = Run(segment, config, vlm, model)
    attempts, stands, accepted = run.pass2(whole)
    assert [(attempt["fps"], attempt["outcome"]) for attempt in attempts] == [(4.95, "UNCERTAIN")]
    assert len(model.requests) == 1 and run.counts["pass2_retried"] == run.counts["pass2_reused"] == 0
    assert stands["outcome"] == "UNCERTAIN" and accepted[0].status == "UNCERTAIN"


def test_a_verdict_citing_an_omitted_object_reports_its_laser_id(segment, config, vlm):
    alias, cited = cut_in_alias(segment), []

    def answer(system, content, schema):
        if schema is PASS1_SCHEMA:
            return scripted(alias)(system, content, schema)
        cited.append(window_json(content)["omitted_objects"][0]["alias"])
        return verdict(hypothesis_of(content), frame=captions(content)[0], changes={"object_aliases": [alias, *cited]})
    run_segment(segment, replace(config, max_objects=1), vlm, FakeModel(answer))
    [report] = results(segment)[1]
    omitted_id = segment.tracks.loc[segment.tracks["alias"] == cited[0], "laser_object_id"].iloc[0]
    assert report["object_ids"] == [OBJECT_IDS["cut_in"], omitted_id]


# --- one event reported twice ---


def found_twice(segment):
    """A fake model's answers, shaped like the live run's INC-005 and INC-006.

    Pass 1 finds the pedestrian crossing, and in its last window a cut-in too, which Pass 2 rejects while it confirms
    the crossing in the same frames.
    """
    walker, cars = alias_of(segment, "pedestrian"), [cut_in_alias(segment), alias_of(segment, "lead")]

    def answer(system, content, schema):
        if schema is PASS1_SCHEMA:
            window = window_json(content)["window"]
            crossing = {**GOOD_CANDIDATE, "labels": ["PEDESTRIAN_INTRUSION"], "start_s": max(14.0, window["start_s"]),
                        "end_s": window["end_s"], "object_aliases": [walker]}
            cut_in = {**GOOD_CANDIDATE, "start_s": 16.0, "end_s": window["end_s"], "object_aliases": cars}
            found = ([crossing] if window["end_s"] > 14.0 else []) + ([cut_in] if window["start_s"] >= 16.0 else [])
            return json.dumps({"candidates": found})
        hypothesis, crossing = hypothesis_of(content), assessment("CONFIRMED", "PEDESTRIAN_INTRUSION")
        changes = {"label_assessments": [crossing]}
        if hypothesis["labels"] == ["SUDDEN_CUT_IN"]:
            changes = {"label_assessments": [assessment("REJECTED"), crossing], "object_aliases": [walker],
                       "start_s": 17.0}
        return verdict(hypothesis, frame=captions(content)[-1], changes=changes)
    return answer


def test_an_event_confirmed_from_two_candidates_keeps_both_reports_and_marks_the_later(segment, config, vlm):
    counts = run_segment(segment, config, vlm, FakeModel(found_twice(segment)))
    candidates, incidents = results(segment)
    assert [item["labels"] for item in candidates["merged"]] == [["PEDESTRIAN_INTRUSION"], ["SUDDEN_CUT_IN"]]
    assert [(report["incident_id"], report["subcategories"], report["object_ids"], report["duplicate_of"])
            for report in incidents] == [("INC-001", ["PEDESTRIAN_INTRUSION"], [OBJECT_IDS["pedestrian"]], None),
                                         ("INC-002", ["PEDESTRIAN_INTRUSION"], [OBJECT_IDS["pedestrian"]], "INC-001")]
    assert counts["reports"] == 2 and counts["duplicates"] == 1


def report_of(incident_id, start_s, end_s, labels=("PEDESTRIAN_INTRUSION",), object_ids=("PEDESTRIAN",)):
    """An accepted report, with only what marking duplicates reads: no labels means Pass 2 rejected them all."""
    return IncidentReport(
        schema_version="0.1", segment_id="synthetic_cut_in", incident_id=incident_id, duplicate_of=None,
        start_s=start_s, end_s=end_s, categories=[], subcategories=list(labels), object_ids=list(object_ids),
        verification_status="CONFIRMED" if labels else "REJECTED", label_assessments=[], vlm_reasoning="",
        driving_impact="", uncertainty_reasons=[], source_frames=[], pipeline_version="", model_id="",
        prompt_version="", preprocessing_version="")


@pytest.mark.parametrize("later, duplicate_of", [
    (report_of("INC-002", 17.0, 20.0), "INC-001"),
    (report_of("INC-002", 20.0, 21.0), "INC-001"),                        # touching counts
    (report_of("INC-002", 20.5, 21.0), None),                             # apart in time
    (report_of("INC-002", 17.0, 20.0, object_ids=("CUT-IN",)), None),     # another object
    (report_of("INC-002", 17.0, 20.0, labels=("SUDDEN_CUT_IN",)), None),  # another label
    (report_of("INC-002", 17.0, 20.0, labels=()), None),                  # every label rejected
], ids=["same event", "touching", "time", "object", "label", "rejected"])
def test_a_duplicate_shares_a_kept_label_an_object_and_its_time(later, duplicate_of):
    reports = [report_of("INC-001", 14.0, 20.0), later]
    mark_duplicates(reports)
    assert [report.duplicate_of for report in reports] == [None, duplicate_of]


def test_a_duplicate_names_the_first_earlier_report_it_repeats():
    reports = [report_of("INC-001", 10.0, 12.0), report_of("INC-002", 11.0, 15.0), report_of("INC-003", 14.0, 16.0),
               report_of("INC-004", 11.5, 14.5)]
    mark_duplicates(reports)
    assert [report.duplicate_of for report in reports] == [None, "INC-001", "INC-002", "INC-001"]


def test_reports_that_name_no_object_are_never_marked():
    """Unlike the merge, which falls back on a shared label: the docs leave this case to M3."""
    debris = [report_of(f"INC-00{number}", 4.0, 6.0, labels=("ROAD_DEBRIS",), object_ids=()) for number in (1, 2)]
    mark_duplicates(debris)
    assert [report.duplicate_of for report in debris] == [None, None]


# --- what's stored, and reuse ---


def test_raw_answers_and_validation_errors_are_stored_apart_from_results(segment, config, vlm):
    alias = cut_in_alias(segment)

    def pass1(content):
        window = window_json(content)["window"]
        if window["start_s"] == 0.0:
            return "not json"
        bad = {**GOOD_CANDIDATE, "start_s": max(8.0, window["start_s"]), "end_s": min(9.0, window["end_s"])}
        if window["start_s"] == 4.0:
            return json.dumps({"candidates": [{**bad, "object_aliases": [alias]}, {**bad, "labels": []}]})
        return json.dumps({"candidates": []})
    run_segment(segment, config, vlm, FakeModel(scripted(alias, pass1=pass1)))

    assert "not json" in [record.get("text") for record in stored(segment, "responses")]
    errors = stored(segment, "errors")
    assert [record["step"] for record in errors] == ["pass1", "pass1"]
    messages = [message for record in errors for message in record["errors"]]
    assert any("Invalid JSON" in message for message in messages) and "W1-1: no labels" in messages
    candidates, incidents = results(segment)
    assert [item["outcome"] for item in candidates["windows"]] == ["invalid"] + ["answered"] * 4
    assert [item["errors"] for item in candidates["candidates"]] == [[], ["no labels"]]
    assert len(candidates["merged"]) == 1 and len(incidents) == 1  # the valid candidate went on alone
    for path in (segment.directory / "stage2" / "errors").glob("*.json"):
        assert (segment.directory / "stage2" / "responses" / path.name).exists()  # each beside its raw answer


def test_an_invalid_verdict_is_stored_and_makes_no_report(segment, config, vlm):
    alias = cut_in_alias(segment)

    def answer(system, content, schema):
        if schema is PASS2_SCHEMA:
            return verdict(hypothesis_of(content), changes={"object_aliases": ["V999"]})
        return scripted(alias)(system, content, schema)
    counts = run_segment(segment, config, vlm, FakeModel(answer))
    candidates, incidents = results(segment)
    assert incidents == [] and candidates["merged"][0]["outcome"] == "invalid" and counts["pass2_invalid"] == 1
    assert [record["step"] for record in stored(segment, "errors")] == ["pass2"]


def test_a_stored_answer_is_reused_with_no_second_call(segment, config, vlm):
    alias = cut_in_alias(segment)
    run_segment(segment, config, vlm, FakeModel(scripted(alias)))
    first = results(segment)
    again = FakeModel(scripted(alias))
    counts = run_segment(segment, config, vlm, again)
    assert again.requests == [] and counts["pass1_reused"] == 5 and counts["pass2_reused"] == 1
    assert results(segment) == first


def test_a_dry_runs_gaps_never_replace_a_full_runs_results(segment, config, vlm, tmp_path):
    """Pass 1 is answered from the store, and Pass 2's wider window goes unsent: the results stay as they were."""
    run_segment(segment, config, vlm, FakeModel(scripted(cut_in_alias(segment)), model_id=vlm.model["id"]))
    stage2 = segment.directory / "stage2"
    before = [(stage2 / name).read_bytes() for name in ("candidates.json", "incidents.json")]
    dry = ClaudeModel(vlm.model["id"], vlm.model["effort"], vlm.model["max_tokens"], dry_run=tmp_path / "dry_run")
    counts = run_segment(segment, config, replace(vlm, margin_s=2.0), dry)
    assert (counts["pass1_reused"], counts["pass2_not_sent"], counts["unsent"]) == (5, 1, 1)
    assert [(stage2 / name).read_bytes() for name in ("candidates.json", "incidents.json")] == before
    assert len(list((tmp_path / "dry_run").glob("*.json"))) == 1


def test_another_model_setting_prompt_schema_or_image_is_a_new_request(segment, config, vlm):
    alias = cut_in_alias(segment)
    run_segment(segment, config, vlm, FakeModel(scripted(alias)))
    other_model = FakeModel(scripted(alias), model_id="another-model")
    run_segment(segment, config, vlm, other_model)
    other_effort = FakeModel(scripted(alias))
    run_segment(segment, config, replace(vlm, model={**vlm.model, "effort": "high"}), other_effort)
    assert len(other_model.requests) == len(other_effort.requests) == 6
    content = request_content(pass1_window(segment, config), segment.directory)
    changed = [{**block, "data": block["data"] + b"\0"} if block["type"] == "image" else block for block in content]
    keys = [request_key("m", vlm.model, PASS1_SYSTEM, content, PASS1_SCHEMA),
            request_key("m", vlm.model, PASS1_SYSTEM, changed, PASS1_SCHEMA),
            request_key("m", vlm.model, PASS1_SYSTEM + " ", content, PASS1_SCHEMA),
            request_key("m", vlm.model, PASS1_SYSTEM, content, PASS2_SCHEMA)]
    assert len(set(keys)) == 4


def test_errors_that_no_longer_apply_are_removed(segment, config, vlm):
    alias = cut_in_alias(segment)
    run_segment(segment, config, vlm, FakeModel(scripted(alias)))
    stale = segment.directory / "stage2" / "errors" / next((segment.directory / "stage2" / "responses").glob("*")).name
    stale.parent.mkdir(exist_ok=True)
    stale.write_text('{"step": "pass1", "errors": ["from older checks"]}')
    run_segment(segment, config, vlm, FakeModel(scripted(alias)))
    assert not stale.exists()


# --- refusals, cut-offs and API errors ---


def test_a_refusal_is_recorded_not_parsed_and_the_run_goes_on(segment, config, vlm):
    alias = cut_in_alias(segment)
    answers = scripted(alias)

    def answer(system, content, schema):
        if schema is PASS1_SCHEMA and window_json(content)["window"]["start_s"] == 8.0:
            return ModelFailure("refusal", category="cyber")
        return answers(system, content, schema)
    counts = run_segment(segment, config, vlm, FakeModel(answer))
    failures = [record for record in stored(segment, "responses") if "failure" in record]
    assert [(record["failure"], record["category"]) for record in failures] == [("refusal", "cyber")]
    assert "text" not in failures[0] and counts["pass1_failed"] == 1
    candidates, incidents = results(segment)
    assert [item["outcome"] for item in candidates["windows"]] == ["answered", "answered", "failed", "answered",
                                                                   "answered"]  # never assessed, not empty
    assert [item["id"] for item in candidates["candidates"]] == ["W1-0"] and len(incidents) == 1
    again = FakeModel(answer)
    assert run_segment(segment, config, vlm, again)["pass1_failed"] == 1 and again.requests == []


def test_a_failure_marked_retry_is_asked_again_on_a_rerun(segment, config, vlm):
    alias = cut_in_alias(segment)

    def busy(system, content, schema):  # as when the fallback model was rate-limited
        if schema is PASS1_SCHEMA and window_json(content)["window"]["start_s"] == 8.0:
            return ModelFailure("refusal", retry=True, category="cyber")
        return scripted(alias)(system, content, schema)
    run_segment(segment, config, vlm, FakeModel(busy))
    again = FakeModel(scripted(alias))
    counts = run_segment(segment, config, vlm, again)
    # The refused window is asked again, and its answer widens the event, so Pass 2 asks about the new span.
    assert [request["schema"] for request in again.requests] == [PASS1_SCHEMA, PASS2_SCHEMA]
    assert counts["pass1_reused"] == 4 and counts["pass1_sent"] == 1 and len(results(segment)[1]) == 1


def test_a_cut_off_verdict_is_recorded_and_makes_no_report(segment, config, vlm):
    alias = cut_in_alias(segment)

    def answer(system, content, schema):
        return ModelFailure("max_tokens") if schema is PASS2_SCHEMA else scripted(alias)(system, content, schema)
    counts = run_segment(segment, config, vlm, FakeModel(answer))
    candidates, incidents = results(segment)
    assert incidents == [] and candidates["merged"][0]["outcome"] == "failed" and counts["pass2_failed"] == 1
    assert [record.get("failure") for record in stored(segment, "responses")].count("max_tokens") == 1


class StubClient:
    """Stands in for anthropic.Anthropic(): keeps each request, then returns or raises `response`, or calls it."""

    def __init__(self, response):
        self.response, self.calls = response, []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response(**kwargs) if callable(self.response) else self.response


THINKING = {"type": "thinking", "thinking": "", "signature": "sig"}


def message(stop_reason="end_turn", text='{"candidates": []}', model="claude-opus-5-5", details=None, blocks=None):
    """A real BetaMessage from the installed SDK, with thinking first, and usage whose iterations show a fallback."""
    from anthropic.types.beta import BetaMessage

    attempt = {"input_tokens": 1100, "output_tokens": 0, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
    return BetaMessage.model_validate({
        "id": "msg_1", "type": "message", "role": "assistant", "model": model, "stop_reason": stop_reason,
        "stop_sequence": None, "stop_details": details,
        "content": [THINKING, {"type": "text", "text": text}] if blocks is None else blocks,
        "usage": {"input_tokens": 1200, "output_tokens": 80, "iterations": [
            {**attempt, "type": "message", "model": "claude-opus-5-5"},
            {**attempt, "type": "fallback_message", "model": model, "input_tokens": 1200, "output_tokens": 80}]}})


def api_error(status: int):
    import anthropic
    import httpx2

    kind = {400: anthropic.BadRequestError, 401: anthropic.AuthenticationError, 404: anthropic.NotFoundError,
            413: anthropic.RequestTooLargeError}[status]
    request = httpx2.Request("POST", "http://127.0.0.1:9/v1/messages")
    return kind(f"http {status}", response=httpx2.Response(status, request=request), body=None)


CONTENT = [{"type": "text", "text": "FRONT t=8.00 s"}, {"type": "image", "ref": "render/FRONT/1__pass1.jpg",
                                                         "data": b"\xff\xd8jpeg"}, {"type": "text", "text": "{}"}]


@needs_sdk
def test_claude_reads_the_text_after_thinking_and_names_the_model_that_answered():
    client = StubClient(message(model="claude-opus-4-8"))  # a server-side fallback answered
    reply = ClaudeModel("claude-opus-5-5", "medium", 16000, client).respond("system", CONTENT, PASS1_SCHEMA)
    assert reply.text == '{"candidates": []}' and reply.model_id == "claude-opus-4-8"
    assert reply.usage["output_tokens"] == 80  # the top level counts only the attempt that answered
    assert [entry["type"] for entry in reply.usage["iterations"]] == ["message", "fallback_message"]


@needs_sdk
@pytest.mark.parametrize("response, reason, category, retry", [
    ({"stop_reason": "refusal", "text": "I can't", "details": {"type": "refusal", "category": "bio"}},
     "refusal", "bio", False),
    ({"stop_reason": "refusal", "text": "", "details": {"type": "refusal", "category": "cyber",
                                                         "recommended_model": "claude-opus-4-8"}},
     "refusal", "cyber", True),  # the fallback couldn't run, so a rerun may get an answer
    ({"stop_reason": "refusal", "text": ""}, "refusal", None, False),
    ({"stop_reason": "max_tokens", "text": '{"candidates": [{"lab'}, "max_tokens", None, False),
    ({"blocks": [THINKING]}, "no_text", None, False),
], ids=["refusal", "fallback couldn't run", "refusal without details", "cut off", "no text block"])
def test_claude_turns_a_response_with_no_answer_into_a_failure(response, reason, category, retry):
    client = StubClient(message(**response, model="claude-opus-4-8"))  # a fallback ran last: the record names it
    with pytest.raises(ModelFailure) as caught:
        ClaudeModel("claude-opus-5-5", "medium", 16000, client).respond("system", CONTENT, PASS1_SCHEMA)
    details = caught.value.details
    assert (caught.value.reason, details.get("category"), details.get("retry", False)) == (reason, category, retry)
    assert details["served_by"] == "claude-opus-4-8" and details["usage"]["input_tokens"] == 1200  # still billed


@needs_sdk
@pytest.mark.parametrize("status", [400, 413])
def test_an_api_error_on_one_request_is_a_failure_a_rerun_asks_again(status):
    with pytest.raises(ModelFailure) as caught:
        ClaudeModel("claude-opus-5-5", "medium", 16000, StubClient(api_error(status))).respond(
            "system", CONTENT, PASS1_SCHEMA)
    assert caught.value.reason == f"http_{status}" and caught.value.details["retry"] is True


@needs_sdk
@pytest.mark.parametrize("status", [401, 404], ids=["no key", "wrong model"])
def test_an_api_error_that_every_request_would_meet_stops_the_run(segment, config, vlm, status):
    error = api_error(status)
    model = ClaudeModel(vlm.model["id"], vlm.model["effort"], vlm.model["max_tokens"], StubClient(error))
    with pytest.raises(type(error)):
        run_segment(segment, config, vlm, model)
    assert not (segment.directory / "stage2" / "responses").exists()  # nothing stored, so a rerun asks again


@needs_sdk
def test_claude_runs_on_a_synthetic_segment_and_reports_the_model_that_answered(segment, config, vlm):
    answers = scripted(cut_in_alias(segment))

    def create(**request):  # a fallback answers every request
        content, schema = request["messages"][0]["content"], request["output_config"]["format"]["schema"]
        return message(text=answers(request["system"], content, schema), model="claude-opus-4-8")
    model = ClaudeModel(vlm.model["id"], vlm.model["effort"], vlm.model["max_tokens"], StubClient(create))
    counts = run_segment(segment, config, vlm, model)  # remote, and allowed: the segment is synthetic
    [report] = results(segment)[1]
    assert model.remote and counts["pass1_sent"] == 5 and counts["pass2_sent"] == 1
    assert report["model_id"] == "claude-opus-4-8" and report["verification_status"] == "CONFIRMED"
    assert {record["served_by"] for record in stored(segment, "responses")} == {"claude-opus-4-8"}


@needs_sdk
def test_a_request_the_api_turns_away_is_recorded_and_the_run_goes_on(segment, config, vlm):
    answers = scripted(cut_in_alias(segment))

    def create(**request):
        content, schema = request["messages"][0]["content"], request["output_config"]["format"]["schema"]
        if schema is PASS2_SCHEMA:
            raise api_error(413)
        return message(text=answers(request["system"], content, schema))
    model = ClaudeModel(vlm.model["id"], vlm.model["effort"], vlm.model["max_tokens"], StubClient(create))
    counts = run_segment(segment, config, vlm, model)
    candidates, incidents = results(segment)
    assert incidents == [] and candidates["merged"][0]["outcome"] == "failed" and counts["pass2_failed"] == 1
    [failure] = [record for record in stored(segment, "responses") if "failure" in record]
    assert failure["failure"] == "http_413" and failure["retry"] is True
    again = run_segment(segment, config, vlm, model)
    assert again["pass1_reused"] == 5 and again["pass2_sent"] == 1  # only the one turned away is asked again


@needs_sdk
def test_claudes_request_has_the_model_betas_fallbacks_output_config_and_images(vlm):
    client = StubClient(message())
    model = ClaudeModel(vlm.model["id"], vlm.model["effort"], vlm.model["max_tokens"], client)
    model.respond("the system prompt", CONTENT, PASS1_SCHEMA)
    [request] = client.calls
    assert request["model"] == "claude-opus-5-5" and request["max_tokens"] == 16000
    assert request["betas"] == ["server-side-fallback-2026-07-01"] == [FALLBACK_BETA]
    assert request["fallbacks"] == "default"
    assert request["output_config"] == {"effort": "medium", "format": {"type": "json_schema", "schema": PASS1_SCHEMA}}
    assert request["system"] == "the system prompt"
    assert not {"thinking", "temperature", "top_p", "top_k", "budget_tokens"} & set(request)
    [turn] = request["messages"]  # one user turn and no assistant prefill
    assert turn["role"] == "user"
    assert turn["content"] == [CONTENT[0], {"type": "image", "source": {
        "type": "base64", "media_type": "image/jpeg", "data": base64.standard_b64encode(b"\xff\xd8jpeg").decode()}},
                               CONTENT[2]]
    other = StubClient(message())  # the settings come from stage2.toml, not from the code
    ClaudeModel("claude-opus-5-5", "high", 8000, other).respond("system", CONTENT, PASS1_SCHEMA)
    assert other.calls[0]["output_config"]["effort"] == "high" and other.calls[0]["max_tokens"] == 8000


def test_claudes_request_matches_the_installed_sdk():
    anthropic = pytest.importorskip("anthropic")
    from anthropic.resources.beta.messages import Messages
    from anthropic.types.anthropic_beta_param import AnthropicBetaParam
    from anthropic.types.beta.beta_output_config_param import BetaOutputConfigParam

    request = ClaudeModel("claude-opus-5-5", "medium", 16000).request("system", CONTENT, PASS1_SCHEMA)
    assert set(request) <= set(inspect.signature(Messages.create).parameters)
    assert FALLBACK_BETA in typing.get_args(typing.get_args(AnthropicBetaParam)[1])
    effort = typing.get_type_hints(BetaOutputConfigParam)["effort"]
    assert "medium" in typing.get_args(typing.get_args(effort)[0])
    assert hasattr(anthropic, "Anthropic")


def test_the_package_and_a_dry_run_work_without_the_sdk(tmp_path):
    code = ("import sys\n"
            "import agatha.vlm.run\n"
            "print('anthropic' in sys.modules)\n"
            "sys.modules['anthropic'] = None  # any import of it now fails\n"
            "from pathlib import Path\n"
            "from agatha.vlm.model import ClaudeModel, NotSent\n"
            f"model = ClaudeModel('claude-opus-5-5', 'medium', 16000, dry_run=Path({str(tmp_path)!r}))\n"
            "try:\n"
            "    model.respond('system', [{'type': 'text', 'text': '{}'}], {})\n"
            "except NotSent:\n"
            "    print('not sent')\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60, cwd=PROJECT)
    assert out.stdout.split() == ["False", "not", "sent"], out.stderr
    assert len(list(tmp_path.glob("*.json"))) == 1


# --- the data guard ---


@pytest.mark.parametrize("remote, context, cleared, allowed", [
    (True, PRIVATE_CONTEXT, False, False),
    (True, PUBLIC_SEGMENT, False, False),  # public, but real data all the same
    (True, "synthetic_cut_in", False, True),
    (True, "a_synthetic_name", False, False),
    (True, f"synthetic_cut_in/../{PRIVATE_CONTEXT}", False, False),  # a segment.json can name anything
    (False, PRIVATE_CONTEXT, False, True),  # a local model may run on anything
    (True, PRIVATE_CONTEXT, True, True),
])
def test_the_guard_holds_a_remote_model_to_synthetic_segments(remote, context, cleared, allowed):
    model = FakeModel(lambda *_: "{}", remote=remote)
    if allowed:
        check_guard(model, context, cleared)
    else:
        with pytest.raises(GuardRefused) as caught:
            check_guard(model, context, cleared)
        assert context not in str(caught.value)


def stage2_toml(path: Path, cleared: str) -> Path:
    """configs/stage2.toml with real_data_cleared = `cleared`, whatever the shipped file says, written to `path`."""
    lines = STAGE2_CONFIG.read_text().splitlines(keepends=True)
    path.write_text("".join(f"real_data_cleared = {cleared}\n" if line.startswith("real_data_cleared") else line
                            for line in lines))
    return path


def test_a_refused_run_sends_and_writes_nothing(private_segment, vlm):
    config, segment = private_segment
    vlm = replace(vlm, real_data_cleared=False)  # the shipped config clears real data since Waymo's answer
    model = FakeModel(scripted(cut_in_alias(segment)), remote=True)
    with pytest.raises(GuardRefused):
        run_segment(segment, config, vlm, model)
    assert model.requests == [] and not (segment.directory / "stage2").exists()
    local = FakeModel(scripted(cut_in_alias(segment)))
    assert run_segment(segment, config, vlm, local)["reports"] == 1


def test_only_a_literal_true_clears_real_data(tmp_path):
    for value, cleared in (("false", False), ('"true"', False), ("1", False), ("true", True)):
        assert load_vlm_config(stage2_toml(tmp_path / "stage2.toml", value)).real_data_cleared is cleared


def test_claude_is_remote_and_a_dry_run_is_not(tmp_path):
    claude = ClaudeModel("claude-opus-5-5", "medium", 16000)
    assert claude.remote is True and claude.client is None  # no SDK client until a request is made
    dry = ClaudeModel("claude-opus-5-5", "medium", 16000, dry_run=tmp_path)
    assert dry.remote is False and dry.model_id == claude.model_id
    with pytest.raises(NotSent):
        dry.respond("system", CONTENT, PASS1_SCHEMA)
    [written] = list(tmp_path.glob("*.json"))
    image = json.loads(written.read_text())["messages"][0]["content"][1]
    assert image["source"]["data"] == "ref:render/FRONT/1__pass1.jpg" and dry.client is None


def test_no_request_names_the_segment(private_segment, vlm):
    config, segment = private_segment
    model = FakeModel(scripted(cut_in_alias(segment)))
    run_segment(segment, config, vlm, model)
    assert len(model.requests) == 6
    for request in model.requests:
        assert PRIVATE_CONTEXT not in request["system"] + "".join(texts(request["content"]))


# --- the CLI ---


@pytest.fixture(scope="module")
def private_outputs(tmp_path_factory):
    """Stage 1's outputs for a synthetic segment under a real-looking name, and the config that reads them."""
    base = tmp_path_factory.mktemp("private")
    ref = write_synthetic_segment(base / "wod", context_name=PRIVATE_CONTEXT)
    config_path = base / "stage1.toml"
    config_path.write_text(STAGE1_CONFIG.read_text().replace("$WOD_ROOT", str(base / "wod"))
                           .replace("$WOD_OUTPUTS", str(base / "outputs")))
    process_segment(load_config(config_path), ref)
    return config_path, base / "outputs" / PRIVATE_CONTEXT


@pytest.fixture
def private_segment(private_outputs, tmp_path):
    """(config, segment): a copy of the private-named outputs, so each test's stage2/ starts empty."""
    config_path, directory = private_outputs
    copy = tmp_path / PRIVATE_CONTEXT
    shutil.copytree(directory, copy, ignore=shutil.ignore_patterns("clips", "qa", "stage2"))
    return load_config(config_path), load_segment(copy)


def agatha_vlm(config_path, *args, context=PRIVATE_CONTEXT, vlm_config=STAGE2_CONFIG,
               **options) -> subprocess.CompletedProcess:
    # No key, and an address nothing listens on: even a bug can't reach the API from here.
    env = {**os.environ, "ANTHROPIC_API_KEY": "", "ANTHROPIC_BASE_URL": "http://127.0.0.1:9"}
    return subprocess.run([sys.executable, "-m", "agatha.vlm.run", "--config", str(config_path), "--vlm-config",
                           str(vlm_config), "--context", context, *args], capture_output=True, text=True,
                          timeout=300, env=env, cwd=PROJECT, **options)


def test_the_dry_run_writes_each_request_and_prints_only_counts_and_the_label(private_outputs):
    config_path, directory = private_outputs
    shutil.rmtree(directory / "stage2", ignore_errors=True)
    run = agatha_vlm(config_path, "--model", "dry-run")
    assert run.returncode == 0, run.stderr
    assert PRIVATE_CONTEXT not in run.stdout + run.stderr and run.stderr == ""
    label = context_label(PRIVATE_CONTEXT)
    assert run.stdout.splitlines()[0].startswith(f"{label}: pass 1: 5 windows: 0 sent, 0 reused, 5 not sent")
    assert "none written, as requests went unsent" in run.stdout
    written = sorted((directory / "stage2" / "dry_run").glob("*.json"))
    assert len(written) == 5
    for path in written:
        request = json.loads(path.read_text())
        assert request["model"] == "claude-opus-5-5" and request["fallbacks"] == "default"
        images = [block["source"]["data"] for block in request["messages"][0]["content"] if block["type"] == "image"]
        assert images and all(data.startswith("ref:render/FRONT/") and data.endswith("__pass1.jpg") for data in images)
        assert PRIVATE_CONTEXT not in path.read_text()
    assert not (directory / "stage2" / "incidents.json").exists()  # a dry run's gaps don't replace results


def test_with_no_model_named_the_cli_does_a_dry_run(private_outputs):
    """A run that sends anything costs money, so it has to be asked for."""
    run = agatha_vlm(private_outputs[0])
    assert run.returncode == 0 and run.stderr == "" and "5 not sent" in run.stdout.splitlines()[0]


def test_stage2_outputs_are_private_whatever_umask_the_caller_has(private_outputs):
    config_path, directory = private_outputs
    shutil.rmtree(directory / "stage2", ignore_errors=True)
    assert agatha_vlm(config_path, "--model", "dry-run", umask=0o022).returncode == 0
    paths = [directory / "stage2", *(directory / "stage2").rglob("*")]
    assert {path.parent.name for path in paths} >= {"dry_run", "windows"}
    assert not [path for path in paths if stat.S_IMODE(path.stat().st_mode) & 0o077]


def test_the_cli_counts_the_reports_marked_duplicate(private_outputs, tmp_path, vlm):
    """Answers a run stored serve the CLI's dry run in full, so it writes the results, and counts the duplicate."""
    config_path, directory = private_outputs
    shutil.copytree(directory, tmp_path / "outputs" / PRIVATE_CONTEXT, ignore=shutil.ignore_patterns("stage2"))
    config = tmp_path / "stage1.toml"
    config.write_text(config_path.read_text().replace(str(directory.parent), str(tmp_path / "outputs")))
    segment = load_segment(tmp_path / "outputs" / PRIVATE_CONTEXT)
    run_segment(segment, load_config(config), vlm, FakeModel(found_twice(segment), model_id=vlm.model["id"]))
    run = agatha_vlm(config)
    assert run.returncode == 0 and run.stderr == ""
    assert run.stdout.splitlines()[1] == (
        f"{context_label(PRIVATE_CONTEXT)}: pass 2: 0 sent, 2 reused, 0 not sent, 0 no frames, 0 failed, 0 invalid, "
        "0 retried; 2 confirmed, 0 rejected, 0 uncertain; incident reports: 2 written, 1 marked duplicate")


def test_a_dry_run_leaves_only_its_own_requests(private_outputs, tmp_path):
    config_path, directory = private_outputs
    high = tmp_path / "stage2.toml"
    high.write_text(STAGE2_CONFIG.read_text().replace('effort = "medium"', 'effort = "high"'))
    assert agatha_vlm(config_path, "--model", "dry-run").returncode == 0
    assert agatha_vlm(config_path, "--model", "dry-run", vlm_config=high).returncode == 0
    written = list((directory / "stage2" / "dry_run").glob("*.json"))
    assert len(written) == 5 and {json.loads(path.read_text())["output_config"]["effort"] for path in written} == {
        "high"}


def test_the_cli_refuses_claude_on_real_data_before_reading_it(private_outputs, tmp_path):
    config_path, directory = private_outputs
    shutil.rmtree(directory / "stage2", ignore_errors=True)
    run = agatha_vlm(config_path, "--model", "claude", vlm_config=stage2_toml(tmp_path / "stage2.toml", "false"))
    assert run.returncode == 1 and "refused" in run.stderr
    assert PRIVATE_CONTEXT not in run.stdout + run.stderr
    assert not (directory / "stage2").exists()


def test_the_cli_refuses_before_it_reads_anything(private_outputs, tmp_path):
    run = agatha_vlm(private_outputs[0], "--model", "claude", context=UNPROCESSED_CONTEXT,
                     vlm_config=stage2_toml(tmp_path / "stage2.toml", "false"))
    assert run.returncode == 1 and "refused" in run.stderr and UNPROCESSED_CONTEXT not in run.stdout + run.stderr


@pytest.mark.parametrize("args, context, says", [
    ([], f"synthetic_cut_in/../{PRIVATE_CONTEXT}", "bad arguments"),  # a path would pass for synthetic
    (["--contxt", PRIVATE_CONTEXT], PRIVATE_CONTEXT, "bad arguments"),  # argparse would echo the argument
    ([], UNPROCESSED_CONTEXT, "failed (FileNotFoundError)"),  # the exception's message holds the path
    (["--split", "validation"], PRIVATE_CONTEXT, "bad arguments"),  # segment.json says training
], ids=["path as context", "misspelled flag", "no outputs", "wrong split"])
def test_no_cli_failure_prints_a_context_name(private_outputs, args, context, says):
    run = agatha_vlm(private_outputs[0], "--model", "dry-run", *args, context=context)
    assert run.returncode != 0 and says in run.stderr and run.stdout == ""
    assert PRIVATE_CONTEXT not in run.stderr and UNPROCESSED_CONTEXT not in run.stderr


# --- gaps that planted bugs found: the caps, the retry, duplicates and the hypothesis ---


def _images(request) -> list[bytes]:
    return [block["data"] for block in request["content"] if block["type"] == "image"]


def _budget_case(segment, config, vlm, share):
    """Pass 2 on the cut-in, UNCERTAIN first, with max_request_mb at `share` of its 5 FPS request's images."""
    alias = cut_in_alias(segment)
    hypothesis = candidate(start_s=8.0, end_s=12.0, object_aliases=[alias])
    probe = FakeModel(scripted(alias), model_id="probe")  # another model, so its answer isn't reused below
    Run(segment, config, vlm, probe).pass2(hypothesis)
    size_mb = sum(map(len, _images(probe.requests[0]))) * 4 / 3e6
    model = FakeModel(scripted(alias, statuses=["UNCERTAIN", "CONFIRMED"]), model_id=f"budget-{share}")
    run = Run(segment, config, replace(vlm, max_request_mb=share * size_mb), model)
    attempts, _, _ = run.pass2(hypothesis)
    return attempts, run.counts, [len(_images(request)) for request in model.requests]


def test_a_retry_the_budget_holds_to_the_same_frames_is_not_sent(segment, config, vlm):
    """Half the budget holds both requests to 15 frames, so a retry would show the model nothing new."""
    attempts, counts, sent = _budget_case(segment, config, vlm, 0.5)
    assert [attempt["outcome"] for attempt in attempts] == ["UNCERTAIN"] and attempts[0]["fps"] == 2.5
    assert sent == [15] and counts["pass2_retried"] == 0


def test_a_retry_over_the_budget_is_rebuilt_from_its_own_rate(segment, config, vlm):
    """Rebuilt from the configured 5 FPS instead, a 10 FPS retry came out under the first request's rate."""
    attempts, counts, sent = _budget_case(segment, config, vlm, 1.5)
    assert [attempt["outcome"] for attempt in attempts] == ["UNCERTAIN", "CONFIRMED"]
    assert attempts[0]["fps"] == vlm.fps < attempts[1]["fps"] < vlm.retry_fps and sent[1] > sent[0] == 30


def test_the_cap_counts_the_margins_in_the_span(segment, config, vlm):
    """2 to 18 s has a 1 to 19 s window: dividing by the candidate's own 16 s sent 44 images for a cap of 40."""
    alias = cut_in_alias(segment)
    model = FakeModel(scripted(alias), model_id="cap-probe")  # its own id, so no stored answer serves it
    Run(segment, config, replace(vlm, max_images=40), model).pass2(
        candidate(start_s=2.0, end_s=18.0, object_aliases=[alias]))
    assert 38 <= len(_images(model.requests[0])) <= 40


def test_a_run_with_one_report_marks_none_duplicate(segment, config, vlm):
    counts = run_segment(segment, config, vlm, FakeModel(scripted(cut_in_alias(segment)), model_id="one-report"))
    assert (counts["reports"], counts["duplicates"]) == (1, 0)


def test_an_uncertain_report_can_be_repeated():
    """An UNCERTAIN label is kept, not rejected, so a later report of it is a duplicate too."""
    earlier = report_of("INC-001", 14.0, 20.0).model_copy(update={"verification_status": "UNCERTAIN"})
    later = report_of("INC-002", 17.0, 20.0)
    mark_duplicates([earlier, later])
    assert later.duplicate_of == "INC-001"


def test_a_split_stays_apart_in_window_order():
    """In Pass 1's own order, W1-0 then W2-0 and W2-1: checking only a group's first member rejoined the split."""
    both = original("W1-0", 8.0, 9.0, labels=("SUDDEN_CUT_IN", "HARD_BRAKING"))
    split = [original("W2-0", 8.0, 9.0), original("W2-1", 8.5, 10.0, labels=("HARD_BRAKING",))]
    assert [sources for sources, _ in merge_candidates([both, *split])] == [["W1-0", "W2-0"], ["W2-1"]]


def test_pass2s_hypothesis_is_the_whole_candidate(segment, config, vlm):
    """Pass 1's evidence and driving impact go to Pass 2 too, not only its labels."""
    alias = cut_in_alias(segment)
    hypothesis = candidate(start_s=8.0, end_s=12.0, object_aliases=[alias])
    model = FakeModel(scripted(alias), model_id="hypothesis-probe")
    Run(segment, config, vlm, model).pass2(hypothesis)
    assert hypothesis_of(model.requests[0]["content"]) == hypothesis.model_dump()


def test_a_capped_rate_reaches_the_json_at_two_decimals(three_cameras, vlm):
    """Unrounded, 1.6166666666666667 went out, and the extra digits change the request key."""
    config, segment = three_cameras
    whole = candidate(start_s=0.0, end_s=segment.meta["last_t_s"], object_aliases=[cut_in_alias(segment)])
    model = FakeModel(scripted(cut_in_alias(segment)), model_id="fps-probe")
    Run(segment, config, vlm, model).pass2(whole)
    assert window_json(model.requests[0]["content"])["window"]["fps"] == 1.62
