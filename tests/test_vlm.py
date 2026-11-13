import base64
import importlib.util
import inspect
import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import threading
import time
import typing
import urllib.error
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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
from agatha.vlm.model import FALLBACK_BETA, IMAGE_TOKENS, ClaudeModel, FakeModel, LocalModel, ModelFailure, NotSent
from agatha.vlm.prompts import PASS1_SYSTEM, PASS2_SYSTEM, PROMPT_VERSION, request_content, window_context
from agatha.vlm.run import (
    PIPELINE_VERSION, GuardRefused, Run, check_guard, incident_report, load_vlm_config, mark_duplicates,
    merge_candidates, request_key, run_segment,
)
from agatha.vlm.schema import (
    CATEGORIES, PASS1_SCHEMA, PASS2_SCHEMA, TAXONOMY, Candidate, IncidentReport, LabelAssessment, Pass1Answer,
    Pass2Answer, check_verdict, fit_candidate, fit_verdict,
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
    # The window is [7, 13) at 5 FPS, so FRONT's frames run from 7.0 to 12.8 s. One between two is fitted first.
    ({"source_frames": [{"camera": "FRONT", "t_s": 12.9}]}, "FRONT at 12.9 s is not a frame in the window"),
    ({"source_frames": [{"camera": "FRONT", "t_s": 6.9}]}, "FRONT at 6.9 s is not a frame in the window"),
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
    fitted, _ = fit_verdict(answer, window)  # as Pass 2 does: each fault here outlasts fitting
    assert any(error in problem for problem in check_verdict(fitted, window, hypothesis))


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


# --- fitting Pass 2's cited frames ---


def cited(*frames) -> dict:
    """A verdict's change: these (camera, t_s) as its source frames."""
    return {"source_frames": [{"camera": camera, "t_s": t_s} for camera, t_s in frames]}


def frames_cited(answer: Pass2Answer) -> list[tuple[str, float]]:
    return [(frame.camera, frame.t_s) for frame in answer.source_frames]


def test_a_frame_cited_between_two_shown_moves_to_the_nearer(segment, config):
    """As Gemma's FRONT at 12.6 s, between frames at 12.5 and 12.7, on 2026-11-04: the track JSON runs at 10 Hz."""
    window, hypothesis, answer = pass2_case(segment, config, cited(("FRONT", 7.13), ("FRONT", 12.67)))
    fitted, changes = fit_verdict(answer, window)
    assert frames_cited(fitted) == [("FRONT", 7.2), ("FRONT", 12.6)]
    assert changes == ["FRONT at 7.13 s: cited as 7.2 s, the nearest frame shown",
                       "FRONT at 12.67 s: cited as 12.6 s, the nearest frame shown"]
    assert len(check_verdict(answer, window, hypothesis)) == 2 and check_verdict(fitted, window, hypothesis) == []
    assert frames_cited(answer) == [("FRONT", 7.13), ("FRONT", 12.67)]  # the answer as given is left alone
    micros = {round(frame.t_s, 2): frame.frame_timestamp_micros for frame in window.frames}
    report = incident_report(fitted, window, "fake-model", "INC-001")  # the report cites the frames shown
    assert [(frame.timestamp_s, frame.frame_timestamp_micros) for frame in report.source_frames] == [
        (7.2, micros[7.2]), (12.6, micros[12.6])]


@pytest.mark.parametrize("name", ["ego", "EGO", "Ego"])
def test_an_ego_alias_is_dropped_as_the_ego_has_none(segment, config, name):
    """As Gemma's HARD_BRAKING answer for random15_12 at temperature 0, on 2026-11-05: its one fault."""
    window, hypothesis, answer = pass2_case(segment, config, cited(("FRONT", 7.0)))
    real = answer.object_aliases
    named = answer.model_copy(update={"object_aliases": [name, *real]})
    assert check_verdict(named, window, hypothesis) == [f"alias {name} is not in the window"]
    fitted, changes = fit_verdict(named, window)
    assert fitted.object_aliases == real and changes == [f"dropped alias {name}: the ego has no alias"]
    assert check_verdict(fitted, window, hypothesis) == []
    again, more = fit_verdict(fitted, window)
    assert again is fitted and more == []


def test_other_unknown_aliases_are_still_refused(segment, config):
    window, hypothesis, answer = pass2_case(segment, config, cited(("FRONT", 7.0)))
    named = answer.model_copy(update={"object_aliases": ["ego_vehicle", "V999", *answer.object_aliases]})
    fitted, changes = fit_verdict(named, window)
    assert fitted == named and changes == []
    assert check_verdict(fitted, window, hypothesis) == ["alias ego_vehicle is not in the window",
                                                         "alias V999 is not in the window"]


def test_a_frame_cited_halfway_between_two_takes_the_earlier(segment, config):
    halfway = [round(7.1 + 0.2 * step, 2) for step in range(29)]  # 7.1, 7.3 and on to 12.7: each 10 Hz time not shown
    window, hypothesis, answer = pass2_case(segment, config, cited(*[("FRONT", t_s) for t_s in halfway]))
    fitted, changes = fit_verdict(answer, window)
    assert frames_cited(fitted) == [("FRONT", round(t_s - 0.1, 2)) for t_s in halfway]
    assert changes[0] == "FRONT at 7.1 s: cited as 7.0 s, the nearest frame shown" and len(changes) == 29
    assert check_verdict(fitted, window, hypothesis) == []


@pytest.mark.parametrize("frame, error", [
    (("FRONT", 6.9), "FRONT at 6.9 s is not a frame in the window"),  # before FRONT's first frame, and the window
    (("FRONT", 12.9), "FRONT at 12.9 s is not a frame in the window"),  # after its last, inside the window's [7, 13)
    (("FRONT_LEFT", 7.1), "FRONT_LEFT at 7.1 s is not a frame in the window"),  # a camera the window doesn't show
], ids=["before the first frame", "after the last frame", "another camera"])
def test_a_frame_cited_outside_its_cameras_frames_stays_as_cited_and_is_refused(segment, config, frame, error):
    window, hypothesis, answer = pass2_case(segment, config, cited(("FRONT", 7.0), frame))
    assert fit_verdict(answer, window) == (answer, [])
    assert check_verdict(answer, window, hypothesis) == [error]


def test_a_frame_cited_again_once_moved_is_dropped_keeping_the_order(segment, config):
    # 7.004 s is captioned 7.00, so it's the frame at 7.0 s cited a second time.
    given = [("FRONT", 7.4), ("FRONT", 7.1), ("FRONT", 7.0), ("FRONT", 7.004), ("FRONT", 7.33), ("FRONT", 7.4)]
    window, hypothesis, answer = pass2_case(segment, config, cited(*given))
    fitted, changes = fit_verdict(answer, window)
    assert frames_cited(fitted) == [("FRONT", 7.4), ("FRONT", 7.0)]
    assert changes == ["FRONT at 7.1 s: cited as 7.0 s, the nearest frame shown",
                       "FRONT at 7.0 s: cited already, so dropped", "FRONT at 7.0 s: cited already, so dropped",
                       "FRONT at 7.33 s: cited as 7.4 s, the nearest frame shown",
                       "FRONT at 7.4 s: cited already, so dropped", "FRONT at 7.4 s: cited already, so dropped"]
    assert check_verdict(fitted, window, hypothesis) == []


def test_each_camera_keeps_its_own_frame_at_a_time_the_cameras_share(three_cameras):
    """The three cameras' frames share their times here. A frame is a camera and a time, so none is dropped."""
    config, segment = three_cameras
    given = [("FRONT", 10.0), ("FRONT_LEFT", 10.0), ("FRONT_RIGHT", 10.1)]
    window, hypothesis, answer = pass2_case(segment, config, cited(*given))
    fitted, changes = fit_verdict(answer, window)
    assert frames_cited(fitted) == [("FRONT", 10.0), ("FRONT_LEFT", 10.0), ("FRONT_RIGHT", 10.0)]
    assert changes == ["FRONT_RIGHT at 10.1 s: cited as 10.0 s, the nearest frame shown"]
    assert check_verdict(fitted, window, hypothesis) == []
    _, _, pair = pass2_case(segment, config, cited(*given[:2]))  # FRONT and FRONT_LEFT at 10.0 s: nothing to fit
    assert fit_verdict(pair, window) == (pair, [])


@pytest.mark.parametrize("t_s", [12.7, 12.699999, 12.7000001])
def test_a_time_off_by_float_noise_is_fitted_as_the_time_it_rounds_to(segment, config, t_s):
    """As Gemma's 12.6 s between 12.5 and 12.7: here FRONT's frames are at 12.6 and 12.8 s, and 12.7 s is a tie."""
    window, hypothesis, answer = pass2_case(segment, config, cited(("FRONT", t_s)))
    assert check_verdict(answer, window, hypothesis) == [f"FRONT at {t_s} s is not a frame in the window"]
    fitted, changes = fit_verdict(answer, window)
    assert frames_cited(fitted) == [("FRONT", 12.6)]  # to the earlier
    assert changes == [f"FRONT at {t_s} s: cited as 12.6 s, the nearest frame shown"]
    assert check_verdict(fitted, window, hypothesis) == []


def test_fitting_a_fitted_answer_changes_nothing(segment, config):
    given = [("FRONT", 7.13), ("FRONT", 7.1), ("FRONT", 6.9), ("FRONT_LEFT", 8.0), ("FRONT", 7.0)]
    window, _, answer = pass2_case(segment, config, cited(*given))
    fitted, changes = fit_verdict(answer, window)  # two moved, one dropped, and two left for check_verdict
    assert frames_cited(fitted) == [("FRONT", 7.2), ("FRONT", 7.0), ("FRONT", 6.9), ("FRONT_LEFT", 8.0)]
    assert len(changes) == 3
    again, more = fit_verdict(fitted, window)
    assert again is fitted and more == []


def test_an_answer_with_nothing_to_fit_comes_back_unchanged(segment, config):
    window, hypothesis, answer = pass2_case(segment, config, cited(("FRONT", 7.0), ("FRONT", 12.8), ("FRONT", 10.004)))
    fitted, changes = fit_verdict(answer, window)  # 10.004 s is captioned 10.0 s: a frame shown
    assert fitted is answer and changes == [] and check_verdict(fitted, window, hypothesis) == []
    rejected = {"status": "REJECTED", "label_assessments": [assessment("REJECTED")], "source_frames": []}
    _, _, none_cited = pass2_case(segment, config, rejected)
    assert fit_verdict(none_cited, window) == (none_cited, [])


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
    assert attempt["changes"] == []  # it cited a frame shown, so fitting changed nothing
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
    assert attempt["changes"] == []  # every attempt has its changes, empty with no answer to fit


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


BETWEEN = "FRONT at 10.13 s: cited as 10.2 s, the nearest frame shown"  # 10.0 s is the [7, 13) window's middle frame
RETRY_BETWEEN = "FRONT at 10.13 s: cited as 10.1 s, the nearest frame shown"  # a frame at 10 FPS, not at 5


def between_frames(alias: str, statuses=("CONFIRMED",)):
    """A fake model whose Pass 2 cites a time between two frames shown, as Gemma did on 2026-11-04, with these
    statuses in turn."""
    statuses = list(statuses)

    def answer(system, content, schema):
        if schema is PASS2_SCHEMA:
            camera, t_s = captions(content)[len(captions(content)) // 2]
            return verdict(hypothesis_of(content), statuses.pop(0), frame=(camera, round(t_s + 0.13, 2)))
        return scripted(alias)(system, content, schema)
    return answer


def test_a_verdict_citing_a_time_between_frames_is_fitted_and_reported(segment, config, vlm):
    counts = run_segment(segment, config, vlm, FakeModel(between_frames(cut_in_alias(segment))))
    candidates, [report] = results(segment)
    [attempt] = candidates["merged"][0]["attempts"]
    assert (attempt["outcome"], attempt["changes"], counts["pass2_invalid"]) == ("CONFIRMED", [BETWEEN], 0)
    assert [(frame["camera"], frame["timestamp_s"]) for frame in report["source_frames"]] == [("FRONT", 10.2)]
    record = json.loads((segment.directory / "stage2" / "responses" / f"{attempt['response']}.json").read_text())
    assert json.loads(record["text"])["source_frames"] == [{"camera": "FRONT", "t_s": 10.13}]  # stored as answered
    assert stored(segment, "errors") == []


def test_a_stored_verdict_the_old_checks_refused_is_accepted_on_a_rerun_with_nothing_sent(segment, config, vlm,
                                                                                         monkeypatch):
    """As the 3 answers Gemma's run on 2026-11-04 stored: a rerun fits them from the store and asks nothing."""
    alias = cut_in_alias(segment)
    with monkeypatch.context() as patch:  # the checks as they were then, with no fitting
        patch.setattr("agatha.vlm.run.fit_verdict", lambda answer, window: (answer, []))
        first = run_segment(segment, config, vlm, FakeModel(between_frames(alias)))
    assert first["pass2_invalid"] == 1 and results(segment)[1] == []
    assert stored(segment, "errors") == [{"step": "pass2", "errors": ["FRONT at 10.13 s is not a frame in the window"]}]
    again = FakeModel(scripted(alias))
    counts = run_segment(segment, config, vlm, again)
    assert again.requests == [] and (counts["pass2_reused"], counts["pass2_invalid"], counts["CONFIRMED"]) == (1, 0, 1)
    candidates, [report] = results(segment)
    assert candidates["merged"][0]["attempts"][0]["changes"] == [BETWEEN]
    assert [(frame["camera"], frame["timestamp_s"]) for frame in report["source_frames"]] == [("FRONT", 10.2)]
    assert stored(segment, "errors") == []  # the old checks' error, removed


def test_a_retrys_cited_frames_are_fitted_to_its_own_frames(segment, config, vlm):
    """The 10 FPS retry shows a frame at 10.1 s, which the first request didn't, and its attempt says what moved."""
    model = FakeModel(between_frames(cut_in_alias(segment), ["UNCERTAIN", "CONFIRMED"]))
    counts = run_segment(segment, config, vlm, model)
    candidates, [report] = results(segment)
    attempts = candidates["merged"][0]["attempts"]
    assert [(attempt["fps"], attempt["outcome"], attempt["changes"]) for attempt in attempts] == [
        (5.0, "UNCERTAIN", [BETWEEN]), (10.0, "CONFIRMED", [RETRY_BETWEEN])]
    assert [(frame["camera"], frame["timestamp_s"]) for frame in report["source_frames"]] == [("FRONT", 10.1)]
    assert (counts["pass2_retried"], counts["pass2_invalid"]) == (1, 0)


def test_a_stored_uncertain_verdict_the_old_checks_refused_gets_its_retry_on_a_rerun(segment, config, vlm,
                                                                                    monkeypatch):
    """Refused, it had no retry, since only an UNCERTAIN outcome gets one. Fitted on a rerun, it's UNCERTAIN, so the
    rerun sends its 10 FPS retry: the one request it sends."""
    alias = cut_in_alias(segment)
    with monkeypatch.context() as patch:  # the checks as they were before fitting
        patch.setattr("agatha.vlm.run.fit_verdict", lambda answer, window: (answer, []))
        first = run_segment(segment, config, vlm, FakeModel(between_frames(alias, ["UNCERTAIN"])))
    assert (first["pass2_invalid"], first["pass2_retried"]) == (1, 0)
    again = FakeModel(between_frames(alias, ["CONFIRMED"]))
    counts = run_segment(segment, config, vlm, again)
    [request] = again.requests
    assert request["schema"] is PASS2_SCHEMA and window_json(request["content"])["window"]["fps"] == 10.0
    assert (counts["pass2_reused"], counts["pass2_sent"], counts["pass2_retried"]) == (1, 1, 1)
    attempts = results(segment)[0]["merged"][0]["attempts"]
    assert [(attempt["fps"], attempt["outcome"], attempt["changes"]) for attempt in attempts] == [
        (5.0, "UNCERTAIN", [BETWEEN]), (10.0, "CONFIRMED", [RETRY_BETWEEN])]


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
    assert candidates["merged"][0]["attempts"][0]["changes"] == []  # no answer, so nothing fitted
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


# --- a local model: Gemma 4 behind vLLM's OpenAI-compatible server ---

GEMMA = "google/gemma-4-31B-it"
SHIPPED = "RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic"  # configs/stage2.toml's [local] id, an FP8 checkpoint
# configs/stage2.toml's [local], but for its id. Its temperature is 0 since 2026-11-05.
LOCAL = {"id": GEMMA, "base_url": "http://127.0.0.1:8000/v1", "max_tokens": 8192, "image_tokens": 280,
         "temperature": 0.0}
# What keys a local answer: [local] but its base_url. With no temperature, as before 2026-11-05, and gemma() sends
# none, so the tests built on these also pin that a config with none asks what it did then.
LOCAL_SETTINGS = {"id": GEMMA, "max_tokens": 8192, "image_tokens": 280}
GREEDY_SETTINGS = {**LOCAL_SETTINGS, "temperature": 0.0}
TEMPERATURE = "temperature = 0.0\n"  # [local]'s line in configs/stage2.toml
NOWHERE = "http://127.0.0.1:9/v1"  # nothing listens there


def completion(text='{"candidates": []}', finish="stop", model=GEMMA, output_tokens=56) -> dict:
    """A chat completion as vLLM returns one."""
    return {"id": "chatcmpl-1", "object": "chat.completion", "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish}],
            "usage": {"prompt_tokens": 1234, "completion_tokens": output_tokens, "total_tokens": 1234 + output_tokens}}


@pytest.fixture
def server():
    """serve(answer) -> (base_url, requests): a fake vLLM on 127.0.0.1. answer(body) gives JSON, or (status, JSON).

    JSON given as bytes is sent as it is. As vLLM does, it turns away a body that doesn't say it's JSON.
    """
    started = []

    def serve(answer):
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                requests.append({"path": self.path, "body": body})
                if self.headers["Content-Type"] == "application/json":
                    reply = answer(body)
                else:  # urllib's default is a form, which vLLM's validate_json_request answers so
                    reply = (400, {"error": {"message": "Unsupported Media Type: Only 'application/json' is allowed"}})
                status, reply = reply if isinstance(reply, tuple) else (200, reply)
                data = reply if isinstance(reply, bytes) else json.dumps(reply).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        started.append(httpd)
        return f"http://127.0.0.1:{httpd.server_port}/v1", requests
    yield serve
    for httpd in started:
        httpd.shutdown()
        httpd.server_close()


def gemma(base_url: str, max_tokens=8192, image_tokens=280, temperature=None) -> LocalModel:
    return LocalModel(GEMMA, base_url, max_tokens, image_tokens, temperature)


def with_temperature(text: str, value: str | None) -> str:
    """A stage2.toml's text with [local]'s temperature set to `value`, as TOML, or with none, as before 2026-11-05."""
    assert text.count(TEMPERATURE) == 1
    return text.replace(TEMPERATURE, "" if value is None else f"temperature = {value}\n")


def as_vllm(answers):
    """A fake model's answers, given vLLM's request body: the pass by its schema's title, images as plain blocks."""
    def answer(body):
        system, user = body["messages"]
        content = [{"type": "image"} if block["type"] == "image_url" else block for block in user["content"]]
        schema = {"Pass1Answer": PASS1_SCHEMA, "Pass2Answer": PASS2_SCHEMA}[
            body["response_format"]["json_schema"]["schema"]["title"]]
        reply = answers(system["content"], content, schema)
        return (400, {"error": {"message": str(reply)}}) if isinstance(reply, Exception) else completion(reply)
    return answer


def test_a_local_request_has_the_system_message_text_images_schema_and_image_budget(server):
    url, requests = server(lambda body: completion())
    gemma(url).respond("the system prompt", CONTENT, PASS1_SCHEMA)
    [request] = requests
    body = request["body"]
    assert request["path"] == "/v1/chat/completions"
    assert set(body) == {"model", "max_completion_tokens", "messages", "response_format", "mm_processor_kwargs"}
    assert (body["model"], body["max_completion_tokens"]) == (GEMMA, 8192)
    system, user = body["messages"]  # one user turn after the system message, and no assistant prefill
    assert system == {"role": "system", "content": "the system prompt"} and user["role"] == "user"
    data = base64.standard_b64encode(b"\xff\xd8jpeg").decode()
    image = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}}
    assert user["content"] == [CONTENT[0], image, CONTENT[2]]
    assert body["response_format"] == {"type": "json_schema", "json_schema": {"name": "answer", "schema": PASS1_SCHEMA}}
    assert body["mm_processor_kwargs"] == {"max_soft_tokens": 280}
    gemma(url + "/", 4096, 560).respond("system", CONTENT, PASS2_SCHEMA)  # the settings come from stage2.toml
    other = requests[1]
    assert other["path"] == "/v1/chat/completions" and other["body"]["max_completion_tokens"] == 4096
    assert other["body"]["mm_processor_kwargs"] == {"max_soft_tokens": 560}
    assert other["body"]["response_format"]["json_schema"]["schema"] == PASS2_SCHEMA


@pytest.mark.parametrize("temperature", [0.0, 0.7])
def test_a_local_request_sends_a_temperature_only_when_one_is_set(server, temperature):
    """With none, LocalModel's default, the body is what it was before 2026-11-05, and vLLM samples with the model's
    own settings."""
    url, requests = server(lambda body: completion())
    LocalModel(GEMMA, url, 8192, 280).respond("system", CONTENT, PASS1_SCHEMA)
    gemma(url, temperature=temperature).respond("system", CONTENT, PASS1_SCHEMA)
    unset, given = (request["body"] for request in requests)
    assert "temperature" not in unset and given == {**unset, "temperature": temperature}


def test_a_local_reply_is_the_answer_the_model_that_wrote_it_and_its_tokens(server):
    url, _ = server(lambda body: completion('{"candidates": []}', model="gemma-as-served"))
    reply = gemma(url).respond("system", CONTENT, PASS1_SCHEMA)
    assert reply == ('{"candidates": []}', "gemma-as-served", {"input_tokens": 1234, "output_tokens": 56})


@pytest.mark.parametrize("reply, reason, retry", [
    # Gemma samples, so a resend usually finishes: 5 of 6 did on 2026-11-04.
    (completion('{"candidates": [{"lab', finish="length", output_tokens=8192), "max_tokens", True),
    # vLLM 0.32 ends an answer at the end of the context too: a server with a longer one may fit it.
    (completion('{"candidates": [{"lab', finish="length", output_tokens=100), "max_tokens", True),
    (completion('{"candidates": []}', finish="abort"), "abort", False),
    (completion(None), "no_text", False),
    (completion(""), "no_text", False),
    ((400, {"error": {"message": "longer than the maximum model length"}}), "http_400", True),
    ((413, {"error": {"message": "too large"}}), "http_413", True),
    ((422, {"error": {"message": "unprocessable"}}), "http_422", True),
], ids=["cut off", "context full", "aborted", "no text", "empty text", "too long", "too large", "unprocessable"])
def test_a_local_response_with_no_answer_is_a_failure(server, reply, reason, retry):
    url, _ = server(lambda body: reply)
    with pytest.raises(ModelFailure) as caught:
        gemma(url).respond("system", CONTENT, PASS1_SCHEMA)
    details = caught.value.details
    assert (caught.value.reason, details.get("retry", False)) == (reason, retry)
    if reason.startswith("http_"):  # the server's reason, to see why on PACE
        assert json.loads(details["message"]) == reply[1]
    else:  # billed all the same
        usage = {"input_tokens": 1234, "output_tokens": reply["usage"]["completion_tokens"]}
        assert details["served_by"] == GEMMA and details["usage"] == usage
        full = reason == "max_tokens" and usage["output_tokens"] == 8192  # only that one says how it ended
        assert set(details) - {"retry"} == {"served_by", "usage", *(["whitespace_share", "tail"] if full else [])}


RUN = ("\n  " * 14)[:40]  # 40 characters of blank lines, which a cut-off on 2026-11-04 repeated 560 times


@pytest.mark.parametrize("text, share, tail", [
    ('{"label_assessments": [' + RUN * 560, 0.999, RUN[20:] + RUN * 7),  # 22,401 of 22,423 characters
    # JSON allows tabs and carriage returns between tokens too: 301 of 323 characters.
    ('{"label_assessments": [' + "\r\n\t" * 100, 0.932, "\r\n\t" * 100),
    ('{"candidates": [{"labels": ["SUDDEN_CUT_IN"], "evidence": ["the car', 0.075,
     '{"candidates": [{"labels": ["SUDDEN_CUT_IN"], "evidence": ["the car'),  # 5 of 67, all of it kept
    (None, 0.0, ""),
], ids=["blank lines", "tabs and returns", "words", "no text"])
def test_a_local_answer_cut_off_at_max_tokens_keeps_its_whitespace_share_and_end(server, text, share, tail):
    """Whether Gemma looped on blank lines between JSON tokens, as on 2026-11-04, or ran on in words."""
    url, _ = server(lambda body: completion(text, finish="length", output_tokens=8192))
    with pytest.raises(ModelFailure) as caught:
        gemma(url).respond("system", CONTENT, PASS2_SCHEMA)
    details = caught.value.details
    assert (caught.value.reason, details["retry"], details["whitespace_share"]) == ("max_tokens", True, share)
    assert details["tail"] == tail and len(tail) <= 300


@pytest.mark.parametrize("temperature", [None, 0.0], ids=["sampled", "greedy"])
@pytest.mark.parametrize("usage", [None, {"prompt_tokens": 1234, "completion_tokens": None}],
                         ids=["no usage", "no output count"])
def test_a_local_cut_off_with_no_token_count_is_still_recorded_and_asked_again(server, usage, temperature):
    """With no count, whether it used all of max_tokens is unknown, so it gets no diagnosis. It's a cut-off still,
    and like a shorter one, it's asked again at temperature 0 too."""
    reply = {**completion('{"candidates": [{"lab', finish="length"), "usage": usage}
    url, _ = server(lambda body: {key: value for key, value in reply.items() if value is not None})
    with pytest.raises(ModelFailure) as caught:  # not a TypeError, which would stop the run with nothing recorded
        gemma(url, temperature=temperature).respond("system", CONTENT, PASS1_SCHEMA)
    details = caught.value.details
    assert (caught.value.reason, details["retry"], details["usage"]["output_tokens"]) == ("max_tokens", True, None)
    assert set(details) == {"retry", "served_by", "usage"}


@pytest.mark.parametrize("temperature, retry", [(None, True), (0.7, True), (0.0, False)],
                         ids=["model's own", "sampled", "greedy"])
@pytest.mark.parametrize("output_tokens", [8192, 100], ids=["full length", "context full"])
def test_a_full_length_cut_off_is_asked_again_only_while_the_model_samples(server, temperature, retry, output_tokens):
    """At temperature 0 a resend gives the same tokens, so it would run away again on every rerun. One cut off
    shorter ended where the context did, which a server with a longer one may fit, so it's asked again at any."""
    full = output_tokens == 8192
    url, _ = server(lambda body: completion('{"label_assessments": [' + RUN * 20, finish="length",
                                            output_tokens=output_tokens))
    with pytest.raises(ModelFailure) as caught:
        gemma(url, temperature=temperature).respond("system", CONTENT, PASS2_SCHEMA)
    details = caught.value.details
    assert (caught.value.reason, details["retry"]) == ("max_tokens", retry or not full)
    assert ("whitespace_share" in details, "tail" in details) == (full, full)  # the diagnosis, at any temperature


@pytest.mark.parametrize("reason, kept", [
    (b"x" * 5000, "x" * 1000),
    (("x" + "é" * 600).encode(), "x" + "é" * 499 + "�"),  # 1,000 bytes end inside a character
], ids=["long", "cut inside a character"])
def test_a_long_reason_from_the_server_is_cut_short(server, reason, kept):
    """A validation error can echo the request, frames and all: only its first 1,000 bytes are kept."""
    url, _ = server(lambda body: (400, reason))
    with pytest.raises(ModelFailure) as caught:
        gemma(url).respond("system", CONTENT, PASS1_SCHEMA)
    assert caught.value.details["message"] == kept


@pytest.mark.parametrize("status", [500, 404], ids=["engine died", "serves another model"])
def test_a_server_error_or_another_model_stops_the_run(server, segment, config, vlm, status):
    """A 404 is the setup's fault, as when serve_vlm.sh started another config's model: every request would meet it."""
    stage2 = segment.directory / "stage2"
    stage2.mkdir()
    for name in ("candidates.json", "incidents.json"):  # another model's results, as Opus's are on PACE
        (stage2 / name).write_text("{}")
    url, requests = server(lambda body: (status, {"error": {"message": "no"}}))
    with pytest.raises(urllib.error.HTTPError):
        run_segment(segment, config, replace(vlm, model=LOCAL_SETTINGS), gemma(url))
    assert len(requests) == 1 and not (stage2 / "responses").exists()  # nothing stored, so a rerun asks again
    assert all((stage2 / name).read_text() == "{}" for name in ("candidates.json", "incidents.json"))


def test_a_local_answer_slower_than_the_timeout_stops_the_run(server):
    url, _ = server(lambda body: time.sleep(1.5) or completion())
    with pytest.raises(TimeoutError):
        LocalModel(GEMMA, url, 8192, 280, timeout_s=0.3).respond("system", CONTENT, PASS1_SCHEMA)
    assert LocalModel(GEMMA, url, 8192, 280).timeout_s == 1800  # half an hour by default, for a whole answer


def test_with_no_server_up_the_run_stops(segment, config, vlm):
    with pytest.raises(urllib.error.URLError) as caught:
        run_segment(segment, config, replace(vlm, model=LOCAL_SETTINGS), gemma(NOWHERE))
    assert not isinstance(caught.value, urllib.error.HTTPError)  # it never answered
    assert not (segment.directory / "stage2" / "responses").exists()


def test_a_local_request_never_goes_through_a_proxy(server):
    """http_proxy names somewhere else, as a cluster's might: the frames still go to this machine's server only."""
    url, requests = server(lambda body: completion())
    code = ("import sys\nfrom agatha.vlm.model import LocalModel\n"
            "reply = LocalModel('m', sys.argv[1], 100, 280).respond('system', [{'type': 'text', 'text': '{}'}], {})\n"
            "print(reply.text)")
    proxied = {**os.environ, "http_proxy": NOWHERE, "HTTP_PROXY": NOWHERE, "all_proxy": NOWHERE, "ALL_PROXY": NOWHERE,
               "no_proxy": "", "NO_PROXY": ""}
    out = subprocess.run([sys.executable, "-c", code, url], capture_output=True, text=True, timeout=60, env=proxied,
                         cwd=PROJECT)
    assert out.stdout == '{"candidates": []}\n', out.stderr
    assert len(requests) == 1


@pytest.mark.parametrize("base_url", ["http://10.0.0.5:8000/v1", "https://api.example.com/v1",
                                      "http://127.0.0.1.example.com:8000/v1", "http://user@example.com/127.0.0.1"])
def test_a_local_model_refuses_a_server_off_this_machine(base_url):
    with pytest.raises(ValueError):
        gemma(base_url)


def test_a_local_model_is_not_remote_and_takes_only_gemmas_image_budgets():
    for base_url in ("http://127.0.0.1:8000/v1", "http://localhost:8000/v1", "http://[::1]:8000/v1"):
        assert gemma(base_url).remote is False
    assert IMAGE_TOKENS == (70, 140, 280, 560, 1120)
    for budget in IMAGE_TOKENS:
        assert gemma(LOCAL["base_url"], image_tokens=budget).image_tokens == budget
    with pytest.raises(ValueError):
        gemma(LOCAL["base_url"], image_tokens=300)


def gemma_config_text() -> str:
    """configs/stage2.toml with [local] naming the 31B, whose BF16 weights the serve tests size by GPU."""
    return STAGE2_CONFIG.read_text().replace(f'id = "{SHIPPED}"', f'id = "{GEMMA}"')


def without_local(path: Path) -> Path:
    """configs/stage2.toml with no [local] section, written to `path`."""
    text = STAGE2_CONFIG.read_text()
    path.write_text(text[:text.index("\n[local]\n") + 1] + text[text.index("\n[data]\n") + 1:])
    return path


def test_stage2_toml_names_gemma_4_in_local_and_leaves_claudes_settings_as_they_were(vlm):
    assert vlm.local == {**LOCAL, "id": SHIPPED}  # at temperature 0, Andrew's choice on 2026-11-05
    assert type(vlm.local["temperature"]) is float
    assert vlm.model == {"id": "claude-opus-5-5", "effort": "medium", "max_tokens": 16000}


def test_a_config_with_no_local_section_still_loads(tmp_path):
    """As the copy for the Sonnet comparison on PACE has none."""
    assert load_vlm_config(without_local(tmp_path / "stage2.toml")).local is None


@pytest.mark.parametrize("value, temperature", [
    (None, None), ("0", 0.0), ("0.0", 0.0), ("-0.0", 0.0), ("0.7", 0.7), ("2", 2.0)])
def test_a_local_temperature_is_optional_and_kept_as_a_float(tmp_path, value, temperature):
    """So 0, 0.0 and -0.0 are one request. With none, [local] is what it was before 2026-11-05."""
    local = load_vlm_config(local_toml(tmp_path / "stage2.toml", LOCAL["base_url"], value)).local
    if temperature is None:
        assert local == {key: value for key, value in LOCAL.items() if key != "temperature"} | {"id": SHIPPED}
    else:  # as the key writes it: -0.0 == 0.0, but json.dumps writes "-0.0", which would key apart
        assert json.dumps(local["temperature"]) == json.dumps(temperature) and type(local["temperature"]) is float


@pytest.mark.parametrize("value, shown", [
    ("-0.5", "-0.5"), ("-1", "-1"), ('"0"', "'0'"), ("true", "True"), ("false", "False"), ("nan", "nan"),
    ("inf", "inf"), ("[0.0]", "[0.0]"),
])  # shown as Python writes it, so "0" in quotes doesn't read as the number 0
def test_a_local_temperature_that_isnt_a_number_0_or_more_is_refused(tmp_path, value, shown):
    with pytest.raises(ValueError) as caught:
        load_vlm_config(local_toml(tmp_path / "stage2.toml", LOCAL["base_url"], value))
    assert str(caught.value) == f"[local] temperature must be a number, 0 or more, not {shown}"


def test_a_temperature_is_part_of_a_local_key_and_none_keys_as_before(monkeypatch):
    """Each temperature is its own request, so the shipped 0 sends every local request again, and answers stored
    without one stay under their own keys. With none, the key is the one computed before temperature was added."""
    monkeypatch.setattr("agatha.vlm.run.PROMPT_VERSION", "pinned")  # so the key below changes with settings only

    def key(settings):
        return request_key(GEMMA, settings, "system", CONTENT, {"type": "object"})
    assert key(LOCAL_SETTINGS) == "6aa4dd017b488620"  # with the code of 2026-11-04
    keys = {key({**LOCAL_SETTINGS, "temperature": temperature}) for temperature in (0.0, 0.7, 1.0)}
    assert len(keys) == 3 and key(LOCAL_SETTINGS) not in keys


def test_the_guard_lets_a_local_model_run_on_a_real_looking_name():
    check_guard(gemma(LOCAL["base_url"]), PRIVATE_CONTEXT, cleared=False)


def test_claude_and_the_local_model_key_the_same_request_apart(segment, config, vlm):
    content = request_content(pass1_window(segment, config), segment.directory)
    claude = request_key(vlm.model["id"], vlm.model, PASS1_SYSTEM, content, PASS1_SCHEMA)
    local = request_key(GEMMA, LOCAL_SETTINGS, PASS1_SYSTEM, content, PASS1_SCHEMA)
    other_budget = request_key(GEMMA, {**LOCAL_SETTINGS, "image_tokens": 560}, PASS1_SYSTEM, content, PASS1_SCHEMA)
    assert len({claude, local, other_budget}) == 3


def test_run_segment_goes_through_the_local_server(segment, config, vlm, server):
    url, requests = server(as_vllm(scripted(cut_in_alias(segment))))
    counts = run_segment(segment, config, replace(vlm, model=LOCAL_SETTINGS), gemma(url))
    [report] = results(segment)[1]
    assert (counts["pass1_sent"], counts["pass2_sent"], counts["reports"]) == (5, 1, 1) and len(requests) == 6
    assert report["model_id"] == GEMMA and report["verification_status"] == "CONFIRMED"
    records = stored(segment, "responses")
    assert {(record["model_id"], record["served_by"]) for record in records} == {(GEMMA, GEMMA)}
    assert all(record["settings"] == LOCAL_SETTINGS for record in records)
    assert all(record["usage"] == {"input_tokens": 1234, "output_tokens": 56} for record in records)
    first = requests[0]["body"]["messages"][1]["content"]  # each frame as the bytes Stage 1 drew
    sent = [base64.standard_b64decode(block["image_url"]["url"].removeprefix("data:image/jpeg;base64,"))
            for block in first if block["type"] == "image_url"]
    frames = default_windows(segment, config)[0].frames
    drawn = [(segment.directory / frame.annotated_ref).read_bytes() for frame in frames]
    assert sent == drawn and len(sent) > 1
    again = run_segment(segment, config, replace(vlm, model=LOCAL_SETTINGS), gemma(NOWHERE))  # all stored
    assert (again["pass1_reused"], again["pass2_reused"]) == (5, 1)


def test_a_local_answer_turned_away_is_recorded_and_asked_again_on_a_rerun(segment, config, vlm, server):
    answers = scripted(cut_in_alias(segment))

    def too_long(system, content, schema):
        return ModelFailure("too long") if schema is PASS2_SCHEMA else answers(system, content, schema)
    url, _ = server(as_vllm(too_long))
    counts = run_segment(segment, config, replace(vlm, model=LOCAL_SETTINGS), gemma(url))
    assert counts["pass2_failed"] == 1 and results(segment)[1] == []
    [failure] = [record for record in stored(segment, "responses") if "failure" in record]
    assert failure["failure"] == "http_400" and failure["retry"] is True
    url, requests = server(as_vllm(scripted(cut_in_alias(segment))))
    again = run_segment(segment, config, replace(vlm, model=LOCAL_SETTINGS), gemma(url))
    assert (again["pass1_reused"], again["pass2_sent"], len(requests)) == (5, 1, 1)


def cut_off_run(segment, config, vlm, server, temperature=None) -> tuple[Path, dict]:
    """Stage 2 on the local server, whose Pass 2 answer runs to max_tokens: (its stored record's path, the record)."""
    settings = LOCAL_SETTINGS if temperature is None else {**LOCAL_SETTINGS, "temperature": temperature}
    answers = as_vllm(scripted(cut_in_alias(segment)))

    def runaway(body):
        if body["response_format"]["json_schema"]["schema"]["title"] == "Pass2Answer":
            return completion("{\n  \n  ", finish="length", output_tokens=8192)
        return answers(body)
    url, _ = server(runaway)
    model = gemma(url, temperature=temperature)
    assert run_segment(segment, config, replace(vlm, model=settings), model)["pass2_failed"] == 1
    [path] = [path for path in (segment.directory / "stage2" / "responses").glob("*.json")
              if "failure" in json.loads(path.read_text())]
    return path, json.loads(path.read_text())


def test_a_local_answer_cut_off_is_asked_again_on_a_rerun(segment, config, vlm, server):
    path, record = cut_off_run(segment, config, vlm, server)
    assert {key: record[key] for key in ("failure", "retry", "whitespace_share", "tail")} == {
        "failure": "max_tokens", "retry": True, "whitespace_share": 0.857, "tail": "{\n  \n  "}
    url, requests = server(as_vllm(scripted(cut_in_alias(segment))))  # this time it finishes
    again = run_segment(segment, config, replace(vlm, model=LOCAL_SETTINGS), gemma(url))
    assert (again["pass1_reused"], again["pass2_sent"], again["CONFIRMED"], len(requests)) == (5, 1, 1, 1)
    assert "failure" not in json.loads(path.read_text()) and len(results(segment)[1]) == 1


def test_a_greedy_cut_off_is_stored_and_not_asked_again(segment, config, vlm, server):
    """At temperature 0 a resend would run away again, so a rerun reuses it: its diagnosis says what happened."""
    path, record = cut_off_run(segment, config, vlm, server, temperature=0.0)
    assert {key: record[key] for key in ("failure", "retry", "whitespace_share", "settings")} == {
        "failure": "max_tokens", "retry": False, "whitespace_share": 0.857, "settings": GREEDY_SETTINGS}
    again = run_segment(segment, config, replace(vlm, model=GREEDY_SETTINGS), gemma(NOWHERE, temperature=0.0))
    assert (again["pass1_reused"], again["pass2_reused"], again["pass2_failed"]) == (5, 1, 1)
    assert json.loads(path.read_text()) == record and results(segment)[1] == []


def test_a_stored_failure_not_marked_retry_is_reused_not_asked_again(segment, config, vlm, server):
    """As a cut-off stored before 2026-11-04's change, which said retry false: delete it, or mark it, to ask again."""
    path, record = cut_off_run(segment, config, vlm, server)
    path.write_text(json.dumps({**record, "retry": False}))
    again = run_segment(segment, config, replace(vlm, model=LOCAL_SETTINGS), gemma(NOWHERE))  # asks nothing
    assert (again["pass1_reused"], again["pass2_reused"], again["pass2_failed"]) == (5, 1, 1)
    assert json.loads(path.read_text())["retry"] is False and results(segment)[1] == []


# The CLI with --model local, on a segment with a real-looking name.


@pytest.fixture
def private_copy(private_outputs, tmp_path):
    """(stage1.toml, directory): a copy of the private-named outputs with no stage2/, and a config reading it."""
    config_path, directory = private_outputs
    shutil.copytree(directory, tmp_path / "outputs" / PRIVATE_CONTEXT, ignore=shutil.ignore_patterns("stage2"))
    config = tmp_path / "stage1.toml"
    config.write_text(config_path.read_text().replace(str(directory.parent), str(tmp_path / "outputs")))
    return config, tmp_path / "outputs" / PRIVATE_CONTEXT


def local_toml(path: Path, base_url: str, temperature: str | None = "0.0") -> Path:
    """configs/stage2.toml with [local] at `base_url` and `temperature` (see with_temperature), and real data not
    cleared: a local model needs no clearing."""
    text = stage2_toml(path, "false").read_text()
    text = text.replace(f'base_url = "{LOCAL["base_url"]}"', f'base_url = "{base_url}"')
    path.write_text(with_temperature(text, temperature))
    return path


def test_the_cli_offers_local_claude_and_dry_run(private_outputs):
    run = agatha_vlm(private_outputs[0], "--help")
    assert run.returncode == 0 and "{local,claude,dry-run}" in run.stdout


def test_the_cli_runs_the_local_model_on_a_real_looking_name_and_prints_only_counts(private_copy, server, tmp_path):
    config, directory = private_copy
    url, requests = server(as_vllm(found_twice(load_segment(directory))))
    run = agatha_vlm(config, "--model", "local", vlm_config=local_toml(tmp_path / "stage2.toml", url))
    assert run.returncode == 0 and run.stderr == "" and PRIVATE_CONTEXT not in run.stdout
    label = context_label(PRIVATE_CONTEXT)
    assert run.stdout.splitlines() == [
        f"{label}: pass 1: 5 windows: 5 sent, 0 reused, 0 not sent, 0 no frames, 0 failed, 0 invalid; 3 candidates, "
        "0 failing checks, 2 after merging",
        f"{label}: pass 2: 2 sent, 0 reused, 0 not sent, 0 no frames, 0 failed, 0 invalid, 0 retried; 2 confirmed, "
        "0 rejected, 0 uncertain; incident reports: 2 written, 1 marked duplicate"]
    assert len(requests) == 7 and all(PRIVATE_CONTEXT not in json.dumps(request["body"]) for request in requests)
    assert {(request["body"]["max_completion_tokens"], request["body"]["mm_processor_kwargs"]["max_soft_tokens"],
             request["body"]["temperature"]) for request in requests} == {(8192, 280, 0.0)}  # [local]'s, not [model]'s
    records = [json.loads(path.read_text()) for path in (directory / "stage2" / "responses").glob("*.json")]
    assert {record["model_id"] for record in records} == {SHIPPED}
    assert all(record["settings"] == {**GREEDY_SETTINGS, "id": SHIPPED} for record in records)  # but its base_url


def test_a_temperature_is_a_new_request_and_a_config_with_none_asks_as_before(private_copy, server, tmp_path):
    """[local] with no temperature, as before 2026-11-05, sends none and keys its answers on the same settings, so
    they serve it. The shipped 0 is a new request, so all are sent again, and the sampled answers stay as they were."""
    config, directory = private_copy
    url, requests = server(as_vllm(found_twice(load_segment(directory))))
    sampled = local_toml(tmp_path / "sampled.toml", url, temperature=None)
    run = agatha_vlm(config, "--model", "local", vlm_config=sampled)
    assert run.returncode == 0 and "5 windows: 5 sent, 0 reused" in run.stdout
    assert "pass 2: 2 sent, 0 reused" in run.stdout
    assert len(requests) == 7 and all("temperature" not in request["body"] for request in requests)
    responses = directory / "stage2" / "responses"
    before = {path.name: path.read_bytes() for path in responses.glob("*.json")}
    assert all(json.loads(data)["settings"] == {**LOCAL_SETTINGS, "id": SHIPPED} for data in before.values())
    greedy = agatha_vlm(config, "--model", "local", vlm_config=local_toml(tmp_path / "greedy.toml", url))
    assert greedy.returncode == 0 and "5 windows: 5 sent, 0 reused" in greedy.stdout
    assert len(requests) == 14 and {request["body"]["temperature"] for request in requests[7:]} == {0.0}
    after = {path.name: path.read_bytes() for path in responses.glob("*.json")}
    assert len(after) == 2 * len(before) and {name: after[name] for name in before} == before
    again = agatha_vlm(config, "--model", "local", vlm_config=sampled)  # the sampled answers serve still
    assert again.returncode == 0 and "0 sent, 5 reused" in again.stdout and len(requests) == 14


def test_the_cli_refuses_a_negative_temperature_and_says_why(private_copy, server, tmp_path):
    config, directory = private_copy
    url, requests = server(lambda body: completion())
    run = agatha_vlm(config, "--model", "local", vlm_config=local_toml(tmp_path / "stage2.toml", url, "-0.5"))
    assert (run.returncode, run.stdout, requests) == (2, "", [])
    assert run.stderr == "agatha-vlm: --vlm-config: [local] temperature must be a number, 0 or more, not -0.5\n"
    assert not (directory / "stage2").exists()


def test_the_cli_refuses_a_stage2_toml_it_cant_read_and_says_where(private_copy, tmp_path):
    """tomllib's message gives a line and column, not the file's text, so it names no segment."""
    config, directory = private_copy
    broken = tmp_path / "stage2.toml"
    broken.write_text("[local\n")
    run = agatha_vlm(config, "--model", "local", vlm_config=broken)
    assert (run.returncode, run.stdout) == (2, "") and run.stderr.startswith("agatha-vlm: --vlm-config: ")
    assert "line 1" in run.stderr and PRIVATE_CONTEXT not in run.stderr
    assert not (directory / "stage2").exists()


def test_where_the_local_server_runs_is_not_part_of_a_key_but_the_provider_is(private_copy, server, tmp_path):
    config, directory = private_copy
    url, _ = server(as_vllm(found_twice(load_segment(directory))))
    assert agatha_vlm(config, "--model", "local", vlm_config=local_toml(tmp_path / "a.toml", url)).returncode == 0
    elsewhere, requests = server(as_vllm(found_twice(load_segment(directory))))
    run = agatha_vlm(config, "--model", "local", vlm_config=local_toml(tmp_path / "b.toml", elsewhere))
    assert run.returncode == 0 and "0 sent, 5 reused" in run.stdout and requests == []
    claude = agatha_vlm(config, "--model", "dry-run", vlm_config=local_toml(tmp_path / "c.toml", elsewhere))
    assert claude.returncode == 0 and "0 reused, 5 not sent" in claude.stdout  # Gemma's answers never serve Claude


def test_the_cli_stops_when_no_local_server_is_up(private_copy, tmp_path):
    config, directory = private_copy
    run = agatha_vlm(config, "--model", "local", vlm_config=local_toml(tmp_path / "stage2.toml", NOWHERE))
    assert run.returncode == 1 and run.stderr == "agatha-vlm: failed (URLError)\n" and run.stdout == ""
    assert not (directory / "stage2" / "responses").exists()


def test_the_cli_refuses_local_with_no_local_section(private_copy, tmp_path):
    config, directory = private_copy
    without_local(tmp_path / "stage2.toml")
    run = agatha_vlm(config, "--model", "local", vlm_config=tmp_path / "stage2.toml")
    assert run.returncode == 2 and "bad arguments" in run.stderr and PRIVATE_CONTEXT not in run.stderr
    assert not (directory / "stage2").exists()


# The PACE scripts, under bash, with stubs for nvidia-smi, curl, vllm, uv and sleep.

STUBS = {
    "nvidia-smi": '#!/bin/bash\n[ -n "$GPU_MIB" ] || exit 9\nprintf "%s\\n" $GPU_MIB\n',
    # /health answers from its HEALTHY_AFTER-th call on, counting every call in the test; 0 is never.
    "curl": '#!/bin/bash\necho "$*" >> "$STUB_LOG/curl"\n'
            'if [[ $* == */v1/models ]]; then\n'
            '  echo \'{"object":"list","data":[{"id":"m","object":"model","max_model_len":131072}]}\'\n'
            '  exit 0\n'
            'fi\n'
            '[ "$HEALTHY_AFTER" -gt 0 ] && [ "$(grep -c /health "$STUB_LOG/curl")" -ge "$HEALTHY_AFTER" ]\n',
    # A server for VLLM_LIVES_S seconds, with a child as vLLM has its engine. VLLM_ON_TERM=ignore leaves it running
    # until it's killed, and slow takes a moment to shut down, then leaves a file saying so.
    "vllm": '#!/bin/bash\n'
            'echo "$HF_HOME offline=$HF_HUB_OFFLINE no_usage_stats=$VLLM_NO_USAGE_STATS'
            ' flashinfer_sampler=$VLLM_USE_FLASHINFER_SAMPLER" > "$STUB_LOG/env"\n'
            'printf "%s\\n" "$@" > "$STUB_LOG/vllm.part"\nmv "$STUB_LOG/vllm.part" "$STUB_LOG/vllm"\n'
            'case $VLLM_ON_TERM in\n'
            '  ignore) trap "" TERM ;;\n'
            '  slow) trap \'/bin/sleep 0.5; touch "$STUB_LOG/shut_down"; exit 0\' TERM ;;\n'
            'esac\n'
            '/bin/sleep "$VLLM_LIVES_S" &\necho $! > "$STUB_LOG/child"\nwait $!\n',
    "uv": '#!/bin/bash\necho "$*" >> "$STUB_LOG/uv"\necho "uv $*"\necho "uv $*" >&2\n'
          'if [ "$1" = venv ]; then  # with the tools build_vllm.sbatch runs from the new environment\n'
          '  mkdir -p "${@: -1}/bin"\n'
          '  cp "$STUB_LOG/../venv_bin/"* "${@: -1}/bin"\n'
          'fi\n'
          '[ -z "$UV_SLEEP" ] || /bin/sleep "$UV_SLEEP"\n'  # a run that takes a while, to be stopped midway
          '[[ -z "$UV_FAILS" || "$*" != *"$UV_FAILS"* ]]\n',  # a call naming UV_FAILS fails, as a run can
    "pace-quota": '#!/bin/bash\n[ -n "$SCRATCH_USED_GB" ] || exit 1\n'
                  'echo "Home:/home/hice1/user        12.0    30.0  40.0%    1000   1000000  0.1%"\n'
                  'echo "Scratch:/storage/ice1/0/0/user   $SCRATCH_USED_GB   300.0  89.6%   719098   1000000  71.9%"\n',
    "sleep": '#!/bin/bash\n[ -z "$SLEEP_S" ] || exec /bin/sleep "$SLEEP_S"\n',  # waiting costs nothing, unless asked
}
# The tools build_vllm.sbatch runs from the environment it makes. The Hub's dry run says NEED_GB are left to download.
VENV_BIN = {
    "python": '#!/bin/bash\n'  # one line a call, though the code given with -c takes several
              'printf "%s" "python HF_HOME=$HF_HOME $*" | tr "\\n" " " >> "$STUB_LOG/venv"\necho >> "$STUB_LOG/venv"\n'
              'if [[ $* == *dry_run=True* ]]; then echo "$NEED_GB"; fi\n',
    "vllm": '#!/bin/bash\necho "vllm HF_HOME=$HF_HOME $*" >> "$STUB_LOG/venv"\n',
}


@pytest.fixture
def pace(tmp_path):
    """pace(script, *args, **env) -> the run: a PACE script run from a directory standing in for ~/agatha.

    Its env.sh points at test directories, serve_vlm.sh and stage2_cutin.sbatch are there for the scripts that
    call them, and the stubs log their arguments under tmp_path/stub. Any server left running is killed after it.
    """
    home = tmp_path / "home"
    waymo = home / "scratch" / "waymo"
    (tmp_path / "infra" / "pace").mkdir(parents=True)
    (tmp_path / "infra" / "pace" / "env.sh").write_text(
        f'export WOD_HOME="{waymo}"\nexport WOD_ROOT="{waymo}/root"\nexport WOD_OUTPUTS="{waymo}/outputs"\n'
        f'export HF_HOME="{home}/scratch/hf"\n')
    for script in ("serve_vlm.sh", "stage2_cutin.sbatch"):
        shutil.copy(PROJECT / "infra" / "pace" / script, tmp_path / "infra" / "pace")
    (home / "scratch" / "hf").mkdir(parents=True)
    for directory, stubs in (("bin", STUBS), ("venv_bin", VENV_BIN)):
        (tmp_path / directory).mkdir()
        for name, text in stubs.items():
            (tmp_path / directory / name).write_text(text)
            (tmp_path / directory / name).chmod(0o755)
    (tmp_path / "stub").mkdir()
    gemma_config = tmp_path / "stage2_gemma.toml"
    gemma_config.write_text(gemma_config_text())

    def run(script, *args, background=False, **env):
        settings = {"GPU_MIB": "46068", "HEALTHY_AFTER": "2", "VLLM_LIVES_S": "20", "VLM_CONFIG": str(gemma_config),
                    "SCRATCH_USED_GB": "100.0", "NEED_GB": "63", **env}
        full = {**os.environ, "PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}", "HOME": str(home),
                "STUB_LOG": str(tmp_path / "stub"), **settings}
        command = ["bash", str(PROJECT / "infra" / "pace" / script), *args]
        if background:  # still running, for the test to stop
            return subprocess.Popen(command, cwd=tmp_path, env=full, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    text=True, stdin=subprocess.DEVNULL)
        return subprocess.run(command, cwd=tmp_path, env=full, capture_output=True, text=True,
                              stdin=subprocess.DEVNULL, timeout=120)
    yield run
    for pidfile in (waymo / "logs").glob("vllm-*.pid"):  # the server and its child: serve_vlm.sh gave them a group
        try:
            os.killpg(int(pidfile.read_text()), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, ValueError):
            pass


BF16, FP8 = ("--dtype", "bfloat16"), ("--quantization", "fp8_per_tensor")


def stub_calls(tmp_path, stub: str) -> list[str]:
    path = tmp_path / "stub" / stub
    return path.read_text().splitlines() if path.exists() else []


def alive(pid: int) -> bool:
    """Whether the process is still there, given a moment to go."""
    for _ in range(40):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        time.sleep(0.05)
    return True


def served(tmp_path) -> list[str]:
    """vllm's arguments, one an item, once the server the script left running has written them."""
    for _ in range(100):
        if (tmp_path / "stub" / "vllm").exists():
            break
        time.sleep(0.05)
    return stub_calls(tmp_path, "vllm")


def server_and_child(tmp_path) -> tuple[int, int]:
    """The pids of the server the script left running and of its child, once it has one."""
    for _ in range(100):
        if (tmp_path / "stub" / "child").exists() and (tmp_path / "stub" / "child").read_text().endswith("\n"):
            break
        time.sleep(0.05)
    [pidfile] = tmp_path.joinpath("home", "scratch", "waymo", "logs").glob("vllm-*.pid")
    return int(pidfile.read_text()), int((tmp_path / "stub" / "child").read_text())


def serve_args(tp="1", precision=FP8, kv="fp8", max_len="auto", images="100", image_tokens="280",
               model=GEMMA, port="8000") -> list[str]:
    """vllm's arguments, one an item."""
    return ["serve", model, "--host", "127.0.0.1", "--port", port, "--tensor-parallel-size", tp, *precision,
            "--kv-cache-dtype", kv, "--max-model-len", max_len, "--gpu-memory-utilization", "0.9",
            "--limit-mm-per-prompt", f'{{"image": {images}}}', "--mm-processor-kwargs",
            f'{{"max_soft_tokens": {image_tokens}}}', "--trust-request-mm-kwargs", "--no-enable-log-requests",
            "--disable-uvicorn-access-log", "--no-enable-prefix-caching"]  # one request, one answer at temperature 0


def starting(label="fp8", tp="1", max_len="auto", need="168288", model=GEMMA) -> str:
    """What start prints as it starts the server. 100 images at 280 tokens, with 1,280 tokens of text each, 4,096
    more and 8,192 to answer in come to 168,288."""
    return (f"vllm: starting {model}, {label}, {tp} GPU(s), max-model-len {max_len}; Pass 2's longest request "
            f"needs about {need}")


UP = "vllm: up, max-model-len 131072"  # the length vLLM took, as its list of models gives it
LOGS = ("home", "scratch", "waymo", "logs")


@pytest.mark.parametrize("gpus, tp, precision, kv, label", [
    ("46068", "1", FP8, "fp8", "fp8"),  # one L40S
    ("40960", "1", FP8, "fp8", "fp8"),  # an A100 40 GB
    ("81559", "1", BF16, "auto", "bf16"),  # an H100 80 GB
    ("81920", "1", BF16, "auto", "bf16"),  # an A100 80 GB
    ("46068 46068", "2", BF16, "auto", "bf16"),  # two L40S hold 90 GB
    ("46068 46068 46068", "2", BF16, "auto", "bf16"),  # tensor parallelism takes 1, 2 or 4
    ("22528 22528", "2", FP8, "fp8", "fp8"),  # two 22 GB cards hold 44 GB
    ("30720 30720 30720", "2", FP8, "fp8", "fp8"),  # the two used hold 60 GB, not all 90
    (" ".join(["81920"] * 8), "4", BF16, "auto", "bf16"),
], ids=["L40S", "A100 40GB", "H100", "A100 80GB", "two L40S", "three L40S", "two small", "three small", "eight"])
def test_the_server_takes_bf16_on_80_gb_or_more_and_fp8_below(pace, tmp_path, gpus, tp, precision, kv, label):
    run = pace("serve_vlm.sh", "start", GPU_MIB=gpus)
    assert run.returncode == 0, run.stderr
    assert served(tmp_path) == serve_args(tp, precision, kv)
    assert run.stdout.splitlines() == [starting(label, tp), UP]
    # The weights from scratch, with no download at start, no usage statistics sent to vLLM's makers, and
    # PyTorch's sampler: FlashInfer's compiles at start, which needs ninja and nvcc, which ICE's nodes lack.
    env = f"{tmp_path}/home/scratch/hf offline=1 no_usage_stats=1 flashinfer_sampler=0"
    assert stub_calls(tmp_path, "env") == [env]


def test_the_server_is_sized_from_stage2_toml(pace, tmp_path):
    """Fewer images at a larger budget, and less room to answer in: 40 * (560 + 1280) + 4096 + 4096 = 81,792."""
    config = tmp_path / "stage2.toml"
    text = gemma_config_text().replace("max_images = 100", "max_images = 40")
    text = text.replace("image_tokens = 280", "image_tokens = 560").replace("max_tokens = 8192", "max_tokens = 4096")
    config.write_text(text)
    run = pace("serve_vlm.sh", "start", VLM_CONFIG=str(config))
    assert run.returncode == 0 and run.stdout.splitlines()[0] == starting(need="81792")
    capacity = '{"vision_soft_tokens_per_image": 560, "vision_config": {"default_output_length": 560}}'
    assert served(tmp_path) == [*serve_args(images="40", image_tokens="560"), "--hf-overrides", capacity]


@pytest.mark.parametrize("temperature", [None, "0.7"], ids=["none", "sampled"])
def test_the_server_ignores_the_temperature_each_request_carries(pace, tmp_path, temperature):
    """The shipped 0.0 is in every other serve test's config. A float in [local] doesn't trip setting()'s awk."""
    config = tmp_path / "stage2.toml"
    config.write_text(with_temperature(gemma_config_text(), temperature))
    run = pace("serve_vlm.sh", "start", VLM_CONFIG=str(config))
    assert run.returncode == 0 and run.stdout.splitlines() == [starting(), UP]
    assert served(tmp_path) == serve_args()


def test_max_model_len_and_kv_cache_dtype_override_the_defaults(pace, tmp_path):
    """MAX_MODEL_LEN asks for a length, so vLLM won't start unless it fits; KV_CACHE_DTYPE=auto keeps BF16's."""
    run = pace("serve_vlm.sh", "start", MAX_MODEL_LEN="168288", KV_CACHE_DTYPE="auto")
    assert run.returncode == 0 and run.stdout.splitlines() == [starting(max_len="168288"), UP]
    assert served(tmp_path) == serve_args(kv="auto", max_len="168288")


def test_a_qat_checkpoint_is_served_as_it_comes(pace, tmp_path):
    config = tmp_path / "stage2.toml"
    config.write_text(gemma_config_text().replace(f'id = "{GEMMA}"', f'id = "{GEMMA}-qat-w4a16-ct"'))
    run = pace("serve_vlm.sh", "start", VLM_CONFIG=str(config))
    assert run.returncode == 0 and run.stdout.splitlines()[0] == starting("w4a16", model=f"{GEMMA}-qat-w4a16-ct")
    assert served(tmp_path) == serve_args(precision=BF16, kv="auto", model=f"{GEMMA}-qat-w4a16-ct")


def test_an_fp8_checkpoint_is_served_as_it_comes(pace, tmp_path):
    """The shipped model, Red Hat's FP8 copy of the 26B A4B: no online quantization over it, and a BF16 KV cache."""
    run = pace("serve_vlm.sh", "start", VLM_CONFIG=str(STAGE2_CONFIG))
    assert run.returncode == 0 and run.stdout.splitlines()[0] == starting("fp8-checkpoint", model=SHIPPED)
    assert served(tmp_path) == serve_args(precision=(), kv="auto", model=SHIPPED)


def test_start_waits_for_health_then_status_and_stop_follow_it(pace, tmp_path):
    run = pace("serve_vlm.sh", "start", HEALTHY_AFTER="4")
    assert run.returncode == 0 and run.stdout.splitlines()[-1] == UP
    health = "-sf --noproxy * -m 5 -o /dev/null http://127.0.0.1:8000/health"
    assert stub_calls(tmp_path, "curl") == [health] * 4 + ["-sf --noproxy * -m 5 http://127.0.0.1:8000/v1/models"]
    again = pace("serve_vlm.sh", "start")  # up already, so not started twice
    assert again.returncode == 0 and again.stdout == UP + "\n"
    status = pace("serve_vlm.sh", "status")
    assert (status.returncode, status.stdout) == (0, "vllm: up\n")
    pid, child = server_and_child(tmp_path)
    stop = pace("serve_vlm.sh", "stop")
    # The server and its child, as vLLM's engine and GPU workers are: the GPU is free again.
    assert (stop.returncode, stop.stdout) == (0, "vllm: stopped\n") and not alive(pid) and not alive(child)
    status = pace("serve_vlm.sh", "status", HEALTHY_AFTER="0")
    assert (status.returncode, status.stdout) == (1, "vllm: down\n")
    assert not list(tmp_path.joinpath(*LOGS).glob("*.pid"))


@pytest.mark.parametrize("on_term", ["slow", "ignore"])
def test_stop_waits_for_the_server_and_kills_one_that_wont_go(pace, tmp_path, on_term):
    """slow shuts down in half a second, inside stop's minute; ignore is still there after it, so it's killed."""
    assert pace("serve_vlm.sh", "start", VLLM_ON_TERM=on_term, VLLM_LIVES_S="60").returncode == 0
    pid, child = server_and_child(tmp_path)
    [pidfile] = tmp_path.joinpath(*LOGS).glob("vllm-*.pid")
    stop = pace("serve_vlm.sh", "stop", SLEEP_S="0.05")  # a minute of waiting takes three seconds here
    assert (stop.returncode, stop.stdout) == (0, "vllm: stopped\n")
    assert not alive(pid) and not alive(child) and not pidfile.exists()
    assert (tmp_path / "stub" / "shut_down").exists() == (on_term == "slow")  # given its time, not killed


def test_a_pid_file_left_by_an_ended_job_is_not_trusted(pace, tmp_path):
    """Slurm ended the server with its job, and the number now belongs to another process: not a server."""
    tmp_path.joinpath(*LOGS).mkdir(parents=True)
    host = subprocess.run(["hostname", "-s"], capture_output=True, text=True).stdout.strip()
    other = subprocess.Popen(["/bin/sleep", "30"])
    tmp_path.joinpath(*LOGS, f"vllm-{host}.pid").write_text(f"{other.pid}\n")
    status = pace("serve_vlm.sh", "status", HEALTHY_AFTER="1")  # and another server answers on 8000
    assert status.returncode == 1 and "not from a server this script started" in status.stdout
    assert pace("serve_vlm.sh", "stop").stdout == "vllm: stopped\n" and other.poll() is None  # left alone
    other.kill()
    other.wait()


def test_the_port_comes_from_base_url(pace, tmp_path):
    """Another user's vLLM had 8000 on ICE on 2026-11-04: a config naming another port moves the server there,
    and the health checks and stop follow it."""
    config = tmp_path / "stage2.toml"
    config.write_text(gemma_config_text().replace("127.0.0.1:8000/v1", "127.0.0.1:8001/v1"))
    run = pace("serve_vlm.sh", "start", VLM_CONFIG=str(config), HEALTHY_AFTER="2")
    assert run.returncode == 0, run.stderr
    assert served(tmp_path) == serve_args(port="8001")
    health = "-sf --noproxy * -m 5 -o /dev/null http://127.0.0.1:8001/health"
    assert stub_calls(tmp_path, "curl") == [health] * 2 + ["-sf --noproxy * -m 5 http://127.0.0.1:8001/v1/models"]
    status = pace("serve_vlm.sh", "status", VLM_CONFIG=str(config))
    assert (status.returncode, status.stdout) == (0, "vllm: up\n")
    assert pace("serve_vlm.sh", "stop", VLM_CONFIG=str(config)).stdout == "vllm: stopped\n"


@pytest.mark.parametrize("base_url, port", [
    ("http://127.0.0.1/v1", "8000"),  # no port named: vLLM's default
    ("http://localhost:8123/v1", "8123"),
    ("http://[::1]:8002/v1", "8002"),
])
def test_the_port_is_read_from_any_loopback_base_url(pace, tmp_path, base_url, port):
    config = tmp_path / "stage2.toml"
    config.write_text(gemma_config_text().replace("http://127.0.0.1:8000/v1", base_url))
    assert pace("serve_vlm.sh", "start", VLM_CONFIG=str(config)).returncode == 0
    assert served(tmp_path) == serve_args(port=port)


def test_another_server_on_the_named_port_is_refused_by_number(pace, tmp_path):
    config = tmp_path / "stage2.toml"
    config.write_text(gemma_config_text().replace("127.0.0.1:8000/v1", "127.0.0.1:8001/v1"))
    run = pace("serve_vlm.sh", "start", VLM_CONFIG=str(config), HEALTHY_AFTER="1")
    assert run.returncode == 1 and "port 8001 answers, but not from a server this script started" in run.stderr


def test_start_refuses_a_port_another_server_answers_on(pace, tmp_path):
    """Another user's vLLM on this shared node, say: frames go only to a server this script started."""
    run = pace("serve_vlm.sh", "start", HEALTHY_AFTER="1")
    assert run.returncode == 1 and "not from a server this script started" in run.stderr
    assert stub_calls(tmp_path, "vllm") == []
    status = pace("serve_vlm.sh", "status", HEALTHY_AFTER="1")
    assert status.returncode == 1 and "not from a server this script started" in status.stdout


@pytest.mark.parametrize("lives_s, says, checks", [
    ("0", "exited while starting", None),
    ("120", "not up after an hour", 361),
], ids=["exits", "never answers"])
def test_a_server_that_never_comes_up_fails_the_start(pace, tmp_path, lives_s, says, checks):
    run = pace("serve_vlm.sh", "start", HEALTHY_AFTER="0", VLLM_LIVES_S=lives_s)
    assert run.returncode == 1 and says in run.stderr
    if checks:  # one for another server on the port, then 360 ten seconds apart: an hour
        assert len(stub_calls(tmp_path, "curl")) == checks
    # No pid file left behind, whose number another process could take and pass for the server.
    assert not list((tmp_path / "home" / "scratch" / "waymo" / "logs").glob("*.pid"))
    assert pace("serve_vlm.sh", "status", HEALTHY_AFTER="0").stdout == "vllm: down\n"  # nothing left running


def test_start_needs_a_gpu_and_a_local_section(pace, tmp_path):
    run = pace("serve_vlm.sh", "start", GPU_MIB="", HEALTHY_AFTER="0")
    assert run.returncode == 1 and "no GPU" in run.stderr and stub_calls(tmp_path, "vllm") == []
    config = without_local(tmp_path / "stage2.toml")
    run = pace("serve_vlm.sh", "start", VLM_CONFIG=str(config), HEALTHY_AFTER="0")
    assert run.returncode == 2 and "[local]" in run.stderr and stub_calls(tmp_path, "vllm") == []
    assert pace("serve_vlm.sh").returncode == 2


def cutin_page(tmp_path):
    """Page 1 is a real-looking segment with camera data in training."""
    waymo = tmp_path / "home" / "scratch" / "waymo"
    (waymo / "selected").mkdir(parents=True)
    (waymo / "selected" / "cutin_top.txt").write_text(f"{PRIVATE_CONTEXT}\n")
    (waymo / "root" / "training" / "camera_image").mkdir(parents=True)
    (waymo / "root" / "training" / "camera_image" / f"{PRIVATE_CONTEXT}.parquet").touch()
    return waymo


def test_stage2_cutin_runs_local_once_its_server_is_up(pace, tmp_path):
    waymo = cutin_page(tmp_path)
    (waymo / "logs").mkdir()
    host = subprocess.run(["hostname", "-s"], capture_output=True, text=True).stdout.strip()
    # A server the script started, as far as it can tell: the process in its pid file is vllm serve.
    server = subprocess.Popen([tmp_path / "bin" / "vllm", "serve", GEMMA], start_new_session=True,
                              env={**os.environ, "STUB_LOG": str(tmp_path / "stub"), "VLLM_LIVES_S": "30"})
    (waymo / "logs" / f"vllm-{host}.pid").write_text(f"{server.pid}\n")
    run = pace("stage2_cutin.sbatch", "1", "local", HEALTHY_AFTER="1")
    assert run.returncode == 0, run.stderr
    [call] = stub_calls(tmp_path, "uv")
    assert call.endswith(f"--split training --context {PRIVATE_CONTEXT} --model local")
    assert run.stdout.splitlines()[0] == "vllm: up" and run.stdout.splitlines()[-1] == "cutin_1: agatha-vlm exit 0"
    assert PRIVATE_CONTEXT not in run.stdout + run.stderr
    assert (waymo / "review" / "stage2" / "cutin_1").is_symlink()
    os.killpg(server.pid, signal.SIGKILL)
    server.wait()


@pytest.mark.parametrize("healthy_after", ["0", "1"], ids=["no server", "another's server"])
def test_stage2_cutin_sends_nothing_to_a_local_server_it_cant_trust(pace, tmp_path, healthy_after):
    cutin_page(tmp_path)
    run = pace("stage2_cutin.sbatch", "1", "local", HEALTHY_AFTER=healthy_after)
    assert run.returncode == 1 and "cutin_1: no local model" in run.stderr and stub_calls(tmp_path, "uv") == []
    assert PRIVATE_CONTEXT not in run.stdout + run.stderr


def test_stage2_cutin_runs_a_dry_run_with_no_server_and_refuses_other_models(pace, tmp_path):
    cutin_page(tmp_path)
    run = pace("stage2_cutin.sbatch", "1", HEALTHY_AFTER="0")
    assert run.returncode == 0 and stub_calls(tmp_path, "uv")[0].endswith("--model dry-run")
    assert stub_calls(tmp_path, "curl") == [] and PRIVATE_CONTEXT not in run.stdout + run.stderr
    run = pace("stage2_cutin.sbatch", "1", "gemma")
    assert run.returncode == 2 and run.stdout == "" and len(stub_calls(tmp_path, "uv")) == 1


QAT = f"{GEMMA}-qat-w4a16-ct"


@pytest.mark.parametrize("args, env, wheel, backend, model", [
    ([], {}, "vllm==0.32.0", "cu130", GEMMA),
    (["qat"], {}, "vllm==0.32.0", "cu130", QAT),  # instead, for a scratch with less room
    ([], {"CUDA": "129"}, "https://github.com/vllm-project/vllm/releases/download/v0.32.0/"
                          "vllm-0.32.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl", "cu129", GEMMA),
], ids=["default", "qat", "cuda 12.9"])
def test_build_vllm_makes_its_own_environment_and_downloads_gemma_4(pace, tmp_path, args, env, wheel, backend, model):
    run = pace("build_vllm.sbatch", *args, **env)
    assert run.returncode == 0, run.stderr
    venv = tmp_path / "home" / "scratch" / "venvs" / "vllm"
    assert stub_calls(tmp_path, "uv") == [f"venv --allow-existing --seed --python 3.12 {venv}",
                                          f"pip install --python {venv}/bin/python --torch-backend {backend} {wheel}"]
    hf_home = f"HF_HOME={tmp_path}/home/scratch/hf"  # scratch, not home's small quota
    vllm, versions, size, download = stub_calls(tmp_path, "venv")
    assert vllm == f"vllm {hf_home} download-kernels" and versions.startswith(f"python {hf_home} -c import torch")
    for call in (size, download):  # the size of what's left to download, then the download, of one model
        assert call.startswith(f"python {hf_home} -c ") and call.endswith(f" {model}") and "snapshot_download" in call
    assert "dry_run=True" in size and "dry_run" not in download
    printed = [line for line in run.stdout.splitlines() if not line.startswith("uv ")]  # the stub's own lines
    assert printed[:-1] == ["vllm download-kernels: exit 0", f"{model}: 63 GB to download, 200 GB free on scratch",
                            f"{model}: downloaded"]


@pytest.mark.parametrize("env, says", [
    ({"SCRATCH_USED_GB": "268.9"}, "free space on scratch first"),  # 31 GB free, against 63 to download
    ({"SCRATCH_USED_GB": ""}, "no Scratch line from pace-quota"),
    ({"NEED_GB": ""}, "no size from the Hub"),
], ids=["no room", "no quota", "no size"])
def test_build_vllm_downloads_nothing_scratch_has_no_room_for(pace, tmp_path, env, says):
    """A download the quota stops leaves partial files that count against it, and a full scratch stops every job."""
    run = pace("build_vllm.sbatch", **env)
    assert run.returncode == 1 and says in run.stderr
    assert sum("snapshot_download" in call for call in stub_calls(tmp_path, "venv")) == 1  # only the size asked


def test_build_vllm_runs_again_on_a_full_scratch_once_the_model_is_there(pace, tmp_path):
    run = pace("build_vllm.sbatch", SCRATCH_USED_GB="299.9", NEED_GB="0")
    assert run.returncode == 0 and f"{GEMMA}: 0 GB to download, 0 GB free on scratch" in run.stdout


def test_build_vllm_takes_only_qat_and_never_touches_agathas_environment(pace, tmp_path):
    run = pace("build_vllm.sbatch", "all")
    assert run.returncode == 2 and stub_calls(tmp_path, "uv") == []
    assert pace("build_vllm.sbatch").returncode == 0
    assert not any("agatha" in call or call.split()[0] in ("sync", "run") for call in stub_calls(tmp_path, "uv"))
    assert 'export HF_HOME="$HOME/scratch/hf"' in (PROJECT / "infra" / "pace" / "env.sh").read_text()


# A scene list other than cutin_top.txt: stage2_cutin.sbatch with LIST, and stage2_list.sh over the whole list.

LISTED = [f"{digit * 20}_100_000_120_000" for digit in "789"]  # real-looking names, for random15.txt


def scene_list(tmp_path, names=LISTED, list_name="random15.txt", without_camera=()):
    """$WOD_HOME/selected/<list_name> naming `names`, each with camera data in training but `without_camera`."""
    waymo = tmp_path / "home" / "scratch" / "waymo"
    (waymo / "selected").mkdir(parents=True, exist_ok=True)
    (waymo / "selected" / list_name).write_text("".join(f"{name}\n" for name in names))
    (waymo / "root" / "training" / "camera_image").mkdir(parents=True, exist_ok=True)
    for name in names:
        if name not in without_camera:
            (waymo / "root" / "training" / "camera_image" / f"{name}.parquet").touch()
    return waymo


def contexts_run(tmp_path) -> list[str]:
    """The context and model of each agatha-vlm run, in order."""
    return [call.split("--context ")[1] for call in stub_calls(tmp_path, "uv")]


@pytest.mark.parametrize("listed, scene", [
    ("random15.txt", "random15_2"), ("fast_left.txt", "fast_left_2"), ("cutin_top.txt", "cutin_2"),
])
def test_stage2_cutin_takes_its_scene_from_the_list_in_LIST(pace, tmp_path, listed, scene):
    """Each list's scenes take its prefix, as review_cutins.sbatch names them: cutin_top.txt's stay cutin_<rank>."""
    waymo = scene_list(tmp_path, list_name=listed)
    run = pace("stage2_cutin.sbatch", "2", LIST=listed)
    assert run.returncode == 0, run.stderr
    assert contexts_run(tmp_path) == [f"{LISTED[1]} --model dry-run"]
    assert f"--context {scene} --model dry-run" in run.stdout  # its name masked as the scene
    assert run.stdout.splitlines()[-1] == f"{scene}: agatha-vlm exit 0"
    assert not any(name in run.stdout + run.stderr for name in LISTED)
    assert os.readlink(waymo / "review" / "stage2" / scene) == str(waymo / "outputs" / LISTED[1] / "stage2")


@pytest.mark.parametrize("listed", ["Random15.txt", "random15", "../selected/cutin_top.txt", "random15.txt.bak", ""])
def test_stage2_cutin_takes_only_a_lists_plain_name_in_LIST(pace, tmp_path, listed):
    """An empty LIST is cutin_top.txt, as with none."""
    scene_list(tmp_path)
    scene_list(tmp_path, list_name="cutin_top.txt")
    run = pace("stage2_cutin.sbatch", "1", LIST=listed)
    if listed:
        assert run.returncode == 2 and run.stdout == "" and "usage" in run.stderr and stub_calls(tmp_path, "uv") == []
    else:
        assert run.returncode == 0 and run.stdout.splitlines()[-1] == "cutin_1: agatha-vlm exit 0"


def test_stage2_cutin_says_which_scene_a_list_lacks(pace, tmp_path):
    scene_list(tmp_path, without_camera=[LISTED[0]])
    run = pace("stage2_cutin.sbatch", "1", LIST="random15.txt")
    assert run.returncode == 1 and run.stderr == "random15_1: no camera data\n"
    run = pace("stage2_cutin.sbatch", "4", LIST="random15.txt")
    assert run.returncode == 2 and run.stderr == "random15_4: no such rank\n" and stub_calls(tmp_path, "uv") == []


def test_stage2_list_serves_gemma_runs_every_scene_then_stops_the_server(pace, tmp_path):
    waymo = scene_list(tmp_path)
    run = pace("stage2_list.sh", "random15.txt", "local")
    assert run.returncode == 0, run.stderr
    assert served(tmp_path)[:2] == ["serve", GEMMA]  # the fixture's VLM_CONFIG names the 31B
    assert contexts_run(tmp_path) == [f"{name} --model local" for name in LISTED]
    lines = run.stdout.splitlines()
    assert lines[:2] == [starting(), UP] and lines.count("vllm: up") == 3  # each scene checks it's still ours
    assert [line for line in lines if "agatha-vlm exit" in line] == [
        f"random15_{rank}: agatha-vlm exit 0" for rank in (1, 2, 3)]
    assert lines[-2:] == ["random15: 3 run, 0 failed", "vllm: stopped"]
    assert not list(tmp_path.joinpath(*LOGS).glob("*.pid"))  # stopped, so the GPU is free again
    assert not alive(int((tmp_path / "stub" / "child").read_text()))
    assert all((waymo / "review" / "stage2" / f"random15_{rank}").is_symlink() for rank in (1, 2, 3))
    assert not any(name in run.stdout + run.stderr for name in LISTED)


def test_stage2_list_runs_only_the_ranks_asked_for_and_defaults_to_local(pace, tmp_path):
    scene_list(tmp_path)
    run = pace("stage2_list.sh", "random15.txt", RANKS="3 1")
    assert run.returncode == 0, run.stderr
    assert contexts_run(tmp_path) == [f"{LISTED[2]} --model local", f"{LISTED[0]} --model local"]
    assert run.stdout.splitlines()[-2:] == ["random15: 2 run, 0 failed", "vllm: stopped"]


def test_a_scene_that_fails_doesnt_stop_the_rest_but_fails_the_list(pace, tmp_path):
    """Scene 1 has no camera data, and agatha-vlm fails on scene 3."""
    scene_list(tmp_path, without_camera=[LISTED[0]])
    run = pace("stage2_list.sh", "random15.txt", "local", UV_FAILS=LISTED[2])
    assert run.returncode == 1
    assert contexts_run(tmp_path) == [f"{LISTED[1]} --model local", f"{LISTED[2]} --model local"]
    assert "random15_1: no camera data" in run.stderr and "random15_3: agatha-vlm exit 1" in run.stdout
    assert run.stdout.splitlines()[-2:] == ["random15: 3 run, 2 failed: random15_1 random15_3", "vllm: stopped"]
    assert not list(tmp_path.joinpath(*LOGS).glob("*.pid"))


def test_a_scene_whose_results_cant_be_linked_fails(pace, tmp_path):
    """agatha-vlm ran, so the last line says exit 0, but the link for review failed: the script's status says so."""
    waymo = scene_list(tmp_path, LISTED[:1])
    (waymo / "review").mkdir()
    (waymo / "review" / "stage2").write_text("in the way of the folder\n")
    run = pace("stage2_list.sh", "random15.txt", "local")
    assert run.returncode == 1 and "random15_1: agatha-vlm exit 0" in run.stdout.splitlines()
    assert run.stdout.splitlines()[-2:] == ["random15: 1 run, 1 failed: random15_1", "vllm: stopped"]


def wait_for_a_scene(tmp_path) -> None:
    for _ in range(200):  # until the first scene's agatha-vlm is running
        if stub_calls(tmp_path, "uv"):
            return
        time.sleep(0.05)


@pytest.mark.parametrize("stop", [signal.SIGTERM, signal.SIGHUP, signal.SIGINT], ids=["TERM", "HUP", "INT"])
def test_stage2_list_stops_the_server_when_it_is_stopped(pace, tmp_path, stop):
    """Slurm's TERM at the time limit, a hang-up, or ctrl-C: the server goes too, and no other scene starts. bash
    waits for the running scene first, and its lines are kept."""
    scene_list(tmp_path)
    job = pace("stage2_list.sh", "random15.txt", background=True, UV_SLEEP="1")
    wait_for_a_scene(tmp_path)
    pid, child = server_and_child(tmp_path)
    job.send_signal(stop)
    out, _ = job.communicate(timeout=60)
    assert job.returncode == 1 and out.splitlines()[-2:] == ["random15_1: agatha-vlm exit 0", "vllm: stopped"]
    assert len(stub_calls(tmp_path, "uv")) == 1 and not alive(pid) and not alive(child)
    assert not list(tmp_path.joinpath(*LOGS).glob("*.pid"))


def test_stage2_list_stops_a_server_still_starting(pace, tmp_path):
    """Slurm's TERM reaches every process of the job step, serve_vlm.sh start too, which ends at once: the server it
    launched is stopped, as the trap is set before start. The server here would live a minute, and answers never."""
    scene_list(tmp_path)
    job = pace("stage2_list.sh", "random15.txt", background=True, HEALTHY_AFTER="0", VLLM_LIVES_S="60",
               SLEEP_S="0.2")
    pid, child = server_and_child(tmp_path)  # launched, and start is waiting for its health
    start = subprocess.run(["pgrep", "-P", str(job.pid)], capture_output=True, text=True).stdout.split()
    for process in [job.pid, *map(int, start)]:
        os.kill(process, signal.SIGTERM)
    out, _ = job.communicate(timeout=30)
    assert job.returncode == 1 and out.splitlines()[-1] == "vllm: stopped"
    assert not alive(pid) and not alive(child) and stub_calls(tmp_path, "uv") == []


def test_stage2_list_prints_a_scenes_lines_as_they_come(pace, tmp_path):
    """Launched detached, its log is all Andrew sees: the running scene's server check is there before it ends.
    agatha-vlm prints its counts as it ends, however they're read."""
    scene_list(tmp_path, LISTED[:1])
    job = pace("stage2_list.sh", "random15.txt", background=True, UV_SLEEP="3")
    os.set_blocking(job.stdout.fileno(), False)
    wait_for_a_scene(tmp_path)
    time.sleep(0.5)
    printed, running = job.stdout.read() or "", job.poll() is None
    job.communicate(timeout=60)
    assert running and printed.splitlines() == [starting(), UP, "vllm: up"]


def test_stage2_list_uses_a_server_up_already_and_stops_it(pace, tmp_path):
    scene_list(tmp_path, LISTED[:1])
    assert pace("serve_vlm.sh", "start").returncode == 0
    pid, child = server_and_child(tmp_path)
    run = pace("stage2_list.sh", "random15.txt")
    assert run.returncode == 0 and run.stdout.splitlines()[0] == UP and starting() not in run.stdout
    assert len(stub_calls(tmp_path, "env")) == 1 and not alive(pid) and not alive(child)  # launched only once


def test_stage2_list_with_claude_starts_no_server(pace, tmp_path):
    scene_list(tmp_path, LISTED[:1])
    key = tmp_path / "home" / ".config" / "agatha" / "anthropic.env"
    key.parent.mkdir(parents=True)
    key.write_text("# a test's: no key in it\n")
    run = pace("stage2_list.sh", "random15.txt", "claude")
    assert run.returncode == 0, run.stderr
    assert stub_calls(tmp_path, "vllm") == [] and stub_calls(tmp_path, "curl") == []
    assert contexts_run(tmp_path) == [f"{LISTED[0]} --model claude"]
    assert run.stdout.splitlines()[-1] == "random15: 1 run, 0 failed"


def test_vlm_config_reaches_the_server_and_every_scene(pace, tmp_path):
    """A config with a higher image budget, as for reading finer detail: the server and agatha-vlm both take it."""
    scene_list(tmp_path, LISTED[:2])
    config = tmp_path / "stage2_detail.toml"
    config.write_text(gemma_config_text().replace("image_tokens = 280", "image_tokens = 1120"))
    run = pace("stage2_list.sh", "random15.txt", VLM_CONFIG=str(config))
    assert run.returncode == 0, run.stderr
    assert '{"max_soft_tokens": 1120}' in served(tmp_path)
    calls = stub_calls(tmp_path, "uv")
    assert len(calls) == 2 and all(f"--vlm-config {config} " in call for call in calls)


def test_stage2_list_runs_no_scene_when_the_server_wont_start(pace, tmp_path):
    scene_list(tmp_path)
    run = pace("stage2_list.sh", "random15.txt", GPU_MIB="", HEALTHY_AFTER="0")
    assert run.returncode == 1 and "no GPU" in run.stderr and stub_calls(tmp_path, "uv") == []


@pytest.mark.parametrize("args, env", [
    ([], {}), (["random15"], {}), (["random15.txt", "dry-run"], {}), (["random15.txt", "gemma"], {}),
    (["missing.txt"], {}), (["../waymo/selected/random15.txt"], {}), (["random15.txt"], {"RANKS": "1,2"}),
    (["random15.txt"], {"RANKS": "0"}), (["random15.txt"], {"RANKS": "1 4"}), (["random15.txt"], {"RANKS": " "}),
], ids=["none", "no .txt", "dry-run", "another model", "no such list", "a path", "commas", "rank 0", "past the list",
        "blank"])
def test_stage2_list_refuses_bad_arguments_before_starting_anything(pace, tmp_path, args, env):
    """Before the server starts: a start takes minutes of the GPU's time."""
    scene_list(tmp_path)
    run = pace("stage2_list.sh", *args, **env)
    assert run.returncode == 2 and run.stdout == "" and "stage2_list.sh" in run.stderr
    assert stub_calls(tmp_path, "vllm") == [] and stub_calls(tmp_path, "uv") == []


def test_stage2_list_refuses_an_empty_list_before_starting_anything(pace, tmp_path):
    scene_list(tmp_path, names=[])
    run = pace("stage2_list.sh", "random15.txt")
    assert (run.returncode, run.stdout, run.stderr) == (2, "", "stage2_list.sh: no scenes to run in random15.txt\n")
    assert stub_calls(tmp_path, "vllm") == []


DETACHED = ("source infra/pace/env.sh; nohup setsid srun --jobid=<job> --overlap bash -l infra/pace/stage2_list.sh "
            "random15.txt local > $WOD_HOME/logs/stage2-random15.log 2>&1 &")


def test_stage2_list_says_how_to_launch_it_detached(tmp_path):
    """The header's line, and docs/stage2.md's, run as written from a fresh login, where only env.sh sets $WOD_HOME:
    srun gets the job, and its output goes to the log."""
    text = (PROJECT / "infra" / "pace" / "stage2_list.sh").read_text()
    assert f"on the login node. env.sh sets $WOD_HOME, which a fresh login lacks:\n#   {DETACHED}\n" in text
    assert f"`{DETACHED}`" in (PROJECT / "docs" / "stage2.md").read_text()
    (tmp_path / "infra" / "pace").mkdir(parents=True)
    (tmp_path / "infra" / "pace" / "env.sh").write_text(f'export WOD_HOME="{tmp_path}/waymo"\n')
    (tmp_path / "waymo" / "logs").mkdir(parents=True)
    (tmp_path / "bin").mkdir()
    for name, body in (("nohup", 'exec "$@"'), ("setsid", 'exec "$@"'), ("srun", 'echo "srun $*"')):
        (tmp_path / "bin" / name).write_text(f"#!/bin/bash\n{body}\n")
        (tmp_path / "bin" / name).chmod(0o755)
    env = {key: value for key, value in os.environ.items() if key != "WOD_HOME"}
    run = subprocess.run(["bash", "-c", DETACHED.replace("<job>", "6808392") + "\nwait"], cwd=tmp_path, timeout=30,
                         env={**env, "PATH": f"{tmp_path / 'bin'}:{env['PATH']}"}, capture_output=True, text=True)
    assert (run.returncode, run.stdout, run.stderr) == (0, "", "")
    assert (tmp_path / "waymo" / "logs" / "stage2-random15.log").read_text() == (
        "srun --jobid=6808392 --overlap bash -l infra/pace/stage2_list.sh random15.txt local\n")
