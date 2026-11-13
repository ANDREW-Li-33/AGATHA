import base64
import html
import json
import os
import random
import re
import shutil
import stat
import subprocess
import sys
import tomllib
from dataclasses import replace
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import pandas as pd
import pytest

from agatha.columns import CAMERA_IDS, TIME
from agatha.config import load_config
from agatha.run_stage1 import process_segment
from agatha.vlm.labels import NO_FILE, HumanLabels, agreement, read_labels, summary, template
from agatha.vlm.pages import list_prefix, write_pages
from agatha.vlm.run import mark_duplicates
from agatha.vlm.schema import CATEGORIES, TAXONOMY, IncidentReport

PROJECT = Path(__file__).parents[1]
STAGE1_CONFIG = PROJECT / "configs" / "stage1.toml"
GEMMA = "RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic"
# Real-looking names: scenes 1 and 2 have Stage 1's outputs, and scene 3 has none.
CONTEXTS = [f"{str(digit) * 20}_100_000_120_000" for digit in (4, 5, 6)]
SCENES = ["random15_1", "random15_2", "random15_3"]


def report(incident_id: str, start_s: float, end_s: float, kept=("SUDDEN_CUT_IN",), status="CONFIRMED", rejected=(),
           duplicate_of=None, reasoning="V002 moves into the ego's lane.") -> IncidentReport:
    """A Stage 2 report: the labels kept at `status`, and the labels rejected. With none kept, it's REJECTED."""
    assessments = [{"label": label, "evidence": ["seen"], "contradicting_evidence": [], "status": status}
                   for label in kept]
    assessments += [{"label": label, "evidence": [], "contradicting_evidence": ["not seen"], "status": "REJECTED"}
                    for label in rejected]
    return IncidentReport(
        schema_version="0.1", segment_id="x", incident_id=incident_id, duplicate_of=duplicate_of, start_s=start_s,
        end_s=end_s, categories=[category for category in CATEGORIES if any(TAXONOMY[label][0] == category
                                                                          for label in kept)],
        subcategories=list(kept), object_ids=["object"], verification_status=status if kept else "REJECTED",
        label_assessments=assessments, vlm_reasoning=reasoning, driving_impact="Less clearance ahead.",
        uncertainty_reasons=["the <b>gap</b> is hard to see"] if status == "UNCERTAIN" and kept else [],
        source_frames=[], pipeline_version="p", model_id=GEMMA, prompt_version="v", preprocessing_version="pp")


def labels(*incidents, scene="random15_1", reviewed=True) -> HumanLabels:
    """Andrew's reviewed labels: each incident as (labels, start_s, end_s), or none for "nothing happened"."""
    return HumanLabels(scene=scene, reviewed=reviewed, no_incident=not incidents, incidents=[
        {"labels": list(names), "start_s": start_s, "end_s": end_s} for names, start_s, end_s in incidents])


# Scene 1's reports: a cut-in with a rejected label, a rejected pedestrian, an uncertain lane closure, and a
# duplicate of the cut-in.
REPORTS = [
    report("INC-001", 8.0, 12.0, rejected=("HARD_BRAKING",), reasoning="V002 <b>cuts</b> in."),
    report("INC-002", 1.5, 5.0, kept=(), rejected=("PEDESTRIAN_INTRUSION",)),
    report("INC-003", 14.0, 18.0, kept=("CONSTRUCTION_LANE_CLOSURE",), status="UNCERTAIN"),
    report("INC-004", 10.0, 12.5, duplicate_of="INC-001"),
]


@pytest.fixture(scope="module")
def stage1(synthetic_root, tmp_path_factory):
    """(config, directory) of Stage 1's FRONT outputs for the synthetic segment, made once: a run takes seconds."""
    root, ref = synthetic_root
    config = load_config(STAGE1_CONFIG, root=root, outputs=tmp_path_factory.mktemp("outputs"))
    directory, _ = process_segment(config, ref)
    return config, directory


@pytest.fixture
def scenes(stage1, tmp_path):
    """(config, home): random15.txt's three scenes in a fresh $WOD_HOME. Scene 1 has Stage 2's reports too."""
    config, directory = stage1
    outputs = tmp_path / "outputs"
    for context in CONTEXTS[:2]:
        shutil.copytree(directory, outputs / context, ignore=shutil.ignore_patterns("clips", "qa"))
    write_reports(outputs / CONTEXTS[0], REPORTS)
    home = tmp_path / "waymo"
    (home / "selected").mkdir(parents=True)
    (home / "selected" / "random15.txt").write_text("".join(f"{context}\n" for context in CONTEXTS))
    return replace(config, outputs=outputs), home


def write_reports(directory: Path, reports: list[IncidentReport]) -> None:
    (directory / "stage2").mkdir(parents=True, exist_ok=True)
    (directory / "stage2" / "incidents.json").write_text(json.dumps([item.model_dump(mode="json") for item in reports]))


def write_outcomes(directory: Path, windows: list[str], merged: list[str]) -> None:
    """stage2/candidates.json as agatha-vlm writes it, with only what the pages read: each request's outcome, and each
    merged candidate's labels and span. Candidate n is ROAD_DEBRIS at 2n to 2n + 1.5 s."""
    (directory / "stage2" / "candidates.json").write_text(json.dumps({
        "windows": [{"id": f"W{number}", "outcome": outcome} for number, outcome in enumerate(windows)],
        "candidates": [], "merged": [{"incident_id": f"INC-{number:03d}", "labels": ["ROAD_DEBRIS"],
                                      "start_s": 2.0 * number, "end_s": 2.0 * number + 1.5, "outcome": outcome}
                                     for number, outcome in enumerate(merged, 1)]}))


def label_file(home: Path, scene: str) -> Path:
    return home / "review" / "random15" / "labels" / f"{scene}.json"


def write_labels(home: Path, value: HumanLabels | dict) -> None:
    value = value.model_dump() if isinstance(value, HumanLabels) else value
    label_file(home, value["scene"]).parent.mkdir(parents=True, exist_ok=True)
    label_file(home, value["scene"]).write_text(json.dumps(value))


def page(home: Path, scene: str) -> str:
    return (home / "review" / "random15" / f"{scene}.html").read_text()


def index_rows(home: Path) -> list[list[str]]:
    """Each row of the index as its cells' text."""
    text = (home / "review" / "random15" / "index.html").read_text()
    return [[html.unescape(re.sub(r"<[^>]+>", "", cell)) for cell in re.findall(r"<td>(.*?)</td>", row)]
            for row in re.findall(r"<tr><td>.*?</tr>", text)]


# --- the label format ---


def test_the_template_is_not_labeled_yet_and_lists_the_taxonomy():
    value = template("random15_3")
    assert value == {"scene": "random15_3", "reviewed": False, "no_incident": None, "incidents": [],
                     "allowed_labels": list(TAXONOMY)}
    assert HumanLabels.model_validate_json(json.dumps(value)).reviewed is False


def test_a_reviewed_label_file_reads_with_whole_seconds_and_no_notes(tmp_path):
    path = tmp_path / "random15_1.json"
    path.write_text(json.dumps({"scene": "random15_1", "reviewed": True, "no_incident": False, "incidents": [
        {"labels": ["SUDDEN_CUT_IN", "HARD_BRAKING"], "start_s": 8, "end_s": 12}]}))
    found = read_labels(path, "random15_1")
    assert isinstance(found, HumanLabels) and found.incidents[0].start_s == 8.0 and found.incidents[0].notes == ""
    assert read_labels(tmp_path / "missing.json", "random15_1") == NO_FILE


def test_an_incident_can_start_at_the_scenes_first_moment(tmp_path):
    path = tmp_path / "random15_1.json"
    path.write_text(json.dumps({**GOOD, "incidents": [{**GOOD["incidents"][0], "start_s": 0.0}]}))
    assert read_labels(path, "random15_1").incidents[0].start_s == 0.0


GOOD = {"scene": "random15_1", "reviewed": True, "no_incident": False,
        "incidents": [{"labels": ["SUDDEN_CUT_IN"], "start_s": 8.0, "end_s": 12.0, "notes": ""}]}


@pytest.mark.parametrize("change, says", [
    ({"incidents": [{**GOOD["incidents"][0], "labels": ["CUT_IN"]}]}, "incidents.0.labels.0"),
    ({"incidents": [{**GOOD["incidents"][0], "labels": []}]}, "incidents.0.labels"),
    ({"incidents": [{**GOOD["incidents"][0], "start_s": "8"}]}, "incidents.0.start_s"),
    ({"incidents": [{**GOOD["incidents"][0], "start_s": -1.0}]}, "incidents.0.start_s"),
    ({"incidents": [{**GOOD["incidents"][0], "start_s": float("nan")}]}, "incidents.0.start_s"),
    ({"incidents": [{**GOOD["incidents"][0], "end_s": float("inf")}]}, "incidents.0.end_s"),
    ({"incidents": [{**GOOD["incidents"][0], "start_s": 13.0}]}, "ends before it starts"),
    ({"incidents": [{**GOOD["incidents"][0], "label": "SUDDEN_CUT_IN"}]}, "incidents.0.label"),
    ({"reviewed": "true"}, "reviewed"),
    ({"incident": []}, "incident"),
    ({"no_incident": None}, "sets no_incident"),
    ({"no_incident": True}, "no_incident must be true with no incidents"),
    ({"incidents": []}, "no_incident must be true with no incidents"),
    ({"scene": "random15_2"}, "scene: random15_2, not random15_1"),
], ids=["unknown label", "no labels", "a time as text", "before the scene", "not a number", "endless", "backwards",
        "misspelled key",
        "a bool as text", "an extra key", "undecided", "nothing yet incidents", "incidents yet none", "another scene"])
def test_a_label_file_that_cant_be_used_says_why(tmp_path, change, says):
    path = tmp_path / "random15_1.json"
    path.write_text(json.dumps({**GOOD, **change}))
    found = read_labels(path, "random15_1")
    assert isinstance(found, str) and says in found


def test_a_label_file_that_isnt_json_says_so(tmp_path):
    path = tmp_path / "random15_1.json"
    path.write_text('{"scene": "random15_1", "reviewed": tru')
    assert "Invalid JSON" in read_labels(path, "random15_1")


@pytest.mark.parametrize("no_incident", [None, True])
def test_an_unreviewed_file_needs_no_decision_yet(tmp_path, no_incident):
    """Andrew can save a half-done file, undecided or not yet agreeing: it shows as not labeled until reviewed."""
    path = tmp_path / "random15_1.json"
    path.write_text(json.dumps({**GOOD, "reviewed": False, "no_incident": no_incident}))
    assert isinstance(read_labels(path, "random15_1"), HumanLabels)


# --- agreement ---


@pytest.mark.parametrize("human, reports, matches, model_only", [
    ([(["SUDDEN_CUT_IN"], 9.0, 11.0)], [report("INC-001", 8.0, 12.0)], [["INC-001"]], []),
    ([(["SUDDEN_CUT_IN"], 12.0, 13.0)], [report("INC-001", 8.0, 12.0)], [["INC-001"]], []),
    ([(["SUDDEN_CUT_IN"], 12.5, 13.0)], [report("INC-001", 8.0, 12.0)], [[]], ["INC-001"]),
    ([(["SUDDEN_CUT_IN"], 2.0, 3.0)], [report("INC-001", 8.0, 12.0)], [[]], ["INC-001"]),
    ([(["HARD_BRAKING"], 9.0, 11.0)], [report("INC-001", 8.0, 12.0)], [[]], ["INC-001"]),
    ([(["SUDDEN_CUT_IN"], 9.0, 11.0)], [report("INC-001", 8.0, 12.0, status="UNCERTAIN")], [["INC-001"]], []),
    ([(["SUDDEN_CUT_IN"], 9.0, 11.0)], [report("INC-001", 8.0, 12.0, kept=(), rejected=("SUDDEN_CUT_IN",))],
     [[]], []),
    ([(["HARD_BRAKING"], 9.0, 11.0)], [report("INC-001", 8.0, 12.0, rejected=("HARD_BRAKING",))],
     [[]], ["INC-001"]),
    ([(["SUDDEN_CUT_IN"], 9.0, 11.0)], [report("INC-001", 8.0, 12.0), report("INC-002", 9.0, 13.0,
                                                                           duplicate_of="INC-001")],
     [["INC-001", "INC-002"]], []),
    ([(["HARD_BRAKING"], 10.0, 12.0)], [report("INC-001", 8.0, 12.0), report(
        "INC-002", 9.0, 13.0, kept=("SUDDEN_CUT_IN", "HARD_BRAKING"), duplicate_of="INC-001")], [["INC-002"]], []),
    ([(["SUDDEN_CUT_IN"], 14.0, 15.0)], [report("INC-001", 8.0, 12.0), report("INC-002", 11.0, 16.0,
                                                                           duplicate_of="INC-001")], [["INC-002"]], []),
    ([(["SUDDEN_CUT_IN"], 15.0, 16.0)], [report("INC-001", 0.0, 5.0),
                                         report("INC-002", 4.0, 9.0, duplicate_of="INC-001"),
                                         report("INC-003", 8.0, 16.0, duplicate_of="INC-002")], [["INC-003"]], []),
    ([(["HARD_BRAKING"], 1.0, 2.0)], [report("INC-001", 8.0, 12.0), report("INC-002", 9.0, 13.0,
                                                                          duplicate_of="INC-001")], [[]], ["INC-001"]),
    ([(["SUDDEN_CUT_IN"], 9.0, 11.0), (["SUDDEN_CUT_IN", "HARD_BRAKING"], 10.0, 15.0)],
     [report("INC-001", 8.0, 12.0), report("INC-002", 14.0, 16.0, kept=("HARD_BRAKING",))],
     [["INC-001"], ["INC-001", "INC-002"]], []),
], ids=["overlap and a shared label", "touching", "after it", "before it", "no shared label", "uncertain matches",
        "rejected never matches", "a label the report rejected", "an event and its duplicate",
        "a duplicate's own label", "a duplicate's later time", "a chain of duplicates", "a duplicate is never alone",
        "many to many"])
def test_an_incident_matches_a_kept_report_sharing_a_kept_label_and_its_time(human, reports, matches, model_only):
    """A duplicate matches as any report does, and a match with it finds the report it repeats too."""
    result = agreement(labels(*human), reports)
    assert result["matches"] == matches and result["model_only"] == model_only
    assert result["human_only"] == [number for number, ids in enumerate(matches, 1) if not ids]
    unmarked = [report.model_copy(update={"duplicate_of": None}) for report in reports]
    mark_duplicates(unmarked)  # the duplicates here are the ones Stage 2 marks
    assert [report.duplicate_of for report in unmarked] == [report.duplicate_of for report in reports]


@pytest.mark.parametrize("human, reports, says", [
    ((), [report("INC-001", 1.0, 2.0, kept=(), rejected=("SUDDEN_CUT_IN",))], "both say nothing happened"),
    ((), [], "both say nothing happened"),
    ((), [report("INC-001", 8.0, 12.0)], "only Andrew says nothing happened"),
    (((["SUDDEN_CUT_IN"], 9.0, 11.0),), [report("INC-001", 1.0, 2.0, kept=(), rejected=("SUDDEN_CUT_IN",))],
     "only Stage 2 says nothing happened"),
    (((["HARD_BRAKING"], 9.0, 11.0),), [report("INC-001", 8.0, 12.0)], "both saw something"),
], ids=["both, a rejected report", "both, no reports", "Andrew only", "Stage 2 only", "neither"])
def test_whether_both_say_nothing_happened(human, reports, says):
    assert summary(agreement(labels(*human), reports)).endswith(says)


def test_with_stage2_incomplete_nothing_happened_is_unknown_unless_it_kept_a_report():
    """Stage 2 kept nothing but didn't assess it all: a report it kept says something happened, whatever it missed."""
    unknown = agreement(labels(), [], complete=False)
    assert unknown["nothing"] == (True, None)
    assert summary(unknown) == "0 of Andrew's matched, 0 Stage 2 only, 0 Andrew only; nothing happened: unknown, as " \
                               "Stage 2 is incomplete"
    kept = agreement(labels(), [report("INC-001", 8.0, 12.0)], complete=False)
    assert kept["nothing"] == (True, False) and summary(kept).endswith("only Andrew says nothing happened")


def test_an_incident_in_a_candidate_stage2_didnt_assess_isnt_andrews_only():
    """Pass 2 failed on a candidate sharing a label and time with Andrew's incident: a pipeline failure, not a miss.
    A match with a report still counts first."""
    failed = [{"incident_id": "INC-002", "labels": ["ROAD_DEBRIS"], "start_s": 8.5, "end_s": 10.5, "outcome": "failed"}]
    human = labels((["ROAD_DEBRIS"], 10.5, 12.0), (["HARD_BRAKING"], 9.0, 10.0), (["ROAD_DEBRIS"], 11.0, 12.0),
                   (["SUDDEN_CUT_IN", "ROAD_DEBRIS"], 9.0, 10.0))
    result = agreement(human, [report("INC-001", 8.0, 12.0)], complete=False, unassessed=failed)
    assert result["matches"] == [[], [], [], ["INC-001"]] and result["unassessed"] == [["INC-002"], [], [], []]
    assert result["human_only"] == [2, 3]
    assert summary(result) == "1 of Andrew's matched, 0 Stage 2 only, 2 Andrew only, 1 not assessed; both saw something"


def test_the_summary_counts_matches_and_each_side_alone():
    result = agreement(labels((["SUDDEN_CUT_IN"], 9.0, 11.0), (["ROAD_DEBRIS"], 2.0, 3.0)), REPORTS)
    assert summary(result) == "1 of Andrew's matched, 1 Stage 2 only, 1 Andrew only; both saw something"
    assert result["model_only"] == ["INC-003"]  # the uncertain closure: the rejected one and the duplicate don't count


# --- the pages ---


def test_each_page_embeds_the_scenes_front_clip_at_its_real_timing(scenes, tmp_path):
    config, home = scenes
    counts = write_pages(config, home, "random15.txt")
    assert counts["pages"] == 3 and counts["clips"] == 2
    text = page(home, "random15_1")
    [(offset, data)] = re.findall(r'<video id="clip" data-offset="([\d.]+)" controls muted playsinline preload="auto" '
                                  r'src="data:video/mp4;base64,([A-Za-z0-9+/=]+)"></video>', text)
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(base64.b64decode(data))
    assert clip.read_bytes()[4:8] == b"ftyp"
    # Stage 1's FRONT frames over the whole scene, each at its own time less the offset, so a time seeks to its frame.
    directory = config.outputs / CONTEXTS[0]
    frames = drawn_front_times(directory, config.profile)
    assert len(frames) > 30 and float(offset) == pytest.approx(frames[0], abs=1e-3)
    times = presentation_times(clip)
    np.testing.assert_allclose(times[:len(frames)] + float(offset), frames, atol=2e-3)


def test_a_clip_that_starts_late_seeks_by_its_offset(scenes, tmp_path):
    """With Stage 1's first FRONT frames gone, the clip starts later, and its offset says where."""
    config, home = scenes
    directory = config.outputs / CONTEXTS[0]
    frames = pd.read_parquet(directory / "frames.parquet")
    for ts in frames.loc[(frames["camera"] == CAMERA_IDS["FRONT"]) & (frames["t_s"] < 3.0), TIME]:
        (directory / "render" / "FRONT" / f"{ts}__{config.profile}.jpg").unlink(missing_ok=True)
    write_pages(config, home, "random15.txt")
    [(offset, data)] = re.findall(r'data-offset="([\d.]+)" .*?base64,([A-Za-z0-9+/=]+)"', page(home, "random15_1"))
    drawn = drawn_front_times(directory, config.profile)
    assert 3.0 <= drawn[0] < 3.6 and float(offset) == pytest.approx(drawn[0], abs=1e-3)
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(base64.b64decode(data))
    np.testing.assert_allclose(presentation_times(clip)[:len(drawn)] + float(offset), drawn, atol=2e-3)


def drawn_front_times(directory: Path, profile: str) -> list[float]:
    """The t_s of every FRONT frame Stage 1 drew in `profile`."""
    frames = pd.read_parquet(directory / "frames.parquet")
    front = frames[frames["camera"] == CAMERA_IDS["FRONT"]]
    return [t_s for ts, t_s in zip(front[TIME], front["t_s"])
            if (directory / "render" / "FRONT" / f"{ts}__{profile}.jpg").exists()]


def presentation_times(clip: Path) -> np.ndarray:
    """Frame times in seconds, read back from the file."""
    dump = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-i", str(clip), "-map", "0:v", "-c", "copy",
                           "-f", "framecrc", "-"], capture_output=True, text=True, check=True).stdout
    numerator, denominator = re.search(r"#tb 0: (\d+)/(\d+)", dump).groups()
    pts = [int(line.split(",")[2]) for line in dump.splitlines() if not line.startswith("#")]
    return np.sort(pts) * int(numerator) / int(denominator)


def test_every_time_on_a_page_seeks_the_clip(scenes):
    config, home = scenes
    write_labels(home, labels((["SUDDEN_CUT_IN"], 9.25, 11.0)))
    write_pages(config, home, "random15.txt")
    text = page(home, "random15_1")
    assert 'function seek(t)' in text and "video.currentTime = Math.max(0, t - Number(video.dataset.offset))" in text
    assert "video.play();" in text  # and plays from there
    seeks = [float(value) for value in re.findall(r'<a href="#" onclick="return seek\(([\d.]+)\)">\1 s</a>', text)]
    assert seeks == [8.0, 12.0, 1.5, 5.0, 14.0, 18.0, 10.0, 12.5, 9.25, 11.0]  # the reports', then Andrew's


def test_a_page_shows_each_reports_status_labels_explanation_reasons_and_duplicate(scenes):
    config, home = scenes
    write_pages(config, home, "random15.txt")
    text = page(home, "random15_1")
    assert f"<h2>Stage 2 ({GEMMA})</h2>" in text
    assert "4 reports: 2 confirmed, 1 rejected, 1 uncertain, 1 marked duplicate." in text
    assert "<td>SUDDEN_CUT_IN, HARD_BRAKING (rejected)</td>" in text and "V002 &lt;b&gt;cuts&lt;/b&gt; in." in text
    assert '<tr class="REJECTED"><td>INC-002</td><td>REJECTED</td><td>PEDESTRIAN_INTRUSION (rejected)</td>' in text
    assert "<td>the &lt;b&gt;gap&lt;/b&gt; is hard to see</td>" in text and "<td>duplicate of INC-001</td>" in text


def test_a_template_is_made_for_each_scene_and_never_overwritten(scenes):
    config, home = scenes
    assert write_pages(config, home, "random15.txt")["templates"] == 3
    for scene in SCENES:
        assert json.loads(label_file(home, scene).read_text()) == template(scene)
    edited = json.dumps({**GOOD, "incidents": [{**GOOD["incidents"][0], "notes": "Andrew's own"}]})
    label_file(home, "random15_1").write_text(edited)
    label_file(home, "random15_2").write_text("not JSON, as Andrew left it")
    assert write_pages(config, home, "random15.txt")["templates"] == 0
    assert label_file(home, "random15_1").read_text() == edited
    assert label_file(home, "random15_2").read_text() == "not JSON, as Andrew left it"
    assert "<td>Andrew&#x27;s own</td>" in page(home, "random15_1")  # the rerun refreshed the page around it


def test_a_page_says_what_it_lacks(scenes):
    config, home = scenes
    write_pages(config, home, "random15.txt")
    assert "No clip: Stage 1 hasn't run on this scene." in page(home, "random15_3")
    for scene in ("random15_2", "random15_3"):
        assert "No results: Stage 2 hasn't written incidents.json for this scene." in page(home, scene)
    for scene in SCENES:
        assert "<p>Not labeled yet.</p>" in page(home, scene) and "<h2>Agreement</h2>" not in page(home, scene)
        assert f"Edit labels/{scene}.json" in page(home, scene)


def test_a_scene_stage1_drew_no_front_frames_for_has_no_clip(scenes):
    config, home = scenes
    shutil.rmtree(config.outputs / CONTEXTS[1] / "render" / "FRONT")
    assert write_pages(config, home, "random15.txt")["clips"] == 1
    assert "No clip: Stage 1 drew no FRONT frames." in page(home, "random15_2")


def test_a_labeled_page_shows_andrews_labels_and_the_agreement(scenes):
    config, home = scenes
    write_labels(home, labels((["SUDDEN_CUT_IN"], 9.0, 11.0), (["ROAD_DEBRIS"], 2.0, 3.0)))
    counts = write_pages(config, home, "random15.txt")
    assert counts["labeled"] == 1
    text = page(home, "random15_1")
    assert "<p>Not labeled yet.</p>" not in text and "<td>2</td><td>ROAD_DEBRIS</td>" in text
    assert ("<h2>Agreement</h2>\n<p>1 of Andrew&#x27;s matched, 1 Stage 2 only, 1 Andrew only; both saw something</p>"
            in text)
    assert re.findall(r"<li>(.*?)</li>", text) == [
        "Andrew&#x27;s 1: matches INC-001, INC-004", "Andrew&#x27;s 2: Andrew only", "INC-003: Stage 2 only"]


def test_nothing_happened_shows_and_agrees(scenes):
    config, home = scenes
    write_reports(config.outputs / CONTEXTS[1], [report("INC-001", 1.0, 2.0, kept=(), rejected=("ROAD_DEBRIS",))])
    write_labels(home, labels(scene="random15_2"))
    write_pages(config, home, "random15.txt")
    assert "<p>Nothing happened.</p>" in page(home, "random15_2")
    assert "both say nothing happened" in page(home, "random15_2")


def test_stage2_reporting_nothing_shows_and_counts_as_results(scenes):
    """The likely case for many random scenes: incidents.json is [], with no table under it."""
    config, home = scenes
    write_reports(config.outputs / CONTEXTS[1], [])
    write_labels(home, labels(scene="random15_2"))
    counts = write_pages(config, home, "random15.txt")
    assert (counts["stage2"], counts["incomplete"], counts["labeled"]) == (2, 0, 1)
    assert "<h2>Stage 2</h2>\n<p>0 reports: Stage 2 reported nothing in this scene.</p>\n<h2>Andrew" in page(
        home, "random15_2")
    assert index_rows(home)[1] == ["random15_2", "0 confirmed, 0 rejected, 0 uncertain, 0 marked duplicate", "yes",
                                   "0 of Andrew's matched, 0 Stage 2 only, 0 Andrew only; both say nothing happened"]


@pytest.mark.parametrize("windows, merged, missed, listed", [
    (["answered", "failed"], [], "1 of 2 windows", []),
    (["invalid", "answered", "no_frames"], [], "2 of 3 windows", []),
    (["answered"], ["CONFIRMED", "failed"], "1 of 2 candidates", ["INC-002: ROAD_DEBRIS, 4.00 s to 5.50 s, failed"]),
    (["answered"], ["invalid", "REJECTED", "no_frames", "UNCERTAIN"], "2 of 4 candidates",
     ["INC-001: ROAD_DEBRIS, 2.00 s to 3.50 s, invalid", "INC-003: ROAD_DEBRIS, 6.00 s to 7.50 s, no_frames"]),
    (["failed", "answered"], ["failed"], "1 of 2 windows and 1 of 1 candidates",
     ["INC-001: ROAD_DEBRIS, 2.00 s to 3.50 s, failed"]),
], ids=["a failed window", "invalid and empty windows", "a failed candidate", "invalid and empty candidates", "both"])
def test_a_stage2_run_that_left_part_unassessed_says_so(scenes, windows, merged, missed, listed):
    """agatha-vlm writes its results though requests failed: with nothing kept, "nothing happened" is unknown.
    The page lists each candidate it didn't assess, with its labels and a span that seeks the clip."""
    config, home = scenes
    write_reports(config.outputs / CONTEXTS[1], [])
    write_outcomes(config.outputs / CONTEXTS[1], windows, merged)
    write_labels(home, labels(scene="random15_2"))
    assert write_pages(config, home, "random15.txt")["incomplete"] == 1
    text = page(home, "random15_2")
    assert f"<h2>Stage 2</h2>\n<p><b>Stage 2 didn't assess {missed}.</b></p>\n" in text and "\n<p>0 reports" in text
    items = re.findall(r"<li>(.*?)</li>", text)
    assert [re.sub(r"<[^>]+>", "", item) for item in items] == listed
    assert all(item.count('onclick="return seek(') == 2 for item in items)
    assert index_rows(home)[1] == [
        "random15_2", f"0 confirmed, 0 rejected, 0 uncertain, 0 marked duplicate; Stage 2 didn't assess {missed}",
        "yes", "0 of Andrew's matched, 0 Stage 2 only, 0 Andrew only; nothing happened: unknown, as Stage 2 is "
        "incomplete"]


def test_andrews_incident_in_a_candidate_stage2_didnt_assess_shows_as_not_assessed(scenes):
    """As in random15_1, where Pass 2 failed on 1 of 6 candidates but kept other reports: the page and the index
    say the incident wasn't assessed, instead of counting it as one Stage 2 missed."""
    config, home = scenes
    write_outcomes(config.outputs / CONTEXTS[0], ["answered"] * 4,
                   ["CONFIRMED", "REJECTED", "UNCERTAIN", "CONFIRMED", "failed"])  # INC-005: 10 to 11.5 s
    write_labels(home, labels((["SUDDEN_CUT_IN"], 9.0, 11.0), (["ROAD_DEBRIS"], 10.5, 11.0),
                              (["HARD_BRAKING"], 10.5, 11.0)))
    write_pages(config, home, "random15.txt")
    agreed = "1 of Andrew's matched, 1 Stage 2 only, 1 Andrew only, 1 not assessed; both saw something"
    text = html.unescape(page(home, "random15_1"))
    assert [re.sub(r"<[^>]+>", "", item) for item in re.findall(r"<li>(.*?)</li>", text)] == [
        "INC-005: ROAD_DEBRIS, 10.00 s to 11.50 s, failed", "Andrew's 1: matches INC-001, INC-004",
        "Andrew's 2: overlaps INC-005, which Stage 2 didn't assess", "Andrew's 3: Andrew only",
        "INC-003: Stage 2 only"]
    assert f"<h2>Agreement</h2>\n<p>{agreed}</p>" in text
    assert index_rows(home)[0][1:] == ["2 confirmed, 1 rejected, 1 uncertain, 1 marked duplicate; Stage 2 didn't "
                                       "assess 1 of 5 candidates", "yes", agreed]


def test_a_stage2_run_that_assessed_everything_says_nothing_more(scenes):
    config, home = scenes
    write_outcomes(config.outputs / CONTEXTS[0], ["answered"] * 4, ["CONFIRMED", "REJECTED", "UNCERTAIN"])
    assert write_pages(config, home, "random15.txt")["incomplete"] == 0
    assert "assess" not in page(home, "random15_1") and "assess" not in index_rows(home)[0][1]


def test_labels_with_no_stage2_results_show_with_no_agreement(scenes):
    config, home = scenes
    write_labels(home, labels((["SUDDEN_CUT_IN"], 9.0, 11.0), scene="random15_3"))
    assert write_pages(config, home, "random15.txt")["labeled"] == 1
    assert "<td>SUDDEN_CUT_IN</td>" in page(home, "random15_3") and "Agreement" not in page(home, "random15_3")


def test_an_unusable_label_file_says_why_on_its_page_and_in_the_index(scenes):
    config, home = scenes
    write_labels(home, {**GOOD, "no_incident": True})
    counts = write_pages(config, home, "random15.txt")
    assert counts["invalid"] == 1 and counts["labeled"] == 0
    text = page(home, "random15_1")
    assert "The label file can&#x27;t be used: no_incident must be true with no incidents, and false with some" in text
    assert "<h2>Agreement</h2>" not in text and index_rows(home)[0][2] == "invalid"


def test_the_index_has_a_row_per_scene_with_counts_labels_and_agreement_and_no_video(scenes):
    config, home = scenes
    write_labels(home, labels((["SUDDEN_CUT_IN"], 9.0, 11.0)))
    write_pages(config, home, "random15.txt")
    text = (home / "review" / "random15" / "index.html").read_text()
    assert "<video" not in text and "<h1>random15: 3 scenes</h1>" in text
    assert [f'<a href="{scene}.html">{scene}</a>' in text for scene in SCENES] == [True] * 3
    assert index_rows(home) == [
        ["random15_1", "2 confirmed, 1 rejected, 1 uncertain, 1 marked duplicate", "yes",
         "1 of Andrew's matched, 1 Stage 2 only, 0 Andrew only; both saw something"],
        ["random15_2", "-", "no", "-"],
        ["random15_3", "-", "no", "-"]]


def test_ranks_write_only_their_pages_and_templates_but_the_index_lists_every_scene(scenes):
    config, home = scenes
    counts = write_pages(config, home, "random15.txt", [2])
    assert (counts["pages"], counts["templates"]) == (1, 1)
    folder = home / "review" / "random15"
    assert sorted(path.name for path in folder.glob("*.html")) == ["index.html", "random15_2.html"]
    assert [path.name for path in (folder / "labels").iterdir()] == ["random15_2.json"]
    assert [row[0] for row in index_rows(home)] == SCENES and index_rows(home)[0][1].startswith("2 confirmed")
    assert [row[2] for row in index_rows(home)] == ["no", "no", "no"]  # no label file isn't an invalid one


@pytest.mark.parametrize("line", ["../escape", "a/b"])
def test_a_scene_that_fails_is_counted_and_the_rest_go_on(scenes, line):
    """A list line that isn't a context name, or Stage 2 results that don't parse."""
    config, home = scenes
    (config.outputs / CONTEXTS[1] / "stage2").mkdir()
    (config.outputs / CONTEXTS[1] / "stage2" / "incidents.json").write_text("[{")
    (home / "selected" / "random15.txt").write_text(f"{CONTEXTS[0]}\n{CONTEXTS[1]}\n{line}\n")
    counts = write_pages(config, home, "random15.txt")
    assert (counts["pages"], counts["failed"]) == (1, 2)
    assert [row[1] for row in index_rows(home)][1:] == ["failed", "failed"]
    assert not (home / "review" / "random15" / "random15_2.html").exists()


def test_cutin_top_keeps_its_cutin_prefix_and_other_lists_take_their_stem():
    assert [list_prefix(name) for name in ("cutin_top.txt", "fast_left.txt", "random15.txt")] == [
        "cutin", "fast_left", "random15"]


# --- the CLI ---


def agatha_pages(config, home, *args, umask=None) -> subprocess.CompletedProcess:
    env = {**os.environ, "WOD_HOME": str(home), "WOD_OUTPUTS": str(config.outputs), "WOD_ROOT": str(config.root)}
    return subprocess.run([sys.executable, "-m", "agatha.vlm.pages", "--config", str(STAGE1_CONFIG), *args],
                          capture_output=True, text=True, env=env, cwd=PROJECT, timeout=300, umask=umask or -1)


def test_the_cli_writes_private_pages_and_prints_only_counts(scenes):
    config, home = scenes
    run = agatha_pages(config, home, "--list", "random15.txt", umask=0o022)
    assert run.returncode == 0 and run.stderr == ""
    assert run.stdout.splitlines() == [
        "random15: 3 pages, 2 with a clip, 1 with Stage 2 results (0 incomplete), 0 labeled, 0 label files invalid, "
        "3 templates made, 0 failed", f"index: {home / 'review' / 'random15' / 'index.html'}"]
    assert not any(context in run.stdout for context in CONTEXTS)
    written = [path for path in (home / "review").rglob("*")]
    assert len(written) == 9 and not [path for path in written if stat.S_IMODE(path.stat().st_mode) & 0o077]
    # One page, with its template made already: every count starts from a page's, and each is still a number.
    again = agatha_pages(config, home, "--list", "random15.txt", "--ranks", "2")
    assert again.returncode == 0 and again.stdout.splitlines()[0] == (
        "random15: 1 pages, 1 with a clip, 0 with Stage 2 results (0 incomplete), 0 labeled, 0 label files invalid, "
        "0 templates made, 0 failed")


@pytest.mark.parametrize("args", [["--list", "random15"], ["--list", "../selected/random15.txt"],
                                  ["--list", "other.txt"], ["--list", "random15.txt", "--ranks", "4"],
                                  ["--list", "random15.txt", "--ranks", "0"]],
                         ids=["no .txt", "a path", "no such list", "past the list", "rank 0"])
def test_the_cli_refuses_a_list_it_cant_read_and_ranks_past_it(scenes, args):
    config, home = scenes
    run = agatha_pages(config, home, *args)
    assert run.returncode == 2 and "bad arguments" in run.stderr and not (home / "review").exists()


def test_agatha_pages_is_a_console_script():
    scripts = tomllib.loads((PROJECT / "pyproject.toml").read_text())["project"]["scripts"]
    assert scripts["agatha-pages"] == "agatha.vlm.pages:main"


# --- pick_random.py ---


def pick(tmp_path, *args, names=(), validation=(), cutin=(), fast_left=()) -> subprocess.CompletedProcess:
    """pick_random.py over a $WOD_ROOT holding lidar_box for `names` in training and `validation` in validation."""
    home, root = tmp_path / "waymo", tmp_path / "waymo" / "root"
    for split, listed in (("training", names), ("validation", validation)):
        (root / split / "lidar_box").mkdir(parents=True, exist_ok=True)
        for name in listed:
            (root / split / "lidar_box" / f"{name}.parquet").touch()
    (home / "selected").mkdir(exist_ok=True)
    for listing, listed in (("cutin_top.txt", cutin), ("fast_left.txt", fast_left)):
        if not (home / "selected" / listing).exists():
            (home / "selected" / listing).write_text("".join(f"{name}\n" for name in listed))
    env = {**os.environ, "WOD_HOME": str(home), "WOD_ROOT": str(root)}
    return subprocess.run([sys.executable, str(PROJECT / "infra" / "pace" / "pick_random.py"), *args], env=env,
                          capture_output=True, text=True, timeout=60, cwd=tmp_path)


NAMES = [f"{index:020d}_100_000_120_000" for index in range(40)]


def picked(tmp_path, name="random15.txt") -> list[str]:
    return (tmp_path / "waymo" / "selected" / name).read_text().splitlines()


def test_pick_random_takes_15_by_its_seed_in_pick_order_from_both_splits(tmp_path):
    run = pick(tmp_path, names=NAMES[:30], validation=NAMES[30:])
    assert run.returncode == 0 and run.stderr == ""
    assert run.stdout == "random15.txt: 15 picked from 40 labeled segments, 0 left out as listed\n"
    assert picked(tmp_path) == random.Random(20261104).sample(sorted(NAMES), 15)
    assert any(name in NAMES[30:] for name in picked(tmp_path))  # validation's count too


def test_pick_random_ignores_what_isnt_a_lidar_box_file(tmp_path):
    (tmp_path / "waymo" / "root" / "training" / "lidar_box").mkdir(parents=True)
    (tmp_path / "waymo" / "root" / "training" / "lidar_box" / "_metadata").touch()
    (tmp_path / "waymo" / "root" / "training" / "camera_image").mkdir()
    (tmp_path / "waymo" / "root" / "training" / "camera_image" / f"{'9' * 20}_1_2_3_4.parquet").touch()
    run = pick(tmp_path, "--n", "3", names=NAMES[:3])
    assert run.returncode == 0 and sorted(picked(tmp_path)) == NAMES[:3]


def test_pick_random_leaves_out_every_listed_segment(tmp_path):
    run = pick(tmp_path, "--n", "30", names=NAMES, cutin=NAMES[:6], fast_left=NAMES[5:10])
    assert run.returncode == 0 and "30 picked from 30 labeled segments, 10 left out as listed" in run.stdout
    assert sorted(picked(tmp_path)) == NAMES[10:]
    other = tmp_path / "other"
    other.mkdir()
    run = pick(other, "--exclude", "mine.txt", "--n", "39", names=NAMES, cutin=NAMES[:6])
    assert run.returncode == 2 and "no list mine.txt" in run.stderr
    (other / "waymo" / "selected" / "mine.txt").write_text(f"{NAMES[0]}\n")
    run = pick(other, "--exclude", "mine.txt", "--n", "39", names=NAMES)
    assert run.returncode == 0 and sorted(picked(other)) == NAMES[1:]  # --exclude replaces the scan's two lists


def test_pick_random_keeps_an_existing_list_unless_forced(tmp_path):
    assert pick(tmp_path, names=NAMES).returncode == 0
    first = picked(tmp_path)
    run = pick(tmp_path, "--seed", "7", names=NAMES)
    assert run.returncode == 1 and "--force replaces it" in run.stderr and picked(tmp_path) == first
    run = pick(tmp_path, "--seed", "7", "--force", names=NAMES)
    assert run.returncode == 0 and picked(tmp_path) == random.Random(7).sample(sorted(NAMES), 15) != first
    assert pick(tmp_path, "--force", names=NAMES).returncode == 0 and picked(tmp_path) == first  # the seed decides


def test_pick_random_writes_another_list_when_asked_and_keeps_it_private(tmp_path):
    run = pick(tmp_path, "--out", "random5.txt", "--n", "5", names=NAMES)
    assert run.returncode == 0 and len(picked(tmp_path, "random5.txt")) == 5
    assert not (tmp_path / "waymo" / "selected" / "random15.txt").exists()
    assert stat.S_IMODE((tmp_path / "waymo" / "selected" / "random5.txt").stat().st_mode) == 0o600


@pytest.mark.parametrize("args", [["--out", "cutin_top.txt", "--force"], ["--out", "fast_left.txt", "--force"],
                                  ["--out", "cutins.txt"],
                                  ["--out", "stage2.txt"], ["--out", "cutin.txt"], ["--out", "Random.txt"],
                                  ["--out", "../random15.txt"], ["--n", "0"], ["--exclude", "missing.txt"]])
def test_pick_random_refuses_a_taken_or_bad_name_and_a_missing_exclude_list(tmp_path, args):
    run = pick(tmp_path, *args, names=NAMES, cutin=NAMES[:2])
    assert run.returncode == 2 and run.stdout == ""
    assert sorted(path.name for path in (tmp_path / "waymo" / "selected").iterdir()) == ["cutin_top.txt",
                                                                                          "fast_left.txt"]
    assert (tmp_path / "waymo" / "selected" / "cutin_top.txt").read_text() == f"{NAMES[0]}\n{NAMES[1]}\n"


def test_pick_random_writes_nothing_when_too_few_are_left(tmp_path):
    run = pick(tmp_path, names=NAMES[:20], cutin=NAMES[:6])
    assert run.returncode == 1 and "only 14 labeled segments" in run.stderr
    assert not (tmp_path / "waymo" / "selected" / "random15.txt").exists()
