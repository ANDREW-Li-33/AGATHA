import base64
import itertools
import json
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from agatha import run_stage1
from agatha.checks import output_hashes, pick_review_frames
from agatha.columns import (
    CAMERA, CAMERA_IDS, CAMERA_IMAGE, LIDAR_BOX, LIDAR_COMPONENTS, REQUIRED_COLUMNS, STAGE1_COMPONENTS, TIMESTAMP,
    VEHICLE_POSE, col,
)
from agatha.config import load_config
from agatha.contracts import SceneWindow
from agatha.loader import SegmentRef, component_path
from agatha.render import boxes_for_image
from agatha.report import VOCABULARY, context_label
from agatha.scene_window import build_scene_window, write_scene_window
from agatha.store import camera_models, load_segment
from agatha.synthetic import LABEL_OFFSET_S, OBJECT_IDS, write_synthetic_segment

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "stage1.toml"
CONTEXT = "synthetic_cut_in"
PRIVATE_CONTEXT = "12345678901234567890_100_000_120_000"  # shaped like a real name, so it gets hashed
THREE_CAMERAS_EDIT = ('cameras = ["FRONT"]', 'cameras = ["FRONT", "FRONT_LEFT", "FRONT_RIGHT"]')

TIME = "frame_timestamp_micros"
NO_RESIDUALS = pd.DataFrame(columns=[TIME, "camera", "column_px", "image_width_px", "e_u_min", "e_u_max", "e_v_min",
                                     "e_v_max", "kind"])
RUNS = ("front", "three_cameras")
CHECKS = ("schema", "joins", "gaps", "residuals", "tracks", "velocity", "ego_velocity", "timing", "nulls")
SPANS = ("0_5000", "4000_9000", "8000_13000", "12000_17000", "16000_20000")

# The sections a run writes, in order, so a stopped run can say which it reached.
SECTIONS = ("schema", "load", "joins", "tables", "windows", "render", "gaps", "residuals", "tracks", "velocity",
            "ego_velocity", "timing", "nulls", "contact_sheet")

# The spec's tables, column for column.
EGO_COLUMNS = """
    frame_timestamp_micros t_s world_from_vehicle pos_local_x_m pos_local_y_m pos_local_z_m yaw_rad v_world_x_mps
    v_world_y_mps speed_mps a_long_mps2 a_lat_mps2 yaw_rate_radps path_s_m""".split()
FRAME_COLUMNS = """
    frame_timestamp_micros t_s camera camera_name gap_before_ms has_gap_before pose_timestamp_s trigger_time_s
    readout_done_time_s shutter_s capture_offset_ms world_from_vehicle world_from_image_vehicle""".split()
TRACK_COLUMNS = """
    frame_timestamp_micros t_s laser_object_id alias type center_x_m center_y_m center_z_m size_x_m size_y_m size_z_m
    heading_rad pos_world_x_m pos_world_y_m pos_world_z_m pos_local_x_m pos_local_y_m pos_local_z_m yaw_world_rad
    v_label_world_x_mps v_label_world_y_mps v_diff_world_x_mps v_diff_world_y_mps v_world_x_mps v_world_y_mps v_source
    a_world_x_mps2 a_world_y_mps2 v_abs_ego_x_mps v_abs_ego_y_mps v_rel_ego_x_mps v_rel_ego_y_mps a_abs_ego_x_mps2
    a_abs_ego_y_mps2 speed_mps stationary along_path_m lateral_offset_m on_driven_path path_clearance_m front_gap_m
    num_lidar_points num_top_lidar_points not_directly_observed fov_cameras synced_camera synced_center_x_m
    synced_center_y_m synced_center_z_m synced_size_x_m synced_size_y_m synced_size_z_m synced_heading_rad""".split()
META_KEYS = """
    schema_version dataset_version split context_name t0_micros last_t_s n_frames local_origin_world_m
    rear_axle_to_front_m preprocessing_version attribution cameras""".split()


def write_config(path: Path, root: Path, outputs: Path, *edits: tuple[str, str]) -> Path:
    """configs/stage1.toml with its paths filled in and `edits` made, which is all a run can vary."""
    text = CONFIG_PATH.read_text().replace("$WOD_ROOT", str(root)).replace("$WOD_OUTPUTS", str(outputs))
    for old, new in edits:
        assert old in text
        text = text.replace(old, new)
    path.write_text(text)
    return path


def cli(*args, **options) -> subprocess.CompletedProcess:
    """The command a job runs, so the exit code, stdout, and stderr are the real ones."""
    return subprocess.run([sys.executable, "-m", "agatha.run_stage1", *args], capture_output=True, text=True,
                          timeout=300, **options)


def report_path(directory: Path) -> Path:
    """Where a segment's report goes: beside the segment folders, named by label, so nothing has to be listed."""
    return directory.parent / "reports" / f"{context_label(directory.name)}.json"


def read_report(directory: Path) -> dict:
    return json.loads(report_path(directory).read_text())


def statuses(report: dict) -> dict[str, str]:
    return {name: section["status"] for name, section in report["sections"].items() if "status" in section}


def leaves(value) -> list:
    """Every string and number under a parsed JSON value."""
    if isinstance(value, dict):
        value = list(value.values())
    return [leaf for item in value for leaf in leaves(item)] if isinstance(value, list) else [value]


@pytest.fixture(autouse=True)
def process_state_is_restored():
    """main takes over the process it runs in, and in these tests that's pytest's."""
    handler, hook, umask = signal.getsignal(signal.SIGTERM), sys.excepthook, os.umask(0o077)
    yield
    signal.signal(signal.SIGTERM, handler)
    sys.excepthook = hook
    os.umask(umask)


@pytest.fixture(scope="module")
def runs(tmp_path_factory) -> dict[str, SimpleNamespace]:
    """FRONT, then the same command with three cameras in the config, as the real rollout goes.

    The second writes over the first, so what's left of each is copied here. The config's root
    doesn't exist: --synthetic has to bring its own.
    """
    base = tmp_path_factory.mktemp("e2e")
    directory = base / "outputs" / CONTEXT
    finished = {}
    for name, cameras, edits in (("front", ["FRONT"], ()),
                                 ("three_cameras", ["FRONT", "FRONT_LEFT", "FRONT_RIGHT"], (THREE_CAMERAS_EDIT,))):
        config = write_config(base / f"{name}.toml", base / "no_such_root", base / "outputs", *edits)
        result = cli("--config", str(config), "--synthetic")
        kept = shutil.copytree(directory, base / name / CONTEXT)
        finished[name] = SimpleNamespace(result=result, directory=kept, written_to=directory, cameras=cameras,
                                         report=read_report(directory), report_text=report_path(directory).read_text(),
                                         contact_sheet=(directory / "qa" / "contact_sheet.html").read_text(),
                                         segment=load_segment(kept))
    return finished


@pytest.fixture(scope="module")
def segment(runs):
    return runs["three_cameras"].segment


def window_path(run, span: str) -> Path:
    [path] = (run.directory / "windows").glob(f"{span}__2fps__{'+'.join(run.cameras)}__pass1__*.json")
    return path


def scene_windows(run) -> list[SceneWindow]:
    return [SceneWindow.model_validate_json(window_path(run, span).read_text()) for span in SPANS]


def rows_of(segment, name: str) -> pd.DataFrame:
    return segment.tracks[segment.tracks["laser_object_id"] == OBJECT_IDS[name]]


def object_in(window: SceneWindow, name: str):
    return next(track for track in window.objects if track.laser_object_id == OBJECT_IDS[name])


@pytest.fixture(scope="module")
def short_root(tmp_path_factory) -> Path:
    """A 4 s segment under a private name, for tests that stop or repeat a run. One window, and quick."""
    root = tmp_path_factory.mktemp("short")
    write_synthetic_segment(root, context_name=PRIVATE_CONTEXT, n_frames=40)
    return root


@pytest.fixture
def short_run(short_root, tmp_path) -> SimpleNamespace:
    """Arguments for `main` on a copy of the short segment, so a test can damage it, with fresh outputs."""
    root = shutil.copytree(short_root, tmp_path / "root")
    config = write_config(tmp_path / "stage1.toml", root, tmp_path / "outputs")
    return SimpleNamespace(root=root, tmp_path=tmp_path, directory=tmp_path / "outputs" / PRIVATE_CONTEXT,
                           ref=SegmentRef("training", PRIVATE_CONTEXT),
                           args=["--config", str(config), "--context", PRIVATE_CONTEXT])


# The two runs the acceptance list names.

@pytest.mark.parametrize("name", RUNS)
def test_every_check_passes(runs, name):
    run = runs[name]
    assert run.result.returncode == 0
    assert run.report["failures"] == []
    assert statuses(run.report) == dict.fromkeys(CHECKS, "pass")


def test_synthetic_flag_writes_under_the_outputs_root(runs):
    outputs = runs["front"].written_to.parent
    assert component_path(outputs / "synthetic", SegmentRef("training", CONTEXT), LIDAR_BOX).exists()


def test_config_alone_brings_in_the_other_cameras(runs):
    """Check 7. Boxes must reach each camera's numbers, or passing says nothing about it."""
    assert set(runs["front"].report["sections"]["residuals"]["by_camera"]) == {"front"}
    section = runs["three_cameras"].report["sections"]["residuals"]
    assert set(section["by_camera"]) == {"front", "front_left", "front_right"}
    for spread in section["by_camera"].values():
        assert spread["boxes"] >= 10 and spread["median"] < 5.0 and spread["p95"] < 30.0
    assert all(spread["boxes"] >= 20 for spread in section["lidar_box_by_camera"].values())


def test_the_report_holds_what_was_drawn_beside_the_label_numbers(runs):
    """The limits read by_camera, and the label's numbers still have to get off PACE: they show pose and timing."""
    section = runs["three_cameras"].report["sections"]["residuals"]
    assert section["drawn"]["boxes"] == sum(spread["boxes"] for spread in section["by_camera"].values())
    assert set(section["lidar_box_by_camera"]) == set(section["frame_pose_by_camera"]) == set(section["by_camera"])
    assert section["label_fallback_by_camera"] == {}  # a box that fits inside one image here is in its synced camera


def test_the_rollout_leaves_only_what_its_last_run_wrote(runs):
    """Windows and clips from the FRONT run would no longer match the tables, and a glob would take them as current."""
    run = runs["three_cameras"]
    windows = sorted(path.name for path in (run.directory / "windows").iterdir())
    assert len(windows) == len(SPANS) and all("__FRONT+FRONT_LEFT+FRONT_RIGHT__" in name for name in windows)
    assert len(list((run.directory / "clips").glob("*.mp4"))) == len(SPANS) * len(run.cameras)
    assert all(window.preprocessing_version == run.report["preprocessing_version"] for window in scene_windows(run))


def test_more_cameras_change_no_alias(runs):
    """Stage 2 validates against aliases, and a rollout must not rename anyone."""
    def alias_by_id(run) -> dict:
        return {track.laser_object_id: track.alias for window in scene_windows(run) for track in window.objects}

    assert alias_by_id(runs["front"]) == alias_by_id(runs["three_cameras"])


@pytest.mark.parametrize("name", RUNS)
def test_contact_sheet_shows_the_review_frames(runs, name):
    run = runs[name]
    section = run.report["sections"]["contact_sheet"]
    images = re.findall(r'src="data:image/jpeg;base64,([^"]+)"', run.contact_sheet)
    assert len(images) == section["frames"] > 0
    reasons = ("worst_residual", "image_edge", "near_fast", "random")
    assert section["frames"] == sum(section[reason] for reason in reasons)
    assert set(re.findall(r"<figcaption>(\w+) at", run.contact_sheet)) == set(run.cameras)
    for image in images:
        decoded = cv2.imdecode(np.frombuffer(base64.b64decode(image), np.uint8), cv2.IMREAD_COLOR)
        assert decoded.shape == (728, 1092, 3)  # the profile's long edge, from 1920x1280


def test_contact_sheet_draws_each_frame_as_the_windows_do(runs, segment):
    """render_annotated is deterministic, so a review frame that's also in a window must match its file."""
    sheet = runs["three_cameras"].contact_sheet
    images = re.findall(r'src="data:image/jpeg;base64,([^"]+)"', sheet)
    captions = re.findall(r"<figcaption>(\w+) at ([\d.]+) s", sheet)
    compared = set()
    for image, (camera, t_s) in zip(images, captions):
        frames = segment.frames[segment.frames["camera_name"] == camera]
        timestamp = frames.loc[frames["t_s"].round(2) == float(t_s), TIME].item()
        cached = runs["three_cameras"].directory / "render" / camera / f"{timestamp}__pass1.jpg"
        if cached.exists():
            assert base64.b64decode(image) == cached.read_bytes()
            compared.add(camera)
    assert compared == {"FRONT", "FRONT_LEFT", "FRONT_RIGHT"}


@pytest.mark.parametrize("name", RUNS)
def test_window_json_validates_against_the_contract(runs, name):
    run = runs[name]
    windows = scene_windows(run)
    assert run.report["sections"]["windows"]["windows"] == len(windows) == len(SPANS)
    for window in windows:
        assert {frame.camera for frame in window.frames} == set(run.cameras)
        assert window.objects and window.ego and window.omitted_objects == []
        assert window.segment_context_name == CONTEXT
        assert (window.cameras, window.profile) == (run.cameras, "pass1")


@pytest.mark.parametrize("name", RUNS)
def test_every_frame_a_window_names_is_on_disk(runs, name):
    run = runs[name]
    for frame in (frame for window in scene_windows(run) for frame in window.frames):
        assert (run.directory / frame.clean_ref).exists() and (run.directory / frame.annotated_ref).exists()


def test_render_counts_each_shared_frame_once(runs):
    """Windows overlap, so a frame they share must be drawn and counted one time, over every window."""
    run = runs["front"]
    cam = camera_models(run.segment.meta)[CAMERA_IDS["FRONT"]]
    shared = {frame.frame_timestamp_micros for window in scene_windows(run) for frame in window.frames}
    drawn = Counter(box.draw_source for timestamp in shared
                    for box in boxes_for_image(run.segment, timestamp, CAMERA_IDS["FRONT"], cam))
    section = run.report["sections"]["render"]
    assert (section["camera_synced_box"], section["lidar_box"]) == (drawn["camera_synced_box"], drawn["lidar_box"])
    assert sum(len(window.frames) for window in scene_windows(run)) > len(shared)  # they do overlap


@pytest.mark.parametrize("name", RUNS)
def test_each_clip_holds_exactly_its_windows_frames(runs, name):
    """Timestamps jitter, so a frame for grid time 8.0 can sit just before a window's nominal start."""
    run = runs[name]
    windows = scene_windows(run)
    assert run.report["sections"]["render"]["clips"] == len(windows) * len(run.cameras)
    for span, window, camera in ((s, w, c) for (s, w) in zip(SPANS, windows) for c in run.cameras):
        stem = f"{window_path(run, span).stem}__{camera}"
        assert (run.directory / "clips" / f"{stem}.mp4").stat().st_size > 0
        script = (run.directory / "clips" / f"{stem}.ffconcat").read_text()
        listed = [int(ts) for ts in re.findall(r"(\d+)__pass1\.jpg'", script)]
        assert listed[:-1] == [frame.frame_timestamp_micros for frame in window.frames if frame.camera == camera]


# What leaves PACE.

@pytest.mark.parametrize("name", RUNS)
def test_stdout_is_counts_and_the_report_path(runs, name):
    run = runs[name]
    assert run.result.stdout.splitlines() == [f"checks: {len(CHECKS)} pass, 0 fail, 0 no_data; failed steps: 0",
                                              f"report: {report_path(run.written_to)}"]
    assert run.result.stderr == ""


def test_the_timing_section_names_the_readout_direction(runs):
    front = runs["front"].report["sections"]["timing"]["cameras"]["front"]
    assert front["rolling_shutter_direction"] == "LEFT_TO_RIGHT"


@pytest.mark.parametrize("name", RUNS)
def test_no_object_id_reaches_stdout_or_the_report(runs, name):
    run = runs[name]
    window_text = next((run.directory / "windows").iterdir()).read_text()
    assert any(object_id in window_text for object_id in OBJECT_IDS.values())  # they're in the data to leak
    assert not any(object_id in run.result.stdout + run.report_text for object_id in OBJECT_IDS.values())


@pytest.mark.parametrize("name", RUNS)
def test_the_report_holds_only_aggregates_and_vocabulary(runs, name):
    values = leaves(runs[name].report["sections"])
    assert {value for value in values if isinstance(value, str)} <= VOCABULARY
    numbers = [abs(value) for value in values if isinstance(value, (int, float))]
    assert max(numbers) < 1e12  # a raw frame timestamp is about 1.55e15


def test_a_private_context_stays_out_of_stdout_and_the_report(short_run, capfd):
    assert run_stage1.main(short_run.args) == 0
    out, err = capfd.readouterr()
    text = report_path(short_run.directory).read_text()
    assert PRIVATE_CONTEXT not in out + err + text
    assert json.loads(text)["context"] == context_label(PRIVATE_CONTEXT)


def test_the_printed_report_path_is_real_and_needs_no_listing(short_run, capfd):
    """Finding a report must not mean listing a directory of context names."""
    run_stage1.main(short_run.args)
    [line] = [line for line in capfd.readouterr().out.splitlines() if line.startswith("report: ")]
    path = Path(line.removeprefix("report: "))
    assert path.exists() and path.parent.name == "reports" and path.stem == context_label(PRIVATE_CONTEXT)


@pytest.mark.parametrize("args, context", [
    ([], f"./{PRIVATE_CONTEXT}"), ([], f"{CONTEXT}/../{PRIVATE_CONTEXT}"),
    (["--synthetic"], f"{CONTEXT}/../{PRIVATE_CONTEXT}"), ([], ".."), ([], ""),
], ids=["relative", "through a synthetic name", "before --synthetic writes", "parent", "empty"])
def test_a_context_that_is_a_path_is_refused_before_anything_is_written(short_run, args, context):
    """Path tidies ./name, and synthetic_cut_in/../<name> passed for synthetic, so its report was named <name>."""
    result = cli("--config", short_run.args[1], *args, "--context", context)
    assert result.returncode == 2 and result.stdout == "" and "bad arguments" in result.stderr
    assert PRIVATE_CONTEXT not in result.stderr and not (short_run.tmp_path / "outputs").exists()


@pytest.mark.parametrize("args", [
    ["--split", "training", PRIVATE_CONTEXT],     # the flag forgotten
    ["--cameras", "FRONT", PRIVATE_CONTEXT],      # taken for a camera choice
])
def test_a_mistyped_command_does_not_repeat_the_context(args):
    result = cli(*args)
    assert result.returncode == 2 and result.stdout == ""
    assert PRIVATE_CONTEXT not in result.stderr


@pytest.mark.parametrize("problem", ["no_such_file", "bad_camera"])
def test_a_bad_config_stays_quiet_about_its_path(tmp_path, problem):
    directory = tmp_path / PRIVATE_CONTEXT
    directory.mkdir()
    config = directory / "stage1.toml"
    if problem == "bad_camera":
        write_config(config, tmp_path, tmp_path, ('cameras = ["FRONT"]', 'cameras = ["FRONTT"]'))
    result = cli("--config", str(config))
    assert result.returncode == 1 and result.stdout == ""
    assert result.stderr == "agatha-stage1: failed outside any step\n"


# A failed step: the report exists, and no message gets out.

@pytest.mark.parametrize("step, function, reached", [
    ("schema", "check_segment", None),
    ("load", "load_projected_boxes", "schema"),
    ("joins", "check_joins", "load"),
    ("tables", "write_segment", "joins"),
    ("windows", "write_scene_window", "tables"),
    ("render", "ensure_rendered", "windows"),
    ("render", "write_clip", "windows"),
    ("checks", "check_velocity", "tracks"),
    ("contact_sheet", "write_contact_sheet", "nulls"),
])
def test_a_failing_step_writes_the_report_with_no_message(short_run, monkeypatch, capfd, step, function, reached):
    message = f"{OBJECT_IDS['lead']} in {PRIVATE_CONTEXT}"

    def explode(*args, **kwargs):
        raise RuntimeError(message)

    monkeypatch.setattr(run_stage1, function, explode)
    with pytest.raises(SystemExit) as exit_info:
        run_stage1.main(short_run.args)
    out, err = capfd.readouterr()

    assert exit_info.value.code == 1
    text = report_path(short_run.directory).read_text()
    assert OBJECT_IDS["lead"] not in out + err + text and PRIVATE_CONTEXT not in out + err + text
    report = json.loads(text)
    assert [(failure["step"], failure["exception"]) for failure in report["failures"]] == [(step, "RuntimeError")]
    assert re.fullmatch(r"run_stage1\.py:\d+", report["failures"][0]["location"])
    expected = SECTIONS[:SECTIONS.index(reached) + 1] if reached else ()
    assert set(report["sections"]) == set(expected)
    assert out.splitlines()[0].endswith("failed steps: 1")


def test_a_killed_rerun_leaves_no_report_that_reads_as_a_pass(short_run, monkeypatch):
    """SIGKILL, which hitting a memory limit sends, runs nothing, so the old report has to be gone already."""
    assert run_stage1.main(short_run.args) == 0
    path = report_path(short_run.directory)
    assert path.exists()

    seen = []
    check_segment = run_stage1.check_segment
    monkeypatch.setattr(run_stage1, "check_segment", lambda *args: seen.append(path.exists()) or check_segment(*args))
    assert run_stage1.main(short_run.args) == 0
    assert seen == [False] and path.exists()


def test_sigterm_still_writes_the_report(short_run, monkeypatch):
    """Slurm sends it at the time limit, and Python's default ends the process with nothing written."""
    monkeypatch.setattr(run_stage1, "ensure_rendered", lambda *args: os.kill(os.getpid(), signal.SIGTERM))
    with pytest.raises(SystemExit) as exit_info:
        run_stage1.main(short_run.args)
    assert exit_info.value.code == 1
    assert [(failure["step"], failure["exception"]) for failure in read_report(short_run.directory)["failures"]] == [
        ("render", "SystemExit")]


def test_outputs_are_private_whatever_umask_the_caller_has(short_run):
    """env.sh sets 077, but a run that skips it shouldn't lean on the parent directory's mode."""
    result = cli(*short_run.args, umask=0o022)
    assert result.returncode == 0
    outputs = short_run.tmp_path / "outputs"
    paths = [*outputs.rglob("*")]
    assert any(path.suffix == ".mp4" for path in paths) and any(path.suffix == ".ffconcat" for path in paths)
    assert not [path for path in paths if stat.S_IMODE(path.stat().st_mode) & 0o077]


def test_a_corrupt_file_fails_closed_in_a_real_process(short_run):
    """pyarrow's own message holds the path, and so the context name."""
    component_path(short_run.root, short_run.ref, LIDAR_BOX).write_bytes(b"not a parquet file")
    result = cli(*short_run.args)
    assert result.returncode == 1 and result.stderr == ""
    assert PRIVATE_CONTEXT not in result.stdout
    report = read_report(short_run.directory)
    assert report["failures"][0]["step"] == "schema" and report["failures"][0]["exception"] == "ArrowInvalid"
    assert PRIVATE_CONTEXT not in json.dumps(report)


def test_a_missing_component_stops_at_the_schema_check(short_run):
    component_path(short_run.root, short_run.ref, LIDAR_BOX).unlink()
    with pytest.raises(SystemExit):
        run_stage1.main(short_run.args)
    report = read_report(short_run.directory)
    assert report["sections"]["schema"]["status"] == "fail"
    assert report["sections"]["schema"]["components"][LIDAR_BOX]["missing"] == 1
    assert report["sections"]["schema"]["components"][VEHICLE_POSE]["missing"] == 0
    assert report["failures"][0]["step"] == "schema" and set(report["sections"]) == {"schema"}


def test_a_missing_column_is_reported_by_position_so_no_second_job_has_to_find_it(short_run):
    path = component_path(short_run.root, short_run.ref, LIDAR_BOX)
    table = pq.read_table(path)
    names = table.column_names
    position = REQUIRED_COLUMNS[LIDAR_BOX].index(col(LIDAR_BOX, "box.heading"))
    pq.write_table(table.rename_columns([name.replace("box.heading", "box.yaw") for name in names]), path)
    with pytest.raises(SystemExit):
        run_stage1.main(short_run.args)
    component = read_report(short_run.directory)["sections"]["schema"]["components"][LIDAR_BOX]
    assert (component["missing"], component["missing_at"]) == (1, [position])


def test_orphans_are_counted_before_they_stop_the_tables(short_run):
    path = component_path(short_run.root, short_run.ref, VEHICLE_POSE)
    pq.write_table(pq.read_table(path).slice(1), path)
    with pytest.raises(SystemExit):
        run_stage1.main(short_run.args)
    report = read_report(short_run.directory)
    assert report["failures"][0]["step"] == "tables"
    assert report["sections"]["joins"]["status"] == "fail"
    assert report["sections"]["joins"]["rows_without_pose"]["camera_image"] == 1


def test_a_camera_with_no_images_fails_closed(short_run):
    with pytest.raises(SystemExit):
        run_stage1.main([*short_run.args, "--cameras", "SIDE_LEFT"])
    report = read_report(short_run.directory)
    assert report["sections"]["joins"]["cameras_without_images"] == 1
    assert report["failures"][0]["step"] == "windows"


def test_outputs_that_cannot_be_written_exit_quietly(short_run, capfd):
    """No report is possible here, but a traceback would print the path, and so the context."""
    outputs = short_run.tmp_path / "outputs"
    outputs.write_text("a file where the directory should be")
    with pytest.raises(SystemExit) as exit_info:
        run_stage1.main(short_run.args)
    assert exit_info.value.code == 1
    assert capfd.readouterr() == ("", "")


def test_a_failed_check_finishes_the_run_and_exits_1(short_run, capfd):
    config = write_config(short_run.tmp_path / "strict.toml", short_run.root, short_run.tmp_path / "outputs",
                          ("residual_median_px = 5.0", "residual_median_px = -1.0"))
    assert run_stage1.main(["--config", str(config), "--context", PRIVATE_CONTEXT]) == 1
    report = read_report(short_run.directory)
    assert report["failures"] == [] and statuses(report)["residuals"] == "fail"
    assert (short_run.directory / "qa" / "contact_sheet.html").exists()  # review matters most on a failed run
    assert "1 fail" in capfd.readouterr().out


@pytest.mark.parametrize("cameras, no_data", [(["FRONT", "FRONT_LEFT"], 1), (["FRONT_LEFT"], 2)],
                         ids=["one_of_two_cameras", "the_only_camera"])
def test_a_camera_with_no_boxes_to_judge_is_no_data_and_not_a_failure(short_run, capfd, cameras, no_data):
    """Only the clamped car beside the ego is in FRONT_LEFT this early. Alone, it also leaves ego_velocity no FRONT."""
    assert run_stage1.main([*short_run.args, "--cameras", *cameras]) == 0
    report = read_report(short_run.directory)
    assert statuses(report)["residuals"] == "no_data"
    assert report["sections"]["residuals"]["cameras_without_boxes"] == 1
    assert f"{no_data} no_data" in capfd.readouterr().out
    assert (short_run.directory / "qa" / "contact_sheet.html").exists()


# Flags and config.

def test_an_interrupted_table_write_leaves_nothing_load_segment_will_take(short_run, monkeypatch):
    """A half-replaced set of tables would load without complaint, with new frames beside old ego rows."""
    assert run_stage1.main(short_run.args) == 0
    original = pq.write_table
    calls = itertools.count()

    def killed_after_the_first_table(table, path):
        if next(calls) == 1:
            raise OSError("job killed")
        original(table, path)

    monkeypatch.setattr(pq, "write_table", killed_after_the_first_table)
    with pytest.raises(SystemExit):
        run_stage1.main(short_run.args)
    monkeypatch.undo()
    with pytest.raises(FileNotFoundError):
        load_segment(short_run.directory)


def test_a_config_change_redraws_every_image(short_run):
    """Files keyed by timestamp and profile name would keep old pixels under a new preprocessing_version."""
    assert run_stage1.main(short_run.args) == 0
    image = next(short_run.directory.rglob("*__pass1.jpg"))
    assert cv2.imdecode(np.frombuffer(image.read_bytes(), np.uint8), cv2.IMREAD_COLOR).shape[1] == 1092

    config = write_config(short_run.tmp_path / "small.toml", short_run.root, short_run.tmp_path / "outputs",
                          ("max_long_edge = 1092", "max_long_edge = 568"))
    assert run_stage1.main(["--config", str(config), "--context", PRIVATE_CONTEXT]) == 0
    assert cv2.imdecode(np.frombuffer(image.read_bytes(), np.uint8), cv2.IMREAD_COLOR).shape[1] == 568
    assert read_report(short_run.directory)["sections"]["render"]["lidar_box"] > 0


def test_a_cameras_flag_counts_as_a_different_preprocessing(short_run):
    """The tables differ, so the provenance must."""
    run_stage1.main(short_run.args)
    only_front = read_report(short_run.directory)["preprocessing_version"]
    run_stage1.main([*short_run.args, "--cameras", "FRONT", "FRONT_RIGHT"])
    assert read_report(short_run.directory)["preprocessing_version"] != only_front
    assert load_segment(short_run.directory).meta["preprocessing_version"] != only_front


def test_a_repaired_image_is_read_again_because_the_damaged_one_was_never_kept(short_run):
    """The clean file is trusted once it exists, so a damaged one would outlive the repair."""
    path = component_path(short_run.root, short_run.ref, CAMERA_IMAGE)
    good = pq.read_table(path)
    image = col(CAMERA_IMAGE, "image")
    damaged = good.set_column(good.column_names.index(image), image, pa.array([b"not a jpeg"] * len(good), pa.binary()))
    pq.write_table(damaged, path)
    with pytest.raises(SystemExit):
        run_stage1.main(short_run.args)
    assert not list(short_run.directory.rglob("*__clean.jpg"))

    pq.write_table(good, path)
    assert run_stage1.main(short_run.args) == 0
    clean = next(short_run.directory.rglob("*__clean.jpg"))
    assert cv2.imdecode(np.frombuffer(clean.read_bytes(), np.uint8), cv2.IMREAD_COLOR) is not None


def test_a_zero_in_a_time_column_stays_out_of_the_report(short_run):
    """A filled-in zero turns one offset into a raw timestamp, and min and max would then carry it out of PACE."""
    path = component_path(short_run.root, short_run.ref, CAMERA_IMAGE)
    table = pq.read_table(path)
    name = col(CAMERA_IMAGE, "pose_timestamp")
    times = table[name].to_numpy().copy()
    times[np.flatnonzero(table[CAMERA].to_numpy() == CAMERA_IDS["FRONT"])[0]] = 0.0  # the camera this run reads
    pq.write_table(table.set_column(table.column_names.index(name), name, pa.array(times)), path)
    with pytest.raises(SystemExit):
        run_stage1.main(short_run.args)
    report = read_report(short_run.directory)
    assert [(failure["step"], failure["exception"]) for failure in report["failures"]] == [("checks", "ValueError")]
    assert "timing" not in report["sections"]
    numbers = [value for value in leaves(report["sections"]) if isinstance(value, (int, float))]
    assert max(abs(number) for number in numbers) < 1e12


def test_a_null_pose_timestamp_fails_at_the_windows_step(short_run):
    """pydantic writes NaN as null and a float field refuses null, so Stage 2 would have been the first to find out."""
    path = component_path(short_run.root, short_run.ref, CAMERA_IMAGE)
    table = pq.read_table(path)
    name = col(CAMERA_IMAGE, "pose_timestamp")
    pq.write_table(table.set_column(table.column_names.index(name), name, pa.nulls(len(table), pa.float64())), path)
    with pytest.raises(SystemExit):
        run_stage1.main(short_run.args)
    failures = read_report(short_run.directory)["failures"]
    assert [(failure["step"], failure["exception"]) for failure in failures] == [("windows", "ValidationError")]


def test_a_near_fast_frame_with_no_image_does_not_lose_the_contact_sheet(short_run):
    """Check 8 matters most on the segment with a camera hole."""
    path = component_path(short_run.root, short_run.ref, CAMERA_IMAGE)
    table = pq.read_table(path)
    assert run_stage1.main(short_run.args) == 0
    segment = load_segment(short_run.directory)
    picks = pick_review_frames(NO_RESIDUALS, segment)
    timestamp, camera = next((ts, cam) for ts, cam, reason in picks if reason == "near_fast")
    frames = table.to_pandas()
    keep = ~((frames[TIMESTAMP] == timestamp) & (frames[CAMERA] == camera))
    pq.write_table(table.filter(pa.array(keep.to_numpy())), path)
    assert run_stage1.main(short_run.args) == 1  # the hole fails check 2, and the rest of the run still finishes
    assert (short_run.directory / "qa" / "contact_sheet.html").exists()
    assert read_report(short_run.directory)["failures"] == []


def test_cameras_flag_overrides_the_config(short_run):
    assert run_stage1.main([*short_run.args, "--cameras", "FRONT", "FRONT_RIGHT"]) == 0
    assert [path.name.split("__")[2] for path in (short_run.directory / "windows").iterdir()] == ["FRONT+FRONT_RIGHT"]
    assert set(read_report(short_run.directory)["sections"]["timing"]["cameras"]) == {"front", "front_right"}
    assert (short_run.directory / "render" / "FRONT_RIGHT").is_dir()


def test_max_objects_comes_from_the_config(short_run):
    config = write_config(short_run.tmp_path / "few.toml", short_run.root, short_run.tmp_path / "outputs",
                          ("max_objects = 20", "max_objects = 2"))
    assert run_stage1.main(["--config", str(config), "--context", PRIVATE_CONTEXT]) == 0
    section = read_report(short_run.directory)["sections"]["windows"]
    assert section["objects"] == [2] and section["omitted_objects"][0] > 0


def test_a_rerun_reuses_the_render_cache(short_run):
    assert run_stage1.main(short_run.args) == 0
    assert read_report(short_run.directory)["sections"]["render"]["lidar_box"] > 0
    assert run_stage1.main(short_run.args) == 0
    render = read_report(short_run.directory)["sections"]["render"]
    assert (render["lidar_box"], render["camera_synced_box"]) == (0, 0)


# Check 6.

@pytest.fixture(scope="module")
def repro(tmp_path_factory) -> SimpleNamespace:
    base = tmp_path_factory.mktemp("repro")
    config = write_config(base / "stage1.toml", base / "no_such_root", base / "outputs")
    result = cli("--config", str(config), "--synthetic", "--repro-check")
    checked = base / "outputs" / "repro_check"
    return SimpleNamespace(result=result, outputs=base / "outputs", first=checked / "first" / CONTEXT,
                           second=checked / "second" / CONTEXT, report=read_report(checked / "first" / CONTEXT))


def test_repro_check_reports_a_match(repro):
    assert repro.result.returncode == 0 and repro.result.stderr == ""
    assert repro.report["sections"]["repro"] == {"status": "pass", "hashes_match": True}
    assert "repro: hashes match" in repro.result.stdout.splitlines()
    assert repro.result.stdout.splitlines()[-1] == f"report: {report_path(repro.first)}"


def test_repro_check_compares_every_kind_of_output(repro):
    hashes = output_hashes(repro.first)
    assert hashes == output_hashes(repro.second)
    paths = list(hashes)
    assert any(path.endswith("__clean.jpg") for path in paths) and any(path.endswith("__pass1.jpg") for path in paths)
    assert any(path.startswith("windows/") for path in paths)
    assert {"segment.json", "frames.parquet", "ego.parquet", "tracks.parquet"} <= set(paths)


def test_repro_check_reports_no_paths_or_hashes(repro):
    hashes = output_hashes(repro.first)
    text = report_path(repro.first).read_text() + repro.result.stdout
    assert not any(digest in text or path in text for path, digest in hashes.items())


def test_repro_check_leaves_the_ordinary_outputs_alone(repro):
    assert not (repro.outputs / CONTEXT).exists()


def test_repro_check_catches_a_difference(short_run, monkeypatch, capfd):
    build_segment_meta = run_stage1.build_segment_meta
    runs = itertools.count()
    monkeypatch.setattr(run_stage1, "build_segment_meta",
                        lambda *args: {**build_segment_meta(*args), "n_frames": next(runs)})
    assert run_stage1.main([*short_run.args, "--repro-check"]) == 1
    report = read_report(short_run.tmp_path / "outputs" / "repro_check" / "first" / PRIVATE_CONTEXT)
    assert report["sections"]["repro"] == {"status": "fail", "hashes_match": False}
    assert "repro: hashes differ" in capfd.readouterr().out


def test_repro_check_starts_from_empty_directories(short_run):
    """ensure_rendered trusts what's on disk, so old images would be reused rather than remade."""
    assert run_stage1.main([*short_run.args, "--repro-check"]) == 0
    second = short_run.tmp_path / "outputs" / "repro_check" / "second" / PRIVATE_CONTEXT
    old_image = next(second.rglob("*__pass1.jpg"))
    old_image.write_bytes(b"left over from an older run")
    assert run_stage1.main([*short_run.args, "--repro-check"]) == 0
    assert old_image.read_bytes() != b"left over from an older run"


def test_a_failed_repro_run_keeps_its_report(short_run, monkeypatch):
    runs = itertools.count()
    render = run_stage1.ensure_rendered

    def fail_in_the_second_run(*args):
        if args[1].directory.parent.name == "second" and next(runs) == 0:
            raise RuntimeError("no message")
        return render(*args)

    monkeypatch.setattr(run_stage1, "ensure_rendered", fail_in_the_second_run)
    with pytest.raises(SystemExit):
        run_stage1.main([*short_run.args, "--repro-check"])
    report = read_report(short_run.tmp_path / "outputs" / "repro_check" / "second" / PRIVATE_CONTEXT)
    assert report["failures"][0]["step"] == "render"


# The tables hold no filtering, and match the scene they came from. The synthetic segment is
# analytic, so these are exact answers: the labels only see the same scene through every transform.

def test_tables_keep_every_row(runs, segment):
    loaded = runs["three_cameras"].report["sections"]["load"]
    assert len(segment.ego) == loaded["poses"] == segment.meta["n_frames"] == 199
    assert len(segment.frames) == loaded["images"] == 3 * 199
    assert len(segment.tracks) == loaded["lidar_boxes"]


def test_tables_and_segment_json_match_the_spec(segment):
    assert list(segment.ego.columns) == EGO_COLUMNS
    assert list(segment.frames.columns) == FRAME_COLUMNS
    assert list(segment.tracks.columns) == TRACK_COLUMNS
    assert sorted(segment.meta) == sorted(META_KEYS)
    assert set(segment.meta["cameras"]) == {"FRONT", "FRONT_LEFT", "FRONT_RIGHT"}


def test_tables_stay_sorted_with_exact_keys(segment):
    for table, keys in ((segment.ego, [TIME]), (segment.frames, [TIME, "camera"]),
                        (segment.tracks, [TIME, "laser_object_id"])):
        assert table[TIME].dtype == np.int64
        assert table[keys].equals(table[keys].sort_values(keys, ignore_index=True))
        assert not table.duplicated(keys).any()


def test_transforms_are_lists_of_16(segment):
    for values in (segment.ego["world_from_vehicle"], segment.frames["world_from_image_vehicle"]):
        assert all(isinstance(value, list) and len(value) == 16 for value in values)


def test_segment_times_start_at_the_first_pose(segment):
    assert segment.meta["t0_micros"] == segment.ego[TIME].iloc[0]
    assert segment.ego["t_s"].iloc[0] == 0.0
    assert segment.meta["last_t_s"] == segment.ego["t_s"].iloc[-1]


@pytest.mark.parametrize("camera", ["FRONT", "FRONT_LEFT", "FRONT_RIGHT"])
def test_gaps_are_per_camera_and_flagged_at_the_two_dropped_frames(segment, camera):
    rows = segment.frames[segment.frames["camera_name"] == camera]
    assert np.isnan(rows["gap_before_ms"].iloc[0])
    assert rows["gap_before_ms"].iloc[1:].between(98.0, 205.0).all()
    assert rows.loc[rows["has_gap_before"], "gap_before_ms"].between(195.0, 205.0).sum() == 2


def test_capture_offsets_follow_each_camera(segment):
    medians = segment.frames.groupby("camera_name")["capture_offset_ms"].median()
    assert medians.round(1).to_dict() == {"FRONT": 50.0, "FRONT_LEFT": 37.5, "FRONT_RIGHT": 62.5}


def test_ego_motion_matches_the_drive(segment):
    ego = segment.ego
    t = ego["t_s"].to_numpy() + LABEL_OFFSET_S
    speed = 14.0 - 0.4 * t
    np.testing.assert_allclose(ego["speed_mps"], speed, atol=0.05)  # the ends are one-sided
    np.testing.assert_allclose(ego["yaw_rate_radps"], speed / 200.0, atol=0.002)
    np.testing.assert_allclose(ego["a_long_mps2"], -0.4, atol=0.03)  # every row, ends included
    np.testing.assert_allclose(ego["a_lat_mps2"], speed * speed / 200.0, atol=0.03)
    s = 14.0 * t - 0.2 * t * t
    np.testing.assert_allclose(ego["path_s_m"], s - s[0], atol=0.02)


@pytest.mark.parametrize("name, lateral_m, clearance_m", [
    ("beside", 3.7, 2.75), ("parked", -6.0, 5.05), ("parked_left", 11.0, 10.05), ("parked_right", -11.0, 10.05),
])
def test_lateral_offsets_are_positive_left_of_the_path(segment, name, lateral_m, clearance_m):
    rows = rows_of(segment, name)
    rows = rows[rows["on_driven_path"]]  # past the ego's last pose the path runs straight, and the road bends
    assert len(rows) > 0
    np.testing.assert_allclose(rows["lateral_offset_m"], lateral_m, atol=0.05)
    np.testing.assert_allclose(rows["path_clearance_m"], clearance_m, atol=0.1)


def test_an_object_on_the_path_has_no_clearance(segment):
    rows = rows_of(segment, "cut_in")
    assert (rows.loc[(rows["t_s"] > 12.5) & rows["on_driven_path"], "path_clearance_m"] == 0.0).all()
    assert (rows.loc[rows["t_s"] > 12.5, "on_driven_path"]).any()
    assert (rows.loc[rows["t_s"] < 7.5, "path_clearance_m"] > 2.5).all()


def test_a_lead_past_the_paths_end_says_its_offset_is_not_on_the_road(segment):
    """The lead stays in the ego's lane, yet on the straight extension it drifts 5 m off the bending road."""
    rows = rows_of(segment, "lead")
    late = rows[rows["t_s"] > 19.0]
    assert late["lateral_offset_m"].abs().min() > 1.0 and not late["on_driven_path"].any()
    assert (rows.loc[rows["on_driven_path"], "lateral_offset_m"].abs() < 0.05).all()


def test_every_sample_of_a_window_carries_the_flag(runs):
    samples = object_in(scene_windows(runs["three_cameras"])[3], "lead").samples
    assert {sample.on_driven_path for sample in samples} == {True, False}


def test_along_path_is_relative_to_the_ego_now(segment):
    rows = rows_of(segment, "beside")
    np.testing.assert_allclose(rows.loc[rows["on_driven_path"], "along_path_m"], 4.5, atol=0.05)
    assert np.isnan(rows["front_gap_m"]).all()  # its tail is behind the bumper
    assert (rows_of(segment, "lead")["front_gap_m"] > 30.0).all()


def test_a_parked_car_stays_put_in_every_frame_but_the_ego(segment):
    """Differencing raw vehicle-frame centers would give it the ego's yaw, as false lateral motion."""
    rows = rows_of(segment, "parked")
    assert rows["stationary"].all()
    assert rows[["pos_local_x_m", "pos_local_y_m"]].std().max() < 1e-6
    assert np.abs(rows[["v_abs_ego_x_mps", "v_abs_ego_y_mps"]].to_numpy()).max() < 0.02
    ego_speed = segment.ego.set_index(TIME).loc[rows[TIME], "speed_mps"].to_numpy()
    np.testing.assert_allclose(rows["v_rel_ego_x_mps"], -ego_speed, atol=0.5)


def test_velocity_falls_back_to_differences_when_labels_are_null(segment):
    rows = rows_of(segment, "lead")
    fallback = rows[rows["v_label_world_x_mps"].isna()]
    assert len(fallback) > 0 and set(fallback["v_source"]) == {"central_difference"}
    assert set(rows["v_source"]) == {"label", "central_difference"}
    assert np.isfinite(fallback[["v_world_x_mps", "v_abs_ego_x_mps", "a_world_x_mps2"]].to_numpy()).all()


def test_the_occluded_vehicle_is_flagged_for_its_hidden_span(segment):
    rows = rows_of(segment, "occluded")
    hidden = rows["t_s"] + LABEL_OFFSET_S
    assert rows["not_directly_observed"].equals((hidden >= 5.0) & (hidden < 9.0))


def test_camera_columns_follow_the_three_camera_run(segment):
    rows = rows_of(segment, "beside")
    assert all(cameras == [CAMERA_IDS["FRONT_LEFT"]] for cameras in rows["fov_cameras"])
    assert (rows["synced_camera"] == CAMERA_IDS["FRONT_LEFT"]).all() and rows["synced_center_x_m"].notna().all()


def test_a_label_in_no_camera_keeps_its_row_with_no_synced_box(segment):
    rows = rows_of(segment, "parked")
    behind = rows[[not cameras for cameras in rows["fov_cameras"]]]
    assert len(behind) > 0
    assert (behind["synced_camera"] == -1).all() and behind["synced_center_x_m"].isna().all()


def test_aliases_go_by_first_appearance_then_id(segment):
    first_rows = segment.tracks.drop_duplicates("laser_object_id")  # already sorted by time, then ID
    assert first_rows["alias"].tolist()[:4] == ["V001", "V002", "V003", "V004"]
    assert set(segment.tracks.loc[segment.tracks["type"] == "pedestrian", "alias"]) == {"P001"}
    assert set(segment.tracks.loc[segment.tracks["type"] == "sign", "alias"]) == {"S001"}
    assert segment.tracks.groupby("alias")["laser_object_id"].nunique().eq(1).all()


# Windows are views: they choose, the tables don't.

def test_a_window_keeps_an_object_no_chosen_camera_sees(runs):
    """A cut-in vehicle is outside FRONT before it arrives, and it matters most then."""
    beside = object_in(scene_windows(runs["front"])[0], "beside")
    assert all(sample.fov_cameras == [] for sample in beside.samples)


def test_a_window_leaves_out_signs_but_keeps_the_parked_car(runs):
    windows = scene_windows(runs["three_cameras"])
    tracked = {track.laser_object_id for window in windows for track in window.objects}
    assert OBJECT_IDS["sign"] not in tracked and OBJECT_IDS["parked"] in tracked


def test_samples_with_no_image_are_marked_as_context(runs):
    for window in scene_windows(runs["three_cameras"]):
        shown = {frame.frame_timestamp_micros for frame in window.frames}
        samples = [sample for track in window.objects for sample in track.samples]
        assert any(sample.context_only for sample in samples) and not all(sample.context_only for sample in samples)
        assert all(sample.context_only == (sample.frame_timestamp_micros not in shown) for sample in samples)


def test_context_reaches_past_the_images_and_stops_at_the_segment(runs):
    first, *_, last = scene_windows(runs["three_cameras"])
    assert first.ego[0].t_s == 0.0 and 6.9 < first.ego[-1].t_s < 7.1  # 2 s past a 5 s window
    assert 13.9 < last.ego[0].t_s < 14.1 and last.ego[-1].t_s <= last.window_end_s + 0.1


def test_a_track_summary_describes_the_whole_track(runs):
    window = scene_windows(runs["three_cameras"])[3]
    lead, parked = object_in(window, "lead"), object_in(window, "parked")
    assert lead.truncated_start and lead.truncated_end
    assert lead.first_t_s == 0.0 and lead.last_t_s > window.window_end_s + 2.0  # past what the window samples
    assert not parked.truncated_start and not parked.truncated_end  # labels stop at 75 m, both ways
    assert 0.0 < parked.first_t_s and parked.last_t_s < 19.0
    assert 73.5 < parked.range_at_first_seen_m <= 75.0  # the ego closes about 1.4 m a frame here


def test_windows_at_a_shared_grid_time_share_frames(runs):
    """2 FPS grids are anchored at the segment, so 4 s overlaps name the same files."""
    first, second, *_ = scene_windows(runs["front"])
    in_first = {frame.frame_timestamp_micros for frame in first.frames}
    in_second = {frame.frame_timestamp_micros for frame in second.frames}
    assert len(in_first & in_second) == 2  # 4.0 and 4.5


def segment_of(runs) -> SimpleNamespace:
    run = runs["three_cameras"]
    config = load_config(CONFIG_PATH, root=run.directory, outputs=run.directory.parent)
    return SimpleNamespace(segment=run.segment, config=config)


def many_on_the_path(segment, count: int):
    """`count` extra pedestrians in every frame, far ahead and astride the path, whose aliases all sort first."""
    crowd = [rows_of(segment, "lead").assign(laser_object_id=f"CROWD-{i:02d}", alias=f"P{100 + i}", type="pedestrian",
                                             path_clearance_m=0.0, center_x_m=60.0 + i, center_y_m=0.0)
             for i in range(count)]
    return replace(segment, tracks=pd.concat([segment.tracks, *crowd], ignore_index=True))


def test_the_nearest_of_a_crowd_on_the_path_is_kept_not_the_lowest_alias(runs):
    """Everything crossing the path ties at 0, and aliases count up by first appearance, so a late cut-in lost."""
    base = segment_of(runs)
    segment = many_on_the_path(base.segment, 24)
    cut_in = rows_of(segment, "cut_in")
    assert cut_in.loc[cut_in["t_s"].between(8.0, 13.0), "path_clearance_m"].min() == 0.0
    window = build_scene_window(segment, base.config, 8.0, 13.0, 2.0, (1,), "pass1")
    assert len(window.objects) == 20
    assert OBJECT_IDS["cut_in"] in {track.laser_object_id for track in window.objects}


def test_a_named_object_stays_whatever_its_rank(runs):
    base = segment_of(runs)
    parked = rows_of(base.segment, "parked")["alias"].iloc[0]
    plain = build_scene_window(base.segment, base.config, 8.0, 13.0, 5.0, (1,), "pass1", max_objects=1)
    assert parked not in {track.alias for track in plain.objects}

    kept = build_scene_window(base.segment, base.config, 8.0, 13.0, 5.0, (1,), "pass1", max_objects=1,
                              keep_aliases=[parked])
    assert [track.alias for track in kept.objects] == [plain.objects[0].alias, parked]  # on top of max_objects
    assert parked not in {omitted.alias for omitted in kept.omitted_objects}
    assert len(kept.omitted_objects) == len(plain.omitted_objects) - 1


def test_every_alias_an_image_draws_is_known_to_its_window(runs):
    """Images label every road user, and Stage 2's referential check has only the window to go on."""
    base = segment_of(runs)
    window = build_scene_window(base.segment, base.config, 8.0, 13.0, 2.0, (1, 2, 3), "pass1", max_objects=3)
    assert len(window.objects) == 3
    models = camera_models(base.segment.meta)
    drawn = {box.alias for frame in window.frames
             for box in boxes_for_image(base.segment, frame.frame_timestamp_micros, CAMERA_IDS[frame.camera],
                                        models[CAMERA_IDS[frame.camera]])}
    known = {track.alias for track in window.objects} | {omitted.alias for omitted in window.omitted_objects}
    assert drawn - {track.alias for track in window.objects} and drawn <= known


def test_an_omitted_object_can_be_traced_without_the_tables(runs):
    base = segment_of(runs)
    window = build_scene_window(base.segment, base.config, 8.0, 13.0, 2.0, (1,), "pass1", max_objects=1)
    omitted = window.omitted_objects[0]
    rows = base.segment.tracks[base.segment.tracks["alias"] == omitted.alias]
    assert (omitted.laser_object_id, omitted.type) == (rows["laser_object_id"].iloc[0], rows["type"].iloc[0])


def test_an_object_on_another_level_ranks_last_and_reads_as_none(runs):
    base = segment_of(runs)
    tracks = base.segment.tracks.copy()
    lead = tracks["laser_object_id"] == OBJECT_IDS["lead"]
    tracks.loc[lead, "path_clearance_m"] = np.nan
    window = build_scene_window(replace(base.segment, tracks=tracks), base.config, 8.0, 13.0, 2.0, (1,), "pass1")
    assert window.objects[-1].laser_object_id == OBJECT_IDS["lead"]
    assert {sample.path_clearance_m for sample in window.objects[-1].samples} == {None}


def test_a_window_says_which_cameras_and_profile_it_asked_for(runs):
    base = segment_of(runs)
    window = build_scene_window(base.segment, base.config, 8.0, 9.0, 2.0, (3, 1), "full")
    assert (window.cameras, window.profile) == (["FRONT", "FRONT_RIGHT"], "full")


@pytest.mark.parametrize("cameras", [(1, 4), ("FRONT",)], ids=["never_processed", "a_name_for_an_id"])
def test_a_camera_stage1_never_ran_is_named_and_not_dropped(runs, cameras):
    """Otherwise Pass 2 can't tell a side view that was empty from one it was never given."""
    base = segment_of(runs)
    with pytest.raises(ValueError, match="no frames for cameras"):
        build_scene_window(base.segment, base.config, 8.0, 9.0, 2.0, cameras, "pass1")


def test_windows_that_differ_get_their_own_files(runs, tmp_path):
    """Two Pass 2 candidates over one range can cite different frames, and each keeps its input for audit."""
    base = segment_of(runs)
    frames = base.segment.frames[base.segment.frames["camera"] == 1][TIME].tolist()
    common = (base.segment, base.config, 8.0, 11.0, 5.0, (1,), "full")
    calls = [{"force_include": frames[90:91]}, {"force_include": frames[91:92]}, {"max_objects": 2}, {"context_s": 0.0}]
    paths = [write_scene_window(build_scene_window(*common, **call), tmp_path) for call in calls]
    assert len(set(paths)) == 4
    assert all(path.name.startswith("8000_11000__5fps__FRONT__full__") for path in paths)


def test_the_same_window_writes_the_same_file(runs, tmp_path):
    base = segment_of(runs)
    window = build_scene_window(base.segment, base.config, 8.0, 11.0, 5.0, (1,), "full")
    assert write_scene_window(window, tmp_path) == write_scene_window(window, tmp_path)
    assert len(list((tmp_path / "windows").iterdir())) == 1


def test_a_window_names_the_lidar_components_stage4_will_need(runs):
    window = scene_windows(runs["front"])[0]
    assert [source.component for source in window.sources] == [*STAGE1_COMPONENTS, *LIDAR_COMPONENTS]


def test_nan_in_a_required_field_fails_when_the_window_is_built(runs):
    """pydantic would write it as null, and the file would then fail its own contract on read."""
    base = segment_of(runs)
    tracks = base.segment.tracks.copy()
    tracks.loc[tracks["laser_object_id"] == OBJECT_IDS["lead"], "center_x_m"] = np.nan
    with pytest.raises(ValueError):
        build_scene_window(replace(base.segment, tracks=tracks), base.config, 8.0, 13.0, 2.0, (1,), "pass1")


def test_every_window_written_reads_back_as_what_was_built(runs, tmp_path):
    base = segment_of(runs)
    window = build_scene_window(base.segment, base.config, 16.0, 19.0, 5.0, (1, 2, 3), "pass1")
    path = write_scene_window(window, tmp_path)
    assert SceneWindow.model_validate_json(path.read_text()) == window
    assert path.read_text().count("\n") == 1  # compact: indentation alone is a third of the file
