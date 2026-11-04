from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from agatha.columns import CAMERA_IDS
from agatha.config import load_config
from agatha.run_stage1 import process_segment
from agatha.scene_window import build_scene_window, write_scene_window
from agatha.store import load_segment
from agatha.synthetic import OBJECT_IDS

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "stage1.toml"
THREE_CAMERAS = ["FRONT", "FRONT_LEFT", "FRONT_RIGHT"]
FRONT = CAMERA_IDS["FRONT"]
ALL_IDS = tuple(CAMERA_IDS[name] for name in THREE_CAMERAS)


@pytest.fixture(scope="module")
def stage1(synthetic_root, tmp_path_factory):
    """Config and processed outputs for three cameras, shared because a run takes seconds."""
    root, ref = synthetic_root
    config = load_config(CONFIG_PATH, root=root, outputs=tmp_path_factory.mktemp("outputs"), cameras=THREE_CAMERAS)
    directory, _ = process_segment(config, ref)
    return config, load_segment(directory)


def window(stage1, start_s=8.0, end_s=13.0, fps=2.0, cameras=(FRONT,), **kwargs):
    config, segment = stage1
    return build_scene_window(segment, config, start_s, end_s, fps, cameras, "pass1", **kwargs)


def test_pass2_at_10_fps_takes_every_frame_of_every_camera(stage1):
    built = window(stage1, fps=10.0, cameras=ALL_IDS)
    ego = stage1[1].ego
    in_range = ego[(ego["t_s"] >= 7.95) & (ego["t_s"] < 12.95)]
    assert len(built.frames) == 3 * len(in_range)
    assert {frame.camera for frame in built.frames} == set(THREE_CAMERAS)
    assert built.achieved_fps == pytest.approx(10.0, rel=0.1)  # two frames are dropped, so a gap is possible


def test_a_tie_on_clearance_breaks_on_range_not_alias(stage1):
    config, segment = stage1
    # Everything on the path ties at 0: the alias order would put a far car first.
    tied = replace(segment, tracks=segment.tracks.assign(path_clearance_m=0.0))
    built = build_scene_window(tied, config, 8.0, 13.0, 2.0, (FRONT,), "pass1", max_objects=1)
    rows = tied.tracks[tied.tracks["t_s"].between(8.0, 13.0) & tied.tracks["type"].isin(["vehicle", "pedestrian"])]
    nearest = rows.assign(range_m=np.hypot(rows["center_x_m"], rows["center_y_m"])).sort_values("range_m").iloc[0]
    assert built.objects[0].alias == nearest["alias"]


def test_a_named_object_survives_the_cap_on_top_of_it(stage1):
    capped = window(stage1, max_objects=1)
    omitted = capped.omitted_objects[-1].alias
    named = window(stage1, max_objects=1, keep_aliases=(omitted,))
    assert [track.alias for track in named.objects] == [capped.objects[0].alias, omitted]
    assert omitted not in {item.alias for item in named.omitted_objects}


def test_kept_and_omitted_cover_each_road_user_once(stage1):
    built = window(stage1, max_objects=2)
    aliases = [track.alias for track in built.objects] + [item.alias for item in built.omitted_objects]
    assert len(aliases) == len(set(aliases))
    tracks = stage1[1].tracks
    in_images = tracks[tracks["frame_timestamp_micros"].isin([frame.frame_timestamp_micros for frame in built.frames])]
    assert set(aliases) == set(in_images.loc[in_images["type"] != "sign", "alias"])


def test_context_samples_are_exactly_the_ones_with_no_frame(stage1):
    built = window(stage1)
    shown = {frame.frame_timestamp_micros for frame in built.frames}
    samples = [sample for track in built.objects for sample in track.samples]
    assert any(sample.context_only for sample in samples)
    assert all(sample.context_only == (sample.frame_timestamp_micros not in shown) for sample in samples)
    assert min(sample.t_s for sample in samples) < 7.0 and max(sample.t_s for sample in samples) > 14.0


def test_a_cited_frame_can_be_forced_into_a_window_at_another_rate(stage1):
    frames = stage1[1].frames
    timestamps = frames.loc[frames["camera"] == FRONT, "frame_timestamp_micros"]
    grid = {frame.frame_timestamp_micros for frame in window(stage1, fps=5.0).frames}
    cited = next(int(ts) for ts in timestamps.iloc[90:110] if int(ts) not in grid)
    assert cited in {frame.frame_timestamp_micros for frame in window(stage1, fps=5.0, force_include=(cited,)).frames}


def test_a_camera_stage1_never_ran_is_refused_by_id(synthetic_root, tmp_path):
    root, ref = synthetic_root
    config = load_config(CONFIG_PATH, root=root, outputs=tmp_path)
    directory, _ = process_segment(config, ref)
    with pytest.raises(ValueError, match=str(CAMERA_IDS["FRONT_LEFT"])):
        build_scene_window(load_segment(directory), config, 8.0, 13.0, 2.0, (CAMERA_IDS["FRONT_LEFT"],), "pass1")


def test_an_unknown_optional_value_is_null_and_not_nan(stage1):
    built = window(stage1)
    sample = next(sample for track in built.objects for sample in track.samples if sample.front_gap_m is None)
    assert sample is not None
    assert "NaN" not in built.model_dump_json()


def test_references_are_relative_so_they_survive_a_host_change(stage1):
    for frame in window(stage1).frames:
        assert not Path(frame.clean_ref).is_absolute() and frame.clean_ref.endswith("__clean.jpg")
        assert not Path(frame.annotated_ref).is_absolute() and frame.annotated_ref.endswith("__pass1.jpg")


def test_building_is_pure_and_repeatable(stage1, tmp_path):
    directory = stage1[1].directory
    before = sorted(path for path in directory.rglob("*"))
    first, second = window(stage1, fps=5.0), window(stage1, fps=5.0)
    assert sorted(path for path in directory.rglob("*")) == before
    assert first.model_dump_json() == second.model_dump_json()
    assert write_scene_window(first, tmp_path).read_bytes() == write_scene_window(second, tmp_path).read_bytes()


def test_the_cut_in_stays_in_its_window_by_default(stage1):
    tracks = stage1[1].tracks
    cut_in = tracks.loc[tracks["laser_object_id"] == OBJECT_IDS["cut_in"], "alias"].iloc[0]
    assert cut_in in {track.alias for track in window(stage1).objects}
