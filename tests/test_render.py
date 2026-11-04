import base64
import re
import subprocess
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import cv2
import imageio_ffmpeg
import numpy as np
import pandas as pd
import pytest

from agatha import render
from agatha.columns import CAMERA_IDS
from agatha.config import RenderProfile, load_config
from agatha.geometry import (
    CameraModel, box_corners, project_camera_points, rot_z, se3, transform_points, wireframe,
)
from agatha.loader import (
    load_cameras, load_image_metadata, load_lidar_boxes, load_poses, load_projected_boxes, load_synced_boxes,
    read_jpegs,
)
from agatha.render import (
    ContactItem, DrawBox, boxes_for_image, draw_boxes, ensure_rendered, render_annotated, write_clip,
    write_contact_sheet,
)
from agatha.scene_window import build_scene_window
from agatha.store import (
    SegmentOutputs, build_frames_table, build_segment_meta, camera_models, load_segment, segment_directory,
    write_segment,
)
from agatha.tracks import build_ego_table, build_tracks_table

FRONT, FRONT_LEFT, FRONT_RIGHT = CAMERA_IDS["FRONT"], CAMERA_IDS["FRONT_LEFT"], CAMERA_IDS["FRONT_RIGHT"]
CAMERAS = (FRONT, FRONT_LEFT, FRONT_RIGHT)
CONFIG_PATH = Path(__file__).parents[1] / "configs" / "stage1.toml"
NO_BOXES = {"camera_synced_box": 0, "lidar_box": 0}
SYNCED_VALUES = ("center_x_m", "center_y_m", "center_z_m", "size_x_m", "size_y_m", "size_z_m", "heading_rad")

PINHOLE = CameraModel((1000.0, 1000.0, 960.0, 640.0, 0, 0, 0, 0, 0), np.eye(4), 1920, 1280)
# Neither transform is the identity and they disagree, so a box that takes the wrong one lands elsewhere.
MOUNTED = CameraModel(PINHOLE.intrinsic, se3(np.eye(3), [1.5, 0.0, 2.0]), 1920, 1280)
CAMERA_FROM_LABEL = se3(rot_z(0.02), [-1.0, 0.4, -2.0])
# The lens from Waymo's camera_model_test.cc, see NOTICE. Its distortion folds over off axis.
WAYMO_LENS = CameraModel((2055.55614936, 2055.55614936, 939.657469886, 641.072182194, 0.032316008498,
                          -0.321412482553, 0.000793258395371, -0.000625749354133, 0.0), np.eye(4), 1920, 1280)

PROFILE = RenderProfile("test", 1920, 3, 0.4, 92)


@pytest.fixture(scope="module")
def built(synthetic_root, tmp_path_factory):
    """The synthetic segment's tables, written and reloaded as a real run does."""
    root, ref = synthetic_root
    config = load_config(CONFIG_PATH, root=root, outputs=tmp_path_factory.mktemp("outputs"))
    poses = load_poses(root, ref)
    ego, path = build_ego_table(poses, config.motion)
    tracks = build_tracks_table(load_lidar_boxes(root, ref), load_synced_boxes(root, ref),
                                load_projected_boxes(root, ref, CAMERAS), poses, ego, path, config.motion,
                                config.rear_axle_to_front_m)
    frames = build_frames_table(load_image_metadata(root, ref, CAMERAS), poses, config.checks.gap_ms)
    directory = segment_directory(config, ref)
    meta = build_segment_meta(ref, poses, load_cameras(root, ref, CAMERAS), config)
    write_segment(directory, meta, frames, ego, tracks)
    return SimpleNamespace(root=root, ref=ref, config=config, segment=load_segment(directory),
                           pass1=config.profiles["pass1"], full=config.profiles["full"])


def fresh(built, directory: Path) -> SegmentOutputs:
    """The segment with an empty render cache, so tests can't see each other's files."""
    return replace(built.segment, directory=directory)


def scene_window(built, segment, start_s=6.0, end_s=8.0, fps=5.0, cameras=CAMERAS, profile="pass1"):
    return build_scene_window(segment, built.config, start_s, end_s, fps, cameras, profile)


def rendered(built, directory: Path, **window_args):
    """A fresh segment and a window, with the window's images on disk."""
    segment = fresh(built, directory)
    window = scene_window(built, segment, **window_args)
    ensure_rendered(window, segment, built.root, built.ref, built.config.profiles)
    return segment, window


def drawn_counts(segment, frames) -> dict:
    """What draw_boxes gives for these FrameRefs, to compare ensure_rendered's counts against."""
    models = camera_models(segment.meta)
    return NO_BOXES | Counter(box.draw_source for frame in frames
                              for box in boxes_for_image(segment, frame.frame_timestamp_micros,
                                                         CAMERA_IDS[frame.camera], models[CAMERA_IDS[frame.camera]]))


def track(**overrides) -> dict:
    """A tracks row: a car ahead whose label and synced boxes differ, so a test can tell which was used."""
    return {"alias": "V001", "type": "vehicle", "not_directly_observed": False, "synced_camera": -1,
            "center_x_m": 20.0, "center_y_m": 1.0, "center_z_m": 1.5,
            "size_x_m": 4.6, "size_y_m": 1.9, "size_z_m": 1.6, "heading_rad": 0.1,
            "synced_center_x_m": 21.0, "synced_center_y_m": 1.5, "synced_center_z_m": 1.4,
            "synced_size_x_m": 4.7, "synced_size_y_m": 2.0, "synced_size_z_m": 1.7, "synced_heading_rad": 0.2,
            **overrides}


def draw(*rows, cam=MOUNTED, camera_from_label_frame=CAMERA_FROM_LABEL) -> list[DrawBox]:
    return draw_boxes(pd.DataFrame(rows), FRONT, cam, camera_from_label_frame)


def box_ahead(box_type="vehicle", occluded=False, center=(12.0, 0.0, 0.0)) -> DrawBox:
    """A big box in the middle of a PINHOLE image, already in the camera frame."""
    return DrawBox("V001", box_type, box_corners(center, (4.0, 6.0, 4.0), 0.0), "lidar_box", occluded)


def flat_jpeg(level: int = 128) -> bytes:
    return cv2.imencode(".jpg", np.full((1280, 1920, 3), level, np.uint8))[1].tobytes()


def decoded(jpeg: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)


def line_color(boxes: list[DrawBox]) -> np.ndarray:
    """BGR at the middle of the first box's front right edge, drawn thick enough to survive JPEG."""
    profile = replace(PROFILE, line_width=9, jpeg_quality=100)
    image = decoded(render_annotated(flat_jpeg(60), boxes, PINHOLE, profile))
    u, v = np.round(wireframe(boxes[0].corners_camera, PINHOLE)[1][6]).astype(int)
    return image[v, u].astype(int)


def label_size(image: np.ndarray) -> tuple[int, int]:
    """Width and height of the black label tag on a flat image."""
    rows, cols = np.nonzero((image < 20).all(axis=2))
    return cols.max() - cols.min() + 1, rows.max() - rows.min() + 1


def presentation_times(clip: Path) -> np.ndarray:
    """Frame times in seconds, read back from the file. Sorted, because B-frames reorder packets."""
    dump = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-i", str(clip), "-map", "0:v", "-c", "copy",
                           "-f", "framecrc", "-"], capture_output=True, text=True, check=True).stdout
    numerator, denominator = re.search(r"#tb 0: (\d+)/(\d+)", dump).groups()
    pts = [int(line.split(",")[2]) for line in dump.splitlines() if not line.startswith("#")]
    return np.sort(pts) * int(numerator) / int(denominator)


def probe(clip: Path) -> str:
    return subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-i", str(clip)], capture_output=True, text=True).stderr


def video_frames(clip: Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(clip))
    frames = []
    while (frame := capture.read()[1]) is not None:
        frames.append(frame)
    return frames


def render_around_a_drop(built, directory: Path, profile=None):
    """FRONT at full rate on both sides of the first dropped frame."""
    profile = profile or built.pass1
    segment = fresh(built, directory)
    frames = segment.frames
    after_drop = frames.loc[(frames["camera"] == FRONT) & frames["has_gap_before"], "t_s"].iloc[0]
    window = scene_window(built, segment, after_drop - 0.55, after_drop + 0.55, 10.0, (FRONT,), profile.name)
    ensure_rendered(window, segment, built.root, built.ref, {profile.name: profile})
    return segment, window


def clip_of(segment, window, out_path: Path, profile_name="clean") -> Path:
    return write_clip(segment, FRONT, window.frames[0].t_s, window.frames[-1].t_s, out_path, profile_name)


def real_times(window) -> np.ndarray:
    return np.array([frame.frame_timestamp_micros for frame in window.frames])


def test_a_box_synced_to_this_camera_takes_the_extrinsic_alone():
    [box] = draw(track(synced_camera=FRONT))
    expected = transform_points(MOUNTED.camera_from_vehicle, box_corners([21.0, 1.5, 1.4], [4.7, 2.0, 1.7], 0.2))
    np.testing.assert_allclose(box.corners_camera, expected)
    assert box.draw_source == "camera_synced_box"


@pytest.mark.parametrize("synced_camera", [-1, FRONT_LEFT])  # no synced box, and one for another camera
def test_any_other_box_goes_through_camera_from_label(synced_camera):
    [box] = draw(track(synced_camera=synced_camera))
    expected = transform_points(CAMERA_FROM_LABEL, box_corners([20.0, 1.0, 1.5], [4.6, 1.9, 1.6], 0.1))
    np.testing.assert_allclose(box.corners_camera, expected)
    assert box.draw_source == "lidar_box"


@pytest.mark.parametrize("missing", [SYNCED_VALUES, *[(name,) for name in SYNCED_VALUES]], ids=["all", *SYNCED_VALUES])
def test_a_synced_row_missing_any_box_value_falls_back_to_the_label(missing):
    """Waymo says some camera_synced_box values are missing. A cut-in must not vanish from its evidence for that."""
    [box] = draw(track(synced_camera=FRONT, **{f"synced_{name}": np.nan for name in missing}))
    assert box.draw_source == "lidar_box"
    expected = transform_points(CAMERA_FROM_LABEL, box_corners([20.0, 1.0, 1.5], [4.6, 1.9, 1.6], 0.1))
    np.testing.assert_allclose(box.corners_camera, expected)


def test_in_image_includes_the_first_pixel_and_excludes_the_edge():
    assert render._in_image(np.array([[0.0, 0.0], [1920.0, 640.0], [960.0, 1280.0]]), PINHOLE).tolist() == [
        True, False, False]


def test_a_box_through_its_own_image_pose_lands_on_the_oracle_rectangle(built):
    """Wiring only, since the synthetic oracle shares this projection. Another image's pose would still show.

    Side-camera boxes are all cut off here, so this applies Waymo's clamp instead of skipping them.
    """
    # A label box is one mostly visible elsewhere, so forcing that path gives far more of them to compare.
    segment = replace(built.segment, tracks=built.segment.tracks.assign(synced_camera=-1))
    models = camera_models(segment.meta)
    aliases = dict(zip(segment.tracks["laser_object_id"], segment.tracks["alias"]))
    oracle = {(row.frame_timestamp_micros, row.camera, aliases[row.laser_object_id]): row
              for row in load_projected_boxes(built.root, built.ref, CAMERAS).itertuples()}
    errors = {camera: [] for camera in CAMERAS}
    for row in segment.frames.itertuples():
        cam = models[row.camera]
        for box in boxes_for_image(segment, row.frame_timestamp_micros, row.camera, cam):
            seen = oracle.get((row.frame_timestamp_micros, row.camera, box.alias))
            if seen is not None:  # none when no corner is valid, and a rectangle has nothing to go on
                proj = project_camera_points(box.corners_camera, cam)
                uv = proj.uv[proj.valid]
                low, high = np.clip([uv.min(axis=0), uv.max(axis=0)], 0.0, (cam.width, cam.height))
                errors[row.camera] += [*(low + high) / 2.0 - (seen.center_x, seen.center_y),
                                       *(high - low) - (seen.size_x, seen.size_y)]
    for camera in CAMERAS:
        assert len(errors[camera]) > 200
        np.testing.assert_allclose(errors[camera], 0.0, atol=1e-6)


def test_only_road_users_are_drawn():
    rows = [track(alias=alias, type=box_type) for alias, box_type in [
        ("V001", "vehicle"), ("P001", "pedestrian"), ("C001", "cyclist"), ("S001", "sign"), ("U001", "unknown")]]
    assert [(box.alias, box.type) for box in draw(*rows)] == [("V001", "vehicle"), ("P001", "pedestrian"),
                                                              ("C001", "cyclist")]


def test_a_sign_in_view_stays_undrawn(built):
    segment = built.segment
    tracks, cam = segment.tracks, camera_models(segment.meta)[FRONT]
    in_view = tracks[(tracks["type"] == "sign") & tracks["fov_cameras"].map(lambda cameras: FRONT in cameras)]
    drawn = {box.alias for timestamp in in_view["frame_timestamp_micros"].iloc[::10].tolist()
             for box in boxes_for_image(segment, timestamp, FRONT, cam)}
    assert drawn and not any(alias.startswith("S") for alias in drawn)


def test_a_car_beside_the_ego_draws_no_ghost_box():
    row = track(center_x_m=1.5, center_y_m=3.7, center_z_m=0.0, heading_rad=0.0)
    corners = box_corners([1.5, 3.7, 0.0], [4.6, 1.9, 1.6], 0.0)
    assert wireframe(corners, WAYMO_LENS)  # pieces survive the lens guard, but none of them is in the image
    assert draw(row, cam=WAYMO_LENS, camera_from_label_frame=np.eye(4)) == []


@pytest.mark.parametrize("center_x_m, center_y_m, drawn", [
    (15.0, 0.0, True),
    (15.0, 14.4, True),     # straddles the left edge of the image
    (1.0, 0.0, True),       # straddles the camera plane
    (15.0, 40.0, False),    # off to the left
    (-15.0, 0.0, False),    # behind the camera
])
def test_a_box_is_drawn_if_any_wireframe_point_is_inside(center_x_m, center_y_m, drawn):
    row = track(center_x_m=center_x_m, center_y_m=center_y_m, center_z_m=0.0)
    assert bool(draw(row, cam=PINHOLE, camera_from_label_frame=np.eye(4))) == drawn


def test_a_box_carries_its_alias_and_occlusion():
    [box] = draw(track(alias="P007", type="pedestrian", not_directly_observed=True))
    assert (box.alias, box.type, box.occluded) == ("P007", "pedestrian", True)
    assert draw(track())[0].occluded is False


@pytest.mark.parametrize("max_long_edge", [480, 960, 1092])
def test_polylines_scale_by_the_same_factor_as_the_image(max_long_edge):
    box = box_ahead()
    image = decoded(render_annotated(flat_jpeg(), [box], PINHOLE, replace(PROFILE, max_long_edge=max_long_edge)))
    scale = max_long_edge / 1920
    assert image.shape[:2] == (round(1280 * scale), max_long_edge)

    uv = np.vstack(wireframe(box.corners_camera, PINHOLE)) * scale
    rows, cols = np.nonzero(np.abs(image.astype(int) - 128).max(axis=2) > 40)
    # The label sits above the box, so the top edge can't be compared.
    np.testing.assert_allclose([cols.min(), cols.max(), rows.max()],
                               [uv[:, 0].min(), uv[:, 0].max(), uv[:, 1].max()], atol=4)


def test_a_profile_larger_than_the_image_does_not_upscale():
    box = box_ahead()
    image = decoded(render_annotated(flat_jpeg(), [box], PINHOLE, replace(PROFILE, max_long_edge=4000)))
    assert image.shape[:2] == (1280, 1920)
    rows, cols = np.nonzero(np.abs(image.astype(int) - 128).max(axis=2) > 40)
    assert abs(cols.max() - np.vstack(wireframe(box.corners_camera, PINHOLE))[:, 0].max()) < 4


def test_downscaling_averages_rather_than_aliases():
    # One bright column in three, at exactly a third of the size. Bilinear reads one column of each three.
    stripes = np.zeros((1280, 1920, 3), np.uint8)
    stripes[:, ::3] = 255
    jpeg = cv2.imencode(".jpg", stripes, [cv2.IMWRITE_JPEG_QUALITY, 100])[1].tobytes()
    image = decoded(render_annotated(jpeg, [], PINHOLE, replace(PROFILE, max_long_edge=640)))
    assert image.mean() == pytest.approx(255 / 3, abs=3) and image.std() < 5


def test_label_size_comes_from_the_profile_not_the_image_scale():
    large = decoded(render_annotated(flat_jpeg(), [box_ahead()], PINHOLE, PROFILE))
    small = decoded(render_annotated(flat_jpeg(), [box_ahead()], PINHOLE, replace(PROFILE, max_long_edge=480)))
    assert label_size(small) == pytest.approx(label_size(large), abs=2)
    bigger = decoded(render_annotated(flat_jpeg(), [box_ahead()], PINHOLE, replace(PROFILE, font_scale=1.2)))
    assert label_size(bigger)[1] > 2 * label_size(large)[1]


def test_the_label_is_the_alias():
    short = decoded(render_annotated(flat_jpeg(), [box_ahead()], PINHOLE, PROFILE))
    long = decoded(render_annotated(flat_jpeg(), [box_ahead()._replace(alias="V001V001")], PINHOLE, PROFILE))
    assert label_size(long)[0] > 1.5 * label_size(short)[0]


@pytest.mark.parametrize("center", [(12.0, -16.0, 7.0), (12.0, 13.0, 8.0)])  # off the top right, off the top left
def test_a_label_stays_whole_at_the_image_edge(center):
    box = box_ahead(center=center)
    image = decoded(render_annotated(flat_jpeg(), [box], PINHOLE, PROFILE))
    whole = decoded(render_annotated(flat_jpeg(), [box_ahead()], PINHOLE, PROFILE))
    assert label_size(image) == pytest.approx(label_size(whole), abs=1)


def test_boxes_are_colored_by_type_and_gray_when_occluded():
    blue, green, red = line_color([box_ahead("vehicle")])
    assert blue > red + 100
    blue, green, red = line_color([box_ahead("pedestrian")])
    assert red > blue + 100 and red > green + 100
    blue, green, red = line_color([box_ahead("cyclist")])
    assert min(green, red) > blue + 100

    for box_type in ("vehicle", "pedestrian", "cyclist"):
        color = line_color([box_ahead(box_type, occluded=True)])
        assert np.ptp(color) < 15 and color.min() > 120  # gray, and well above the background's 60


def test_the_nearest_box_is_drawn_last():
    near = box_ahead("pedestrian")
    far = near._replace(type="vehicle", corners_camera=near.corners_camera * 2.0)  # same pixels, twice the depth
    np.testing.assert_array_equal(line_color([near, far]), line_color([near]))
    np.testing.assert_array_equal(line_color([far, near]), line_color([near]))


def test_clean_images_are_byte_identical_to_the_source(built, tmp_path):
    segment, window = rendered(built, tmp_path)
    assert len({frame.camera for frame in window.frames}) == 3
    for frame in window.frames:
        source = read_jpegs(built.root, built.ref, CAMERA_IDS[frame.camera], [frame.frame_timestamp_micros])
        assert (tmp_path / frame.clean_ref).read_bytes() == source[frame.frame_timestamp_micros]


def test_annotated_images_follow_the_profile_and_leave_the_clean_one_alone(built, tmp_path):
    _, window = rendered(built, tmp_path, cameras=(FRONT,))
    frame = window.frames[0]
    assert decoded((tmp_path / frame.annotated_ref).read_bytes()).shape == (728, 1092, 3)
    assert decoded((tmp_path / frame.clean_ref).read_bytes()).shape == (1280, 1920, 3)


def test_jpeg_quality_comes_from_the_profile():
    noise = np.random.default_rng(0).integers(0, 255, (1280, 1920, 3), np.uint8)
    jpeg = cv2.imencode(".jpg", noise, [cv2.IMWRITE_JPEG_QUALITY, 100])[1].tobytes()
    low, high = (render_annotated(jpeg, [], PINHOLE, replace(PROFILE, jpeg_quality=quality)) for quality in (30, 90))
    assert len(low) < len(high) / 2


def test_ensure_rendered_reports_what_it_drew_then_reuses_every_file(built, tmp_path):
    segment = fresh(built, tmp_path)
    window = scene_window(built, segment, 6.0, 12.0)
    first = ensure_rendered(window, segment, built.root, built.ref, built.config.profiles)
    assert first == drawn_counts(segment, window.frames)
    assert min(first.values()) > 0

    files = {path: path.stat().st_mtime_ns for path in tmp_path.rglob("*.jpg")}
    assert len(files) == 2 * len(window.frames)
    # A root with nothing under it proves that no JPEG gets read.
    assert ensure_rendered(window, segment, tmp_path / "no_dataset", built.ref, built.config.profiles) == NO_BOXES
    assert {path: path.stat().st_mtime_ns for path in tmp_path.rglob("*.jpg")} == files


def test_a_second_profile_reuses_the_clean_files(built, tmp_path):
    segment, _ = rendered(built, tmp_path)
    clean = {path: path.stat().st_mtime_ns for path in tmp_path.rglob("*__clean.jpg")}

    window = scene_window(built, segment, profile="full")
    counts = ensure_rendered(window, segment, built.root, built.ref, built.config.profiles)

    assert counts == drawn_counts(segment, window.frames)
    assert {path: path.stat().st_mtime_ns for path in tmp_path.rglob("*__clean.jpg")} == clean
    assert decoded((tmp_path / window.frames[0].annotated_ref).read_bytes()).shape == (1280, 1920, 3)


def test_rates_share_frames_and_each_camera_is_read_once(built, tmp_path, monkeypatch):
    reads = []
    read_jpegs_for_real = render.read_jpegs

    def counting_read(root, ref, camera_id, timestamps):
        reads.append((camera_id, list(timestamps)))
        return read_jpegs_for_real(root, ref, camera_id, timestamps)

    monkeypatch.setattr(render, "read_jpegs", counting_read)
    segment, five_fps = rendered(built, tmp_path)
    assert sorted(camera_id for camera_id, _ in reads) == sorted(CAMERAS)

    reads.clear()
    two_fps = scene_window(built, segment, fps=2.0)
    ensure_rendered(two_fps, segment, built.root, built.ref, built.config.profiles)

    # 2 and 5 FPS grids meet only once a second, so most of the 2 FPS frames are new.
    done = {(frame.camera, frame.frame_timestamp_micros) for frame in five_fps.frames}
    new = [frame for frame in two_fps.frames if (frame.camera, frame.frame_timestamp_micros) not in done]
    assert 0 < len(new) < len(two_fps.frames)
    assert sorted(camera_id for camera_id, _ in reads) == sorted(CAMERAS)
    assert sorted(ts for _, timestamps in reads for ts in timestamps) == sorted(
        frame.frame_timestamp_micros for frame in new)


def test_only_lost_files_come_back(built, tmp_path):
    segment, window = rendered(built, tmp_path)
    first, second = window.frames[:2]

    profiles = built.config.profiles
    (tmp_path / first.annotated_ref).unlink()
    assert ensure_rendered(window, segment, built.root, built.ref, profiles) == drawn_counts(segment, [first])

    (tmp_path / second.clean_ref).unlink()
    assert ensure_rendered(window, segment, built.root, built.ref, profiles) == NO_BOXES
    assert (tmp_path / second.clean_ref).exists()


def test_the_window_names_the_profile_so_pixels_cannot_land_under_another(built, tmp_path):
    """A pass1 window can't be rendered at full size and leave that in the cache under the pass1 name."""
    segment = fresh(built, tmp_path)
    window = scene_window(built, segment, cameras=(FRONT,))
    ensure_rendered(window, segment, built.root, built.ref, {"pass1": built.pass1, "full": built.full})
    assert decoded((tmp_path / window.frames[0].annotated_ref).read_bytes()).shape == (728, 1092, 3)


def test_a_damaged_image_stops_before_it_is_kept_as_clean(built, tmp_path, monkeypatch):
    """A clean file is trusted once it exists, so bytes that can't decode must never become one."""
    segment = fresh(built, tmp_path)
    window = scene_window(built, segment, cameras=(FRONT,))
    monkeypatch.setattr(render, "read_jpegs", lambda *args: dict.fromkeys(args[3], b"not a jpeg"))
    with pytest.raises(AttributeError):
        ensure_rendered(window, segment, built.root, built.ref, built.config.profiles)
    assert not list(tmp_path.rglob("*.jpg"))


def test_a_killed_write_leaves_nothing_that_looks_finished(built, tmp_path, monkeypatch):
    segment = fresh(built, tmp_path)
    window = scene_window(built, segment, cameras=(FRONT,))
    write_for_real = Path.write_bytes

    def killed_halfway(path, data):
        write_for_real(path, data[:100])
        raise OSError("job killed")

    monkeypatch.setattr(Path, "write_bytes", killed_halfway)
    with pytest.raises(OSError):
        ensure_rendered(window, segment, built.root, built.ref, built.config.profiles)
    monkeypatch.undo()
    assert not list(tmp_path.rglob("*.jpg"))

    ensure_rendered(window, segment, built.root, built.ref, built.config.profiles)
    frame = window.frames[0]
    source = read_jpegs(built.root, built.ref, FRONT, [frame.frame_timestamp_micros])
    assert (tmp_path / frame.clean_ref).read_bytes() == source[frame.frame_timestamp_micros]
    assert not list(tmp_path.rglob("*.partial"))


def test_two_renders_give_identical_bytes(built, tmp_path):
    def render_into(directory: Path) -> dict:
        rendered(built, directory)
        return {path.relative_to(directory): path.read_bytes() for path in directory.rglob("*.jpg")}

    first, second = render_into(tmp_path / "first"), render_into(tmp_path / "second")
    assert first and first == second


def test_rendering_prints_nothing(built, tmp_path, capfd):
    segment, window = rendered(built, tmp_path, cameras=(FRONT,))
    clip_of(segment, window, tmp_path / "clip.mp4")
    assert capfd.readouterr() == ("", "")


def test_clip_times_match_real_timestamps_across_a_dropped_frame(built, tmp_path):
    segment, window = render_around_a_drop(built, tmp_path)
    real = real_times(window)
    assert np.diff(real).max() > 150_000

    times = presentation_times(clip_of(segment, window, tmp_path / "clips" / "drop.mp4"))

    # A fixed rate would flatten the gap, and ffmpeg's default image rate rounds every frame to 40 ms.
    np.testing.assert_allclose(times[:len(real)], (real - real[0]) / 1e6, atol=2e-6)
    # The last image is listed twice, to close its duration.
    assert len(times) == len(real) + 1
    assert times[-1] - times[-2] == pytest.approx(0.1, abs=2e-6)


def test_clip_script_holds_each_frame_until_the_next_timestamp(built, tmp_path):
    segment, window = render_around_a_drop(built, tmp_path)
    clip = clip_of(segment, window, tmp_path / "clips" / "drop.mp4")

    script = clip.with_suffix(".ffconcat").read_text()
    durations = [float(value) for value in re.findall(r"^duration (\S+)$", script, re.MULTILINE)]
    np.testing.assert_allclose(durations, [*np.diff(real_times(window)) / 1e6, 0.1], atol=1e-6)
    assert all(Path(path).is_absolute() and Path(path).exists()
               for path in re.findall(r"^file '(.+)'$", script, re.MULTILINE))


def test_clip_is_h264_yuv420p_and_plays_the_real_images(built, tmp_path):
    segment, window = render_around_a_drop(built, tmp_path)
    clip = clip_of(segment, window, tmp_path / "clip.mp4")

    stream = next(line for line in probe(clip).splitlines() if "Video:" in line)
    # Not yuvj420p. Whether the range also prints as tv varies by ffmpeg build, so the pixel check
    # below is what proves it: full-range data read as limited would come out visibly stretched.
    assert "h264" in stream and "yuv420p(" in stream and "1920x1280" in stream
    frames = video_frames(clip)
    assert len(frames) == len(window.frames) + 1
    source = decoded((tmp_path / window.frames[3].clean_ref).read_bytes())
    assert np.abs(frames[3].astype(int) - source).mean() < 3


def test_clip_length_includes_the_last_frame(built, tmp_path):
    segment, window = render_around_a_drop(built, tmp_path)
    hours, minutes, seconds = re.search(r"Duration: (\d+):(\d+):([\d.]+)", probe(
        clip_of(segment, window, tmp_path / "clip.mp4"))).groups()
    span_s = (real_times(window)[-1] - real_times(window)[0]) / 1e6
    assert 3600 * int(hours) + 60 * int(minutes) + float(seconds) == pytest.approx(span_s + 0.1, abs=0.011)


def test_clip_of_an_annotated_profile_with_odd_dimensions_comes_out_even(built, tmp_path):
    odd = RenderProfile("odd", 1093, 2, 0.5, 90)
    segment, window = render_around_a_drop(built, tmp_path, odd)
    assert decoded((tmp_path / window.frames[0].annotated_ref).read_bytes()).shape == (729, 1093, 3)
    assert video_frames(clip_of(segment, window, tmp_path / "clip.mp4", "odd"))[0].shape == (728, 1092, 3)


def test_clip_keeps_to_its_time_range(built, tmp_path):
    segment, window = render_around_a_drop(built, tmp_path)
    inner = window.frames[2:5]
    times = presentation_times(write_clip(segment, FRONT, inner[0].t_s, inner[-1].t_s, tmp_path / "clip.mp4"))
    assert len(times) == len(inner) + 1


def test_clip_leaves_out_frames_that_were_never_rendered(built, tmp_path):
    segment, window = rendered(built, tmp_path, fps=2.0, cameras=(FRONT,))
    times = presentation_times(clip_of(segment, window, tmp_path / "clip.mp4"))
    real = real_times(window)
    np.testing.assert_allclose(times[:len(real)], (real - real[0]) / 1e6, atol=2e-6)
    assert np.diff(times[:len(real)]).min() > 0.35


def test_clip_handles_a_path_with_a_space(built, tmp_path):
    segment, window = render_around_a_drop(built, tmp_path / "Application Support")
    assert len(video_frames(clip_of(segment, window, tmp_path / "my clips" / "clip.mp4"))) == len(window.frames) + 1


def test_clip_from_a_relative_output_directory(built, tmp_path, monkeypatch):
    # ffmpeg reads a concat script's paths relative to the script, not to the working directory.
    monkeypatch.chdir(tmp_path)
    segment, window = render_around_a_drop(built, Path("outputs"))
    assert len(video_frames(clip_of(segment, window, Path("clips") / "clip.mp4"))) == len(window.frames) + 1


def test_clip_does_not_claim_the_level_of_its_microsecond_timing(built, tmp_path):
    segment, window = render_around_a_drop(built, tmp_path)
    data = clip_of(segment, window, tmp_path / "clip.mp4").read_bytes()
    assert data[data.index(b"avcC") + 7] == 50  # x264 would take 6.2 from the frame rate it thinks it sees


def test_clip_with_nothing_rendered_says_so_without_naming_anything(built, tmp_path):
    with pytest.raises(ValueError) as error:
        write_clip(fresh(built, tmp_path), FRONT, 0.0, 1.0, tmp_path / "clip.mp4")
    assert str(tmp_path) not in str(error.value) and built.ref.context_name not in str(error.value)
    assert not (tmp_path / "clip.mp4").exists()


def test_a_failed_encode_keeps_paths_and_ffmpeg_output_out_of_sight(built, tmp_path, capfd):
    directory = tmp_path / built.ref.context_name
    segment, window = rendered(built, directory, cameras=(FRONT,))
    with pytest.raises(RuntimeError) as error:
        clip_of(segment, window, directory / "clip.not_a_format")  # ffmpeg's real complaint quotes this path
    assert built.ref.context_name not in str(error.value)
    assert capfd.readouterr() == ("", "")


def test_contact_sheet_embeds_images_and_references_nothing_outside(tmp_path):
    jpegs = [flat_jpeg(40), flat_jpeg(200)]
    path = write_contact_sheet(tmp_path / "qa" / "contact_sheet.html",
                               [ContactItem(jpegs[0], "FRONT random"), ContactItem(jpegs[1], "FRONT worst_residual")])
    page = path.read_text()

    sources = re.findall(r'src="([^"]*)"', page)
    assert all(source.startswith("data:image/jpeg;base64,") for source in sources)
    assert [base64.b64decode(source.split(",", 1)[1]) for source in sources] == jpegs
    assert decoded(base64.b64decode(sources[1].split(",", 1)[1])).shape == (1280, 1920, 3)
    assert "FRONT random" in page and "FRONT worst_residual" in page
    without_images = re.sub(r'src="[^"]*"', "", page)
    assert not re.search(r"https?:|file:|href|url\(|@import|<script|<link|<iframe|src=", without_images)
    assert str(tmp_path) not in without_images


def test_contact_sheet_escapes_captions(tmp_path):
    page = write_contact_sheet(tmp_path / "sheet.html", [ContactItem(flat_jpeg(), "<img src=x> & more")]).read_text()
    assert "&lt;img src=x&gt; &amp; more" in page
    assert "<img src=x>" not in page
