import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from agatha.columns import (
    CAMERA, CAMERA_CALIBRATION, CAMERA_IMAGE, LIDAR_BOX, OBJECT, OPTIONAL_COLUMNS, PROJECTED_BOX,
    REQUIRED_COLUMNS, SEGMENT, STAGE1_COMPONENTS, SYNCED_BOX, TIMESTAMP, VEHICLE_POSE, col,
)
from agatha.geometry import rot_z, se3
from agatha.loader import (
    SegmentRef, check_segment, component_path, load_cameras, load_image_metadata, load_lidar_boxes,
    load_poses, load_projected_boxes, load_synced_boxes, read_jpegs,
)

REF = SegmentRef("training", "synthetic_loader")
T0 = 1_550_083_467_369_629
IMAGE = col(CAMERA_IMAGE, "image")
SPEED = [col(LIDAR_BOX, f"speed.{a}") for a in "xyz"]
ACCEL = [col(LIDAR_BOX, f"acceleration.{a}") for a in "xyz"]


def _default(name: str, n: int) -> pa.Array:
    """Row numbers in the column's real Arrow type (see docs/research/2026-11-01/schema.md)."""
    rows = np.arange(n)
    if name == TIMESTAMP:
        return pa.array(T0 + rows * 100_000, type=pa.int64())
    if name in (SEGMENT, OBJECT):
        return pa.array([f"id{i}" for i in rows], type=pa.string())
    if name == IMAGE:
        return pa.array([b"jpeg%d" % i for i in rows], type=pa.binary())
    if name.endswith(".transform"):
        return pa.array([np.eye(4).ravel().tolist()] * n, type=pa.list_(pa.float64(), 16))
    if "linear_velocity" in name:
        return pa.array(rows.astype(np.float32), type=pa.float32())
    if name.endswith(("_name", ".type", "rolling_shutter_direction")):
        return pa.array(rows.astype(np.int8), type=pa.int8())
    if name.endswith(("_in_box", ".width", ".height")):
        return pa.array(rows, type=pa.int64())
    return pa.array(rows.astype(np.float64), type=pa.float64())


def _write(root, tag, rows: dict, drop=(), arrays=None) -> None:
    """Write a `tag` file: `rows` in each column's real type, defaults for the rest, then `arrays` as given."""
    n = len(next(iter(rows.values())))
    columns = {name: _default(name, n) for name in [*REQUIRED_COLUMNS[tag], *OPTIONAL_COLUMNS.get(tag, [])]}
    columns |= {name: pa.array(values, type=columns[name].type) for name, values in rows.items()}
    columns = {name: values for name, values in columns.items() if name not in drop} | (arrays or {})
    path = component_path(root, REF, tag)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(columns), path)


def _spy_on_read_table(monkeypatch) -> list[dict]:
    """Record the keyword arguments of each pq.read_table call."""
    read_table, calls = pq.read_table, []

    def spy(path, **kwargs):
        calls.append(kwargs)
        return read_table(path, **kwargs)

    monkeypatch.setattr(pq, "read_table", spy)
    return calls


def test_component_path_layout(tmp_path):
    expected = tmp_path / "training" / "lidar_box" / "synthetic_loader.parquet"
    assert component_path(tmp_path, REF, LIDAR_BOX) == expected


@pytest.mark.parametrize("transform_type", [pa.list_(pa.float64(), 16), pa.list_(pa.float64())])
def test_poses_sort_shuffled_rows_and_read_row_major(tmp_path, transform_type):
    transforms = pa.array([se3(rot_z(0.3), [i, 5.0, 7.0]).ravel().tolist() for i in (2.0, 0.0, 1.0)],
                          type=transform_type)
    _write(tmp_path, VEHICLE_POSE, {TIMESTAMP: [T0 + 200, T0, T0 + 100]},
           arrays={col(VEHICLE_POSE, "world_from_vehicle.transform"): transforms})
    poses = load_poses(tmp_path, REF)
    assert poses.timestamps.tolist() == [T0, T0 + 100, T0 + 200]
    assert poses.timestamps.dtype == np.int64
    assert poses.world_from_vehicle.dtype == np.float64 and poses.world_from_vehicle.shape == (3, 4, 4)
    # Row-major: the translation sits in the last column, and it follows its timestamp.
    np.testing.assert_allclose(poses.world_from_vehicle[:, :3, 3], [[0, 5, 7], [1, 5, 7], [2, 5, 7]])
    np.testing.assert_allclose(poses.world_from_vehicle[:, 3], [[0, 0, 0, 1]] * 3)


def test_boxes_sort_by_time_then_object_and_rows_stay_whole(tmp_path):
    rows = {TIMESTAMP: [T0 + 9, T0, T0 + 9, T0], OBJECT: ["a", "b", "b", "a"]}
    _write(tmp_path, LIDAR_BOX, rows | {col(LIDAR_BOX, "box.heading"): [2.0, 1.0, 3.0, 0.0]})
    _write(tmp_path, SYNCED_BOX, rows | {col(SYNCED_BOX, "camera_synced_box.heading"): [2.0, 1.0, 3.0, 0.0]})
    for df in (load_lidar_boxes(tmp_path, REF), load_synced_boxes(tmp_path, REF)):
        assert df["frame_timestamp_micros"].tolist() == [T0, T0, T0 + 9, T0 + 9]
        assert df["laser_object_id"].tolist() == ["a", "b", "a", "b"]
        assert df["heading"].tolist() == [0.0, 1.0, 2.0, 3.0]
        assert df.index.tolist() == [0, 1, 2, 3]


def test_camera_tables_sort_with_camera_before_object(tmp_path):
    _write(tmp_path, CAMERA_IMAGE, {TIMESTAMP: [T0 + 9, T0, T0], CAMERA: [1, 3, 1]})
    meta = load_image_metadata(tmp_path, REF, [1, 3])
    assert list(zip(meta["frame_timestamp_micros"], meta["camera"])) == [(T0, 1), (T0, 3), (T0 + 9, 1)]

    _write(tmp_path, PROJECTED_BOX, {TIMESTAMP: [T0, T0, T0 + 9], CAMERA: [3, 1, 1], OBJECT: ["a", "b", "a"]})
    boxes = load_projected_boxes(tmp_path, REF, [1, 3])
    assert list(zip(boxes["frame_timestamp_micros"], boxes["camera"], boxes["laser_object_id"])) == [
        (T0, 1, "b"), (T0, 3, "a"), (T0 + 9, 1, "a")]


def test_tidy_names_match_the_contract(tmp_path):
    for tag in (CAMERA_IMAGE, LIDAR_BOX, SYNCED_BOX, PROJECTED_BOX):
        _write(tmp_path, tag, {TIMESTAMP: [T0]})
    assert load_lidar_boxes(tmp_path, REF).columns.tolist() == [
        "frame_timestamp_micros", "laser_object_id", "center_x", "center_y", "center_z",
        "size_x", "size_y", "size_z", "heading", "type", "num_lidar_points", "num_top_lidar_points",
        "speed_x", "speed_y", "speed_z", "accel_x", "accel_y", "accel_z"]
    assert load_synced_boxes(tmp_path, REF).columns.tolist() == [
        "frame_timestamp_micros", "laser_object_id", "most_visible_camera", "center_x", "center_y", "center_z",
        "size_x", "size_y", "size_z", "heading"]
    assert load_projected_boxes(tmp_path, REF, [0]).columns.tolist() == [
        "frame_timestamp_micros", "camera", "laser_object_id", "center_x", "center_y", "size_x", "size_y", "type"]
    assert load_image_metadata(tmp_path, REF, [0]).columns.tolist() == [
        "frame_timestamp_micros", "camera", "pose_timestamp_s", "trigger_time_s", "readout_done_time_s",
        "shutter_s", "world_from_image_vehicle",
        "linear_velocity_x_mps", "linear_velocity_y_mps", "linear_velocity_z_mps"]


def test_each_field_reaches_its_own_name(tmp_path):
    fields = {col(LIDAR_BOX, "box.size.x"): 4.5, col(LIDAR_BOX, "box.size.y"): 2.0, col(LIDAR_BOX, "box.center.z"): 1.0,
              col(LIDAR_BOX, "num_lidar_points_in_box"): 90, col(LIDAR_BOX, "num_top_lidar_points_in_box"): 70,
              SPEED[1]: -3.0, ACCEL[1]: 0.25}
    _write(tmp_path, LIDAR_BOX, {TIMESTAMP: [T0]} | {name: [value] for name, value in fields.items()})
    box = load_lidar_boxes(tmp_path, REF).iloc[0]
    assert (box["size_x"], box["size_y"], box["center_z"]) == (4.5, 2.0, 1.0)
    assert (box["num_lidar_points"], box["num_top_lidar_points"]) == (90, 70)
    assert (box["speed_y"], box["accel_y"]) == (-3.0, 0.25)

    shutter = {col(CAMERA_IMAGE, "pose_timestamp"): 1550083467.40108,
               col(CAMERA_IMAGE, "rolling_shutter_params.camera_trigger_time"): 1550083467.369629,
               col(CAMERA_IMAGE, "rolling_shutter_params.camera_readout_done_time"): 1550083467.423879,
               col(CAMERA_IMAGE, "rolling_shutter_params.shutter"): 0.009991644}
    _write(tmp_path, CAMERA_IMAGE, {TIMESTAMP: [T0]} | {name: [value] for name, value in shutter.items()})
    meta = load_image_metadata(tmp_path, REF, [0]).iloc[0]
    assert meta["pose_timestamp_s"] == 1550083467.40108 and meta["shutter_s"] == 0.009991644
    assert meta["trigger_time_s"] == 1550083467.369629 and meta["readout_done_time_s"] == 1550083467.423879
    assert meta["world_from_image_vehicle"].shape == (16,)


def test_absent_optional_columns_load_as_nan(tmp_path):
    _write(tmp_path, LIDAR_BOX, {TIMESTAMP: [T0, T0 + 1]}, drop=SPEED + ACCEL)
    boxes = load_lidar_boxes(tmp_path, REF)
    for name in ("speed_x", "speed_y", "speed_z", "accel_x", "accel_y", "accel_z"):
        assert boxes[name].dtype == np.float64 and boxes[name].isna().all()


def test_optional_columns_of_arrow_null_type_load_as_nan(tmp_path):
    """Every value null can come back as Arrow's null type, not double, and pandas then holds objects."""
    nulls = {name: pa.nulls(2) for name in SPEED + ACCEL}
    _write(tmp_path, LIDAR_BOX, {TIMESTAMP: [T0, T0 + 1]}, drop=SPEED + ACCEL, arrays=nulls)
    boxes = load_lidar_boxes(tmp_path, REF)
    for name in ("speed_x", "speed_y", "speed_z", "accel_x", "accel_y", "accel_z"):
        assert boxes[name].dtype == np.float64 and boxes[name].isna().all()


def test_null_speed_loads_as_nan_only_where_null(tmp_path):
    _write(tmp_path, LIDAR_BOX, {TIMESTAMP: [T0, T0 + 1], SPEED[0]: [None, 2.5]})
    speed = load_lidar_boxes(tmp_path, REF)["speed_x"]
    assert speed.dtype == np.float64 and np.isnan(speed[0]) and speed[1] == 2.5


def test_absent_required_column_fails_rather_than_loading_as_nan(tmp_path):
    _write(tmp_path, LIDAR_BOX, {TIMESTAMP: [T0]}, drop=[col(LIDAR_BOX, "box.heading")])
    with pytest.raises(pa.ArrowInvalid):
        load_lidar_boxes(tmp_path, REF)


def test_extra_column_is_tolerated(tmp_path):
    _write(tmp_path, VEHICLE_POSE, {TIMESTAMP: [T0]}, arrays={"index": pa.array([7])})
    _write(tmp_path, LIDAR_BOX, {TIMESTAMP: [T0]}, arrays={"index": pa.array([7])})
    assert len(load_poses(tmp_path, REF).timestamps) == 1
    assert "index" not in load_lidar_boxes(tmp_path, REF)
    assert check_segment(tmp_path, REF)[VEHICLE_POSE].extra == 1


def _with_pandas_metadata(tmp_path, tag: str, index_column: str) -> None:
    """The file as pandas would write it with `index_column` as its index, which to_pandas restores."""
    path = component_path(tmp_path, REF, tag)
    table = pq.read_table(path)
    frame = table.to_pandas().set_index(index_column)
    pq.write_table(pa.Table.from_pandas(frame), path)


@pytest.mark.parametrize("index_column", [TIMESTAMP, OBJECT])
def test_a_key_the_file_records_as_its_index_stays_a_column(tmp_path, index_column):
    """to_pandas would move it out of the columns, and reindex would quietly fill it with NaN."""
    rows = {TIMESTAMP: [T0 + 9, T0, T0 + 2], OBJECT: ["c", "a", "b"]}  # uneven, or pandas stores a range and no column
    _write(tmp_path, LIDAR_BOX, rows)
    _with_pandas_metadata(tmp_path, LIDAR_BOX, index_column)
    boxes = load_lidar_boxes(tmp_path, REF)
    assert boxes["frame_timestamp_micros"].tolist() == [T0, T0 + 2, T0 + 9]
    assert boxes["laser_object_id"].tolist() == ["a", "b", "c"]
    assert boxes["frame_timestamp_micros"].dtype == np.int64


def test_a_dictionary_typed_id_sorts_by_value_not_by_the_files_dictionary(tmp_path):
    """Unsorted dictionary order would reorder rows, and with them aliases, in a re-encoded copy of a file."""
    ids = pa.DictionaryArray.from_arrays(pa.array([0, 1], type=pa.int32()), pa.array(["b", "a"]))
    _write(tmp_path, LIDAR_BOX, {TIMESTAMP: [T0, T0]}, drop=[OBJECT], arrays={OBJECT: ids})
    boxes = load_lidar_boxes(tmp_path, REF)
    assert boxes["laser_object_id"].tolist() == ["a", "b"]
    assert not isinstance(boxes["laser_object_id"].dtype, pd.CategoricalDtype)


def test_check_segment_reports_missing_file_and_column(tmp_path):
    _write(tmp_path, VEHICLE_POSE, {TIMESTAMP: [T0]})
    _write(tmp_path, LIDAR_BOX, {TIMESTAMP: [T0]}, drop=[col(LIDAR_BOX, "box.heading")] + SPEED)
    checks = check_segment(tmp_path, REF)
    assert list(checks) == list(STAGE1_COMPONENTS)
    assert checks[VEHICLE_POSE].ok
    assert not checks[LIDAR_BOX].ok and checks[LIDAR_BOX].missing == [col(LIDAR_BOX, "box.heading")]
    assert checks[LIDAR_BOX].missing_optional == SPEED
    assert not checks[CAMERA_IMAGE].ok and checks[CAMERA_IMAGE].missing == ["<no file>"]


def test_check_segment_reads_no_rows(tmp_path, monkeypatch):
    _write(tmp_path, VEHICLE_POSE, {TIMESTAMP: [T0]})
    calls = _spy_on_read_table(monkeypatch)
    assert check_segment(tmp_path, REF)[VEHICLE_POSE].ok
    assert calls == []


def test_velocities_come_back_float64_and_timestamps_stay_int64(tmp_path):
    velocity = col(CAMERA_IMAGE, "velocity.linear_velocity.x")
    _write(tmp_path, CAMERA_IMAGE, {TIMESTAMP: [T0 + 1], velocity: [0.1]})
    assert pq.read_schema(component_path(tmp_path, REF, CAMERA_IMAGE)).field(velocity).type == pa.float32()
    meta = load_image_metadata(tmp_path, REF, [0])
    for axis in "xyz":
        assert meta[f"linear_velocity_{axis}_mps"].dtype == np.float64
    assert meta["linear_velocity_x_mps"][0] == float(np.float32(0.1))
    assert meta["frame_timestamp_micros"].dtype == np.int64 and meta["frame_timestamp_micros"][0] == T0 + 1


def test_camera_ids_filter_rows(tmp_path):
    keys = {TIMESTAMP: [T0, T0, T0], CAMERA: [1, 2, 3], OBJECT: ["a", "a", "a"]}
    _write(tmp_path, PROJECTED_BOX, keys)
    assert load_projected_boxes(tmp_path, REF, (1, 3))["camera"].tolist() == [1, 3]
    _write(tmp_path, CAMERA_IMAGE, {TIMESTAMP: keys[TIMESTAMP], CAMERA: keys[CAMERA]})
    assert load_image_metadata(tmp_path, REF, (2,))["camera"].tolist() == [2]


def test_image_column_is_not_read(tmp_path, monkeypatch):
    _write(tmp_path, CAMERA_IMAGE, {TIMESTAMP: [T0], CAMERA: [1]})
    calls = _spy_on_read_table(monkeypatch)
    meta = load_image_metadata(tmp_path, REF, [1])
    assert len(meta) == 1
    assert len(calls) == 1 and calls[0]["columns"] is not None and IMAGE not in calls[0]["columns"]


def test_read_jpegs_filters_by_camera_and_timestamp(tmp_path):
    jpegs = pa.array([b"late-1", b"early-1", b"early-2", b"late-2"], type=pa.binary())
    _write(tmp_path, CAMERA_IMAGE, {TIMESTAMP: [T0 + 9, T0, T0, T0 + 9], CAMERA: [1, 1, 2, 2]}, arrays={IMAGE: jpegs})
    assert read_jpegs(tmp_path, REF, 2) == {T0: b"early-2", T0 + 9: b"late-2"}
    assert read_jpegs(tmp_path, REF, 1, [T0 + 9]) == {T0 + 9: b"late-1"}
    assert read_jpegs(tmp_path, REF, 1, np.array([T0, T0 + 9], dtype=np.int64)) == {T0: b"early-1", T0 + 9: b"late-1"}
    assert read_jpegs(tmp_path, REF, 1, [T0 + 1]) == {}
    assert read_jpegs(tmp_path, REF, 1, []) == {}  # every frame cached: read nothing, not everything


def test_read_jpegs_filters_inside_pyarrow(tmp_path, monkeypatch):
    _write(tmp_path, CAMERA_IMAGE, {TIMESTAMP: [T0], CAMERA: [1]})
    calls = _spy_on_read_table(monkeypatch)
    read_jpegs(tmp_path, REF, 1, [T0])
    assert {name for name, _, _ in calls[0]["filters"]} == {CAMERA, TIMESTAMP}
    assert sorted(calls[0]["columns"]) == sorted([TIMESTAMP, IMAGE])


def test_load_cameras_builds_models_for_requested_cameras(tmp_path):
    extrinsic = se3(rot_z(0.5), [1.5, -0.1, 2.0])
    intrinsics = {col(CAMERA_CALIBRATION, f"intrinsic.{k}"): [float(i), 0.0]
                  for i, k in enumerate(("f_u", "f_v", "c_u", "c_v", "k1", "k2", "p1", "p2", "k3"))}
    _write(tmp_path, CAMERA_CALIBRATION,
           intrinsics | {CAMERA: [2, 1], col(CAMERA_CALIBRATION, "width"): [1920, 1920],
                         col(CAMERA_CALIBRATION, "height"): [1280, 1280],
                         col(CAMERA_CALIBRATION, "rolling_shutter_direction"): [2, 4],
                         col(CAMERA_CALIBRATION, "extrinsic.transform"): [extrinsic.ravel().tolist()] * 2})
    cameras = load_cameras(tmp_path, REF, (2,))
    assert list(cameras) == [2]
    cam = cameras[2]
    assert cam.intrinsic == tuple(float(i) for i in range(9))
    assert (cam.width, cam.height, cam.rolling_shutter_direction) == (1920, 1280, 2)
    np.testing.assert_allclose(cam.extrinsic, extrinsic)
