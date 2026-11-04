import io

import numpy as np
from PIL import Image

from protobuf_encode import (f_bytes, f_double, f_doubles_packed, f_doubles_unpacked, f_varint, map_point,
                             tfrecord, transform)
from scenario_miner.data.wire import Message
from scenario_miner.data.waymo_v1 import extract_segment
from scenario_miner.geometry import rot_z, se3, transform_points
from scenario_miner.sample import load_sample


def test_wire_scalars_and_repeated():
    msg = Message(f_varint(1, 7) + f_varint(2, -5) + f_doubles_unpacked(3, [1.5, 2.5])
                  + f_doubles_packed(4, [3.0, 4.0, 5.0]) + f_bytes(5, b"hi"))
    assert msg.int(1) == 7 and msg.int(2) == -5 and msg.int(9, default=-1) == -1
    assert msg.doubles(3).tolist() == [1.5, 2.5]
    assert msg.doubles(4).tolist() == [3.0, 4.0, 5.0]
    assert msg.string(5) == "hi" and msg.message(9) is None


def _jpeg() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 48), (90, 90, 90)).save(buf, format="JPEG")
    return buf.getvalue()


YAW, ORIGIN, OFFSET = 0.6, np.array([500.0, -300.0, 10.0]), np.array([2.0, -1.0, 0.0])


def _pose(i: int) -> np.ndarray:
    """Vehicle drives along its own +x at 10 m/s (1 m per 0.1 s frame)."""
    return se3(rot_z(YAW), ORIGIN + rot_z(YAW) @ np.array([1.0 * i, 0.0, 0.0]))


def _to_map(p_vehicle0) -> tuple:
    """Map coordinates = world + map_pose_offset; authored relative to frame 0's vehicle frame."""
    return tuple(transform_points(_pose(0), p_vehicle0)[0] + OFFSET)


def _frame(i: int) -> bytes:
    out = b""
    if i == 0:
        calib = (f_varint(1, 1) + f_doubles_unpacked(2, [50.0, 50.0, 32.0, 24.0, 0, 0, 0, 0, 0])
                 + f_bytes(3, transform(se3(np.eye(3), [1.5, 0.0, 2.0]))) + f_varint(4, 64) + f_varint(5, 48))
        stats = f_bytes(2, b"Night") + f_bytes(3, b"location_sf") + f_bytes(4, b"rain")
        out += f_bytes(1, f_bytes(1, b"seg123") + f_bytes(2, calib) + f_bytes(4, stats))
        lane = f_varint(2, 2) + b"".join(f_bytes(8, map_point(*_to_map([x, 0.0, 0.0]))) for x in range(-10, 80, 5))
        crosswalk = b"".join(f_bytes(1, map_point(*_to_map([x, y, 0.0]))) for x, y in [(20, -5), (24, -5), (24, 5), (20, 5)])
        stop = f_bytes(2, map_point(*_to_map([30.0, -4.0, 1.0])))
        far_lane = b"".join(f_bytes(8, map_point(*_to_map([x, 500.0, 0.0]))) for x in range(0, 50, 5))
        out += f_bytes(10, f_varint(1, 101) + f_bytes(3, lane))
        out += f_bytes(10, f_varint(1, 102) + f_bytes(8, crosswalk))
        out += f_bytes(10, f_varint(1, 103) + f_bytes(7, stop))
        out += f_bytes(10, f_varint(1, 104) + f_bytes(3, far_lane))  # outside map_radius: dropped
    out += f_varint(2, 1_000_000 + i * 100_000)
    out += f_bytes(3, transform(_pose(i)))
    out += f_bytes(4, f_varint(1, 1) + f_bytes(2, _jpeg()) + f_bytes(3, transform(_pose(i))))
    box = (f_double(1, 15.0) + f_double(2, 0.5) + f_double(3, 1.0) + f_double(4, 2.0) + f_double(5, 4.5)
           + f_double(6, 1.6) + f_double(7, 0.1))
    out += f_bytes(6, f_bytes(1, box) + f_bytes(2, f_double(1, 3.0) + f_double(2, 4.0)) + f_varint(3, 1)
                   + f_bytes(4, b"car1") + f_varint(7, 250))
    out += f_bytes(11, f_double(1, OFFSET[0]) + f_double(2, OFFSET[1]) + f_double(3, OFFSET[2]))
    return out


def test_extract_segment_roundtrip(tmp_path):
    path = tmp_path / "segment-seg123_with_camera_labels.tfrecord"
    path.write_bytes(tfrecord([_frame(i) for i in range(3)]))

    written = extract_segment(path, tmp_path / "out", every_n=2)  # keeps frame 1 (i % 2 == 1)
    assert len(written) == 1
    s = load_sample(written[0])
    assert s.segment_id == "seg123" and s.camera == "FRONT" and s.timestamp_us == 1_100_000
    assert s.context["time_of_day"] == "Night" and s.context["weather"] == "rain"
    assert abs(s.ego_speed - 10.0) < 1e-6
    assert s.camera_model.intrinsic[:4] == (50.0, 50.0, 32.0, 24.0) and s.camera_model.width == 64

    (box,) = s.boxes
    assert box.type == "vehicle" and box.object_id == "car1" and box.num_lidar_points == 250
    np.testing.assert_allclose(box.center, [15.0, 0.5, 1.0])
    np.testing.assert_allclose(box.size, [4.5, 2.0, 1.6])  # length, width, height
    assert abs(box.speed - 5.0) < 1e-9

    kinds = {f.kind: f for f in s.map_features}
    assert set(kinds) == {"lane", "crosswalk", "stop_sign"}
    # Frame 1's vehicle is 1 m further along, so map x shifts by -1; offset and world rotation cancel out.
    lane = kinds["lane"]
    assert lane.subtype == 2
    np.testing.assert_allclose(lane.points[:, 1:], 0.0, atol=1e-6)
    np.testing.assert_allclose(lane.points[[0, -1], 0], [-11.0, 59.0], atol=1e-6)  # cropped at map_radius=60
    np.testing.assert_allclose(kinds["crosswalk"].points[:, :2], [[19, -5], [23, -5], [23, 5], [19, 5]], atol=1e-6)
    np.testing.assert_allclose(kinds["stop_sign"].points, [[29.0, -4.0, 1.0]], atol=1e-6)
    Image.open(s.image_path).verify()
