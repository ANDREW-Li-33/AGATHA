# Stage 1 architecture

Implements milestone M1 of the design doc. [stage1_setup_and_decisions.md](stage1_setup_and_decisions.md) explains why each choice was made, and [research/2026-11-01/](research/2026-11-01/) holds the sources.

## Principles

1. **Full-rate tables are the product.** `frames`, `ego`, and `tracks` hold everything at 10 Hz. A SceneWindow is a pure function of them, so Pass 2 can ask for any range at 5 to 10 FPS and Stage 4 can read the trajectories directly.
2. **Tables are never filtered.** Filtering happens only in views: what a SceneWindow keeps, and what gets drawn.
3. **Only the report leaves PACE.** It holds aggregates. Nothing else in a real run is safe for a Claude session to read.
4. **float64 for geometry, int64 for joins.** `frame_timestamp_micros` is the key. `t_s` is for reading and windowing only.
5. **Every line gets explained in an interview.** Prefer the shortest correct code. Match `scenario-miner/src/scenario_miner/`: `from __future__ import annotations`, NumPy over loops where it reads better, one-line docstrings unless there's a real gotcha, and comments that say why rather than what.

## Layout

```
configs/stage1.toml
infra/pace/                   env, ssh config, Slurm templates
src/agatha/
  columns.py     done         component tags, column names, enums, check_schema
  geometry.py    done         SE(3), box corners, projection, wireframes
  config.py      done         Stage1Config from TOML
  contracts.py   done         SceneWindow pydantic models
  files.py                    atomic writes
  loader.py                   read Parquet components into tidy frames
  sampling.py                 gap report, frame selection, window arithmetic
  tracks.py                   central differences, EgoPath, ego table, tracks table
  store.py                    frames table, segment.json, read/write of outputs
  scene_window.py             build_scene_window, write_scene_window
  render.py                   overlays, render cache, clips, contact sheet
  report.py                   sanitized report and guarded steps
  checks.py                   acceptance checks
  synthetic.py                synthetic segment in the real Parquet layout
  run_stage1.py               CLI
tests/
```

## Output layout

```
$WOD_OUTPUTS/reports/<context_label>.json   the only file that leaves PACE
$WOD_OUTPUTS/<context_name>/
  segment.json
  frames.parquet
  ego.parquet
  tracks.parquet
  windows/<start_ms>_<end_ms>__<fps>fps__<CAMS>__<profile>__<content_hash>.json
  render/<CAMERA>/<frame_timestamp_micros>__clean.jpg      original bytes, never re-encoded
  render/<CAMERA>/<frame_timestamp_micros>__<profile>.jpg  wireframes
  clips/...
  qa/contact_sheet.html       for Andrew's eyes, in OnDemand
```

Reports sit apart and go by label, so a session can fetch one without listing a directory of context names. A run deletes its old report first, so no report means no finished run.

## Tables

`t0_micros` is the first `vehicle_pose` timestamp. `t_s = (ts - t0_micros) / 1e6`. Transforms are stored as `list<double>` of length 16, row-major. Unknown numeric values are NaN.

**`ego.parquet`**, one row per frame, sorted by time:

| Column | Meaning |
|---|---|
| `frame_timestamp_micros` (int64), `t_s` | |
| `world_from_vehicle` | frame pose, list of 16 |
| `pos_local_x_m`, `pos_local_y_m`, `pos_local_z_m` | segment_local |
| `yaw_rad` | `yaw_of(world_from_vehicle)` |
| `v_world_x_mps`, `v_world_y_mps` | central difference of position, in world axes |
| `speed_mps` | |
| `a_long_mps2`, `a_lat_mps2` | central difference of velocity over `accel_baseline_s`, rotated into ego axes |
| `yaw_rate_radps` | central difference of unwrapped yaw |
| `path_s_m` | arc length along the ego path at this pose |

Ego velocity comes from pose differences rather than `camera_image` so the same code serves the label-only scan. `checks.py` cross-checks it against FRONT `linear_velocity`.

**`frames.parquet`**, one row per (frame, camera), sorted by time then camera:

| Column | Meaning |
|---|---|
| `frame_timestamp_micros`, `t_s` | |
| `camera` (int8), `camera_name` (str) | |
| `gap_before_ms` | spacing to the previous frame, NaN for the first |
| `has_gap_before` (bool) | `gap_before_ms > checks.gap_ms` |
| `pose_timestamp_s`, `trigger_time_s`, `readout_done_time_s`, `shutter_s` | float64 seconds |
| `capture_offset_ms` | `pose_timestamp_s * 1e3 - frame_timestamp_micros / 1e3` |
| `world_from_vehicle`, `world_from_image_vehicle` | lists of 16 |

**`tracks.parquet`**, one row per `lidar_box` row, sorted by time then object:

| Column | Meaning |
|---|---|
| `frame_timestamp_micros`, `t_s`, `laser_object_id`, `alias` | |
| `type` (str) | per sample, from `BOX_TYPES` |
| `center_x_m`, `center_y_m`, `center_z_m` | vehicle frame, as labeled. This is ego_at_sample. |
| `size_x_m`, `size_y_m`, `size_z_m`, `heading_rad` | length, width, height, yaw in the vehicle frame |
| `pos_world_x_m`, `pos_world_y_m`, `pos_world_z_m` | |
| `pos_local_x_m`, `pos_local_y_m`, `pos_local_z_m` | segment_local |
| `yaw_world_rad` | wrapped to [-pi, pi) |
| `v_label_world_x_mps`, `v_label_world_y_mps` | Waymo `speed`, which is absolute, in world axes |
| `v_diff_world_x_mps`, `v_diff_world_y_mps` | central difference of `pos_world` |
| `v_world_x_mps`, `v_world_y_mps`, `v_source` | label if present, else difference, else NaN and `"none"` |
| `a_world_x_mps2`, `a_world_y_mps2` | central difference of `v_world`, over the time each velocity describes |
| `v_abs_ego_x_mps`, `v_abs_ego_y_mps` | first two entries of `R.T @ (vx, vy, 0)`, with `R` from this frame's `world_from_vehicle` |
| `v_rel_ego_x_mps`, `v_rel_ego_y_mps` | the same with `(v_obj - v_ego)` |
| `a_abs_ego_x_mps2`, `a_abs_ego_y_mps2` | |
| `speed_mps`, `stationary` (bool) | `stationary` is `speed_mps < stationary_mps`, False when speed is unknown |
| `along_path_m`, `lateral_offset_m`, `on_driven_path`, `path_clearance_m` | see EgoPath |
| `front_gap_m` | `min(corner_x) - rear_axle_to_front_m` if positive, else NaN |
| `num_lidar_points`, `num_top_lidar_points` | |
| `not_directly_observed` (bool) | `num_top_lidar_points == 0` |
| `fov_cameras` (list of int) | cameras with a `projected_lidar_box` row. In frame, not necessarily visible. |
| `synced_camera` (int, -1 if none) | `most_visible_camera_name` |
| `synced_center_x_m`, `synced_center_y_m`, `synced_center_z_m`, `synced_size_x_m`, `synced_size_y_m`, `synced_size_z_m`, `synced_heading_rad` | `camera_synced_box`, NaN if none |

Never use the `synced_*` columns for trajectories: each is shifted to a per-object capture time.

**`segment.json`**: `schema_version`, `dataset_version`, `split`, `context_name`, `t0_micros`, `last_t_s`, `n_frames`, `local_origin_world_m`, `rear_axle_to_front_m`, `preprocessing_version`, `attribution`, and `cameras`, a map from camera name to `{camera_id, intrinsic, extrinsic, width, height, rolling_shutter_direction}`.

`preprocessing_version` is `<package version>+<source hash>.<config hash>`. The source hash covers this package's `.py` files, because `sync.sh` copies the working tree and a commit wouldn't see an uncommitted edit. The config hash covers the TOML file and the cameras, which `--cameras` can override.

## Interfaces

### `loader.py`

```python
@dataclass(frozen=True)
class SegmentRef:
    split: str
    context_name: str

class Poses(NamedTuple):
    timestamps: np.ndarray          # int64 (N,), sorted
    world_from_vehicle: np.ndarray  # float64 (N, 4, 4)

def component_path(root: Path, ref: SegmentRef, tag: str) -> Path
    # root / split / tag / f"{context_name}.parquet"
def check_segment(root, ref) -> dict[str, SchemaCheck]
    # uses pq.read_schema only. A missing file gives missing=["<no file>"].
def load_poses(root, ref) -> Poses
def load_cameras(root, ref, camera_ids) -> dict[int, CameraModel]
def load_image_metadata(root, ref, camera_ids) -> pd.DataFrame
    # frame_timestamp_micros, camera, pose_timestamp_s, trigger_time_s, readout_done_time_s,
    # shutter_s, world_from_image_vehicle (arrays of 16), linear_velocity_{x,y,z}_mps.
    # Never reads the image column. float32 velocities are cast to float64.
def load_lidar_boxes(root, ref) -> pd.DataFrame
    # frame_timestamp_micros, laser_object_id, center_{x,y,z}, size_{x,y,z}, heading, type,
    # num_lidar_points, num_top_lidar_points, speed_{x,y,z}, accel_{x,y,z}.
    # An absent or all-null optional column loads as float64 NaN. Sorted by time then object.
def load_synced_boxes(root, ref) -> pd.DataFrame
    # frame_timestamp_micros, laser_object_id, most_visible_camera, center_{x,y,z},
    # size_{x,y,z}, heading
def load_projected_boxes(root, ref, camera_ids) -> pd.DataFrame
    # frame_timestamp_micros, camera, laser_object_id, center_x, center_y, size_x, size_y, type
def read_jpegs(root, ref, camera_id, timestamps=None) -> dict[int, bytes]
    # filters by camera and timestamps through pyarrow, to stay within memory
```

Row order differs across components, so sort after every read and join only on keys. Rows for other segments can't occur, because each file is one segment. Reads ignore a file's pandas metadata and decode dictionary columns to their values: the first could move a key into the index, and the second would sort IDs by the file's dictionary order and so change aliases.

### `sampling.py`

```python
class GapReport(NamedTuple):
    n_frames: int
    duration_s: float
    min_dt_ms: float
    median_dt_ms: float
    max_dt_ms: float
    n_gaps: int

class Selection(NamedTuple):
    timestamps: np.ndarray  # int64, sorted, unique
    achieved_fps: float     # intervals / span of the selected frames, 0 if fewer than 2
    max_gap_ms: float       # largest spacing within the selection, 0 if fewer than 2

class Window(NamedTuple):
    start_s: float
    end_s: float
    closed_end: bool        # True only for a segment's last window

def gap_report(timestamps, gap_ms) -> GapReport
def nearest_frame(timestamps, t0_micros, t_s) -> int
def select_frames(timestamps, t0_micros, fps, start_s, end_s, closed_end=False, force_include=()) -> Selection
def compute_windows(last_t_s, window_s, stride_s, min_tail_s=1.0) -> list[Window]
```

`select_frames` anchors its grid at `t0_micros + k / fps`, never at the window start, so overlapping windows share frames. Each grid time in `[start_s, end_s)`, or `[start_s, end_s]` when `closed_end`, takes the nearest timestamp if it's within half a grid period. Results are deduplicated, then `force_include` is added. Never select every Nth row: frames can be missing and spacing varies.

`force_include` must hold exact frame timestamps, and anything else raises `ValueError`. `nearest_frame` turns a time in seconds into one. `start_s` and `end_s` bound grid times, not frames: jitter can put a selected frame just outside them, and a forced frame can be anywhere. So `SceneWindow.frames` is the authority on what a window holds, and a citation is checked against it, not against the bounds.

`compute_windows` steps `start` by `stride_s`. The last window ends at `last_t_s` with `closed_end=True`. A tail shorter than `min_tail_s` merges into the previous window. For `last_t_s = 19.7` with 5 and 4 it gives (0,5), (4,9), (8,13), (12,17), (16,19.7).

### `tracks.py`

```python
def central_difference(values, t_s, baseline_s, max_baseline_s) -> tuple[np.ndarray, np.ndarray]
```

`values` has shape (N,) or (N, k). For each sample, widen a symmetric index window `w = 1, 2, ...`, clipped to the array, until `t[b] - t[a] >= 0.95 * baseline_s`. The 0.95 allows for timestamp jitter. Ends are therefore one-sided. The result is `(v[b] - v[a]) / (t[b] - t[a])`, or NaN if no pair qualifies or `t[b] - t[a] > max_baseline_s`. Any NaN input in the pair gives NaN. Single-step differences are too noisy to use: in one public sample they read 7.37 and 8.51 m/s against a label of 7.97.

It also returns the time each result describes, `(t[a] + t[b]) / 2`, which at the ends and beside a dropped frame isn't the sample's own. A second pass must run over those times. Over `t_s`, constant braking reads 20% low at the ends. A label describes its own frame, so an object's acceleration takes the row's time where the velocity is a label.

```python
class EgoPath:
    """The ego's own driven path in segment_local: post hoc, and a proxy for lane position."""
    def __init__(self, points_xy: np.ndarray, start_yaw_rad: float, end_yaw_rad: float, extend_m: float = 200.0)
    length_m: float  # of the driven part, 0 if the ego never got 1 m from its first pose
    def locate(self, points_xy) -> tuple[np.ndarray, np.ndarray]
        # (s, d): arc length with s = 0 at the first ego pose, and signed lateral
        # offset, left of travel positive
```

The constructor drops consecutive points closer than 0.05 m, then extends the polyline straight past both ends by `extend_m`, along the first and last poses' headings. Never along the first and last steps: a step can be 5 cm of pose noise, and at 46 m that's meters of lateral error. An ego that never gets 1 m from its first pose is parked, so its path is that one point. `locate` uses the nearest point on the polyline.

A heading is the way the nose points, so a reversing ego's extensions run against its travel, and within the driven path "left of travel" is the ego's right. Reversing is rare in these segments and isn't special-cased.

```python
def build_ego_table(poses: Poses, motion: MotionConfig) -> tuple[pd.DataFrame, EgoPath]
def assign_aliases(boxes: pd.DataFrame) -> dict[str, str]
def build_tracks_table(boxes, synced, projected, poses, ego, path, motion, rear_axle_to_front_m) -> pd.DataFrame
```

`assign_aliases` orders objects by first appearance, then ID. The prefix is the first type's letter (`V`, `P`, `S`, `C`, `U`), with a per-prefix three-digit counter from 001, such as `V012`. Aliases must be deterministic: Stage 2 validates against them.

`on_driven_path`: the box center's `s` is within `[0, path.length_m]`. Past either end, `along_path_m` and `lateral_offset_m` are against a straight line that the road need not follow.

`path_clearance_m`: take the 4 footprint corners in segment_local, and `locate` them. If the signed lateral offsets differ in sign, it's 0. Otherwise it's the smallest absolute offset. It's NaN for a box center more than 4 m from the ego's height at that `s`, since an object on an overpass isn't in the lane. Only `on_driven_path` rows can be ruled out this way: past the ends the path's height is unknown, and a hill would look like a level. `along_path_m` is the box center's `s` minus the ego's `path_s_m` at that frame. Compute differences per object over its own observations, using real timestamps.

### `store.py`

```python
@dataclass(frozen=True)
class SegmentOutputs:
    directory: Path
    meta: dict             # segment.json
    frames: pd.DataFrame
    ego: pd.DataFrame
    tracks: pd.DataFrame

def build_frames_table(image_meta, poses, gap_ms) -> pd.DataFrame
def build_segment_meta(ref, poses, cameras, config) -> dict
def write_segment(directory, meta, frames, ego, tracks) -> None
    # segment.json is removed first and written last, so load_segment can't mix two runs' tables
def load_segment(directory) -> SegmentOutputs
def camera_models(meta) -> dict[int, CameraModel]
def segment_directory(config, ref) -> Path
```

### `scene_window.py`

```python
def build_scene_window(segment: SegmentOutputs, config, start_s, end_s, fps, cameras, profile,
                       closed_end=False, max_objects=None, context_s=None, force_include=(),
                       keep_aliases=()) -> SceneWindow
def default_windows(segment, config) -> list[SceneWindow]
def write_scene_window(window, directory) -> Path
```

`build_scene_window` is pure: it writes nothing, and it only computes references. Images come from `render.ensure_rendered`.

- `cameras` are ids. A camera with no rows in `frames` raises `ValueError` naming it: `frames` holds only the cameras Stage 1 ran on, and a dropped camera would look like an empty view. The window records `cameras` and `profile`.
- Select image frames with `select_frames`. Each (timestamp, camera) row in `frames` becomes a `FrameRef`.
- Track context: select on the same `fps` grid over `[start_s - context_s, end_s + context_s]`, clipped to the segment. Times outside the image selection get `context_only=True`.
- Keep objects whose type at any image-selection sample is vehicle, pedestrian, or cyclist.
- Rank by the minimum `path_clearance_m` over the image selection, then minimum range to the ego, then alias. Range matters because everything crossing the path ties at 0, and aliases count up by first appearance. Objects with no clearance rank last.
- Keep the first `max_objects`, plus any `keep_aliases`, which don't count against it: Pass 2's candidate must be in its own input. `omitted_objects` lists the rest as `{alias, laser_object_id, type}`, because images label every road user, so a citation of one of these is valid and needs no tables to trace. `selection_rule` states the rule.
- Keep stationary objects and objects outside every camera's FOV. Both matter later: the first for obstruction events, the second for a cut-in vehicle before it reaches FRONT.
- `first_t_s`, `last_t_s`, and `range_at_first_seen_m` use the object's whole track, not just the window. `truncated_start` and `truncated_end` compare against the segment's first and last frames.
- NaN becomes None for Optional fields. NaN in any other field raises when the window is built, because it would be written as null and fail the contract on read.

`write_scene_window` writes compact JSON. Its file name ends in 8 hex characters of the content's sha256, because the rest of the name can't say everything that shaped a window: two Pass 2 candidates over one range can cite different frames. The same window always writes the same file.

### `render.py`

```python
class DrawBox(NamedTuple):
    alias: str
    type: str
    corners_camera: np.ndarray  # (8, 3)
    draw_source: str            # "camera_synced_box" or "lidar_box"
    occluded: bool              # not_directly_observed

def draw_boxes(tracks_at_time: pd.DataFrame, camera_id: int, cam: CameraModel,
               camera_from_label_frame: np.ndarray) -> list[DrawBox]
def uses_synced(row, camera_id: int) -> bool
def render_annotated(jpeg: bytes, boxes, cam, profile: RenderProfile) -> bytes
def ensure_rendered(window: SceneWindow, segment: SegmentOutputs, root, ref, profiles: dict[str, RenderProfile]) -> dict
def write_clip(segment, camera_id, start_s, end_s, out_path, profile_name="clean") -> Path
def write_contact_sheet(path, items: list[ContactItem]) -> Path
```

Drawing rule:

- Skip sign and unknown types.
- If `uses_synced`, meaning `synced_camera == camera_id` and all seven synced values are there, build corners from the `synced_*` columns and use `cam.camera_from_vehicle` only (extrinsic-only, Waymo's own path for these boxes).
- Otherwise, which includes a synced row missing any value, build corners from `center_*`, `size_*`, `heading_rad` and use `camera_from_label(...)`.
- Do no time extrapolation.
- Draw a box only if some wireframe point is inside the image.

Scale to `max_long_edge` with `INTER_AREA`, and scale the polylines by the same factor. Label each box with its alias, sized by the profile so it survives resizing. Color by type, and use gray for occluded boxes. Clean images are written from the original bytes.

`ensure_rendered` skips files that exist, so windows and rates share frames. It reads each camera's JPEGs once. It takes the profile named by `window.profile`, so pixels can't land under another profile's name, and it writes the annotated image first, because decoding is what finds a damaged JPEG and a clean file is trusted once it exists. It returns counts per `draw_source`.

Because it trusts what exists, a run clears `render/` when the existing `segment.json` has a different `preprocessing_version`, or none. `windows/`, `clips/`, and `qa/` go on every run, since they describe the tables being replaced.

`write_clip` uses the concat demuxer with per-frame durations taken from real timestamps, because a fixed frame rate misstates timing at dropped frames. It uses `imageio_ffmpeg.get_ffmpeg_exe()`. Its output is H.264, `yuv420p`, even dimensions, with stdout and stderr captured and not printed.

`write_contact_sheet` writes one self-contained HTML file with images embedded as base64, so it opens anywhere inside OnDemand. `ContactItem` is a NamedTuple of `jpeg: bytes` and `caption: str`.

### `report.py`

```python
class Report:
    def __init__(self, context_name: str, preprocessing_version: str)
    def add(self, section: str, **metrics) -> None
    def fail(self, step: str, exc: BaseException) -> None
    def write(self, path) -> Path

def context_label(context_name: str) -> str
@contextmanager
def guarded(report: Report, step: str)
```

This is the data boundary, so it fails closed.

- `add` accepts bool, int, float, None, and dicts and lists of those, recursively. NaN and inf become None.
- A number of 1e12 or more raises `ValueError`, and so does a list of more than 32 items. Aggregates stay well inside both: a raw timestamp in ms or microseconds is past the first, and a column is past the second.
- A string value is accepted only if it's in a fixed vocabulary: rolling-shutter names, `pass`, `fail`, and `no_data`. Anything else raises `ValueError`. So an object ID or context name can't slip in by accident.
- Section names and dict keys must match `^[a-z][a-z0-9_]*$`.
- `context_label` returns the name for the public tutorial segment `10023947602400723454_1120_000_1140_000`, and for names that are `synthetic_` and then only letters, digits and underscores, the whole name matched, so `synthetic_x/../<name>` doesn't pass. Otherwise it returns `sha256:` plus 12 hex characters.
- `fail` records the exception's type name and the file:line of the innermost frame inside `agatha`. It never records `str(exc)`, because messages can echo Dataset values.
- `guarded` wraps a step: on an exception it calls `fail`, writes nothing else, and re-raises as `SystemExit(1)` with `from None`, so no traceback text reaches Slurm logs.

### `checks.py`

Each returns a dict of aggregates ready for `Report.add`, with a `status` of `pass`, `fail`, or `no_data`.

```python
def check_joins(poses, boxes, synced, projected, image_meta, camera_ids) -> dict
def check_gaps(poses, image_meta, config) -> dict
def box_residuals(segment, projected, cameras, config) -> pd.DataFrame
def check_residuals(residuals, config) -> dict
def check_tracks(segment, config) -> dict
def check_velocity(segment) -> dict
def timing_summary(frames, cameras) -> dict
def null_summary(boxes) -> dict
def pick_review_frames(residuals, segment, seed=0) -> list[tuple[int, int, str]]
def output_hashes(directory) -> dict[str, str]
```

`check_joins`: exact int64 equality, zero orphans, no duplicate keys. An orphan is a `lidar_box`, `camera_image`, synced, or projected timestamp with no pose, or a synced or projected object with no `lidar_box` row at that timestamp.

`check_gaps`: `gap_report` fields plus frames missing per camera. No fixed frame count.

`box_residuals`: one row per rectangle that qualifies, with columns `frame_timestamp_micros`, `camera`, `ego_speed_mps`, `ego_yaw_rate_radps`, `object_speed_mps`, `column_px` (the oracle's center), `image_width_px`, and the four signed edge errors `e_u_min`, `e_u_max`, `e_v_min`, `e_v_max`, rectangle minus oracle, plus `kind`. The oracle is the `projected_lidar_box` rectangle, `center ± size / 2`. A rectangle qualifies only if every corner is valid and it and the oracle both sit at least `border_margin_px` inside the image, because Waymo clamps rectangles to the image. This table never leaves PACE.

A box can add one row of each `kind`:

- `lidar_box`: `box_rectangle` of the box through `camera_from_label`.
- `frame_pose`: the same corners through the extrinsic alone, the reading to hold the image pose against.
- `camera_synced_box`: the extrinsic-only synced rectangle, if `synced_camera` is this camera and the synced values are there.
- `drawn`: what `render.draw_boxes` shows, by its same `uses_synced` test. That's the `camera_synced_box` rectangle if the box has one, and otherwise the `lidar_box` rectangle. Sign and unknown types get none, because render skips them. Nor does a synced box get one if its rectangle doesn't qualify: render still draws that box, so the label can't answer for it.
- `label_fallback`: the `drawn` rows that came from the label.

`check_residuals`: each spread is the pooled median, p95, and max of the errors' sizes over the four edges, plus the signed medians in u and v. A wrong pose or clock pushes boxes one way, which sizes can't show, and by column the signed u median finds the column where global shutter is exact. It reports `drawn` overall and, as `by_camera`, per camera. Beside those it reports `lidar_box` and `camera_synced_box` overall, `lidar_box_by_camera`, `frame_pose_by_camera`, `label_fallback_by_camera`, `lidar_box` by ego speed bin (0-1, 1-5, 5-15, 15+ m/s), by object speed bin (the same edges), and by 5 equal column bins, and `lidar_box`, `camera_synced_box`, and `frame_pose` per camera on stationary objects.

The limits judge the boxes an image shows wholly inside its border. Waymo clamps the oracle at the border, so nothing past it can be measured, and that leaves out most boxes the label draws, because a box outside its most visible camera usually straddles the border. On the two real segments the label drew 18 of 236 and 304 of 2,247 overlays, but it has only 11 of 1,185 and 551 of 13,049 `drawn` rows, counted before signs left `drawn`.

`lidar_box` alone can't carry the limits, because it ignores rolling shutter, so at speed it can't be exact. On the fastest training segment, at 29 m/s, it reached p95 68 px on FRONT_LEFT and 80 px on FRONT_RIGHT, yet it drew only 18 of the 236 boxes shown, and the drawn boxes stayed under 5 px at p95 on every camera. Synced boxes overall measured median 0.017 and p95 0.62 px on 1,174 boxes, most of them moving at 15 m/s or more, which also shows that `projected_lidar_box` carries object motion. Section 10 of the setup doc has the full table. Its drawn numbers were measured before signs left `drawn`, so a rerun can move them. It can change a status too: all 12 of FRONT_LEFT's rows at 29 m/s are on stationary objects, so if those are signs, that camera has no `drawn` boxes and the check reads `no_data`.

`status` fails if any of these holds:

- `drawn`, overall or for any one camera, is over `residual_median_px` or `residual_p95_px`, whichever path drew the boxes. Each camera must clear the limits alone, because pooled, a camera with few boxes could be any amount wrong and still pass. The same goes for a path inside a camera: a p95 can't see a group under 5 percent of its rows, so these limits see a label fault only where label rows reach about that share. At 29 m/s they were 0.5 percent of FRONT's `drawn` rows and 4.3 percent of FRONT_RIGHT's.
- `lidar_box` on FRONT is over `residual_median_px` or `residual_p95_px`. A synced box never reads `center_*`, so at speed this is the one verdict that always reads the label geometry, and that geometry is `tracks.parquet`, which ranks every SceneWindow through `path_clearance_m`. FRONT can carry it because its image pose is the frame pose: its `lidar_box` and `frame_pose` spreads match to 10 digits on both real segments. Only rolling shutter is left, and that read median 0.23 and p95 3.3 px at 29 m/s. In a rolling-shutter model, 96 harsh scenes stayed under the limits, the worst at p95 27 px with 30 cars crossing 8 to 20 m ahead. This verdict has a resolution: on the synthetic segment a label 0.5 m to the side fails and 0.25 m passes.
- `lidar_box` is over the slow limits on rows under `slow_speed_mps` with a yaw rate under 0.05 rad/s. On a slow, steady ego it should be exact on anything standing still. Yaw moves the image during readout at any range, about 7 px at the edges at 0.15 rad/s, so slow doesn't mean small error unless the ego is steady too. Moving objects are left out, because the oracle moves them and the label doesn't, so a stopped ego beside cross traffic would otherwise fail on correct code.
- On stationary objects, a camera's `camera_synced_box` p95 is over its `lidar_box` p95 by more than 5 px, with at least 5 boxes of each. Standing still takes object motion out of the comparison, so a synced box has no reason to be the worse of the two. This is what tests the frame of the synced center, which is an inference that no vector backs, and it matters because that path draws most overlays. FRONT can't show it, so it needs a side camera.

Otherwise `status` is `no_data` if a configured camera has no `drawn` boxes, because it has shown nothing either way.

`frame_pose_by_camera` and `label_fallback_by_camera` set no status, and `lidar_box_by_camera` sets none past FRONT's. `frame_pose` is there to compare by eye, and the place to read it is `stationary_by_camera`, at p95, on a side camera, at speed. That's where the image pose should win. Medians and moving traffic can go the other way: traffic moving with the ego barely moves against the label's frame, so the frame pose reads it better. At 29 m/s, over each side camera's rows, the image pose won at p95, 68 against 112 px and 80 against 93 px, and lost at the median, 7.68 against 6.59 px and 3.87 against 3.39 px. So this stays a reading and isn't a rule: in the rolling-shutter model a rule on it fails correct code on most scenes with traffic moving with the ego. FRONT can't tell, because its image pose is the frame pose.

No status can be relied on to catch a wrong pose chain. A synced box uses no pose, and FRONT's label can't feel one. On a slow ego the two poses are millimeters apart, so a pose fault stays under the slow limits, short of a swap near the top of their speed gate. At speed, the limits see it only where label rows reach their share of a camera's `drawn` rows. So a run can show it only in these numbers, and what pins the chain is `tests/test_geometry.py`, against Waymo's own vector.

A spread over fewer than 5 boxes reports only its count, because its numbers come close to being the rows. Verdicts use the full numbers first.

`check_tracks`: counts and percentiles of internal frame gaps, type changes, a count of spikes in implied speed above `speed_spike_mps` from a single step, zero-point rows, and track length. A spike's speed isn't reported, since one spike's value is one object's.

`check_velocity`: across rows with a label and a difference where the difference speed exceeds 1 m/s, compare the difference against three readings of the label: world axes as is, vehicle axes rotated into world, and ego-relative vehicle axes rotated into world plus ego velocity. Rows where the vehicle-axes reading is within 1 m/s of the label are left out: near due east the two are the same reading, and if every row is like that the status is `no_data`. Report the median residual of each, and pass if world axes is under half of both others.

`timing_summary(frames, cameras)` takes the camera models: per camera, the `rolling_shutter_direction` name, and the median and range of `capture_offset_ms`, of `(trigger + done) / 2 - pose`, of readout `done - trigger - shutter`, and of `shutter`.

`null_summary`: the share of null in `speed` and `accel`, and the share of exact zeros in `speed_z`.

`pick_review_frames`: up to 12 random frames, 6 with the worst `drawn` residuals, 6 with a `drawn` box near a left or right image edge, and 6 with a near fast object, deduplicated, and only frames with an image. Worst and edge rank on `drawn` so the list holds what the reviewer can see: at speed, the label's worst frames are rolling shutter that no image shows. Returns `(timestamp, camera, reason)`.

`output_hashes`: sha256 of each output file except `qa/` and `clips/`, keyed by relative path. This is check 6, and the file names and hashes stay on PACE. Only a run-to-run match or mismatch goes into the report.

### `synthetic.py`

```python
def write_synthetic_segment(root, split="training", context_name="synthetic_cut_in", n_frames=199,
                            cameras=("FRONT", "FRONT_LEFT", "FRONT_RIGHT"), seed=0) -> SegmentRef
```

Writes the six Stage 1 components in the real layout with the exact column names from `columns.py`, plus an extra `index` column to prove extras are tolerated. It exists so everything can run end to end with no Dataset access, and so overlays can be shown on Nov 5.

The scene is analytic, built in a world frame with a remote origin, and pushed through the same transforms as real data, so a transform bug shows up as residuals.

- **Ego**: decelerating along a gentle curve, with yaw rate near 0.05 rad/s. This is what reproduces the false lateral term from differencing vehicle-frame positions.
- **Objects**: a cut-in vehicle crossing from the adjacent lane into the ego path, a lead vehicle, a parked vehicle, a vehicle beside the ego (the ghost-box case), a crossing pedestrian, a sign, and one vehicle with `num_top_lidar_points=0` for part of its track.
- **Timing**: 10 Hz with sub-ms jitter and two dropped frames. `vehicle_pose` is the ego pose at about +50 ms. Image poses are at the nominal +37.5, +50, and +62.5 ms.
- **Labels**: box state at about +50 ms in the `vehicle_pose` frame. `speed` and `acceleration` are in world axes, null for a few rows, with `speed.z` always 0.
- **Oracle**: `projected_lidar_box` comes from the same projection, clamped to the image and omitted when fully outside. So label residuals are near zero by construction, and that only proves the wiring. It has no object motion, unlike Waymo's, so synced boxes on moving objects read up to about 3 px. Say so in the docstring.
- **Synced boxes**: the center moved to the most visible camera's capture time, expressed in that image's vehicle frame.
- **Images**: 1920x1280 JPEGs with flat-shaded boxes drawn from the same projection.
- **Parquet**: row order shuffled per component, because real files don't share an order.

Calibration uses Waymo's fixture from `camera_model_test.cc` for FRONT, with `rot_z(±45°)` yaw for the other two.

### `run_stage1.py`

```python
def process_segment(config, ref) -> tuple[Path, Report]
def main(argv=None) -> int
```

Flags: `--config`, `--split`, `--context` (a bare name, only letters, digits and underscores: a path is refused as bad arguments before anything is written), `--cameras` (override, passed to `load_config` so that it counts in `preprocessing_version`), `--synthetic` (write a synthetic segment under the outputs root first), `--repro-check` (process twice into separate directories and report only whether the hashes match).

Steps, each in `guarded`: schema check, load, tables, windows, render, checks, contact sheet. Print only counts and the report path, which holds the label rather than the context name. The report is written even on failure, and that includes SIGTERM, which is how Slurm ends a job at its time limit. Only SIGKILL leaves none, and with the old report deleted first, that reads as what it is. A missing column appears in the schema section as its position in `REQUIRED_COLUMNS[tag]`, since the report takes no names.

`main` sets `umask 077` before anything is written, so file privacy doesn't rest on the caller sourcing `env.sh`. Bad arguments and anything else no step guards, such as a bad config, print one fixed line, because argparse quotes what it rejects and a traceback prints paths, and either can hold a context name. `load_config` refuses an `outputs` path with an unset variable, which would otherwise be written under the working directory.

## Acceptance

These are the numbered checks in section 6 of the setup doc. On synthetic data everything must pass with no Dataset access. On real data the pixel limits are provisional until calibrated on the first segment.
