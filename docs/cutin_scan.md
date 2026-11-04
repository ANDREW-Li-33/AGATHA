# Cut-in scan

Rank the 1,000 labeled segments (training 798, validation 202) by how likely each holds a sudden cut-in, from `lidar_box`, `vehicle_pose`, `stats` and `camera_calibration` alone, a few GB in all. Camera images are fetched only for the listed segments, which Andrew judges by eye. Test segments have no labels.

Labels can't prove a cut-in: the scan finds candidates, and a person decides. The list is for picking a demo clip, so its top must be right. Losing a real cut-in is fine, a non-event near the top is not. The best clip is a cut-in the ego reacted to: it braked, and maybe steered a little.

The versions compared are round 2, round 3 before the steering test, round 3's curvature test, and now. Numbers are from synthetic segments with an exact ego unless a table says otherwise. A population is 1,000 segments, ranked in six mixes: the benchmark's full and few-dramatic ones, and the reviewers' light, adv, soft and roll.

## What counts, from labels alone

Perception v2 has no lane lines, so the ego's own driven path stands in for its lane. Stage 1's `build_ego_table` and `build_tracks_table` measure every box against it (`lateral_offset_m`, `along_path_m`, `on_driven_path`), given empty synced and projected tables, since the scan reads no camera component.

A **cut-in event** is one vehicle track with an onset and an entry:

- **Onset**: the last sample clear of the path, `|lateral_offset_m| >= 2.5` (`clear_m`), an adjacent lane, as lanes are 3.5 to 3.7 m wide. The last, as a car can wait in the next lane for 15 s.
- **Entry**: the first sample after onset in the path, `|lateral_offset_m| <= 1.0` (`inside_m`), within 6 s (`max_crossing_s`), as lane changes take 0.5 to 6 s. An entry uses up its onset, so each entry is its own event.
- **The stay**: `|lateral_offset_m| <= 1.5` (`stay_m`) from entry through its first sample 1.0 s (`min_stay_s`) or more later, or to the track's end. Crossing traffic fails it. A hole in a track is not its end.
- **Driving along the road before**: at the car's sample nearest 2 s (`own_move_lead_s`) before onset, its heading (`yaw_world_rad`) is within 0.5 rad (`max_heading_rad`) of the ego's path where the car is, the ego's yaw there or at its nearer end, and its speed is known and 2 m/s (`min_object_speed_mps`) or more.

At entry the car is **ahead**, `0 < front_gap_m <= 30` (`max_gap_m`); **roughly parallel to the ego**, `|heading_rad| <= 0.5`, which also stops crossing traffic up to about 2.5 m/s; **moving** at 2 m/s or more, with the ego at 3 m/s (`min_ego_speed_mps`) or more; **on the path, or just past its end**; and **in FRONT's view**, `|atan2(center_y_m, center_x_m - 1.5)| <= 25°`, 1.5 m being about where FRONT sits (`front_camera_x_m`) and 25° about its half field of view (`front_half_fov_deg`), close enough to need no calibration.

**The stay's 1.5 m** is the research template's "within 1.5 m of lane center after the maneuver". At 1.0 m, label noise would end a stay one sample in.

| Cut-ins settling 0.8 m off the line | 1.0 m | now |
|---|---|---|
| Fixes: lost to 5 and 10 cm of label noise | up to 24%, 70% | none |

**The heading** tells a lane change from a turn-in, still across the road 2 s before onset. Against the path, not the ego's nose, a curve ahead or the ego's own turn doesn't count against the car.

| 2 s before onset | no rule | against the nose | now |
|---|---|---|---|
| Fixes: side-street turn-ins found, all with short times to collision | 37% | none: 0.53 rad or more on curves | none: 1.06 rad or more, 0.84 at a 1 s lead |
| Costs: real cut-ins over 0.5 rad on tight curves, in a second turn lane, after the ego's turns | none | 9%, 15%, 39% | none: 0.02 rad at most |

**The speed** drops a car pulling out of a parking spot, standing still 2 s before onset, and none of 1,564 moving real cut-ins.

| 150 scenes of each kind | round 2 | now |
|---|---|---|
| Fixes: parking pull-outs found, of 50 | 50 | 0 |
| Costs: cut-ins found after a standstill at a light, slowing behind a blocked lane, or crawling in a queue | 150, 126, 70 | 117, 110, 45 |

33 standstill starts were under 2 m/s then, none found; at a 1 s lead, only 5 would be. A car rolling along the curb for 1.5 s or more first, like a bus leaving its stop, is kept: a known risk.

**The end of the segment.** Needing `on_driven_path` at entry blinded each segment's last seconds, missing cut-ins that make the ego stop short. So entry may sit up to 8 m (`path_end_allowance_m`) past the ego's last pose, on the path's straight extension, the car's place along the path being its `along_path_m` plus the ego's `path_s_m`. Anything before the path's start or past the allowance is refused, and an entry that needed it sets `past_path_end`.

| Synthetic scenes | no allowance | 8 m, now | 12 m |
|---|---|---|---|
| Fixes: queue stops found, where a car cuts in and both stop, of 25 | 8 | 25 | |
| Costs: false events, of 120 segments ending in a tight turn | | none | 6 |

## Severity

Severity is braking, in m/s², from onset to 3 s after entry (`braking_window_s`), as a planner brakes once it sees the car start over.

- `required_decel_mps2`: the hardest braking the ego would have needed, `closing² / (2 × front_gap_m)` at its largest, with `closing = -v_rel_ego_x_mps`, over the car's samples ahead (`front_gap_m > 0`) and closing, from entry on also in the path, `|lateral_offset_m| <= 1.5`. It's 0 when nothing closes, never NaN, and capped at 100 m/s², ten times any car's braking, so a millimeter's gap can't make a number the report refuses.
- `ego_min_a_long_mps2`: the ego's lowest `a_long_mps2` over the span.
- `severity_mps2`: the larger of the two, braking being `-ego_min_a_long_mps2`, and 0 if unknown. Never below 0.

Round 2 ranked by time to collision after entry, but an early-braking planner often opens the gap by then, and noise on a steady gap gave 2 to 300 s. Required braking, the deceleration rate to avoid a crash (DRAC), is 0 when nothing closes, grows smoothly, and shares braking's unit.

| Synthetic scenes | round 2 | now |
|---|---|---|
| Fixes: 864 cut-ins braked for early, each among 27 mild ones: median place, and top-10 places | 28, 105 | 1, 864 |
| ... a same-speed zipper merge outranks a cut-in braked for before entry, of 40 noise seeds | 40 | 0 |
| Costs: 150 dramatic cut-ins braked for only before onset: top-10 placements of 750, full mix (few-dramatic) | 0 (0) | 154 (369) |

Braked for at the car's first drift, they place 517 (695); counting braking from 2 s before onset would give 515 (715), but also counts braking for other things and gives rolling pull-outs 2 more places. Braking for anything counts, as for a light, so a non-event whose own move gets through ranks near the top: the steering test ranks most last.

At entry the scan also keeps `headway_s = front_gap_m / ego speed_mps`, `closing_mps = -v_rel_ego_x_mps`, `crossing_s` from onset to entry, and `peak_lateral_mps`, the largest change in `lateral_offset_m` per second from onset to entry.

## The car's own move

When the ego later moves into a car's lane, the car's `lateral_offset_m` falls to 0 though it never moved. `own_move_m` measures the car against where the ego was going before anyone moved:

1. The pose: the ego row nearest 2 s (`own_move_lead_s`) before onset, clamped to the segment.
2. The circle the ego drove there: curvature `yaw_rate_radps / speed_mps`, capped at 0.2 per meter as no car turns tighter than about 5 m, and a straight line below 1 m/s, where that ratio is mostly noise.
3. The car's signed distances from the circle at onset and at entry, from `pos_local_x_m` and `pos_local_y_m` in the pose's frame.
4. `own_move_m = |distance at onset| - |distance at entry|`, NaN only when the circle or a position is unknown.

A car that holds its lane scores about 0, and a real cut-in 1.5 or more while the circle follows the lane. Under 1.0 m (`min_own_move_m`), or unknown, the event ranks last: the ego probably moved. It's never dropped, since where the road's curvature changes the circle misjudges real cut-ins.

**Why 2 s.** Round 2's 1 s pose often sat inside the ego's own maneuver, so the ego's move read as the car's.

| Ranked last | round 2, 1 s | now |
|---|---|---|
| Fixes: the ego's own lane changes, 2 to 8 s long, and the 8 s ones | 86%, 50% | 98%, 92% |
| ... aborted passes, of 400 | 308 | 400 |
| Costs: real cut-ins near the ego's own 90° turn, turning alongside it in a second turn lane, where the curvature changes | 60%, 48%, 39% | 88%, 73%, 50% |
| ... dramatic ones starting in the second after the ego's own lane change (of 114), on a highway where a curve starts or ends (100) | 18, 10 | 109, 23 |

Without the steering test, the lead alone ranks 373 of those aborted passes last and 88 of those cut-ins. From 1 to 2 s after the ego's lane change, 20 of 117 rank last, and none later. On straight roads and constant curves real cut-ins keep their own move, also at 0.2 mrad of yaw noise; at 0.5 mrad, 7 and 8 of 100 highway ones don't.

## The steering test

The circle stands for the lane only while the ego keeps driving it, and leaving it pulls the ego sideways by its curvature's change times its speed squared. So the scan takes `|curvature - the pose's curvature| × speed²` at each frame from the pose to 0.5 s (`steer_window_end_s`) before onset, never less than the pose's own frame; the largest is `lateral_change_mps2`. Past 0.5 m/s² (`max_lateral_change_mps2`), the ego may have steered, out to pass, over a lane or where the road bends, so the event ranks last, as with an unknown pull: an unknown speed, or an unknown yaw rate at 1 m/s or more, where the circle isn't simply straight. The own move is kept either way.

In an aborted pass, say, the pose sits late in the pull-out, turning back, so the circle bends across the car's lane: the car seems to come 2 m nearer, and would rank on the ego's braking.

| 30 populations at 3 cm of wander and 0.2 mrad of yaw noise, half the reacting egos nudging away once the car is 2.5 m out, unless said | round 3 before the steering test | round 3's curvature test | now |
|---|---|---|---|
| Real top-10 places of 1,800, six mixes | 1,648 | 1,741 | 1,727 |
| ... exact ego, no nudges | 1,669 | 1,770 | 1,730 |
| ... with early nudges, 0.2 to 1.2 s after the car starts over | 1,648 | 1,741 | 1,724 |
| Fixes: of 400 extra draws, exact ego, keeping an own move: aborted passes (3 to 4 s pull-outs), dual-turn lane changes | 27, 7 | 0, 3 | 0, 3 |
| Costs: dramatic cut-ins ranked last, of 2,812, and with early nudges | 1, 0 | 982, 957 | 1, 49 |
| ... targeted, exact ego: dramatic, starting in the last second of the ego's own lane change (of 113), in the second after it (114), where a bend starts or ends (145) | 71, 88, 23 | 113, 110, 60 | 111, 109, 26 |

**Why m/s².** Curvature is that pull over speed squared, so a curvature limit is strictest at city speeds. Round 3's curvature test, 0.002 per meter to entry, took wander and nudges for steers. **Why 0.5.** At 0.25 m/s², 35 of those 2,812 dramatic cut-ins rank last, 147 with early nudges. At 1, 20 of the 400 aborted passes keep an own move.

**Why 0.5 s before onset.** A planner may react to the car's first drift, and a nudge away from the car widens its offset from the ego's path, delaying onset itself.

| 30 populations, 3 cm and 0.2 mrad, unless said | window to onset | now |
|---|---|---|
| Fixes: with early nudges, ranked last: dramatic cut-ins, of 2,812; highway and moderate ones, of 400 and 396 extra draws | 235; 49, 53 | 49; 36, 33 |
| ... dramatic places in those few-dramatic top 10s, of 90 | 79 | 89 |
| ... targeted, exact ego, ranked last: 2 cm jolts (290 dramatic, 300 highway), re-centring (1,172 dramatic), one pose 5 mrad off (300 highway) | 91, 112, 376, 300 | 63, 83, 316, 245 |
| Costs: of 400 extra draws, keeping an own move: lane changes where a bend starts or ends, plain ego lane changes | 27, 13 | 28, 16 |
| ... real top-10 places, exact ego, of 1,800 | 1,737 | 1,730 |

Real frames come 99.9 to 100.1 ms apart (the repo's own Stage 1 reports), so the frame 0.5 s before onset often lands a hair after the window's end: on real data the window ends 0.5 to 0.6 s before onset. Ending it at 0.6 s: exact ego 1,728 real places against 1,730; early nudges 1,723 against 1,724, and 27 dramatic cut-ins ranked last against 49.

## The list, in two parts

`cutin_top.txt` holds the ranked top `--top`, then the top `--top` of round 3 before the steering test's ranking, as if the ego never steered (`lateral_change_mps2` read as 0), less any already listed, in that order: up to 20 at the scan job's `--top 10`. Pages 1 to `scan.json`'s `top.segments` are the ranked top 10, and the next `top.steering_test_off` pages rank there only with the test off; Andrew judges both. Over the benchmark's 30 populations the second part holds mostly aborted passes, plus 13 to 15 real cut-ins per few-dramatic mix with early nudges. Only the pages show what the test costs: `car_moved` and `ego_steered` barely change with early nudges, and `top.steering_test_off` counts aborted passes too.

`max_lateral_change_mps2 = 1000` in `configs/scan.toml` switches the test off, as no ego pulls that hard: the curvature cap allows 0.4 × speed², 640 m/s² at 40 m/s. Switch it off for a later run only if pages 11 to 10+K, K being `top.steering_test_off`, hold more real cut-ins than pages 11−K to 10, which they would replace; a tie keeps it on. In 1,260 synthetic lists, switching off always swapped exactly those pages, and the rule fired 21 times, all with early reactions. A second-part page reading "pull before onset unknown", as beside lost frames, stays last even with the test off: leave it and page 11−K out.

## Ranking

A segment ranks by its most severe event. Events sort by:

1. Last when the own move is under `min_own_move_m` (1.0 m) or unknown, or the pull is past `max_lateral_change_mps2` (0.5 m/s²) or unknown.
2. `severity_mps2`, largest first.
3. `headway_s`, smallest first.
4. Split and name, so a rerun lists the same segments.

Headway alone ranks a car tucking in just past the bumper as the gap opens above one that makes the ego brake hard.

| The earlier review's four example segments | by headway | now |
|---|---|---|
| Fixes: the place of the one real hazard, which needed 3.1 m/s² and braked 4.5 | 4th | 1st |

All thresholds live in `configs/scan.toml`, as Stage 1's `preprocessing_version` hashes `configs/stage1.toml` and tuning the scan must not invalidate Stage 1 outputs.

## A fast segment with traffic on the left

Stage 1's open gap is FRONT_LEFT accuracy at highway speed: its check 3 measures only boxes wholly inside the image, 2 px in (`border_margin_px` in `stage1.toml`), and the fastest segment tried gave it none. So the scan counts `left_at_speed`: vehicle boxes taken while the ego goes 20 m/s (`fast_ego_speed_mps`) or more whose eight corners all fall inside FRONT_LEFT's image with that margin, through the segment's own calibration by Stage 1's `load_cameras`, `row_corners`, `transform_points`, `box_rectangle` and inside test, as check 3 projects them. The top 3 above 0 go to `fast_left.txt`, for Stage 1's checks.

| The last review's two synthetic highway segments | by bearing and range | now |
|---|---|---|
| Fixes: samples counted from three cars drifting past in the next lane, which FRONT_LEFT never shows whole | 399 | 0 |

`download_labels.sbatch` fetches `camera_calibration`, a row per camera, with the labels. A segment whose FRONT_LEFT calibration can't be read (missing, empty, cut short, or its row missing or unusable) keeps its events, counts 0 with `front_left_calibrated` false, and `scan.json` counts it. Round 3 before the steering test failed such a segment outright.

## Outputs

| Path | What it holds |
|---|---|
| `$WOD_OUTPUTS/scan/events.parquet` | One row per event: split, context name, object ID, alias, onset and entry timestamps (int64) and `t_s`, `headway_s`, `closing_mps`, `required_decel_mps2`, `ego_min_a_long_mps2`, `severity_mps2`, `crossing_s`, `peak_lateral_mps`, `own_move_m`, `lateral_change_mps2`, `past_path_end` |
| `$WOD_OUTPUTS/scan/segments.parquet` | One row per segment: split, context name, frame count, ego speed median and max, vehicle tracks, events, best `headway_s`, `left_at_speed`, `front_left_calibrated`, and `time_of_day`, `weather`, `location` from `stats`' first row |
| `$WOD_HOME/selected/cutin_top.txt` | Context names, mode 600: the ranked top `--top`, then the second part |
| `$WOD_HOME/selected/fast_left.txt` | Up to 3 context names, mode 600 |
| `$WOD_HOME/review/cutins/cutin_<rank>.html` | The review pages |
| `$WOD_OUTPUTS/reports/scan.json` | Aggregates only, through `report.Report` |

All but the report stay on PACE, private. `scan.json` holds `config`, every threshold, and:

- `segments`: `scanned`, `failed`, `failed_by_type` by exception type, `with_left_at_speed`, `without_calibration`.
- `events`: `events`, `segments_with_events`, `car_moved` (an own move of 1 m or more, pull within the limit), `ego_steered` (pull past it or unknown), `past_path_end`, and spreads of `headway_s`, `required_decel_mps2`, `ego_min_a_long_mps2`, `severity_mps2`, `crossing_s` and `lateral_change_mps2`: the count, and from 10 values the min, quartiles and max, NaNs dropped. From fewer, those five could be the events' own values.
- `conditions`: counts by `time_of_day` and `weather`, as Waymo's documented values or other.
- `top`: `segments`, the ranked part, `steering_test_off`, the second part, and `daytime`, over the whole list.

It never holds names, labels, or per-event or per-segment values. A list of labels is a context list.

## Interfaces

`src/agatha/scan.py`:

```python
@dataclass(frozen=True)
class ScanConfig:  # from configs/scan.toml
    clear_m: float = 2.5
    inside_m: float = 1.0
    max_crossing_s: float = 6.0
    min_stay_s: float = 1.0
    stay_m: float = 1.5
    max_gap_m: float = 30.0
    max_heading_rad: float = 0.5
    min_object_speed_mps: float = 2.0
    min_ego_speed_mps: float = 3.0
    front_camera_x_m: float = 1.5
    front_half_fov_deg: float = 25.0
    path_end_allowance_m: float = 8.0
    braking_window_s: float = 3.0
    own_move_lead_s: float = 2.0
    min_own_move_m: float = 1.0
    max_lateral_change_mps2: float = 0.5
    steer_window_end_s: float = 0.5
    fast_ego_speed_mps: float = 20.0

def load_scan_config(path) -> ScanConfig
def find_cut_ins(tracks: pd.DataFrame, ego: pd.DataFrame, config: ScanConfig) -> pd.DataFrame
def scan_segment(root, ref, stage1: Stage1Config, config: ScanConfig) -> tuple[pd.DataFrame, dict]
def main(argv=None) -> int  # console script agatha-scan
```

`find_cut_ins` is pure, on Stage 1's own table shapes, so it runs on a Stage 1 segment as on the scan's light tables. `scan_segment` reads `stats`' first row, its columns in `columns.py`, and FRONT_LEFT's calibration if it can.

The CLI takes `--config configs/stage1.toml`, `--scan-config configs/scan.toml`, `--splits training validation`, `--workers N` (a process pool, a segment per task), `--top K`, and `--limit N` to try a few segments. It prints only counts, `top: K, steering test off: M` among them, and the report's path. A segment's failure is counted by type, never by message, and the run goes on. A worker killed outright, by the out-of-memory killer say, breaks the pool: the run exits 1, with `BrokenProcessPool` in `scan.json`. Workers take SIGTERM's default action: a forked one would otherwise catch the pool's SIGTERM with main's handler and keep a broken pool waiting forever.

## Review

`agatha-scan review` writes a self-contained page per listed segment, `$WOD_HOME/review/cutins/cutin_<rank>.html`, once Stage 1 has run on it with three cameras. A scan or review clears only `review/cutins/`, so Stage 1 contact sheets beside it stay.

Each event gets FRONT frames at 2 FPS from onset minus 1 s to entry plus 2 s, in the `full` profile, captioned with the alias, the time and onset or entry, and the cut-in vehicle drawn thicker. Its line gives the headway, the severity as "needed 3.1 m/s² to avoid it, braked 4.5 m/s²" (braking as severity counts it), the crossing time and the own move. Under `min_own_move_m`, it says the ego probably moved, not the car. Past `max_lateral_change_mps2`, it gives the pull: "The ego pulled 0.79 m/s² sideways before onset, so the own move may be wrong". An unknown pull reads "pull before onset unknown", claiming no steer, and an unknown own move "unknown". Andrew opens the pages in OnDemand.

## Running it on PACE

In the order each script's top comment also gives:

1. `download_labels.sbatch`: labels, poses, stats and camera calibration for every labeled segment, printing counts per split and component and the total size. gcloud's own output goes nowhere, as its errors name objects.
2. `scan.sbatch`, at `--top 10`: writes `cutin_top.txt` and `fast_left.txt`. Fetch `scan.json` with `pull_report.sh scan`.
3. `download_listed.sbatch cutin_top.txt`, then `download_listed.sbatch fast_left.txt`: each listed segment's camera components, its split found by its labels, and `camera_calibration` again, harmlessly. It prints line numbers and exit codes.
4. `review_cutins.sbatch cutin_top.txt`: Stage 1 with three cameras on each, then the pages. Judge pages 1 to `top.segments`, then the rest. If the job times out, no page is written; once Stage 1 has run, `uv run --frozen agatha-scan review --config configs/stage1.toml --scan-config configs/scan.toml` writes the pages on its own, in a short job sourcing `infra/pace/env.sh`.
5. `review_cutins.sbatch fast_left.txt`: Stage 1 with three cameras on each fast-left segment, whose own report judges FRONT_LEFT.

The two list jobs have 4 hours for up to 20 segments; single-segment Stage 1 jobs took 69 to 85 s of wall time on ICE, and a single-segment download 16 s (Slurm's accounting, Nov 1 and 2, 2026). If the labels predate the calibration, run `download_labels.sbatch` again first; rsync fetches only what's missing, and without it `without_calibration` counts every segment. The jobs print counts, ranks, line numbers and exit codes, never names. The lists and tables name contexts, so don't read them, and keep Slurm logs private.

## Known misses

Real cut-ins missed by decision:

- **Slow ones, and some on curves**, still angled over 0.5 rad to the ego's nose at entry, as a car at 3 to 5 m/s with a short lane change is.
- **One the ego steers around.** The driven path follows the ego away, so the car never enters it.
- **One in a segment's last second or so, worst on a curve.** Entry can lie past the allowance, as for a car entering 25 m ahead of an ego 1.1 s from the end at 12 m/s. On a curve the straight extension leaves the road, so the offset swings, +0.72 to -1.73 m within 0.8 s on a 120 m curve, and the stay fails.
- **A car slow 2 s before onset**: after a standstill, behind a blocked lane, or in a queue.

Found, but ranked last more often: real cut-ins near the ego's own turns, where the road's curvature changes, and around the end of the ego's own lane change. Lane-keeping wander bends the pose's circle too: at 5 cm and 0.5 mrad, round 3 before the steering test ranks last 68 of 390 mild, 47 of 396 moderate and 69 of 400 highway cut-ins, and now 68, 47 and 73.

The steering test ranks last a real cut-in whose ego moves sideways in its window, even briefly; the second part lists it if it would rank in the top 10 otherwise. The default populations hold neither:

- **A reaction more than about half a second before onset.** With early nudges, 49 of 2,812 dramatic cut-ins rank last, and 36 of 400 highway and 33 of 396 moderate ones (5 and 8 with no test). Nudging 0 to 0.5 s after the car starts over, or in the 1.5 s before, ranks 201 and 626 of about 2,810 dramatic ones last; few-dramatic top 10s keep 84 and 67 of 90 dramatic places, the second part listing most of the rest. A nudge spanning the pose, about 2 s before onset, bends the circle itself, so both parts lose most: 443 of 523 reacting cut-ins nudged 2 to 2.5 s before onset rank last, 374 with no test.
- **A lane-centering correction, a jolt, or one bad pose.** 316 of 1,172 dramatic cut-ins whose ego re-centres by 0.1 to 0.3 m over 1 to 2.5 s, from 3 s before the car starts over to 1 s after, rank last (158 with no test); a 2 cm jolt sideways and back within 0.6 to 1.2 s, 63 of 290 dramatic and 83 of 300 highway ones (3 and 5); one pose 5 mrad off, 245 of 300 highway ones (19). 1 cm, or 3 mrad, costs at most 2.

Real cut-ins braked for only before onset, on the car's turn signal say, rank lower than they should: see Severity.

## Known risks

These non-events can still reach the top 10 in 30 exact-ego populations of the reviewers' mixes, each a guess at what real segments hold:

| Non-event | Per 1,000 | Few-dramatic top-10 places, of 300 | Full mix |
|---|---|---|---|
| A lane change before a light, braking at 2 to 3.5 m/s² | 40 | 23, at 2 to 6 | 7 |
| ... at 1.5 to 2.5 m/s² | 20 | 7, at 3 to 8 | 1 |
| A lane change where a bend starts or ends, braking for a light | 10 | 7, at 3 to 6 | 0 |
| A lane change just out of a dual turn lane | 10 | 3, at 3 to 5 | 1 |
| A pull-out that rolls along the curb first, like a bus leaving its stop | 20 | 7, at 7 to 10 | 0 |

In the first four the road's curvature changes between the pose and the car, so the circle leaves the car's lane, and the ego meets the change after onset, where no steering test before onset can see it: 26 of the 28 lane changes before a light that keep an own move, 9 of 10 with gentler braking and all 16 at a bend or dual turn. Of the 3 with no bend, 2 are at 25 m/s or more, where the circle reaches 100 m ahead. They leak about as with no test. At 3 cm and 0.2 mrad, lane changes before a light take 24 places, and rolling pull-outs 19 to 31 when egos react at or before the car's first move, as real cut-ins move to pages 11 and on.

**A jolt at the pose**, on any road. A 1 to 2 cm jolt sideways and back across the pose bends its circle, and the frames after it pass the test, so a lane change behind a braking car can look like a cut-in, its page showing only the own move. One per segment, at 3 cm and 0.2 mrad, 1 cm jolts lift lane changes before a light from 24 to 29 light-few places. 2 cm ones give ego merges and lane changes while stopping 13 of the 1,500 places in the five few mixes (18 for round 3 before the steering test), at places 4 to 10.

Two more turned up in targeted draws placed into the populations:

- **An aborted pass pulling out over 5 to 8 s**, not the benchmark's 3 to 4, turns back so gently, 0.17 to 0.5 m/s², that 70 of 708 keep an own move (81 before the steering test, 4 with round 3's curvature test). Braking at about 3 m/s², those take a few-dramatic top-10 place 98.6% of the time: a 10% chance over all 708.
- **A highway lane change over 8 s**, braking at 1 to 2.5 m/s² behind a slower car. Its pose sits inside the lane change, and the 100 m circle at 27 m/s crosses into the car's lane: 152 of 374 keep an own move in every version, and those take a place 39.1% of the time: a 16% chance over all 374. Its pull, 0.06 m/s² at the median, is wander's, so no limit can catch it.

Two rules were measured for these and not taken:

- The car also moving 2 m/s 4 s before onset: 6 of the 7 rolling pull-out places go, but so do 79 of the 117 standstill starts and 6 of the 45 crawling-queue cut-ins.
- Counting only the ego's braking beyond the car's own, so braking for a light counts for neither: lane changes before a light drop from 23 places to 4, but 150 cut-ins braked for once in front from 639 of 750 full-mix placements to 611, and noise in the car's acceleration lifts rolling pull-outs from 7 places to 12, or 224 unclipped, as speeding away reads as braking.

## Tests

`tests/test_scan.py`, on synthetic data, covers every rule and threshold from both sides, read from `scan.toml`; the synthetic segment; turn-ins, pull-outs, crossing traffic and track holes; severity, the own move and the path-end allowance; the steering test's formula, window, unknowns, wander and nudges; the ranking, on the ego's own maneuvers and three known risks; the list's order, cap, single entries and short lists; `left_at_speed` and `fast_left.txt`; Stage 1's tables against the scan's; `scan.json`; failures and SIGTERM; the review pages; and the PACE jobs under bash with stub `uv` and `gcloud`: no name printed, 20 segments, 4-hour limits.
