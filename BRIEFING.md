# Briefing for Q&A models: AGATHA

*Written by the main Claude Code session that built this project with Andrew, and brought fully up to date by a side session at Andrew's request. Snapshot: 2026-11-04 11:05 EST. The main session keeps section 8, "Current status", up to date. This file is not part of any build or polish round: agents doing that work, leave it alone.*

## 1. Your job and the rules

You answer Andrew's questions about this project. Andrew prefers short, plain, natural answers to dense technical reports. Lead with the answer, then give the one or two facts behind it, and cite files as `path:line`.

- **The code is the source of truth.** This file is a snapshot. Before you answer anything about code, configs, docs or results, read the current file. If this briefing and a file disagree, trust the file and say so.
- **Read only.** Don't edit, create, move or delete anything in this folder. The main session, its side sessions and their background agents edit these files, nothing is in git, and there is no undo. If Andrew wants a change, describe it, so Andrew can hand it to the main session.
- **Don't run the tests or benchmarks** unless Andrew asks: a background round may be running them. If asked, run `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider` from this folder (about 2 minutes). Read-only commands such as `ls`, `cat`, `sed -n`, `grep`, `find` and `wc` are fine.
- **Never touch PACE.** Don't run `infra/pace/ice.sh`, `sync.sh`, `submit.sh`, `pull_report.sh`, `ssh` or `scp`. They act on Andrew's account and on real data. Never read or print an API key file.
- **The data boundary.** No real Waymo data enters any AI session, yours included: no frames or images, table rows, box values, tracks or segment (context) names, no list of report labels (a label is an unsalted hash of a name), and nothing from a real segment's Stage 2 reports beyond counts. Only aggregate numbers come back from PACE: the JSON files in `reports/` and the counts in `docs/stage2.md`. Never ask Andrew for screenshots of the review pages or reports, or for real data. If Andrew offers some, decline, and ask for page numbers or counts instead. The public tutorial segment's name is the one exception.
- **What Waymo cleared.** On 2026-11-04 Andrew reported that Waymo answered in writing: Stage 2 may send real camera frames and track data to Claude's API. That clears the pipeline, not these sessions, so the rule above stands until Andrew says otherwise. The background is in `docs/stage1_setup_and_decisions.md`, section 3, which was written before the answer.
- **Andrew's pronouns aren't stated.** Use "Andrew" or they/them.
- **If you can't read files** (for example, you're a chat model this was pasted into), say when an answer depends on code you haven't seen, and ask Andrew to paste the current file. Don't guess line numbers.

## 2. Staying current

Files change while you answer: the main session, its side sessions and their background agents keep editing code and docs. Before each answer:

1. From this folder, list what changed since your last check:
   ```bash
   find . -type f -newermt "2026-11-04 11:05" -not -path './.venv/*' -not -path './.pytest_cache/*' -not -path '*/__pycache__/*'
   ```
   Put in the time of your last check, which `date '+%Y-%m-%d %H:%M'` gives you. The first time, use this file's snapshot time.
2. Re-read every changed file the question touches. If `BRIEFING.md` itself changed, re-read all of it: the main session corrects it, as well as updating section 8.
3. Recount anything countable instead of quoting this file: tests, `scan.py`'s lines (`wc -l`), the scan doc's words (`wc -w`), and Stage 2's counts and costs (`docs/stage2.md`).

Moving right now (section 8): nothing. No background work is editing files.

## 3. The project

**What.** AGATHA (`agatha` 0.1.0 in `pyproject.toml`): driving incident detection, retrieval, and 3D reconstruction on the Waymo Open Dataset. Until 2026-11-04 it was called `driving-incident-reconstruction`, with the package `waymo_ingest`. It uses Python 3.12 and uv. It implements Andrew's design doc v0.1 of 2026-11-01, at `~/Downloads/waymo_incident_project_implementation_v0_1.docx` (read it with `pandoc -t markdown`).

**Why now.** Andrew, an MS student at Georgia Tech, has a recruiter screen for Waymo's Simulation intern role (Summer 2027, job 5442) on Thursday 2026-11-05, 11:10 to 11:25 AM PT. It's a 15-minute recruiter call, not a technical round. `scenario-miner/PREP.md` is Andrew's prep plan: the pitch, likely questions, and a map from the job description to the code. Andrew must be able to explain every line in later technical rounds, so the code is kept short (`docs/stage1_architecture.md`, principle 5).

**Data.** Waymo Open Dataset Perception v2.0.1: Parquet files at `gs://waymo_open_dataset_v_2_0_1/<split>/<component>/<context_name>.parquet`, one segment per file. A segment is a log of about 20 s at 10 Hz. 1,000 segments have labels: 798 in training and 202 in validation. The test split has none.

**Compute.** PACE ICE, Georgia Tech's instructional cluster, which one of Andrew's courses covers. The real data stays there (section 1).

**Stages.** This is the outline; the design doc has the details.
- **Stage 1** (milestone M1): one segment's Parquet components become full-rate tables (frames, ego, tracks), SceneWindows, overlays, clips and a contact sheet, plus a report of aggregate checks. Done, on synthetic and real data.
- **The cut-in scan**: built on Stage 1 to find real cut-ins from labels alone. Its list feeds the review pages, and its fast-left list closed Stage 1's last gap. Done, and run on real data.
- **Stage 2** (milestone M2): a two-pass Claude VLM. Pass 1 finds candidate events in each SceneWindow at about 2 FPS. Pass 2 verifies each candidate at 5 to 10 FPS and writes an IncidentReport. The design doc puts it in `src/vlm_detection/`; it lives in `src/agatha/vlm/`. **Built, run live on the synthetic segment, and, once Waymo cleared it, run with Claude on the real segments of all ten cut-in pages, on PACE** (section 8, `docs/stage2.md`, "Live run" and "First real run"). The old fallback for real data, an open-weight model on an ICE L40S GPU, wasn't needed.
- **M3, evaluation:** Pass 1's recall and Pass 2's precision, against 20 to 30 hand-annotated segments. Not started. Andrew's verdicts on the ten pages, set against Pass 2's, are a first look (section 8).
- **Stage 3** (M4): incidents in Supabase PostgreSQL with private storage, and a Streamlit explorer. Not started. Whether Waymo's answer covers Supabase isn't settled.
- **Stage 4** (M5): a PINGS mesh from the LiDAR, camera and poses, and a CARLA replay of the cut-in. Its two spikes are deferred: Spike A replays the cut-in on a stock CARLA map, and Spike B loads a PINGS mesh into CARLA.
- **M6:** the MVP end to end. One cut-in segment is classified by the two passes, found in Streamlit, watched as plain video, and replayed in CARLA. Not started. Andrew picked page 1's segment, but Pass 2 rejected its cut-in and confirmed one only on pages 5 and 9, so the choice may change. That's Andrew's call.

**Relation to scenario-miner.** The `scenario-miner/` folder holds Andrew's earlier prototype from 2026-10-23 ("VLM + 3D sensor/HD-map scenario mining and road-change detection on the Waymo Open Dataset"), kept as it was as the style reference, and `PREP.md`, Andrew's interview prep.

## 4. Where things live

In this folder:

| Path | What it holds |
|---|---|
| `src/agatha/` | The one package: Stage 1, the cut-in scan, and Stage 2 in `vlm/` (section 5) |
| `scenario-miner/` | Andrew's earlier prototype, kept as it was, with `PREP.md` (interview prep) |
| `configs/stage1.toml` | Stage 1's settings and check limits. Hashed into `preprocessing_version` |
| `configs/scan.toml` | The scan's thresholds, kept apart so tuning the scan doesn't invalidate Stage 1 outputs. Each key has a comment saying why |
| `configs/stage2.toml` | Stage 2's model (`claude-opus-5-5`, effort medium), Pass 2's settings, and the data guard: `real_data_cleared = true` since Waymo's answer on 2026-11-04, with a comment saying what it covers |
| `tests/` | 1,272 tests at the last full run, all on synthetic data. `test_scan.py` is the largest, and `test_vlm.py` (143) covers Stage 2. The guard's tests set `real_data_cleared = false` themselves, so they don't depend on the shipped value |
| `infra/pace/` | The PACE scripts (section 5) |
| `docs/stage1_architecture.md` | Stage 1's principles, layout, tables and per-module interfaces |
| `docs/stage1_setup_and_decisions.md` | Gates (§1), setup (§2), the data boundary (§3), changes from the design doc (§4), defaults (§5), the acceptance checks (§6), drafts of the email to Waymo and the note to the instructor (§7), what to show on Nov 5 (§8), a PACE quick reference (§9), first real results (§10). §3 and §7 predate Waymo's answer |
| `docs/cutin_scan.md` | The cut-in scan, end to end. Kept under 5,000 words by rule |
| `docs/stage2.md` | Stage 2: the taxonomy, both passes, the contracts, validation, outputs, Claude, the data guard and its clearing, running it on PACE, the synthetic live run, the real runs on all ten pages, and what's left for M3 |
| `docs/research/2026-11-01/` | Source notes with URLs: license, PACE, schema, geometry, critique |
| `reports/` | Aggregate reports from PACE (sections 6 and 7), and three from the public tutorial segment |
| `pyproject.toml`, `uv.lock`, `.venv/` | The environment. The CLI entry points are `agatha-stage1`, `agatha-scan` and `agatha-vlm`. The Anthropic SDK is an optional extra (`uv sync --extra claude`). The Mac's environment has it, and so does PACE's, through the Stage 2 script's `uv run --extra claude` |

Nothing is in git. The project has no README; `scenario-miner/README.md` is the old prototype's.

On PACE ICE, for reference only, since you can't reach it:
- The login node is `login-ice.pace.gatech.edu`, user `ali497`, through a master SSH connection Andrew opens with Duo (`infra/pace/ssh_config`). It dropped twice on 2026-11-04, and Andrew opened it again each time.
- The code is at `~/agatha`, copied there by `sync.sh`, which leaves out `scenario-miner/`. The copy from before the rename, `~/driving-incident-reconstruction`, and its environment are still there, unused, until Andrew says to delete them.
- The environment is `~/scratch/venvs/agatha`, with uv 0.9.17 from `module load`, and the Anthropic SDK since 2026-11-04. `build_env.sbatch`'s `uv sync --frozen` removes the SDK, and the next Stage 2 run puts it back.
- Claude's API key is in `~/.config/agatha/anthropic.env` (mode 600, in a mode 700 folder), outside `~/agatha` so `sync.sh` never touches it. The Mac keeps its own copy in the same place.
- For PACE work, use job 6798591, Andrew's OnDemand VS Code session on an L40S GPU node (16 CPUs, 96 GB), until about 18:00 EST on 2026-11-04. Commands run inside it with `srun --jobid=6798591 --overlap` from the login node. After that, Andrew starts a new job.
- `WOD_HOME` is `~/scratch/waymo`. In it:
  - `v2_0_1/` holds the data (`WOD_ROOT`).
  - `outputs/` holds the Stage 1, scan and Stage 2 outputs (`WOD_OUTPUTS`), with reports in `outputs/reports/<label>.json` and Stage 2's under each segment's `stage2/`.
  - `selected/` holds the scan's lists, `cutin_top.txt` and `fast_left.txt`.
  - `review/cutins/cutin_<page>.html` are the review pages, and `review/stage2/cutin_<page>/` links to each page's Stage 2 results.
  - `logs/` holds the Slurm logs and the masked Stage 2 logs.
  - `tools/` holds the Stage 2 batch driver and its helpers (section 5).
- Andrew views the review pages in OnDemand (<https://ondemand-ice.pace.gatech.edu/>), from a Jupyter session: right-click a page in its file list and choose Open in New Browser Tab. VS Code and plain file viewers show the raw HTML instead of the page. Stage 2's results are JSON, so any viewer there shows them.

## 5. Code map

`src/agatha/`:

| Module | Role |
|---|---|
| `run_stage1.py` | The `agatha-stage1` CLI: one segment to tables, windows, images and a report. Flags: `--config`, `--split` (default training), `--context` (default `synthetic_cut_in`), `--cameras`, `--synthetic`, `--repro-check`. It prints only counts and the report's path |
| `loader.py` | Reads Perception v2 components into sorted, tidy DataFrames. Joins go by key, never by position |
| `columns.py` | Component tags, column names and enums, written by hand from Waymo's v2 dataclasses. The `waymo-open-dataset` package isn't used, since it pins TensorFlow |
| `geometry.py` | SE(3) transforms, Waymo's camera projection, box corners and wireframes. Its docstring defines the world, vehicle and camera frames |
| `tracks.py` | Full-rate ego and object motion, by central differences in world axes. `EgoPath`, the ego's own driven path, stands in for its lane. Builds the ego and tracks tables, and aliases such as `V012` |
| `sampling.py` | Gap reports, frame selection on real timestamps, and window arithmetic |
| `scene_window.py` | SceneWindows, as pure views of the full-rate tables |
| `contracts.py` | The SceneWindow contract (pydantic), with its three frames: world, segment_local and ego_at_sample |
| `store.py` | The frames table, `segment.json` (the commit point), and reading and writing a segment's outputs |
| `render.py` | Wireframe overlays, the render cache, H.264 clips, and the one-file HTML contact sheet |
| `checks.py` | Stage 1's acceptance checks (section 6) |
| `report.py` | The data boundary in code. The report is the only file that leaves PACE, and it fails closed: only numbers and a fixed vocabulary get in. `context_label` and `guarded` live here |
| `config.py` | Loads the TOML config. `preprocessing_version` includes a hash of the package's source |
| `files.py` | `write_atomic` |
| `synthetic.py` | A synthetic segment in the real v2 Parquet layout: an ego and nine objects on a curved road. The tests and `--synthetic` runs use it |
| `scan.py` | The cut-in scan and the `agatha-scan` CLI: `scan` (the default) or `review`, with `--config`, `--scan-config`, `--splits` (default training and validation), `--workers`, `--top K` (default 10) and `--limit N` (section 7) |
| `vlm/schema.py` | Stage 2's types: the 14-label taxonomy in 3 categories, the strict answer models whose JSON schemas constrain the model's output, the `IncidentReport`, and the referential checks: `fit_candidate` (Pass 1, which clips a span to its window rather than lose the event) and `check_verdict` (Pass 2, strict, with the event taking its labels' strongest status) |
| `vlm/model.py` | The model interface (`VisionModel`: `model_id`, `remote`, `respond`), `ClaudeModel` (the Anthropic SDK, imported only when a request is sent, with server-side refusal fallbacks on; with `dry_run` set it writes requests and sends nothing) and `FakeModel` (tests) |
| `vlm/prompts.py` | The two system prompts (rule, taxonomy, how to read the inputs, the task), `PROMPT_VERSION`, and request content built from a SceneWindow: captioned images, then compact JSON |
| `vlm/run.py` | Pass 1, the merge (at most one candidate per window in a group), Pass 2 with its UNCERTAIN retry (only when it shows more frames), its `max_images` cap and `max_request_mb` budget, `duplicate_of` marking, the response store under `<segment>/stage2/`, the data guard (`check_guard`), and the `agatha-vlm` CLI: `--config`, `--vlm-config`, `--split` (must match `segment.json`), `--context` (a bare name), `--model claude` or `dry-run` (the default). It prints only counts and the label. `docs/stage2.md` has the details |

`src/agatha/vlm/` is 4 modules and 742 lines. Each module opens with a docstring that says what it's for, and most functions have a one-line docstring. To find a function, run `grep -n "def <name>" src/agatha/*.py src/agatha/vlm/*.py`.

`infra/pace/`:
- `env.sh` is sourced by every job. It sets `umask 077`, the `WOD_*` paths and uv's directories on scratch, and loads uv.
- `ssh_config` and `ice.sh` run commands on ICE through Andrew's master connection, with `BatchMode yes`, so a missing connection fails at once instead of burning password attempts.
- `sync.sh` copies the code to ICE (`rsync --delete`), never data, outputs or reports.
- `submit.sh` runs on ICE. It submits a job with its log in `~/scratch/waymo/logs/`.
- `pull_report.sh <label>` fetches one report by its exact label. Reports are never listed.
- The jobs: `build_env`, `test` (the synthetic suite), `egress`, `install_gcloud`, `download_segment <split> <context>` and `run_stage1 <split> <context>`. For the scan, in order: `download_labels` (labels, poses, stats and camera calibration for all 1,000 segments), `scan` (`--workers 8 --top 10`), `download_listed <list>` (camera data for each segment on a list) and `review_cutins <list>` (Stage 1 with three cameras, then the review pages, for `cutin_top.txt`).
- `stage2_cutin <rank> [claude|dry-run]`: Stage 2 on the segment at that rank in `cutin_top.txt`, so rank 1 is page 1. `dry-run`, the default, sends nothing. `claude` exports the key for `agatha-vlm` only. It prints only counts, with `cutin_<rank>` in place of both the label and the name, and links the results to `review/stage2/cutin_<rank>`. It runs as a job or inside one.

On PACE only, in `~/scratch/waymo/tools/`, outside the repo: `stage2_batch.sh <ranks>` runs Stage 2 on several pages in turn. For each page it runs Pass 1 alone (`pass1_only.py`), then a dry run, then costs Pass 2's unsent requests (`stage2_usage.py`). It holds back a page projected over $7 and stops starting pages after $65, then sends the rest and counts the reports by label and status (`stage2_labels.py`). `mask.sed` there holds the ten context names, to mask any that reach a log, so nobody reads it.

## 6. Stage 1: checks and results

The checks are in `checks.py`, with the full text in setup doc §6 and the limits in `configs/stage1.toml` under `[checks]`:
1. Joins use exact int64 keys, with no orphans and no duplicates.
2. A gap report replaces any fixed frame count.
3. Box residuals: each drawn box's edges against Waymo's `projected_lidar_box`, on boxes wholly inside the image. Each camera must meet a median of 5 px and a p95 of 30 px. FRONT's label path must meet them too, and also 1 px and 3 px on still objects while the ego is nearly stopped (under 1 m/s) and barely turning.
4. Track integrity: gaps, type changes, and speed jumps no object could make.
5. Waymo's `speed` field is in world axes, tested against two other readings.
6. Two runs give identical output hashes.
7. FRONT_LEFT and FRONT_RIGHT pass check 3 by a config change alone.
8. Frames are picked for Andrew's visual review.

Results:
- **2026-11-02** (setup doc §10): the public tutorial segment passes every check, and two runs matched. The fastest training segment (29 m/s) passes all but check 3 on FRONT_LEFT, which had no road user wholly in view, so it reads `no_data`. Andrew's visual review of both contact sheets passed.
- **2026-11-03**: the scan's fast-left list closed that gap. On its top segment, every check passes. FRONT_LEFT has 1,248 boxes there, all at 15 m/s or more, with a median residual of 0.46 px and a p95 of 7.1 px (FRONT 0.019 and 0.48 px, FRONT_RIGHT 0.62 and 5.3 px). The report is `reports/fast_left_1.json`.
- **Page 4's segment** (`reports/cutin_4.json`): Stage 1 exited 1, because check 3 failed on FRONT_LEFT. It had only 14 boxes, in a slow segment, with a median of 5.07 px against the 5.0 limit and a p95 of 34 against 30, probably from close-range boxes. FRONT and FRONT_RIGHT pass. The cut-in comes from the labels, and the pages show only FRONT, so page 4 still stands. Stage 2 ran on it without trouble, since it uses FRONT only.

## 7. The cut-in scan

**Goal** (`docs/cutin_scan.md`, top): rank the 1,000 labeled segments by how likely each holds a sudden cut-in, from labels alone (`lidar_box`, `vehicle_pose`, `stats` and `camera_calibration`, a few GB). Labels can't prove a cut-in, so the scan finds candidates and Andrew decides by eye. The list is for picking a demo clip, so its top must be right: missing a real cut-in is fine, but a non-event near the top isn't. The best clip is a cut-in the ego reacted to.

**How an event is found.** The logic is in `scan.py`, and the keys are in `configs/scan.toml`.
- Perception v2 has no lane lines, so the ego's own driven path stands in for its lane. Each car's sideways offset is measured from that path.
- Onset is the last sample at least `clear_m` (2.5 m) out. Entry is the first sample after it within `inside_m` (1.0 m), within `max_crossing_s` (6 s) of onset. The car must then stay in the path (`min_stay_s`, 1 s, within `stay_m`, 1.5 m).
- At entry, the car must be ahead of the ego by at most `max_gap_m` (30 m), roughly parallel to it (`max_heading_rad`, 0.5), moving at least `min_object_speed_mps` (2 m/s), and roughly in FRONT's view. The ego must be going at least `min_ego_speed_mps` (3 m/s).
- **Severity**, in m/s²: the larger of the braking the ego would have needed to avoid the car and its actual hardest braking, from onset to 3 s after entry.
- **Own move** (`own_move_m`): how far the car closed on the ego's line by its own motion, measured from the ego's pose 2 s before onset. Under `min_own_move_m` (1 m), the event ranks last: the ego probably moved, not the car.
- **Steering test** (`lateral_change_mps2`): how hard the ego pulled sideways off the circle it was driving, from that pose until 0.5 s before onset. Over `max_lateral_change_mps2` (0.5 m/s²), the event ranks last, because the own move may then be wrong. An unknown pull ranks last too.

**The lists.**
- Segments rank by their most severe event, with own-move and steering failures last (`_rank`).
- `cutin_top.txt` holds the ranked top K (10), then up to K more that rank there only with the steering test off. That's the review pages' order.
- `fast_left.txt` holds the few segments with the most vehicle boxes FRONT_LEFT shows whole while the ego goes 20 m/s or more.

**Outputs.** On PACE, `events.parquet`, `segments.parquet`, `cutin_top.txt` and `fast_left.txt` name contexts, so they never leave PACE and nobody reads them. `scan.json` is the aggregate report (`reports/scan.json` here). The review pages, one per listed segment, show FRONT frames at 2 FPS from 1 s before onset to 2 s after entry. The cut-in car's box is drawn thicker, and a line of numbers sits above: the headway, "needed X m/s² to avoid it, braked Y m/s²", the crossing time and the own move.

**How it was built.** Five rounds of build, attack and fix on 2026-11-02 and 03, each a Claude workflow: one agent built, independent agents tried to break it with synthetic benchmarks and planted bugs, and one fixed what held up. Round 5 is frozen as "round5final" (`scan.py`'s sha256 starts with `82563c0a`), with 1,119 tests passing, and that exact code ran on PACE. On the synthetic benchmark (30 seeds, 6 population mixes), 1,730 of 1,800 top-10 places are real cut-ins with an exact ego, 1,727 with 3 cm of pose noise, and 1,724 with early steering nudges. `docs/cutin_scan.md` has the details, the known misses and the known risks.

Decisions worth knowing:
- Precision at the top matters more than recall.
- A cut-in right after the ego's own lane change ranks last. That cost is accepted.
- Not adopted: relative braking, a 4 s pull-out rule, and a "median circle" for the own move. The jolt-at-the-pose risk that the median circle would address is documented and pinned by a test.
- On real frame times, the steering window effectively ends 0.5 to 0.6 s before onset. That's documented, with no code change.
- Turn the steering test off for a later run only if the pages past the top 10 hold more real cut-ins than the top 10's last pages they would replace. A tie keeps it on.

**The real run** (2026-11-03, `reports/scan.json`):
- 1,000 segments scanned in 2 min 6 s, with 0 failures.
- 23 events in 22 segments. In 11, the car clearly moved (`car_moved`); 1 failed the steering test (`ego_steered`); and 11 had a small own move. All 10 top segments have a credible best event.
- The steering-test-off part added nothing, so there are exactly 10 pages.
- Fast left: 3 segments, out of 80 with left-side traffic at speed.
- Real cut-ins were gentle. The needed braking was at most 0.34 m/s², the ego's hardest braking was 4.1 m/s², and the headway had a median of 1.5 s (minimum 0.69 s). 9 of the top 10 are in daytime.

The job chain that evening: tests 6791362 (1,119 passed on ICE), scan 6791363, downloads 6791364 and 6791365 (all exited 0, and the data on ICE is now 5.5 GB), and reviews 6791366 and 6791367. Stage 1 exited 0 on 12 of 13 segments; the exception is page 4's (section 6). Job 6786134 is Andrew's own OnDemand session.

Andrew judged all ten pages real cut-ins (section 8). Stage 2's Pass 2, held to the taxonomy's definition, confirmed a sudden cut-in on two of them (section 8).

## 8. Current status

*Updated by the main session. Last update: 2026-11-04 11:05 EST, a full update by a side session at Andrew's request.*

- **Stage 2 on all ten cut-in pages (2026-11-04).** With Claude on PACE, inside job 6798591. In all: **$29.07**, 93 requests (about 6.75M input and 104k output tokens), no failures and no fallbacks. That came to 39 reports: 8 confirmed, 29 rejected and 2 uncertain, 3 of them duplicates of an earlier report.
  - **Cut-ins.** Pass 1 proposed a SUDDEN_CUT_IN on every page, 13 in all after merging. **Pass 2 confirmed one only on pages 5 and 9.** Page 5 is one Andrew judged clearly real, and page 9 the one Andrew leaned unsure on. The likely reason for the rejections is the one the synthetic run showed: the taxonomy's cut-in needs limited clearance, and these cut-ins were gentle. Only the reports can confirm that.
  - **Page 1, the demo:** $5.90 and 7 reports. Pass 2 rejected its cut-in. It confirmed construction (lane closures, temporary traffic control) and a cross-traffic conflict there. Besides those and the two cut-ins, the only other confirmed event is page 3's temporary traffic control.
  - **How.** Page 1 went in three steps: Pass 1 alone to check its cost, a dry run that priced Pass 2 at about $4, then the full run. Pages 2 to 10 went through the batch driver in `~/scratch/waymo/tools/` (section 5), starting at about 10:18 EST and done in under 40 minutes, at $1.02 to $4.44 a page. The SSH connection dropped midway, and the batch finished regardless.
  - **Where.** Each page's results are in `~/scratch/waymo/review/stage2/cutin_<page>/` on PACE, for Andrew in OnDemand. The per-page table is in `docs/stage2.md`, "All ten cut-in pages".
- **Waymo answered, and Stage 2 was cleared for real data (2026-11-04).** At about 09:45 EST Andrew reported Waymo's written answer: Stage 2 may send real camera frames and track data to Claude's API. So `configs/stage2.toml` has `real_data_cleared = true`, with a comment saying so. The local open-weight model, the planned fallback, wasn't built. The key and the SDK are on PACE (section 4), and `infra/pace/stage2_cutin.sbatch` is new. The guard's code didn't change, and four of its tests now set `real_data_cleared = false` themselves. 1,272 tests pass on the Mac and on PACE, where the 14 SDK tests now run too. Two questions stay open: whether the answer also covers AI sessions seeing real data, and Stage 3's Supabase. Until Andrew says so, sessions stay counts-only. Setup doc §3 and §7 still describe the time before the answer.
- **Open for Andrew:**
  - Read Stage 2's reports in OnDemand, starting with page 1's rejected cut-in and pages 5 and 9's confirmed ones.
  - Decide whether the demo stays page 1 or moves to page 5.
  - Decide whether to try Claude Sonnet 5.5 for Stage 2. It costs half as much ($2 and $10 per million tokens, against Opus 5.5's $4 and $20).
  - Decide whether to delete the old `~/driving-incident-reconstruction` on ICE.
  - Start a new GPU job for PACE work after 18:00 EST today, when job 6798591 ends.
- **Stage 2: built, checked, and run live on synthetic data (2026-11-04).** `src/agatha/vlm/` (4 modules, 742 lines) adds the two-pass VLM and the `agatha-vlm` CLI, with `configs/stage2.toml` and `docs/stage2.md`. The model is `claude-opus-5-5` at effort medium, with server-side refusal fallbacks on (`fallbacks="default"`).
  - **The live run** on the synthetic segment: 12 requests (Pass 1: 5, Pass 2: 7 including one 10 FPS retry), about 895k input and 16k output tokens, **$3.91**, no failures and no fallbacks. Pass 1 found 9 candidates, merged to 6. Pass 2 confirmed 2 and rejected 4. The confirmed one is pedestrian P001 crossing the ego's path at about 19 s, found twice, so one report is marked `duplicate_of` the other.
  - **The synthetic cut-in was rejected:** V002 stayed 10 to 16 m ahead and was faster than the ego, which doesn't meet SUDDEN_CUT_IN's "limited clearance". So the taxonomy's cut-in is stricter than the scan's, as the real pages then showed. Full-resolution frames cost about 3,500 tokens each.
  - **Two rounds of independent checks and fixes followed.** Among the fixes: merging at most one candidate per window; retrying only when the retry shows more frames; requiring reasons for UNCERTAIN; `duplicate_of`; a `max_request_mb` budget (24 MB), because real frames could push a request past the API's 32 MB; and the data guard using the same synthetic test as the report labels. All 19 planted bugs that still apply are caught. A replay of the live run with the final code sent nothing and gave the same results.
  - **Stage 1 changed in two lines:** `agatha-stage1` now refuses a `--context` that isn't a bare name, and report labels pass only `synthetic_<name>` and the public segment through unhashed.
- **Renamed to AGATHA and moved (2026-11-04).** The project now lives in `~/AGATHA`, with the old prototype in `scenario-miner/`. The package is `agatha`, and the commands are `agatha-stage1` and `agatha-scan`. Behavior didn't change: all 1,121 tests passed after the rename, and again in `~/AGATHA` after the move, with the environment rebuilt by `uv sync --frozen`. `scenario-miner/` has no environment of its own now. PACE got the renamed code overnight, with its environment rebuilt there.
- **Polish round: done.** The round fixed the scan doc's slips and some style notes without changing any output: its identity check found 0 differences from round5final in 26 comparisons, and 1,121 tests passed. `scan.py` stayed byte-identical to round5final, because the build's edits to it changed behavior and were undone. Its own final review was cut off when the app restarted, so Andrew chose a quick check by the main session instead. That passed at 00:42 EST on 2026-11-04. The polished files were copied to PACE at 00:53 EST.
- **Review pages: judged.** On 2026-11-04, Andrew called all 10 real cut-ins: pages 1 to 8 clearly, page 9 leaning unsure, and page 10 a merge where the lane ends rather than a lane change. So the scan's top 10 holds no non-event. Andrew picked page 1, the top-ranked candidate, as the demo clip. Per setup doc §8, it's described on Nov 5, not shown.
- **Google sign-in on ICE: revoked** at 01:17 EST on 2026-11-04 (checked: no credentialed accounts left). New downloads would need Andrew to sign in again.
- **Nov 5:** per setup doc §8, the screen needs no visuals. If Andrew shows anything, it's code, architecture, synthetic overlays and aggregate numbers, never Dataset frames. Describe scene events, not a real vehicle's performance. Stage 2's results on the ten pages are aggregates, under the same rule. Don't count on ICE being up during the call.

## 9. Glossary

- **Segment, context:** one Waymo log of about 20 s. Its name is the context name, and real context names never leave PACE.
- **Label:** `sha256:` plus the first 12 hex digits of a context name's sha256 (`report.context_label`). Public and synthetic names are kept as they are. Reports on PACE are named by label.
- **Page, rank, `cutin_<rank>`:** a review page's number is its segment's rank in `cutin_top.txt`. PACE scripts print `cutin_<rank>` in place of a label or a name.
- **Ego:** the Waymo vehicle that recorded the log.
- **FRONT, FRONT_LEFT, FRONT_RIGHT:** the three forward cameras that Stage 1 uses. Stage 2 uses FRONT.
- **`projected_lidar_box`:** Waymo's own 2D boxes for the 3D labels, the reference for check 3.
- **Residual:** how far a drawn box's edge sits from Waymo's projected box, in pixels.
- **Label path, synced box:** the two ways a 3D box gets drawn: from the label through the image's own pose, or from `lidar_camera_synced_box`.
- **Onset, entry, crossing time, own move, pull, severity, headway:** see section 7.
- **`car_moved`, `ego_steered`:** `scan.json`'s counts of events that pass both the own-move and steering tests, and of events that fail the steering test.
- **round5final, the polish:** the frozen round 5 code that ran on PACE, and the wording-only round after it, which changed the tests, the scan doc and two sbatch comments.
- **Pass 1, Pass 2, candidate, merged candidate:** Stage 2's two passes. Pass 1 lists candidate events in each 5 s window; candidates that overlapping windows found twice merge; Pass 2 tests each merged candidate and calls it CONFIRMED, REJECTED or UNCERTAIN.
- **IncidentReport, `duplicate_of`:** Stage 2's output record, one per accepted Pass 2 answer, rejected and uncertain ones included. A later report of the same event names the earlier one in `duplicate_of`.
- **SUDDEN_CUT_IN, the scan's cut-in:** the taxonomy's cut-in needs limited clearance. The scan's is a lane change into the ego's path that ends 30 m ahead or closer. A scan hit is a candidate for SUDDEN_CUT_IN, not a match.
- **The data guard, `real_data_cleared`:** a remote model (Claude) runs on a real segment only when `configs/stage2.toml` sets `real_data_cleared` to a literal `true`. It has been true since 2026-11-04.
- **Dry run:** `agatha-vlm --model dry-run` writes each request to `stage2/dry_run/` and sends nothing. Answers a real run stored still serve it.
- **`WOD_HOME`, `WOD_ROOT`, `WOD_OUTPUTS`:** paths on PACE (section 4).
- **PACE, ICE, Slurm, OnDemand:** Georgia Tech's research computing service, its instructional cluster, the job scheduler, and the cluster's web portal.
- **`preprocessing_version`:** the package version plus hashes of the code and of `stage1.toml`, stamped on every output.

## 10. Where to look for common questions

| Question | Look at |
|---|---|
| What exactly counts as a cut-in? | `docs/cutin_scan.md` "What counts, from labels alone", `configs/scan.toml`, `scan.py` (`_crossings`, `_could_enter`, `find_cut_ins`). Stage 2's definition: `vlm/schema.py` `TAXONOMY` |
| How are segments ranked? | `scan.py` `_rank`, `docs/cutin_scan.md` "Ranking" and "The list, in two parts" |
| What's on a review page? | `docs/cutin_scan.md` "Review", `scan.py` `_review_page` and `_event_line` |
| What did the real scan find? | `reports/scan.json`, section 7 |
| What does check 3 measure, and why did page 4 fail it? | `checks.py` `check_residuals`, `configs/stage1.toml` `[checks]`, `reports/cutin_4.json`, setup doc §6 |
| Is FRONT_LEFT accurate at speed? | `reports/fast_left_1.json`, section 6 |
| Which coordinate frame is this in? | The docstrings of `geometry.py` and `contracts.py`, `docs/stage1_architecture.md` |
| How do I run something on PACE? | `infra/pace/`, `docs/cutin_scan.md` "Running it on PACE", `docs/stage2.md` "Running it", setup doc §9 |
| What may Andrew show or say on Nov 5? | Setup doc §8, `scenario-miner/PREP.md`, section 8 |
| What may leave PACE, and what did Waymo clear? | Section 1, `configs/stage2.toml` `[data]`, `docs/stage2.md` "The data guard", `report.py`, setup doc §3 (written before the answer) |
| What comes after Stage 1? | The design doc, setup doc §4, `contracts.py` |
| How was the scan tested? | `tests/test_scan.py`, and `docs/cutin_scan.md` "Tests", "Known misses" and "Known risks" |
| How does Stage 2 work? | `docs/stage2.md`, `src/agatha/vlm/`, `configs/stage2.toml` |
| What did Stage 2 find, and what did it cost? | `docs/stage2.md` "Live run" (synthetic), "First real run" and "All ten cut-in pages" (real), section 8 |
| Why did Pass 2 reject a real cut-in? | Only the report says, and it stays on PACE for Andrew. The likely reason is in `docs/stage2.md` "Live run" (limited clearance) |
