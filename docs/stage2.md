# Stage 2: the two-pass VLM

Implements milestone M2 of the design doc (sections 5.1 to 5.3 and the IncidentReport contract in section 6). The code is `src/agatha/vlm/`, the settings `configs/stage2.toml`, and the tests `tests/test_vlm.py`. It was built on synthetic data. Claude has run on the synthetic segment (see [Live run](#live-run)) and, since Waymo cleared it on 2026-11-04, on page 1's real segment on PACE (see [First real run](#first-real-run)). Every test runs offline.

## What it does

Stage 2 reads one segment that Stage 1 has processed. **Pass 1** looks at each of Stage 1's own windows and lists every candidate event that might fit the taxonomy, favoring recall. Candidates that overlapping windows found twice are **merged**. **Pass 2** then tests each merged candidate on a wider window at a higher frame rate and image size, and its answer becomes an **IncidentReport**: confirmed, rejected or uncertain. Every answer is parsed into a strict schema and checked against the window it was about before anything is trusted.

The model sits behind a small interface (`VisionModel` in `model.py`): a model id, whether it's `remote`, and `respond(system, content, schema)`, which returns a `Reply`: the raw answer text, the model that wrote it (a fallback can stand in for the one asked), and token usage. `ClaudeModel` uses the Anthropic SDK, or with `dry_run` set writes its requests instead of sending them, and `FakeModel` answers tests. An open-weight model on PACE can implement the same interface.

## The rule and the taxonomy

An event is relevant if unexpected road-user behavior poses a potential hazard, or if construction or an obstruction materially changes the expected driving space, route or traffic control. Mere visibility of roadside construction doesn't qualify, and a detection is no proof that the recorded vehicle failed.

The 14 labels of taxonomy v1, in 3 categories (`TAXONOMY` in `schema.py`, with each label's definition from the design doc):

| Category | Labels |
|---|---|
| DYNAMIC_HAZARD | SUDDEN_CUT_IN, HARD_BRAKING, PEDESTRIAN_INTRUSION, CYCLIST_INTRUSION, CROSS_TRAFFIC_CONFLICT |
| CONSTRUCTION_DISRUPTION | CONSTRUCTION_LANE_CLOSURE, TEMPORARY_LANE_SHIFT, TEMPORARY_TRAFFIC_CONTROL, CONFLICTING_LANE_GUIDANCE, WORK_ZONE_ENCROACHMENT |
| ROAD_OBSTRUCTION | STOPPED_VEHICLE_OBSTRUCTION, DOUBLE_PARKED_OBSTRUCTION, ROAD_DEBRIS, OTHER_PHYSICAL_OBSTRUCTION |

An event can take several labels, each with its own evidence. Both system prompts carry the rule and the full taxonomy.

## Pass 1

- **Windows**: Stage 1's `default_windows`, so 5 s with 1 s of overlap at about 2 FPS, FRONT, in the `pass1` render profile.
- **Content**: each annotated frame after a caption with its camera and `t_s`, then compact JSON: the window's bounds, frame rate and cameras (the prompt says these aren't all the vehicle's); each kept object's alias, type, span in the segment, flags (stationary, present at the segment's start or end, not directly observed at times, past the driven path at times, type changes) and rows of `t_s, x_m, y_m, lateral_m, front_gap_m, v_rel_x_mps, v_rel_y_mps`; the omitted objects; the ego's speed, accelerations and yaw rate; and the window's `SceneLimits`. Numbers are rounded to 2 decimals. Nothing sent names the segment.
- **Answer**: a list of candidates, each with labels, `start_s`, `end_s`, object aliases, evidence and a driving-impact note.
- **Fitting**: Pass 1 is for recall, so a candidate is fitted to its window rather than dropped (`fit_candidate`). A span is clipped to the window, since the JSON's 2 s of track context on each side invites one past it, and an alias the window lacks is dropped. Each change is noted in the candidate's `changes`. Only a candidate with no labels, or a span that misses the window, is kept out of the merge.
- **Merge**: two candidates are one event when their times overlap (touching counts), their labels share a category, and they share an object. Candidates that name no object, as with cones, barriers and debris, which have no boxes, must share a label instead. Grouping is transitive, but a group holds at most one candidate from each window: a candidate joins a group only if no member came from its window. So a window's own candidates stay apart, as the model split them, even when a neighboring window's candidate overlaps both: it ends up with the first of the two, and the other stays on its own. A merged candidate takes the union of labels (in taxonomy order), objects and evidence, and the widest span.

## Pass 2

- **Window**: the merged candidate's span plus `margin_s` (1 s) on each side, clipped to the segment, at 5 FPS in the `full` render profile, rendered on the spot. `keep_aliases` keeps the candidate's objects in it whatever their rank. A window long enough to need more than `max_images` (100) images, counting every camera's, takes a lower frame rate instead: a span of d s at r FPS holds d·r + 1 frames per camera, so a candidate that spans a 20 s segment gets 100 frames at 4.95 FPS, not 101, and with three cameras 99 images at 1.62 FPS. Then the images' size is checked: when the drawn images, as base64 (4 bytes for every 3), come to more than `max_request_mb` (24 MB), the window is built once more at a frame rate lowered in proportion. The API turns away a request over 32 MB, and real camera frames run several hundred KB, so 100 of them could pass it. A window with no frames isn't sent.
- **Content**: as Pass 1's, with three more columns (`path_clearance_m`, `a_x_mps2`, `a_y_mps2`), then the hypothesis. The prompt asks for evidence for and against each label, and the schema puts that evidence before each status, so the answer gives it before its verdict.
- **Answer**: a status for each label and for the event (CONFIRMED, REJECTED or UNCERTAIN), uncertainty reasons, the event's own times and aliases, the frames that show the evidence (by camera and `t_s` as captioned), a concise evidence-based explanation, and a driving-impact note. The schema calls it `explanation`, not reasoning: a request for a model's reasoning can be refused as reasoning extraction.
- **Retry**: an UNCERTAIN event is asked once more at 10 FPS, within `max_images` and `max_request_mb` too, but only when that gives the retry more frames than the first request had. Otherwise the model would see the same frames again, so it isn't asked or counted as retried: a candidate the whole segment long, held to 4.95 FPS either way, gets no retry, nor does a window that `max_request_mb` holds to the same frames. A retry that fails, or whose answer fails the checks, keeps the first answer. Rejected and uncertain results are kept.

## Contracts

The input is Stage 1's `SceneWindow` (`contracts.py`). The output is an `IncidentReport` (`schema.py`) with the design doc's fields:

`schema_version`, `segment_id`, `incident_id` (`INC-001` and on, in time order), `start_s`, `end_s`, `categories`, `subcategories` (the labels Pass 2 didn't reject), `object_ids` (the aliases' `laser_object_id`), `verification_status`, `label_assessments` (each label's evidence, contradicting evidence and status), `vlm_reasoning` (the explanation), `driving_impact`, `uncertainty_reasons`, `source_frames` (`camera`, `timestamp_s` and the exact `frame_timestamp_micros`), `pipeline_version` (the package version and a hash of `vlm/`), `model_id` (the model that answered), and three more: `prompt_version`, `preprocessing_version` and `duplicate_of`.

`duplicate_of` is null, or the `incident_id` of an earlier report of the same event. Pass 2 can find, in one candidate's window, the event another candidate was about: in the live run, it rejected a cut-in and confirmed the pedestrian crossing in the same frames, which its own candidate had already reported. Two reports are one event when they share a label neither one rejected (`subcategories`), share an object (`object_ids`), and overlap in time (touching counts). The later one names the first earlier report it repeats. Both are kept, and the CLI counts the duplicates. Reports that name no object, as for debris, are never marked.

## Validation

An answer is trusted only after both steps:

1. **It parses** into strict pydantic models with no defaults and no extra keys. Labels and statuses are enums, so a label outside the taxonomy fails here, and so does a time that isn't a finite number. The same models give the JSON schema that constrains decoding, and a test holds it to strings, numbers, booleans, string enums, arrays and closed objects.
2. **The referential checks pass**. Pass 1's candidates are fitted to their window, as above. Pass 2's answer is held strictly (`check_verdict`): its times form a span inside the window, give or take 0.05 s; every alias is in the window, among its objects or its omitted objects; every cited frame is one of the window's frames; every hypothesized label is assessed once; the event's status is its labels' strongest, CONFIRMED over UNCERTAIN over REJECTED, so a report's status and categories agree; a CONFIRMED event cites at least one frame; and an UNCERTAIN event gives at least one reason, since M3 reports abstentions. The prompt states these last three rules.

## Outputs

Under `<segment directory>/stage2/`, each file written with `write_atomic`:

| Path | Holds |
|---|---|
| `responses/<key>.json` | each raw answer or failure, with the model asked and the one that answered, settings, prompt version, latency, and the response's full `usage`, whose `iterations` count every attempt a fallback made. Billed failures, such as a refusal or a cut-off, keep their usage too |
| `errors/<key>.json` | why that answer failed validation, kept apart from it |
| `windows/` | every SceneWindow a request was built from |
| `dry_run/` | what the last run would have sent but didn't: each run clears it first |
| `candidates.json` | `windows`: each Pass 1 window's request, response key and outcome (answered, failed, invalid, not sent, no frames), so a window never assessed doesn't pass for one with nothing in it; `candidates`: every Pass 1 candidate as fitted, with its changes and errors; `merged`: each merged candidate with its sources, its Pass 2 `attempts` (frame rate, window, response key, outcome), its outcome, and `response`, the key of the answer behind the outcome and the report |
| `incidents.json` | the accepted IncidentReports, rejected and uncertain ones included. Each one's `incident_id` leads to its merged candidate, and `duplicate_of` to an earlier report of the same event |

The key hashes the whole request: model id, settings, prompt version, system prompt, content (each image by its bytes) and schema. A stored answer to the same request is reused instead of calling again, and a stored refusal or cut-off too, since the same request would most likely fail the same way. Two failures are asked again on a rerun: a refusal whose fallback couldn't run (`stop_details.recommended_model` set) and a request the API turned away (HTTP 400 or 413). Delete a response to ask again otherwise. `candidates.json` and `incidents.json` are written only when every request was answered or failed, so a dry run's gaps never replace a full run's results.

## Claude

`ClaudeModel` calls `client.beta.messages.create` on `anthropic.Anthropic()`. The SDK is an optional extra, `uv sync --extra claude`, imported only when a request is sent, so the package, its tests and the dry run work without it. PACE's environment gets it from the Stage 2 script's `uv run --extra claude`. Credentials come from `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN` or an `ant auth login` profile.

- The model is `claude-opus-5-5`, with `effort` set explicitly in `output_config`, beside the JSON schema. Thinking is always on with this model, so no thinking settings, sampling settings or assistant prefill are sent.
- **Server-side refusal fallbacks are on**: `betas=["server-side-fallback-2026-07-01"]` and `fallbacks="default"`, so a declined request reruns on the model recommended for its refusal category. The reply records the model that actually answered.
- The stop reason is checked before the content. Anything but `end_turn`, such as a `refusal` (with its `stop_details.category`) or `max_tokens`, is recorded as a failure, never parsed. The answer is the first text block, after any thinking blocks, and a response with none is a failure too.
- `max_tokens` is 16000, within what the SDK allows a call that doesn't stream. The SDK retries 429 and 5xx itself. A 400 or 413 is that request's fault, such as its size, so it's recorded and the run goes on. Other API errors, such as a missing key, stop the run, and a rerun reuses every answer stored before them.
- `tests/test_vlm.py` checks the request against a stub client and the SDK's own response types, and checks every argument name against the installed SDK.

## The data guard

A remote model (Claude) runs only on synthetic segments, whose context names start with `synthetic_`, unless `configs/stage2.toml` sets `real_data_cleared = true`. The public tutorial segment is real data, so it's refused too. Only a literal `true` clears it, so a typo fails closed. The CLI checks before it reads the segment, and every request checks again before it's sent. Local models, such as an open-weight model on PACE, and the dry run, which sends nothing, may run on anything.

**Cleared on 2026-11-04.** Waymo answered in writing that Stage 2 may send real camera frames and track data to Claude's API, so `real_data_cleared` is now `true`. That covers the pipeline, not AI sessions: Stage 2's outputs on real data stay on PACE, Andrew reviews them in OnDemand, and only counts come back. The guard's code didn't change, and its tests now set `real_data_cleared = false` themselves, so they don't depend on the shipped value.

## Config

`configs/stage2.toml`: `[model]` (id, effort, max_tokens), `[pass2]` (margin, frame rate, retry and its rate, render profile, `max_images`, `max_request_mb`) and `[data]` (`real_data_cleared`). Pass 1's windows come from `configs/stage1.toml`. Editing `[model]` changes every request's key, so it means new calls.

## Running it

After Stage 1 has run on the segment:

```bash
agatha-vlm --config configs/stage1.toml --vlm-config configs/stage2.toml --split training \
    --context synthetic_cut_in --model dry-run
```

`--model dry-run`, the default, writes each request Claude would get to `stage2/dry_run/`, with images by reference rather than base64, and sends nothing, so the whole flow runs without a key. Pass 2's requests depend on Pass 1's answers, so a dry run on a fresh segment shows only Pass 1's. `--model claude` sends them, and needs the SDK extra, a key and Andrew's go-ahead, since calls cost money. `--split` must match the segment's `segment.json`, and `--context` must be a bare name. Pass 2 renders new images, so it reads the source data under `$WOD_ROOT`: after `agatha-stage1 --synthetic`, that's `$WOD_OUTPUTS/synthetic`.

The CLI prints only counts and the context's label, never a name, on failure too: per pass, what became of its requests; Pass 1's candidates before and after merging; Pass 2's retries and verdicts; and the reports written, with how many are marked `duplicate_of` another.

On PACE, `infra/pace/stage2_cutin.sbatch <rank> [claude|dry-run]` runs it on the segment at that rank in `cutin_top.txt` (1 is the demo, page 1), after `review_cutins.sbatch` has run Stage 1 there. `dry-run` is the default. With `claude`, it exports the key from `~/.config/agatha/anthropic.env`, which sits outside `~/agatha`, so `sync.sh` never touches it. It prints `cutin_<rank>` in place of the label, since a label is a hash of the name, and masks the name too. After a run it links `$WOD_HOME/review/stage2/cutin_<rank>` to the segment's `stage2/`, so Andrew finds the reports in OnDemand by page number. It runs as a job or inside one: `srun --jobid=<job> --overlap bash infra/pace/stage2_cutin.sbatch 1 claude`. PACE's environment has the SDK now, since the script runs `uv run --extra claude`.

## Live run

On 2026-11-04, Stage 2 ran with Claude on the synthetic segment, `claude-opus-5-5` at effort `medium`, in three steps, each reusing what the steps before it had stored: Pass 1 alone, to check its cost; then `agatha-vlm --model claude` with the retry off; then again with it on, which sent only the retry.

| | Requests | Input tokens | Output tokens | Cost |
|---|---|---|---|---|
| Pass 1 | 5 | 87,383 | 4,206 | $0.43 |
| Pass 2 | 7, one of them a 10 FPS retry | 807,963 | 12,089 | $3.47 |
| Total | 12 | 895,346 | 16,295 | $3.91 |

No request failed, and no fallback answered. Pass 1 took about 55 s, and each Pass 2 request 25 to 36 s. A full-size frame took about 3,500 input tokens. The largest request, the retry, held 44 frames, 4.05 MB as base64.

Pass 1 found 9 candidates, merged into 6. Pass 2 confirmed 2 and rejected 4; the one UNCERTAIN, a yellow object far off near 6 to 8.5 s, came back REJECTED at 10 FPS.

- **Confirmed**: P001's pedestrian intrusion, 14 to 20 s, reported twice. INC-005 was its own candidate. INC-006's hypothesis was a cut-in by V002 and V003 at 16 to 20 s: Pass 2 rejected it, and added and confirmed PEDESTRIAN_INTRUSION for P001 from the same frames. That is why `duplicate_of` exists: INC-006 now names INC-005.
- **Rejected**: a parked car at 1.5 to 5 s, the far yellow object, the scripted cut-in at 8 to 12.5 s, and P001 at 11 to 13 s, standing well off the road.

The synthetic cut-in was rejected. V002 does enter the ego's path between about 8 and 12 s, but it stays 10 to 16 m ahead and faster than the ego, so the gap keeps opening, and that doesn't meet "limited clearance".

What that means: the taxonomy's SUDDEN_CUT_IN is stricter than the label-only scan's cut-in ([cutin_scan.md](cutin_scan.md)). The scan counts a lane change into the ego's path that ends 30 m ahead or closer; the taxonomy also asks for limited clearance. A scan hit is a candidate for SUDDEN_CUT_IN, not a match, and the synthetic segment's gentle cut-in is a negative for it. M3's annotations need the taxonomy's definitions, not the scan's.

The stored answers still serve the code as it is now: a rerun over a copy of that run's outputs, with no key, reuses all 12 and sends none.

## First real run

On 2026-11-04, once Waymo had cleared it, Stage 2 ran with Claude on page 1's segment, the demo cut-in, on PACE inside job 6798591. The model and settings were as in the live run. It went in three steps, each reusing what the steps before it had stored: Pass 1 alone, to check its cost; a dry run, which built Pass 2's 7 requests (16 to 42 frames each) without sending them, an estimated $4; then `stage2_cutin.sbatch 1 claude`. Only counts came back. The reports stay on PACE.

| | Requests | Input tokens | Output tokens | Cost |
|---|---|---|---|---|
| Pass 1 | 5 | 114,253 | 3,610 | $0.53 |
| Pass 2 | 9, two of them 10 FPS retries | 1,254,568 | 17,692 | $5.37 |
| Total | 14 | 1,368,821 | 21,302 | $5.90 |

No request failed, and no fallback answered. Pass 1 took 89 s in all, and Pass 2 about 29 s a request.

Pass 1 found 9 candidates, one of which failed its checks, and merged them into 7. Pass 2 confirmed 5, rejected 1 and left 1 uncertain, after retrying 2. That's 7 reports, 2 of them marked `duplicate_of` an earlier one. By label, over the 5 reports that aren't duplicates:

- **Confirmed**: CONSTRUCTION_LANE_CLOSURE in 2, TEMPORARY_TRAFFIC_CONTROL in 1, CROSS_TRAFFIC_CONFLICT in 1.
- **Uncertain**: CONSTRUCTION_LANE_CLOSURE in 1.
- **Rejected**: SUDDEN_CUT_IN, HARD_BRAKING, TEMPORARY_LANE_SHIFT, TEMPORARY_TRAFFIC_CONTROL and WORK_ZONE_ENCROACHMENT, once each.

No report confirms a cut-in. One merged candidate proposed SUDDEN_CUT_IN, and Pass 2 rejected it. That fits the live run's finding: the scan's real cut-ins were gentle (at most 0.34 m/s² of braking needed, and a median headway of 1.5 s, in `reports/scan.json`), and the taxonomy's SUDDEN_CUT_IN asks for limited clearance. Only the report can say whether Pass 2 rejected the scan's own car and time, so Andrew reads it in OnDemand, in `review/stage2/cutin_1/incidents.json`.

A rerun with the final script, as a dry run, reused all 14 answers and sent none.

### All ten cut-in pages

The same day, pages 2 to 10 ran the same way, one after another. A driver in `~/scratch/waymo/tools/` on PACE, outside the repo, took each page through the three steps above. It held back any page whose Pass 1 cost plus Pass 2's estimate passed $7, and would have stopped starting pages after $65. No page came near either limit. The SSH connection to ICE dropped midway, and the batch ran on regardless.

| Page | Candidates, merged | Pass 2 requests | Reports | Confirmed, rejected, uncertain | SUDDEN_CUT_IN candidates | Cost |
|---|---|---|---|---|---|---|
| 1 | 9, 7 | 9 (2 retries) | 7 (2 duplicates) | 5, 1, 1 | 1 rejected | $5.90 |
| 2 | 3, 1 | 1 | 1 | 0, 1, 0 | 1 rejected | $1.02 |
| 3 | 11, 5 | 5 | 5 | 1, 4, 0 | 1 rejected | $3.52 |
| 4 | 3, 1 | 1 | 1 | 0, 1, 0 | 1 rejected | $1.52 |
| 5 | 10, 8 | 8 | 8 | 1, 7, 0 | 1 confirmed, 1 rejected | $4.44 |
| 6 | 8, 5 | 5 | 5 | 0, 5, 0 | 2 rejected | $3.32 |
| 7 | 5, 3 | 3 | 3 | 0, 3, 0 | 1 rejected | $1.82 |
| 8 | 4, 3 | 3 | 3 | 0, 3, 0 | 1 rejected | $1.79 |
| 9 | 5, 4 | 6 (2 retries) | 4 (1 duplicate) | 1, 2, 1 | 1 confirmed, 1 rejected | $4.43 |
| 10 | 2, 2 | 2 | 2 | 0, 2, 0 | 1 rejected | $1.31 |
| All | 60, 39 | 43 | 39 (3 duplicates) | 8, 29, 2 | 2 confirmed, 11 rejected | $29.07 |

With Pass 1's 50 requests, that's 93 requests, about 6.75M input and 104k output tokens. None failed, and no fallback answered.

- **Pass 1 proposed a cut-in on every page.** Each page had at least one merged SUDDEN_CUT_IN candidate, 13 in all. That's the recall Pass 1 is for.
- **Pass 2 confirmed it on 2 pages, 5 and 9**, and rejected it on the other 8, page 1 among them. Page 9 is the one Andrew judged leaning unsure, and page 5 one of those judged clearly real. Page 9's duplicate is an UNCERTAIN report that repeats its confirmed cut-in.
- **Everything else Pass 2 confirmed is on pages 1 and 3**, all of it construction (lane closures, temporary traffic control) except page 1's cross-traffic conflict.

So the scan and Andrew's eye found real cut-ins, and Pass 2 rejects most of them as sudden cut-ins. The likely reason is the live run's: the taxonomy asks for limited clearance, and these cut-ins were gentle. But only the reports say why, and whether a report is about the scan's own car and time. Each page's results are at `~/scratch/waymo/review/stage2/cutin_<page>/` for Andrew to read in OnDemand.

## Left for M3

- **Evaluation**: Pass 1's recall, Pass 2's precision and end-to-end scores against 20 to 30 hand-annotated segments, reporting UNCERTAIN as abstentions.
- **Tuning**: effort, image size, margins, frame rates, `max_images`, `max_request_mb` and the merge rule, all on that set. Request size matters. At about 3,500 input tokens a full-size frame, a Pass 2 request's 30 frames come to about 105,000 tokens, and 100 frames to about 350,000, $1.40 at $4 per million. Real camera frames run several hundred KB against the synthetic ones' 63 KB, so on real data `max_request_mb` rather than `max_images` will set a long window's rate. When it holds both a request and its retry, the retry gets at most a frame or two more than the first, or none and isn't sent, so a rule that asks for clearly more frames may be worth it. Check the vision page's current limits on images per request before raising `max_images`.
- **Duplicates**: `duplicate_of` needs a shared object, so two reports of the same debris are never marked. M3 can say whether that matters.
- **Real data**: cleared for Claude on 2026-11-04 (see [The data guard](#the-data-guard)), and run on all ten cut-in pages for $29.07 (see [First real run](#first-real-run)). Those ten are a start on M3's annotated set, and a test of the cut-in definition: Pass 2 confirmed 2 of the 10 cut-ins that Andrew judged real. An open-weight model on PACE could still stand in through the same interface, say for runs too large to pay for.
- **Cost**: each stored response, failures included, has its full usage, so a run's cost can be added up. Nothing does that yet.
