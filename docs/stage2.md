# Stage 2: the two-pass VLM

Implements milestone M2 of the design doc (sections 5.1 to 5.3 and the IncidentReport contract in section 6). The code is `src/agatha/vlm/`, the settings `configs/stage2.toml`, and the tests `tests/test_vlm.py`. It was built on synthetic data. Claude has run on the synthetic segment (see [Live run](#live-run)) and, since Waymo cleared it on 2026-11-04, on page 1's real segment on PACE (see [First real run](#first-real-run)). Gemma 4, an open-weight model served by vLLM on a PACE GPU node, can stand in for Claude (see [Gemma 4 on PACE](#gemma-4-on-pace)). Every test runs offline.

## What it does

Stage 2 reads one segment that Stage 1 has processed. **Pass 1** looks at each of Stage 1's own windows and lists every candidate event that might fit the taxonomy, favoring recall. Candidates that overlapping windows found twice are **merged**. **Pass 2** then tests each merged candidate on a wider window at a higher frame rate and image size, and its answer becomes an **IncidentReport**: confirmed, rejected or uncertain. Every answer is parsed into a strict schema and checked against the window it was about before anything is trusted.

The model sits behind a small interface (`VisionModel` in `model.py`): a model id, whether it's `remote`, and `respond(system, content, schema)`, which returns a `Reply`: the raw answer text, the model that wrote it (a fallback can stand in for the one asked), and token usage. `ClaudeModel` uses the Anthropic SDK, or with `dry_run` set writes its requests instead of sending them. `LocalModel` asks an open-weight model, Gemma 4, through vLLM's OpenAI-compatible server on the same machine. `FakeModel` answers tests.

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
2. **The referential checks pass**. Pass 1's candidates are fitted to their window, as above. Pass 2's cited frames are fitted too (`fit_verdict`): a time between two frames shown of the same camera moves to the nearer one, the earlier on a tie, and a frame cited again after that is dropped, keeping the order. It also drops an alias `ego`, in any case, unless the window has an object by that name: the ego has no alias, and Gemma named it once (2026-11-05). The frames shown run at 5 FPS and the track JSON's rows at 10 Hz, so a model can cite a time no frame has: on 2026-11-04 Gemma did so in 3 of its 69 Pass 2 answers, such as FRONT at 12.6 s between frames at 12.5 and 12.7, and nothing else in those answers was wrong. Nothing else moves: a time before a camera's first frame shown or after its last, or a camera the window doesn't show, stays as cited and is refused below. Each change is noted in its Pass 2 attempt's `changes`, and the report cites the fitted frames. Then the answer is held strictly (`check_verdict`): its times form a span inside the window, give or take 0.05 s; every alias is in the window, among its objects or its omitted objects; every cited frame is one of the window's frames; every hypothesized label is assessed once; the event's status is its labels' strongest, CONFIRMED over UNCERTAIN over REJECTED, so a report's status and categories agree; a CONFIRMED event cites at least one frame; and an UNCERTAIN event gives at least one reason, since M3 reports abstentions. The prompt states these last three rules.

## Outputs

Under `<segment directory>/stage2/`, each file written with `write_atomic`:

| Path | Holds |
|---|---|
| `responses/<key>.json` | each raw answer or failure, with the model asked and the one that answered, settings, prompt version, latency, and the response's full `usage`, whose `iterations` count every attempt a fallback made. Billed failures, such as a refusal or a cut-off, keep their usage too |
| `errors/<key>.json` | why that answer failed validation, kept apart from it |
| `windows/` | every SceneWindow a request was built from |
| `dry_run/` | what the last run would have sent but didn't: each run clears it first |
| `candidates.json` | `windows`: each Pass 1 window's request, response key and outcome (answered, failed, invalid, not sent, no frames), so a window never assessed doesn't pass for one with nothing in it; `candidates`: every Pass 1 candidate as fitted, with its changes and errors; `merged`: each merged candidate with its sources, its Pass 2 `attempts` (frame rate, window, response key, outcome, and `changes`, what fitting the answer's cited frames changed, empty when nothing did), its outcome, and `response`, the key of the answer behind the outcome and the report |
| `incidents.json` | the accepted IncidentReports, rejected and uncertain ones included. Each one's `incident_id` leads to its merged candidate, and `duplicate_of` to an earlier report of the same event |

The key hashes the whole request: model id, settings, prompt version, system prompt, content (each image by its bytes) and schema. A stored answer to the same request is reused instead of calling again, and a stored refusal or Claude's cut-off too, since the same request would most likely fail the same way. Three failures are asked again on a rerun: a refusal whose fallback couldn't run (`stop_details.recommended_model` set), a request the server turned away (Claude's HTTP 400 or 413, or any 4xx from the local model's), and a local model's cut-off, unless it used all of `max_tokens` at temperature 0, where a resend would repeat it (see [Gemma 4 on PACE](#gemma-4-on-pace)). Each stored failure says which in `retry`. Delete a response to ask again otherwise. A stored answer is fitted and checked again on every run, so a change to the fitting or the checks reaches it with nothing sent. The exception is an answer the change turns into a valid UNCERTAIN one: like any UNCERTAIN answer, it then gets its 10 FPS retry, as in [Pass 2](#pass-2), and that retry is sent. `candidates.json` and `incidents.json` are written only when every request was answered or failed, so a dry run's gaps never replace a full run's results.

## Claude

`ClaudeModel` calls `client.beta.messages.create` on `anthropic.Anthropic()`. The SDK is an optional extra, `uv sync --extra claude`, imported only when a request is sent, so the package, its tests and the dry run work without it. PACE's environment gets it from the Stage 2 script's `uv run --extra claude`. Credentials come from `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN` or an `ant auth login` profile.

- The model is `claude-opus-5-5`, with `effort` set explicitly in `output_config`, beside the JSON schema. Thinking is always on with this model, so no thinking settings, sampling settings or assistant prefill are sent.
- **Server-side refusal fallbacks are on**: `betas=["server-side-fallback-2026-07-01"]` and `fallbacks="default"`, so a declined request reruns on the model recommended for its refusal category. The reply records the model that actually answered.
- The stop reason is checked before the content. Anything but `end_turn`, such as a `refusal` (with its `stop_details.category`) or `max_tokens`, is recorded as a failure, never parsed. The answer is the first text block, after any thinking blocks, and a response with none is a failure too.
- `max_tokens` is 16000, within what the SDK allows a call that doesn't stream. The SDK retries 429 and 5xx itself. A 400 or 413 is that request's fault, such as its size, so it's recorded and the run goes on. Other API errors, such as a missing key, stop the run, and a rerun reuses every answer stored before them.
- `tests/test_vlm.py` checks the request against a stub client and the SDK's own response types, and checks every argument name against the installed SDK.

## Gemma 4 on PACE

`agatha-vlm --model local` asks Gemma 4 31B (`google/gemma-4-31B-it`), served by vLLM on the PACE GPU node the run is on, so real frames never leave PACE. Claude stays available as `--model claude`, and the dry run stays the default. Built on 2026-11-04 and tested offline only: `tests/test_vlm.py` checks `LocalModel` against a fake OpenAI-compatible server on 127.0.0.1, and runs the PACE scripts under bash with stubs for `nvidia-smi`, `curl`, `vllm`, `uv` and `pace-quota`. The 31B hasn't run on PACE yet. A first try there the same day installed vLLM 0.32.0 on ICE and served Gemma 4 26B (Red Hat's FP8 copy) on the L40S: that settled the driver, and showed that FlashInfer's sampler has to be turned off (see below).

**Why this model.** Gemma 4 31B is the largest Gemma 4: a dense model of about 31B parameters, with image input, a 256K context, and the Apache 2.0 license, not gated. 50 of its 60 layers attend over a sliding window of 1,024 tokens, which keeps a long context's memory down.

**Memory**, for the weights plus about 20%, without the KV cache:

| Precision | Memory | Fits on |
|---|---|---|
| BF16 | 69.9 GB | one 80 GB GPU (A100, H100), or two L40S with tensor parallelism |
| 8-bit (FP8) | 34.9 GB | one L40S (48 GB), Andrew's current GPU |
| 4-bit (QAT W4A16) | 17.5 GB | one L40S, with the most room left for context |

**BF16 or FP8, by GPU.** `infra/pace/serve_vlm.sh` reads the GPUs' memory from `nvidia-smi`. It uses tensor parallelism over 1, 2 or 4 of them, since the model's 4 global KV heads split no further, and serves BF16 when those hold 80 GB or more. Otherwise it serves FP8, which vLLM makes from the BF16 weights as it loads them, with no calibration (`--quantization fp8_per_tensor`, vLLM 0.32.0's documented online FP8). It keeps the KV cache in FP8 too (`--kv-cache-dtype fp8`, which vLLM's Gemma 4 recipe suggests), since the weights leave an L40S little room for context; `KV_CACHE_DTYPE=auto` keeps that at BF16. Ada GPUs such as the L40S compute in FP8; Turing and Ampere keep only the weights in it. So one L40S gets FP8, and an H100 or two L40S get BF16. Google's QAT W4A16 checkpoint for vLLM, `google/gemma-4-31B-it-qat-w4a16-ct`, comes quantized: set it as `[local]`'s `id` to use it. The precision, the KV cache's included, isn't part of a request's key, so answers stored from an FP8 server would serve a BF16 one: compare precisions on separate copies of the outputs.

**The image budget.** Gemma 4 turns each image into 70, 140, 280, 560 or 1,120 visual tokens. 280 is its default, and more reads finer detail at more compute. `[local] image_tokens` sets it, at 280. vLLM takes it per request, as `"mm_processor_kwargs": {"max_soft_tokens": 280}` in the request body, and `serve_vlm.sh` starts the server with the same budget, raising the vision tower's limit (`--hf-overrides`) when it's over 280, as vLLM's Gemma 4 recipe does. Each budget is a different request, so it's part of the key. At 280, 100 images come to 28,000 tokens, where Claude took about 3,500 a full-size frame.

**The request.** `LocalModel` posts to `/v1/chat/completions` with the standard library: the system prompt as a system message, then one user turn with the content Claude gets, each image as a base64 JPEG data URL. The answer is held to the same JSON schema through `response_format` (`json_schema`), vLLM's structured outputs, and read from `choices[0].message.content`. Nothing asks Gemma 4 to think, and the schema leaves no room for it, so `max_tokens` (8,192), sent as `max_completion_tokens` since vLLM deprecates `max_tokens`, is the answer's alone. `[local] temperature`, 0 in the shipped config, is sent as `temperature` (see "Sampling"). A config with none sends none, as before 2026-11-05, and vLLM then samples with the model's own settings. The reply records the model the server names, and the usage as input and output tokens.

**Sampling.** Gemma answers at temperature 0, greedy decoding: one request, one answer. Until 2026-11-05 no temperature was sent, so vLLM sampled with the model's own settings, and verdicts varied: on 2026-11-04 the same Pass 2 request, random15_1's INC-001, came back REJECTED once and CONFIRMED once. Pass 1 was steadier: it proposed exactly one candidate in each of 30 tries over 10 windows (the A/B below, without the setting). On 2026-11-05 Andrew chose to have Gemma give the same answer every time, and to rerun the 15 random scenes with it before he labels them, so `[local]` has `temperature = 0.0`. The temperature is one of `[local]`'s settings, so it's in every local request's key: setting it changed every key, the rerun sends every request again, and the sampled answers stay in `responses/` under their old keys. Another temperature is another request, stored apart, and a config with none keys as before, so the sampled answers still serve it. At temperature 0 a cut-off that used all of `max_tokens` isn't asked again (see "Failures"). Greedy decoding gives the same answer only as far as the arithmetic is the same, and vLLM doesn't promise that by default. Batching can change it, which Stage 2 rules out by sending one request at a time. Prefix caching, on by default in vLLM, can too: a resend can find its prompt cached and compute only the rest, a little differently. On ICE on 2026-11-05, at temperature 0 with the cache on, 13 requests each sent twice came back different every time, and the INC-001 above went REJECTED, then CONFIRMED. So `serve_vlm.sh` turns the cache off (`--no-enable-prefix-caching`): with it off, 3 requests (that Pass 2 one and two Pass 1 windows) sent 3 times each came back identical, about 15 s a Pass 2 answer. vLLM's `VLLM_BATCH_INVARIANT=1` made them identical too, but took about 22 s. Stage 2's requests share little but the system prompt, so the cache saved little.

**Runaway answers.** On 2026-11-04, 3 of Gemma's 69 Pass 2 answers on the random scenes ran to the 8,192-token limit. Each was resent twice by hand, as the same request, and 5 of the 6 tries finished at about 400 tokens. The sixth ran away again: its text was 92% whitespace, the same 40 characters of newlines and spaces 560 times over. Gemma loops on blank lines between JSON tokens, which the schema's grammar allows. vLLM can forbid that whitespace, with `--structured-outputs-config '{"backend": "xgrammar", "disable_any_whitespace": true}'`, and that stopped the runaway: 4 of 4 resends of that request finished, at 377 to 561 tokens, and parsed. But it broke Pass 1. On the same 10 windows (5 each of random15_15 and random15_7), tried 3 times each, 26 of 30 answers with the setting held no candidates, and random15_7's were empty in all 15 tries; without it, all 30 proposed exactly one, as the stored answers did. Forced into compact JSON, Gemma leaves its own formatting and closes the list at once. So the server doesn't use the setting, and `serve_vlm.sh` says why. Instead `LocalModel` marks a cut-off `retry`, and a rerun resends it (below), but only while Gemma samples: the 5 of 6 resends that finished were sampled. At temperature 0 a resend gives the same tokens, so a cut-off that used all 8,192 would run away again on every rerun, and it isn't resent. Its diagnosis is kept either way.

**Failures.** A finish reason of `length` is a `max_tokens` failure, as with Claude, and any other finish reason but `stop` is a failure under its own name. Unlike Claude's, a local cut-off is asked again on a rerun while Gemma samples, with no temperature or one above 0, since a resend usually finishes. At temperature 0, one that used all of `max_tokens` is stored with `"retry": false`, since a resend would repeat it: only another temperature, a new request, asks again. Nothing is resent within a run: rerunning `stage2_list.sh` resends each failure marked `retry`. A cut-off that used all of `max_tokens` also keeps a short diagnosis in its record: `whitespace_share`, the share of its text that is whitespace, and `tail`, its last 300 characters. A cut-off stored before 2026-11-04's change, such as the 3 above, says `"retry": false`, so a rerun reuses it: delete it, or set `retry` to `true`, to ask again. Since vLLM 0.32.0, `max_tokens` is only a bound, not room reserved in the context, so an answer can also stop where the context ends, short of its 8,192 tokens, which a server with a longer context may fit, so that one is asked again at any temperature. A 4xx is that request's fault, such as a prompt longer than the server's context, so it's recorded with the start of the server's reason (the first 1,000 bytes, since a validation error can echo the request), the run goes on, and a rerun asks again, as for Claude's 400 and 413. A 404 is the exception: it says the server serves another model, as when `serve_vlm.sh` started one config's model and `agatha-vlm` runs with another's. Every request would meet it, so like a 5xx, or no server at all, it stops the run, before empty results are written over the top-level ones. A rerun reuses every answer stored before it. An answer comes back whole, not streamed, so the client waits up to 30 minutes for one, then stops the run.

**Keeping it on the node.** The server listens on 127.0.0.1 only. `LocalModel` refuses a `base_url` anywhere but this machine and never goes through a proxy, whatever `http_proxy` says, so it's not `remote`, and the data guard lets it run on real segments. ICE's GPU nodes can be shared, and 8000 is vLLM's default port, so `serve_vlm.sh` keeps a pid file per node and trusts the port only when the server it started answers there. The port is the one in `[local] base_url`, 8000 when it names none: on 2026-11-04 another user's vLLM had 8000 on our node, so a config with `base_url = "http://127.0.0.1:8001/v1"` moved ours. In short: `start` refuses a port another server answers on, and `stage2_cutin.sbatch <rank> local` runs only when `serve_vlm.sh status` says it's up. The process in the pid file must still be `vllm serve`: Slurm ends the server with its job but leaves the file, and another process can take its number later. The server runs in its own process group, so `stop` reaches its engine and GPU workers too: TERM, then after a minute KILL, and if anything is still left after another minute, it says so and keeps the pid file. vLLM's request logging and access log are off, and it sends no usage statistics.

**Keys.** A local answer's key covers `[local]`'s `id`, `max_tokens`, `image_tokens` and `temperature`, when it's set, but not `base_url`: where the server runs doesn't change an answer. A config with no temperature keys as it did before 2026-11-05. The model id differs from Claude's, so neither model's answers serve the other. Claude's requests, and so its keys, didn't change: every stored Claude answer still serves.

**Context sizing.** `serve_vlm.sh` lets vLLM take the longest context the GPUs hold (`--max-model-len auto`), and `start` prints it beside what Pass 2's longest request needs. That request holds `max_images` (100) images, each with its caption and its share of the track JSON, plus the system prompt, the hypothesis and room for the answer: 100 × (280 + 1,280) + 4,096 + 8,192 = 168,288 tokens. The 1,280 tokens of text a frame is a guess. The JSON is mostly digits, about a token a character, and on the synthetic segment's Pass 2 windows it took 85 to 130 characters an object a timestamp, so 20 objects take some 1,700 to 2,600 a timestamp, shared by the cameras shown: one camera with 20 objects takes about twice the guess, and three cameras less than it.

One GPU probably holds less than that. In the first try on ICE, Gemma 4 26B at FP8 held 474,585 tokens of context in about 14 GiB on the L40S, some 31 KB a token. The 31B has twice the layers and twice the KV heads, so about 125 KB a token with a BF16 KV cache, and half that in FP8. The L40S's 40 GiB at 0.9, less about 33 GiB for the FP8 weights and the activations, leaves room for roughly 100,000 to 150,000 tokens in an FP8 KV cache, one 80 GB GPU at BF16 for roughly 90,000, and two L40S at BF16 for roughly 150,000. These are estimates until it runs. Asking for a length that doesn't fit stops vLLM at start, so the script takes what fits instead. A longer request then comes back as a 400, when the prompt alone is too long, or cut off at the context's end, and either is asked again on a rerun. `MAX_MODEL_LEN=<tokens>` asks for a length, and vLLM then won't start unless it fits; its log says the longest that does. By the same estimate, page 1's Pass 2 requests, 16 to 42 frames each, would come to about 37,000 to 78,000 tokens, more with one camera and 20 objects.

**Results.** A local run writes `candidates.json` and `incidents.json` over the top-level results, as the Sonnet comparison did, while every answer stays in `responses/` under its own key. A dry run with Claude's config afterwards rewrites Opus's results from its stored answers and sends nothing.

### Installing, serving and running it

On PACE, from `~/agatha`:

1. Make room on scratch. `pace-quota` showed 268.9 of its 300 GB in use on 2026-11-04, about 31 GB free. The BF16 weights take 62.6 GB, and Google's QAT checkpoint 23.3 GB. Red Hat's FP8 copy of Gemma 4 26B from the first try, `~/scratch/hf/hub/models--RedHatAI--gemma-4-26B-A4B-it-FP8-dynamic`, would free 27 GB if it's no longer needed; that and about 5 GB more, from uv's cache say, make room for the BF16 weights. The QAT checkpoint alone fits now.
2. `infra/pace/submit.sh build_vllm.sbatch`, or with `qat` for the QAT checkpoint instead: run it twice for both, and set `[local]`'s `id` to the QAT checkpoint to serve it. It makes `~/scratch/venvs/vllm` with uv and vLLM 0.32.0 (the newest release more than two weeks old; Gemma 4 needs 0.19.1 or later), apart from AGATHA's environment. Then it asks the Hub how much is left to download, and downloads it into `HF_HOME`, `~/scratch/hf`, only if `pace-quota` shows that much free: a download the quota stops leaves partial files that still count against it, and a full scratch stops every other job's writes. With the model already there, nothing is left to download, so a rerun needs no room. Besides uv's own lines, which list packages, it prints only versions, an exit status and sizes. PyPI's vLLM 0.32.0 is built for CUDA 13.0, which ICE's GPU nodes ran in the first try (torch 2.13.0+cu130, GPU visible); `CUDA=129` takes vLLM's own CUDA 12.9 wheel, for an older driver.
3. In a GPU job, from a shell that stays open, such as the OnDemand session's terminal: `bash infra/pace/serve_vlm.sh start`. It sizes the server from `configs/stage2.toml` (or `VLM_CONFIG`), logs to `~/scratch/waymo/logs/vllm-<node>.log`, and waits up to an hour for `/health`. It prints what Pass 2's longest request needs as it starts, and the context vLLM took once it's up. The 26B took about 4 minutes to start in the first try. Slurm ends whatever an `srun` step started when that step ends, so a server started by `srun ... serve_vlm.sh start` alone wouldn't outlive it; the script's header shows starting it and running Stage 2 in one step instead.
4. In the same job: `bash infra/pace/stage2_cutin.sbatch <rank> local`. It checks `serve_vlm.sh status` first, and prints only counts and `cutin_<rank>`, as with Claude.
5. `bash infra/pace/serve_vlm.sh stop` when done, to free the GPU. `status` says whether it's up.

### Untested until it runs on PACE

- Whether online FP8 loads Gemma 4 31B on the L40S with an FP8 KV cache, and how its answers compare with BF16's. The first try ran a checkpoint already in FP8, the 26B's, with a BF16 KV cache.
- The context vLLM takes on one L40S and on one 80 GB GPU, against the estimates above, and how many tokens real requests take. Sending one real Pass 2 request's body to vLLM's `/tokenize` on PACE would count them, and show whether each caption sits just before its image in the rendered prompt.
- Whether vLLM takes `max_soft_tokens` per request as its recipe says, and whether `--hf-overrides` is needed, or enough, over 280.
- Whether other FlashInfer kernels need ninja and nvcc too. FlashInfer's sampler compiles itself at start, which stopped vLLM in the first try, since ICE's GPU nodes have neither, so `serve_vlm.sh` turns it off (`VLLM_USE_FLASHINFER_SAMPLER=0`) and vLLM samples with PyTorch. The build's `vllm download-kernels` installs FlashInfer's prebuilt kernels where it has them, and the build goes on if it fails.
- Whether uv 0.9.17 takes `--torch-backend cu130`: the first try installed vLLM without it.
- Whether a rerun's resend of a sampled cut-off answer finishes, as 5 of 6 resends by hand did on 2026-11-04. Marking a cut-off `retry`, and its diagnosis, were added that day and tested offline only.
- How often Gemma runs away at temperature 0. Greedy decoding tends to repeat itself more than sampling does, and a full-length cut-off there isn't resent, so each leaves a candidate its page counts as not assessed. Whether answers stay identical across a server restart: the 2026-11-05 check resent requests within one server's run.
- Whether `pace-quota` prints its Scratch line inside a batch job, as the build's check needs. Without it, the build stops before downloading.
- How long the 31B takes to start, and how fast it answers.
- Whether `nvidia-smi` inside a job lists only that job's GPUs, as the GPU count assumes.
- Whether Gemma 4 keeps to the schema and the referential checks as well as Claude, and how its verdicts compare. That's M3's question.

## The data guard

A remote model (Claude) runs only on synthetic segments, whose context names start with `synthetic_`, unless `configs/stage2.toml` sets `real_data_cleared = true`. The public tutorial segment is real data, so it's refused too. Only a literal `true` clears it, so a typo fails closed. The CLI checks before it reads the segment, and every request checks again before it's sent. Local models, such as Gemma 4 on PACE (`LocalModel`, whose server must be on this machine), and the dry run, which sends nothing, may run on anything.

**Cleared on 2026-11-04.** Waymo answered in writing that Stage 2 may send real camera frames and track data to Claude's API, so `real_data_cleared` is now `true`. That covers the pipeline, not AI sessions: Stage 2's outputs on real data stay on PACE, Andrew reviews them in OnDemand, and only counts come back. The guard's code didn't change, and its tests now set `real_data_cleared = false` themselves, so they don't depend on the shipped value.

## Config

`configs/stage2.toml`: `[model]` (Claude's id, effort, max_tokens), `[pass2]` (margin, frame rate, retry and its rate, render profile, `max_images`, `max_request_mb`), `[local]` (the local model's id, `base_url`, `max_tokens`, `image_tokens` and an optional `temperature`, shipped at 0) and `[data]` (`real_data_cleared`). Pass 1's windows come from `configs/stage1.toml`. Editing `[model]` changes every Claude request's key, and editing `[local]`, but for its `base_url`, every local one, so either means new calls. A config with no `[local]`, such as an older copy, still loads; only `--model local` needs it. A `temperature` must be a number, 0 or more: anything else, such as `-0.5`, `"0"` or `false`, stops `agatha-vlm` with a message saying so, whatever the model, and 0, 0.0 and -0.0 are one request. A `stage2.toml` that isn't valid TOML stops it the same way, with the line and column.

## Running it

After Stage 1 has run on the segment:

```bash
agatha-vlm --config configs/stage1.toml --vlm-config configs/stage2.toml --split training \
    --context synthetic_cut_in --model dry-run
```

`--model dry-run`, the default, writes each request Claude would get to `stage2/dry_run/`, with images by reference rather than base64, and sends nothing, so the whole flow runs without a key. Pass 2's requests depend on Pass 1's answers, so a dry run on a fresh segment shows only Pass 1's. `--model claude` sends them, and needs the SDK extra, a key and Andrew's go-ahead, since calls cost money. `--model local` sends them to `[local]`'s server on this machine (see [Gemma 4 on PACE](#gemma-4-on-pace)). `--split` must match the segment's `segment.json`, and `--context` must be a bare name. Pass 2 renders new images, so it reads the source data under `$WOD_ROOT`: after `agatha-stage1 --synthetic`, that's `$WOD_OUTPUTS/synthetic`.

The CLI prints only counts and the context's label, never a name, on failure too: per pass, what became of its requests; Pass 1's candidates before and after merging; Pass 2's retries and verdicts; and the reports written, with how many are marked `duplicate_of` another.

On PACE, `infra/pace/stage2_cutin.sbatch <rank> [local|claude|dry-run]` runs it on the segment at that rank in `cutin_top.txt` (1 is the demo, page 1), after `review_cutins.sbatch` has run Stage 1 there. `dry-run` is the default. With `claude`, it exports the key from `~/.config/agatha/anthropic.env`, which sits outside `~/agatha`, so `sync.sh` never touches it. It prints `cutin_<rank>` in place of the label, since a label is a hash of the name, and masks the name too. After a run it links `$WOD_HOME/review/stage2/cutin_<rank>` to the segment's `stage2/`, so Andrew finds the reports in OnDemand by page number. It runs as a job or inside one: `srun --jobid=<job> --overlap bash infra/pace/stage2_cutin.sbatch 1 claude`. `VLM_CONFIG=<path>` runs it with another `stage2.toml`, as for the Sonnet comparison below. `LIST=<list>.txt` takes the rank from another list, as in the next section. PACE's environment has the SDK now, since the script runs `uv run --extra claude`.

## Random scenes and review pages

To see how Stage 2 does on scenes nobody chose, Gemma labels 15 scenes picked at random, and Andrew labels the same scenes by hand. A review page per scene shows its video, Gemma's reports and Andrew's labels side by side, and how far they agree. Every PACE script that takes a list takes any `$WOD_HOME/selected/<list>.txt` named in `[a-z0-9_]`, and names its scenes `<prefix>_<rank>`: `cutin` for `cutin_top.txt`, as before, and otherwise the list's stem, so `random15_3` is line 3 of `random15.txt`.

On PACE, from `~/agatha`, in this order:

1. **Pick.** `source infra/pace/env.sh`, then `uv run --frozen python infra/pace/pick_random.py`. It writes `$WOD_HOME/selected/random15.txt`: 15 of the labeled segments (the `lidar_box` files of training and validation), none of them on `cutin_top.txt` or `fast_left.txt`, picked with the fixed seed 20261104, one name a line in pick order. It keeps a list that's there already unless given `--force`, so the set stays the one Andrew labels. `--n`, `--seed`, `--exclude` and `--out` change the defaults. It prints only counts.
2. **Download.** `infra/pace/submit.sh download_listed.sbatch random15.txt`: each scene's camera data.
3. **Stage 1.** `infra/pace/submit.sh review_cutins.sbatch random15.txt`: Stage 1 on each scene, printed as `random15_<rank>`. The scan's own review pages are still for `cutin_top.txt` only.
4. **Label with Gemma.** Inside a GPU job, `infra/pace/stage2_list.sh random15.txt local`. It starts `serve_vlm.sh`, runs Stage 2 on each scene through `stage2_cutin.sbatch` with `LIST=random15.txt`, which prints each scene's counts and links its results to `$WOD_HOME/review/stage2/random15_<rank>`, and stops the server however it ends, one that was up already too. It exits 1 if any scene failed, naming them. Launch it detached, so a dropped SSH connection doesn't stop it: from `~/agatha` on the login node, `source infra/pace/env.sh; nohup setsid srun --jobid=<job> --overlap bash -l infra/pace/stage2_list.sh random15.txt local > $WOD_HOME/logs/stage2-random15.log 2>&1 &`. `env.sh` is there for `$WOD_HOME`, which a fresh login doesn't set. Each scene's lines reach the log as it prints them, and its counts come as it ends. `RANKS="3 7"` runs only those scenes, checked against the list before the server starts, and `VLM_CONFIG=<path>` a config with a higher `image_tokens`, which the server takes too.
5. **Pages.** `uv run --frozen agatha-pages --config configs/stage1.toml --list random15.txt`, with `env.sh` sourced as in step 1. It writes `$WOD_HOME/review/random15/`: a page per scene, `random15_<rank>.html`, an `index.html`, and `labels/random15_<rank>.json`, a label template for each scene that has no label file yet. `--ranks 3 7` writes only those pages, and the index lists every scene still. It prints only counts. Andrew then labels each scene in OnDemand, by editing its file, and runs `agatha-pages` again: a label file is never overwritten, so the rerun refreshes the pages around the labels.

**A page** is one file, so it opens in OnDemand like the cut-in pages. It holds the scene's video: Stage 1's FRONT frames (the `pass1` render at about 2 FPS, boxes drawn) over the whole scene as one H.264 clip, embedded as base64. Stage 1 writes a clip per 5 s window, so the page joins the same frames with `render.write_clip`. Then Stage 2's reports, each with its status, the labels it kept and rejected, its times, the explanation, the uncertainty reasons and any `duplicate_of`; then Andrew's labels, or "not labeled yet"; then, once Andrew has labeled, the agreement. Every time on the page is a link that seeks the video to it. The pages hold real frames, so they stay on PACE, mode 600.

**The label format** is `HumanLabels` in `src/agatha/vlm/labels.py`:

```json
{"scene": "random15_3", "reviewed": true, "no_incident": false,
 "incidents": [{"labels": ["SUDDEN_CUT_IN"], "start_s": 8.0, "end_s": 12.0, "notes": "merges late"}],
 "allowed_labels": ["SUDDEN_CUT_IN", "..."]}
```

- `scene` is the file's own scene, so a file copied from another is caught.
- `reviewed` stays false until Andrew is done. Until then the page says "not labeled yet", and a half-done file is fine.
- `no_incident` is true if nothing happened, false if something did, and null until decided. A reviewed file sets it: true with no incidents, false with some.
- Each incident has one or more labels from the taxonomy's 14 (above), a span in the scene's seconds, as the page's times give them, from start to end, and optional `notes`.
- `allowed_labels` is the template's reminder of the 14 labels, and is ignored.

The file is strict: an unknown label or key, a time in quotes or `"true"` in quotes isn't read as labeled, and the page says what's wrong.

**Agreement.** One of Andrew's incidents matches a report that's CONFIRMED or UNCERTAIN, keeps one of its labels (in `subcategories`), and overlaps it in time, touching counting. A report marked `duplicate_of` matches too: it can keep a label the earlier report doesn't, or run longer. A match with a duplicate finds the report it repeats as well, so only the first report of an event can be listed as Stage 2's only. The page lists each of Andrew's incidents with the reports it matches, or as not assessed (see below), or as Andrew's only, and each kept report that no incident matches, as Stage 2's only. It also says whether both say nothing happened: Andrew's `no_incident` against Stage 2 keeping no report. The index has a row per scene, without the video: Stage 2's counts, whether the scene is labeled (yes, no, or invalid), and the agreement in a line.

**An incomplete run.** `agatha-vlm` writes its results when every request was sent, even if some failed or came back invalid. So the page also reads `stage2/candidates.json`, and counts the Pass 1 windows that weren't answered and the candidates Pass 2 gave no verdict on (failed, invalid or no frames). If there are any, the page and the index say so, such as "Stage 2 didn't assess 1 of 6 candidates" or "Stage 2 didn't assess 1 of 5 windows and 1 of 6 candidates", leaving out a part that's 0, and `agatha-pages` counts the scene as incomplete. Under that line, the page lists each candidate Pass 2 gave no verdict on: its id, Pass 1's labels, its span, which seeks the video, and what became of it, such as "INC-003: ROAD_DEBRIS, 8.50 s to 10.50 s, failed". One of Andrew's incidents that no report matches, but that such a candidate would match by the rule above, is listed as "overlaps INC-003, which Stage 2 didn't assess", not as Andrew's only, and the agreement line adds "1 not assessed". That's a request that failed, not an event Stage 2 missed. A Pass 1 window that wasn't answered has no candidates, so it's only counted. If Stage 2 then kept no report, whether it says nothing happened is unknown, and the agreement says so.

**What a rerun fixes.** Gemma's run on the 15 scenes on 2026-11-04 left 6 of 69 Pass 2 requests without a verdict, in scenes 1, 4, 8, 9 and 15. A rerun of `stage2_list.sh` on the same list resends only the failures marked `retry`, and fits and checks every stored answer again. Three answers cited a FRONT time between two frames shown, such as 12.6 s between 12.5 and 12.7, and were otherwise valid, so the rerun turns them into reports from the store, and each move shows in its attempt's `changes` in `candidates.json`. That sends nothing, unless one of them is UNCERTAIN: refused, it got no 10 FPS retry, so the rerun sends that retry. Nobody has checked yet which status the three gave. The other three ran to 8,192 tokens, and were stored before a cut-off was marked `retry`, with `"retry": false`: a rerun reuses them unless they're deleted or set to `"retry": true`, and then resends them (see [Gemma 4 on PACE](#gemma-4-on-pace), "Runaway answers"). Then `agatha-pages` again refreshes the pages. Since 2026-11-05, the shipped config asks at temperature 0, which gives every request a new key, so a rerun with it sends everything again (see "Sampling" there).

**What happened.** On 2026-11-05, 00:34 to 00:46 EST, that rerun (with the three cut-offs moved aside to `~/scratch/agatha_ops/runaway/stored/`) sent 3 requests, all of which finished, and turned the three between-frames answers into reports: all 69 answered, 22 confirmed and 47 rejected. Then Andrew chose temperature 0, and from 01:59 to 02:49 EST all 15 scenes ran again, every request sent anew, with prefix caching off. The sampled results are kept in each scene's `stage2/by_model/gemma-4-26b-a4b-fp8-sampled/`.

| | Sampled (2026-11-04) | Temperature 0 (2026-11-05) |
|---|---|---|
| Pass 2 requests | 69 | 63 |
| Confirmed | 22, in 13 scenes | 9, in 6 scenes (3, 8, 10, 12, 13, 15) |
| Rejected | 47 | 54 |
| Ran away | 3 (resent) | 0 |
| Invalid | 3 (fitted since) | 1 (fitted since) |

At temperature 0 Gemma is much more careful: it rejects pedestrians far ahead or crossing at a junction and cars parked at the edge, which it had confirmed when sampling. Of its 8, 5 match a sampled confirmation (scenes 3, 8, 12, 13, 15) and 3 are new (scene 10's parked car, a second pedestrian in scenes 12 and 15). Citation fitting moved 3 between-frames citations (scenes 6 and 15), which the old checks would have refused. The one invalid answer was scene 12's HARD_BRAKING candidate, which named `ego` among its objects; the ego has no alias, so `check_verdict` refused it, and at temperature 0 a resend would repeat it. So `fit_verdict` now drops an `ego` alias (any case, unless the window has an object by that name) and records it in `changes`, and a rerun of scene 12 at 03:10 EST turned the stored answer into a report with nothing sent: Gemma had CONFIRMED the hard braking, at 17.0 to 19.5 s, so scene 12 has 3 confirmations and all 63 requests have a verdict. Which run is closer to the truth is for Andrew's labels to say.

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

### Sonnet 5.5 against Opus 5.5, pages 1 to 5

The same day, `claude-sonnet-5-5` ran on pages 1 to 5 with every other setting unchanged, effort `medium` included. The config was a copy of `stage2.toml` with only the model's id changed, passed in through `VLM_CONFIG`. Sonnet costs half as much per token ($2 and $10 per million, against $4 and $20). Andrew had approved about $15 for all ten pages. At the rate the first pages showed, all ten would have cost about $22, so Andrew stopped it after page 5. The shipped default is still Opus. Each page's results for both models are in its `stage2/by_model/<model>/`, and the top-level results are Opus's again.

| Page | Candidates, merged | Pass 2 requests | Confirmed, rejected, uncertain | SUDDEN_CUT_IN | Cost |
|---|---|---|---|---|---|
| 1 | 14, 10 (Opus 9, 7) | 13 (Opus 9) | 4, 4, 2 (Opus 5, 1, 1) | uncertain (Opus rejected) | $3.63 (Opus $5.90) |
| 2 | 6, 3 (3, 1) | 3 (1) | 0, 3, 0 (0, 1, 0) | rejected (rejected) | $0.89 ($1.02) |
| 3 | 13, 12 (11, 5) | 13 (5) | 0, 11, 1 (1, 4, 0) | 2 rejected (1 rejected) | $3.83 ($3.52) |
| 4 | 7, 5 (3, 1) | 5 (1) | 0, 5, 0 (0, 1, 0) | 4 rejected (1 rejected) | $1.50 ($1.52) |
| 5 | 15, 10 (10, 8) | 10 (8) | 1, 9, 0 (1, 7, 0) | confirmed, and 1 rejected (the same) | $3.00 ($4.44) |
| All | 55, 40 (36, 22) | 44 (24) | 5, 32, 3 (7, 14, 1) | 1 confirmed (1) | $12.85 ($16.40) |

- **Cost: 78% of Opus, not half.** Sonnet's Pass 1 proposed half again as many candidates, so its Pass 2 sent nearly twice as many requests. Pass 1's requests, the same for both models, came to the same input tokens on both.
- **Speed:** Sonnet took about 17 s a Pass 2 request, against Opus's 29 s.
- **Validity:** no failures, no invalid answers and no fallbacks for either model. Sonnet keeps to the strict schema and checks as well as Opus.
- **Cut-ins:** both models confirmed page 5's cut-in, the same event, and nothing else as one. Sonnet called page 1's cut-in candidate uncertain, where Opus rejected it.
- **Agreement on what they kept** (confirmed or uncertain, by label, with overlapping spans, duplicates left out): each kept 8 labels. Sonnet also kept 6 of Opus's 8, and Opus 5 of Sonnet's 8. Of Opus's 6 confirmed labels, Sonnet kept 5. The miss is page 3's temporary traffic control, where Sonnet kept only an uncertain hard braking. On page 1, Sonnet also confirmed a temporary lane shift, which Opus didn't keep.

That's five pages, too few to rank the models. M3's annotated segments are what can. Using Opus for Pass 1 and Sonnet for Pass 2 would avoid Sonnet's extra candidates and halve Pass 2's price. On these pages that would have cost about 60% of Opus alone, though this is untested and needs a setting per pass.

## Left for M3

- **Evaluation**: Pass 1's recall, Pass 2's precision and end-to-end scores against 20 to 30 hand-annotated segments, reporting UNCERTAIN as abstentions.
- **Tuning**: the model (see [Sonnet 5.5 against Opus 5.5](#sonnet-55-against-opus-55-pages-1-to-5)), effort, image size, margins, frame rates, `max_images`, `max_request_mb` and the merge rule, all on that set. Request size matters. At about 3,500 input tokens a full-size frame, a Pass 2 request's 30 frames come to about 105,000 tokens, and 100 frames to about 350,000, $1.40 at $4 per million. Real camera frames run several hundred KB against the synthetic ones' 63 KB, so on real data `max_request_mb` rather than `max_images` will set a long window's rate. When it holds both a request and its retry, the retry gets at most a frame or two more than the first, or none and isn't sent, so a rule that asks for clearly more frames may be worth it. Check the vision page's current limits on images per request before raising `max_images`.
- **Duplicates**: `duplicate_of` needs a shared object, so two reports of the same debris are never marked. M3 can say whether that matters.
- **Real data**: cleared for Claude on 2026-11-04 (see [The data guard](#the-data-guard)), and run on all ten cut-in pages for $29.07 (see [First real run](#first-real-run)). Those ten are a start on M3's annotated set, and a test of the cut-in definition: Pass 2 confirmed 2 of the 10 cut-ins that Andrew judged real. Gemma 4 on PACE can now stand in through the same interface, say for runs too large to pay for, once it has run there.
- **Cost**: each stored response, failures included, has its full usage, so a run's cost can be added up. Nothing does that yet.
