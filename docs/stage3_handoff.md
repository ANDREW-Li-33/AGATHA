# Stage 3 handoff: incident database, storage and Streamlit explorer

Written by the main session on 2026-11-04, for a new session or another coding agent that builds Stage 3 alongside the main session's Stage 1 and 2 work. Read it once, end to end. Then read `BRIEFING.md` (sections 1, 4, 5 and 8) and the design doc's sections 6 and 7.

Andrew prefers short, plain answers in everyday words. Avoid tool names and jargon in updates unless Andrew asks. Use "Andrew" or they/them.

---

## 1. The project in two minutes

AGATHA (`~/AGATHA` on Andrew's Mac, package `agatha`) finds and explains driving incidents in the Waymo Open Dataset (Perception v2.0.1). It follows Andrew's design doc, `~/Downloads/waymo_incident_project_implementation_v0_1.docx`. If pandoc fails, read `word/document.xml` from the zip.

| Stage | What it does | State |
|---|---|---|
| 1 | Waymo Parquet to full-rate tables, 5-second scene windows, rendered frames, clips, checks | Done, runs on real data on PACE |
| Cut-in scan | Ranks all 1,000 labeled scenes for cut-ins, from labels alone | Done. Andrew judged the top 10 real |
| 2 | Two-pass vision-language model: Pass 1 finds candidate incidents, Pass 2 verifies each and writes an IncidentReport | Done. Runs with Claude (`--model claude`) or Gemma 4 on PACE's GPU (`--model local`) |
| **3** | **Database, private file storage, Streamlit explorer** | **Not started. Your job** |
| 4 | 3D reconstruction (PINGS) and CARLA replay | Not started |

The design doc's MVP goal: one selected cut-in can be processed, classified by the two-pass VLM, retrieved in Streamlit, watched as plain video, and later replayed in CARLA.

**Data rules:** lifted on 2026-11-04. Andrew has full permission from Waymo, so you may read real results (scene names, incident reports, frames), store them in Supabase, and show them in Streamlit. Ordinary care still applies:
- Keep keys and tokens private. Ask Andrew to put them in a private file, never in chat.
- Don't publish Dataset content outside the project unless Andrew asks.

Older text in `docs/` and `infra/pace/` that says "counts only" describes the time before the rules were lifted.

---

## 2. What Stage 3 must do (design doc §7, with §6's contracts)

**Database (Supabase PostgreSQL).** Tables, as the doc names them:
- `segments`: source provenance, meaning the context name, split, dataset version, and Stage 1's `preprocessing_version`.
- `incidents`: time range, event-level status, human-readable text (`vlm_reasoning`, `driving_impact`), the **raw IncidentReport as JSONB**, and the pipeline, model and prompt versions.
- `incident_labels`: one row per assessed label, with its status and evidence. This comes from `label_assessments`.
- `incident_objects`: the tracked objects involved, from `object_ids` (Waymo `laser_object_id`).
- `incident_evidence`: supporting observations and frame references, from `source_frames` and each label's evidence.
- `incident_assets`: the incident's video clips and files, as pointers to storage. Videos never go in rows.
- `simulation_jobs`: the Stage 3 to Stage 4 contract, with job id, incident id, segment refs, time range, reconstruction artifact, a status of QUEUED, RUNNING, SUCCEEDED or FAILED, output clip path and error.

Use foreign keys and unique keys, so that ingesting the same results twice changes nothing. Index status, segment and time, category, and labels.

**Storage (private Supabase Storage).** Short front-camera MP4s with a little context before and after each incident. The design wants them **unannotated**. Store pointers in Postgres, and serve short-lived signed URLs.

**Search.** Filters for category, any or all subcategories, verification status, object class, and duration, with AND/OR. Show confirmed incidents by default, with explicit toggles for uncertain and rejected. No natural-language search for the MVP.

**Streamlit UI.**
- A **Scenario Explorer** listing searchable incidents.
- **Incident Details** showing, in order:
  1. plain source video;
  2. a title;
  3. subcategory labels under it;
  4. a heading "VLM reasoning" with the explanation;
  5. timestamps, status and source info;
  6. a **Run Simulation** button.
- The button only creates a `simulation_jobs` row and shows its state. The Streamlit process must never run PINGS or CARLA. Stage 4 isn't built, so the jobs just stay QUEUED for now.

Suggested code layout, adapting the design doc's §11 to the `agatha` package:
- `src/agatha/database/`: schema helpers, ingestion and asset upload;
- `apps/streamlit_app/`: the UI;
- `infra/sql/`: Supabase migrations and indexes;
- `tests/test_stage3_*.py`: tests, synthetic data only.

---

## 3. Stage 3's input: what Stage 2 writes

### IncidentReport (`src/agatha/vlm/schema.py`, class `IncidentReport`)
One per incident, listed in each scene's `stage2/incidents.json`:

```json
{
  "schema_version": "0.1",
  "segment_id": "<Waymo context name>",
  "incident_id": "INC-003",
  "duplicate_of": null,                      // or an earlier incident_id: one event reported twice
  "start_s": 8.2, "end_s": 11.4,             // seconds since the scene's first frame
  "categories": ["DYNAMIC_HAZARD"],
  "subcategories": ["SUDDEN_CUT_IN"],        // labels Pass 2 didn't reject
  "object_ids": ["<laser_object_id>"],
  "verification_status": "CONFIRMED",        // or REJECTED / UNCERTAIN. Rejected and uncertain are kept on purpose
  "label_assessments": [{"label": "SUDDEN_CUT_IN", "evidence": ["..."], "contradicting_evidence": ["..."], "status": "CONFIRMED"}],
  "vlm_reasoning": "Evidence-based explanation.",
  "driving_impact": "Potentially reduces clearance along ego path.",
  "uncertainty_reasons": [],
  "source_frames": [{"camera": "FRONT", "timestamp_s": 8.5, "frame_timestamp_micros": 1550000000000000}],
  "pipeline_version": "0.1.0+<hash>", "model_id": "RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic",
  "prompt_version": "...", "preprocessing_version": "..."
}
```

Notes:
- **The taxonomy:** 14 labels in 3 categories (DYNAMIC_HAZARD, CONSTRUCTION_DISRUPTION, ROAD_OBSTRUCTION). They're in `TAXONOMY` and `CATEGORIES` in the same file.
- `incident_id` is unique within a scene only. Key incidents on (segment_id, model_id, prompt_version, incident_id), or a hash of the report.
- Results from different models for the same scene sit side by side (see section 4).
- `stage2/candidates.json` links every Pass 1 window and Pass 2 attempt to its raw model response, under `stage2/responses/<key>.json`, which holds the raw text, the model and the token use. Validation errors are in `stage2/errors/`. Keep the raw payload: the design wants the original model response preserved.
- `frame_timestamp_micros` is the exact frame key, for cutting clips at the right frames.

### Human labels (being built now by the main session)
Andrew is labeling 15 random scenes by hand, to compare with Gemma. One JSON file per scene goes at `$WOD_HOME/review/random15/labels/random15_<n>.json`, in roughly this shape:

```json
{"scene": "random15_3", "reviewed": false, "no_incident": null,
 "incidents": [{"labels": ["SUDDEN_CUT_IN"], "start_s": 8.0, "end_s": 11.5, "notes": ""}]}
```

The final format is documented in `docs/stage2.md`, section "Random scenes and review pages", once the main session finishes. Check there before writing the ingestion. The design's evaluation stage (M3) wants these stored too, for example as a `human_labels` table linked to segments, so you can show "VLM vs human" in the UI.

### Video
Stage 1 writes frames and clips under each scene's output folder (layout in `docs/stage1_architecture.md`, "Output layout"):
- `render/<CAMERA>/<frame_timestamp_micros>__clean.jpg`: original camera frames, unannotated. Use these for the design's plain clips.
- `render/<CAMERA>/<frame_timestamp_micros>__<profile>.jpg`: frames with boxes drawn.
- `clips/...`: Stage 1's MP4s. Check whether they're annotated. The design wants unannotated incident clips, so you'll likely cut your own from the `__clean.jpg` frames over [start_s - context, end_s + context]. `src/agatha/render.py`'s `write_clip` shows how to keep real frame timing (ffmpeg concat; on Linux it needs `-fps_mode passthrough`).

---

## 4. Where everything lives

### On the Mac (`~/AGATHA`)
`src/agatha/` (Stage 1, the scan, `vlm/` = Stage 2), `tests/` (about 1,340 tests, all synthetic), `configs/` (stage1, scan, stage2 TOML), `docs/`, `infra/pace/` (PACE scripts), `reports/` (small JSON summaries), `scenario-miner/` (Andrew's older prototype: read-only), `BRIEFING.md` (the full picture, kept current). Environment: `uv` with `.venv/`. Run tests with:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider
```

### On PACE ICE (Georgia Tech's cluster)
- **Code:** `~/agatha`, copied from the Mac by `infra/pace/sync.sh`. That's `rsync --delete`, so **edit on the Mac and sync. Never edit in `~/agatha` on PACE, or the next sync deletes your work.**
- **Paths:** `source infra/pace/env.sh` (from `~/agatha`) sets them:
  - `WOD_HOME=~/scratch/waymo`;
  - `WOD_ROOT=$WOD_HOME/v2_0_1` (Waymo Parquet, by split and component);
  - `WOD_OUTPUTS=$WOD_HOME/outputs` (`<context>/` per scene, Stage 1 and 2 outputs);
  - `HF_HOME=~/scratch/hf` (model weights).
- **Environments:** AGATHA's in `~/scratch/venvs/agatha` (`uv run --frozen ...` from `~/agatha`), and vLLM's in `~/scratch/venvs/vllm`.
- **Scene lists:** `$WOD_HOME/selected/`: `cutin_top.txt` (10 cut-in scenes), `fast_left.txt` (3), `random15.txt` (the 15 random ones, being made now).
- **Stage 2 results by scene:** `$WOD_HOME/review/stage2/<list prefix>_<rank>` are links to each scene's `$WOD_OUTPUTS/<context>/stage2/`. For the cut-in pages, the top-level `incidents.json` is Claude Opus's. Other models' results are in `by_model/<model>/` (`claude-opus-5-5`, `claude-sonnet-5-5` for pages 1 to 5, `gemma-4-26b-a4b-fp8`).
- **Review pages:** cut-in pages in `$WOD_HOME/review/cutins/`. The random-scene pages, with video, Gemma's labels and Andrew's, will be in `$WOD_HOME/review/random15/`.
- **Logs:** `$WOD_HOME/logs/`.
- **Disk:** scratch is at about 270 of 300 GB, and most of that is Andrew's other projects. Don't fill it.

### Seeing PACE from the Mac
1. **Andrew opens the connection**, since it needs his password and Duo. Never type credentials yourself:
   ```bash
   ssh -M -S ~/.ssh/cm-ice ali497@login-ice.pace.gatech.edu
   ```
   It drops after a while, and sometimes the login node doesn't answer at all. When a command fails with "Control socket ... No such file" or "Broken pipe", ask Andrew to log in again.
2. **Run commands** from `~/AGATHA` on the Mac: `infra/pace/ice.sh '<command>'`. It uses that connection, with `BatchMode`, so a dropped connection fails fast instead of asking for a password.
3. **Copy files** with `scp -F infra/pace/ssh_config ice:<path on PACE> <local path>`, and push code with `infra/pace/sync.sh`.
4. **Compute:** the login node is for light work only. Heavier work goes in a Slurm job, either `infra/pace/submit.sh <script>.sbatch [args]` or a step in Andrew's interactive job: `srun --jobid=<job> --overlap ...`.
   - Andrew's current GPU job is **6808392** (OnDemand VS Code, L40S GPU). Check it with `squeue -u $USER`.
   - A step started over SSH can die with the connection. Launch long ones detached, from `~/agatha`, with `env.sh` sourced first so `$WOD_HOME` is set (without it the log goes to `/logs/...` and nothing starts): `cd ~/agatha && source infra/pace/env.sh && nohup setsid srun --jobid=<job> --overlap bash -l <script> > $WOD_HOME/logs/<name>.log 2>&1 < /dev/null &`.
5. **Andrew views pages** in OnDemand (<https://ondemand-ice.pace.gatech.edu/>), through a Jupyter session: right-click an HTML file, then "Open in New Browser Tab". VS Code shows HTML as raw text.

---

## 5. Don't collide with the main session

The main session is running the 15-random-scene work right now, and will keep working on Stages 1 and 2. To stay out of each other's way:
- **Don't edit:**
  - `src/agatha/` outside a new `src/agatha/database/`;
  - `configs/stage1.toml`, `scan.toml` and `stage2.toml`;
  - `infra/pace/` scripts that exist now (add new ones instead);
  - existing tests.
- **`pyproject.toml` and `uv.lock`:** the main session may be editing them, for example to add a console script. Add your dependencies (`supabase`, `streamlit`, maybe `psycopg`) as an **optional extra**, `stage3`, so PACE's Stage 1 and 2 environment stays lean. Do it when section 8 of `BRIEFING.md` says the random-scene work is done, or ask Andrew to check with the main session first.
- **Don't use GPU job 6808392**, and don't start or stop the vLLM server.
- **Results folders on PACE** (`$WOD_OUTPUTS/*/stage2/`, `$WOD_HOME/review/`) are read-only for you. Write Stage 3's clips and exports under `$WOD_HOME/stage3/`.
- **Nothing is in git.** Suggest that Andrew initialize a repository before two sessions edit the same folder, and do it only with Andrew's OK.
- When you finish something, add one line to section 8 of `BRIEFING.md`. Leave the other lines alone.

---

## 6. Suggested first steps

1. **Read** `BRIEFING.md`, `docs/stage2.md`, `src/agatha/vlm/schema.py`, and design doc §6, §7 and §11.
2. **Supabase setup with Andrew.** Andrew creates the project and puts its URL and keys in a private file, for example `~/.config/agatha/supabase.env` with mode 600. Load it the way `stage2_cutin.sbatch` loads the Anthropic key: sourced only by the command that needs it, never printed. Make the storage bucket private.
3. **Migrations** in `infra/sql/`: the tables from section 2, constraints, indexes, and a `human_labels` table.
4. **Ingestion** (`src/agatha/database/ingest.py`) of one scene's `incidents.json`, `candidates.json` and raw responses, idempotent and validated with the `IncidentReport` model. Test it on the synthetic segment: `agatha-stage1 --synthetic`, then Stage 2 with the existing `FakeModel` test helpers.
5. **Clip export:** cut unannotated incident clips from `__clean.jpg` frames, upload them, and record them in `incident_assets`. Run this on PACE, where the frames are.
6. **Streamlit app** (`apps/streamlit_app/`): Scenario Explorer, Incident Details and the Run Simulation stub, reading Supabase with signed URLs for video.
7. **Real data:**
   - ingest the 10 cut-in pages, with every model's results tagged by `model_id`;
   - ingest the 15 random scenes, plus Andrew's labels once they exist;
   - show "VLM vs human" for those.

## 7. Done looks like
- Migrations apply cleanly to an empty Supabase project, and re-running ingestion changes nothing.
- All of today's results are in the database: 10 cut-in scenes (Opus, Sonnet and Gemma) and 15 random scenes (Gemma, plus Andrew's labels).
- The Streamlit app filters by category, label, status, object class and duration, plays each incident's plain clip, shows the "VLM reasoning", and creates simulation job rows.
- New tests pass offline on synthetic data, and the existing suite stays green.
- `docs/stage3.md` explains setup and running, and `BRIEFING.md` section 8 has a Stage 3 line.
