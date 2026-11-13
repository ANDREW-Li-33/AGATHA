# Stage 3: incident database and explorer

Stage 3 imports Stage 1/2 results into PostgreSQL, uploads plain FRONT clips to
private Supabase Storage, and serves a local Streamlit explorer. Simulation buttons
only queue jobs. PINGS and CARLA are not part of this stage.

The implementation and offline tests are available. **No Supabase project exists
yet, so cloud migrations, real-data ingestion and live Storage playback remain
pending.** Human reviews can be imported again once Andrew finishes them.

## Install

From the repository root:

```bash
uv sync --frozen --extra stage3
```

Add `--extra claude` if this environment also needs Stage 2's Claude SDK. All new
dependencies are in the optional `stage3` extra; Stage 1/2's base install stays lean.
On PACE, use a separate environment, as the new batch script does.

## Supabase setup

1. Create a private research Supabase project. Get its PostgreSQL connection URL
   (direct connection or **session pooler**), project URL, and a server-side secret
   key (or legacy `service_role` key). The database connection needs privileges to
   apply migrations and read/write the tables. Use `sslmode=require` for Supabase.
2. Put these values in `~/.config/agatha/supabase.env`, outside the repository.
   Set its directory to mode 700 and the file to mode 600. Do not paste credentials
   in chat, put them in source control, or pass them as CLI arguments. File shape:

   ```bash
   AGATHA_DATABASE_URL='postgresql://USER:PASSWORD@HOST:5432/postgres?sslmode=require'
   SUPABASE_URL='https://PROJECT.supabase.co'
   SUPABASE_SECRET_KEY='SERVER_SIDE_SECRET'
   ```

   Percent-encode special characters in the database password when using a URL.
   The database password and Storage secret key are different credentials.
3. Load that private file only into the shell running Stage 3:

   ```bash
   set -a
   source "$HOME/.config/agatha/supabase.env"
   set +a
   uv run --frozen --extra stage3 agatha-stage3 migrate
   ```

`infra/sql/001_stage3.sql` creates the database tables and indexes.
`002_private_storage.sql` creates the private `agatha-incidents` bucket, allowing
MP4 and JSON files up to 50 MiB. Both migrations run in one transaction. The
migration ledger records checksums; repeating the command does nothing, and an
edited migration that was already applied is refused. Add a new numbered file
for later schema changes.

All application tables have row-level security enabled, with no browser policies.
`anon`, `authenticated`, and `PUBLIC` receive no table access. The server connects
directly to PostgreSQL. Storage uploads and signing use the server key; no key is
sent to the browser. Storage has no public policies, and the client checks that
the bucket is still private before uploading or signing. Signed video URLs last
five minutes, and can be refreshed in the explorer. They are bearer links: anyone
given one can use it until it expires. This follows Supabase's
[private bucket model](https://supabase.com/docs/guides/storage/buckets/fundamentals).

## Validate and import

Validation requires no Supabase account or credentials:

```bash
uv run --frozen agatha-stage3 validate --segment /path/to/outputs/CONTEXT
```

The input directory must contain `segment.json`, `frames.parquet`, `tracks.parquet`
and `stage2/incidents.json` plus `candidates.json`. It discovers results in
`stage2/` and every `stage2/by_model/*/` folder. Pass `--stage2 /path/to/model/results`
to choose just one. Response files must be present, either beside those results or
in the segment's shared `stage2/responses/` cache. Request windows are preserved
when available; absent historical window files are listed in the run's audit.
Responses, token usage, errors and candidate/attempt links are kept even when
verification failed. Nothing invokes a VLM or changes the source outputs.

To validate and export clips locally, add `--clips /path/to/stage3/clips`. The clip
directory must be outside the input segment directory. This command also provides
a useful dry run on PACE before enabling cloud access.

After setup, import a scene and its clips:

```bash
uv run --frozen --extra stage3 agatha-stage3 ingest \
  --segment /path/to/outputs/CONTEXT --clips /path/to/stage3/clips
```

`--context-s 1` is the default context before and after an incident. The encoder
reads only `render/FRONT/<exact_timestamp>__clean.jpg`. It includes a boundary
frame on either side when available, clips to the scene's available footage, and
preserves real frame timing with ffmpeg concat and `-fps_mode passthrough`.
Stage 1/2 may have rendered only 2–5 FPS; Stage 3 does not invent missing frames.
The explorer shows the actual start/end, frame count and largest frame gap.
Missing clean footage is an error, never a reason to use annotated footage.

Clips are cached locally with frame-content hashes and timing manifests. Storage
keys use the MP4's SHA-256, so models sharing a preview share its bytes. The
database stores only the bucket/path, checksum, size and timing metadata. It never
stores video bytes or expiring URLs. A failed upload leaves the report usable;
rerunning repairs missing assets. If an upload succeeds but its database insert
fails, retrying reuses the uploaded object. No automatic deletion of old assets is
performed.

`--metadata-only` explicitly imports reports without Storage. It is useful for
database setup/testing; the UI explains that video is not yet available.

For scene lists and human reviews:

```bash
uv run --frozen --extra stage3 agatha-stage3 ingest \
  --list "$WOD_HOME/selected/cutin_top.txt" --outputs "$WOD_OUTPUTS" \
  --clips "$WOD_HOME/stage3/clips"

uv run --frozen --extra stage3 agatha-stage3 ingest \
  --list "$WOD_HOME/selected/random15.txt" --outputs "$WOD_OUTPUTS" \
  --human-labels "$WOD_HOME/review/random15/labels" \
  --clips "$WOD_HOME/stage3/clips"
```

List order is the review-page order: `cutin_top.txt` maps to `cutin_<rank>`; other
lists map to `<stem>_<rank>`. A single label file needs both `--human-labels FILE`
and `--scene random15_3`, so files copied to the wrong scene are caught. A supplied
label directory must have each requested scene's file; missing files fail instead
of silently reading as negative reviews. Unreviewed templates are stored as
unreviewed. Reviewed negatives remain distinct from unfinished reviews and failed
model requests. The CLI processes the other scenes if one fails and exits nonzero
if any failed. It prints counts and exception types, never service error messages
that might expose credentials.

On PACE, after syncing the Mac's code and applying the migrations, use:

```bash
infra/pace/submit.sh stage3_ingest.sbatch cutin_top.txt random15.txt
```

This is a CPU job with its own `~/scratch/venvs/agatha-stage3` environment. It reads
the private credentials on PACE, leaves the GPU/server alone, and writes only under
`$WOD_HOME/stage3/`. It imports all model results that actually exist: Sonnet was
run only on the first five cut-in scenes. It cannot create missing model runs or
human annotations. Do not sync or submit while another session is changing the
source results being imported.

## Explorer

Load the same private environment on the machine hosting Streamlit, then:

```bash
bash apps/streamlit_app/run.sh
```

Open `http://127.0.0.1:8501`. The launcher binds to localhost and disables usage
statistics; the app refuses a public binding. Use an SSH tunnel if viewing a remote
server. This MVP has no shared-user login and is not a public deployment.

The **Scenario Explorer** supports category, any/all subcategories, status, object
class and duration. AND/OR combines the category, label, class and duration groups;
minimum/maximum duration stay one range. Status is always a separate gate: OR does
not expose rejected/uncertain reports unless explicitly selected. Confirmed events
are the default, and duplicates are hidden unless requested. Rejected reports are
searchable by their assessed labels, since their retained `subcategories` is empty.
A rejected secondary label never makes a confirmed event match that label.

Details show the plain source video, title, label statuses, **VLM reasoning**,
driving impact, times/status/source/version information, then **Run Simulation**.
Evidence and original reports are expandable. Search results are paginated.

**VLM vs human** includes every imported scene, even one with zero model reports.
It uses the existing review pages' agreement rule (a shared retained label and
overlapping time), keeps uncertain reports visible, and flags incomplete runs and
unassessed candidates. This is a review aid, not M3's one-to-one evaluation metrics.
Each immutable model run can be inspected with its raw responses and errors.

## Data identities and Stage 4 contract

- `segments`: unique context, split, dataset version and preprocessing version;
  original metadata and logical camera/LiDAR/pose component references survive
  host-path changes. A metadata change without a version change is refused.
- `incidents`: content identity includes the full original report, segment
  provenance and resolved duplicate target. Same scene-local `INC-001` from two
  models or prompts stays separate. Changed reports become new revisions rather
  than rewriting old jobs. Identical copies of reports are deduplicated.
- `incident_labels`, `incident_objects`, `incident_evidence`: assessed label
  status/evidence, tracked IDs with classes during the event, exact frame keys and
  supporting/contradictory observations. Type changes remain searchable.
- `stage2_runs` / `run_incidents`: immutable audit snapshot and its reports,
  including negative and incomplete runs. Model fallbacks remain recorded.
- `human_labels`: content-versioned reviews; the latest newly imported version per
  scene is displayed. Reimporting an old version does not make it current.
- `incident_assets`: private Storage pointers, never video data.
- `simulation_jobs`: UUID, incident/segment IDs, original component references,
  scene-relative replay start/end, nullable reconstruction artifact (a JSON object
  for its path/version/config), QUEUED/RUNNING/SUCCEEDED/FAILED status, nullable
  output clip path and error, creation/update times. Times are clamped to the
  actual scene bounds. Only one QUEUED/RUNNING job per incident is allowed;
  simultaneous clicks return the same job. After failure or success, another
  click can queue a new job. A future worker must set `updated_at` on transitions.

PostgreSQL ingestion is transactional per model run; all discovered model inputs
are validated before a scene starts writing. Uploading assets is a retryable next
step because PostgreSQL and Storage do not share a transaction. Reimporting an
unchanged scene creates no new rows and performs no duplicate uploads.

## Tests and remaining live checks

Verified on 2026-11-05: the full suite passed (1,543 tests). After the final
incomplete-run correction and additional integration coverage, all 47 Stage 3
tests passed, including the real PostgreSQL → Streamlit → simulation queue flow.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider
```

The new tests use generated frames/reports plus one actual synthetic Stage 1 →
FakeModel Stage 2 → Stage 3 path. They cover validation, shared response caches,
immutable identities, negative scenes, human reviews, ffmpeg output/timestamps,
private uploads/signed URLs and Streamlit interactions.

Database tests require local PostgreSQL's `initdb` and `pg_ctl` on PATH. They start
a disposable instance on a Unix socket with no TCP listener, apply the actual SQL,
test idempotence/rollback/search/privacy and concurrent job submission, then stop
and remove it. They skip if PostgreSQL or the optional driver is absent. UI tests
skip without Streamlit. The Storage migration is tested against a Supabase-shaped
local bucket table; this does not replace testing Supabase itself.

Still needed once the project exists: run both migrations against Supabase, import
the cut-in and random-scene lists, check the scene/model counts, replay the imports
to confirm unchanged counts, and watch a signed plain video in the explorer. The
human reviews remain pending until Andrew supplies them. Stage 4 jobs will remain
QUEUED until a separate worker is built.
