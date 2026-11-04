# Stage 1 setup and decisions

Prepared 2026-11-01 against the implementation doc v0.1. Source notes, with URLs, are in [research/2026-11-01/](research/2026-11-01/).

Nothing here has run on PACE or on real Waymo data yet. The license sections report what the text says and are not legal advice.

## 1. Gates

| Gate | What it blocks | Who | Status |
|---|---|---|---|
| ICE login works | All PACE work | You | Unconfirmed. Port 22 is reachable, but that says nothing about your account. |
| This project is OK on ICE | All PACE work | You, then instructor or TA | Not cleared. PACE neither allows nor bans it. |
| Waymo license accepted | Any download | You | Unconfirmed. `gcloud` on the Mac has no logged-in account. |
| Waymo's written answer on third-party processing | Real data reaching the Claude API, Claude Code, or Supabase | You send it, Waymo replies | Not sent. Plan on no reply before Nov 5. |

**Why ICE is a gate.** ICE access comes through courses and workshops, and students can't apply themselves. PACE sends research use to Phoenix. If your access is course-based and this project isn't part of the course, ask your instructor or TA. PACE doesn't support ICE students directly.

If ICE doesn't work out:

- A faculty advisor can sponsor Phoenix.
- A College of Computing student can ask about a school allocation or HCE@CoC.
- PACESHIP review typically takes up to 2 weeks, so it won't help before Nov 5.
- The Mac can hold one segment (about 0.33 GB) but it has only 5.0 GB free, and real data would then sit on the machine running Claude.

**Why the license is a gate.** Downloading or using the data counts as accepting the terms, and they cover Non-commercial Purposes only: research, teaching, scientific publication, and personal experimentation. They exclude purposes "primarily intended for or directed towards commercial advantage or monetary compensation."

The text says nothing about portfolios or job searches. A genuine research or learning project that you also talk about in interviews reads as personal experimentation. A project built mainly to win paid work could be read the other way. Only you know which is the honest description, so that part is your call.

## 2. Setup steps

### Yours

Only you can do these, because they involve your credentials or accepting terms.

**1. Log in to ICE once.** Connect GlobalProtect first, unless you're on a wired campus-building connection. Since August 2026, eduroam and dorm networks need the VPN too. I took `ali497` from your `gtae` SSH entry.

```bash
ssh ali497@login-ice.pace.gatech.edu
```

Then run this on the login node:

```bash
hostname; id -Gn; pace-quota
```

If the connection closes right after your password, suspect that you have no active ICE account. That's an inference from one third-party report.

**2. Clear ICE use** with your instructor or TA if the project isn't part of a course. There's a draft in section 7.

**3. Register at <https://waymo.com/open/licensing/>** with the Google account you want to use.

- Access follows that account, and `gcloud` must later log in as the same one.
- Check that the terms shown are the March 2025 version.
- The form was failing until a fix reported on Oct 20. If you see `LICENSE_TERMS_NOT_ACCEPTED`, retry, then use GitHub issue 1014 or open-dataset@waymo.com.
- Enter Institution truthfully, without implying that you accept on Georgia Tech's behalf. The license has you warrant authority if an institution exercises the rights.

**4. Send the Waymo email** in section 7. I don't send messages for you.

**5. Turn off model improvement** at <https://claude.ai/settings/data-privacy-controls>. This is a backstop, because the boundary in section 3 already keeps real data out. On consumer plans, turning it off doesn't undo training that has already started.

**6. Later, sign gcloud in on PACE.** After I install it, run this on PACE:

```bash
gcloud auth login --no-browser
```

It prints a long `gcloud auth login --remote-bootstrap=...` command. Run that on your Mac, sign in with the registered account, and paste the resulting URL back into PACE.

### Mine

These assume you approve SSH use. Without it, I write the scripts and you run them.

1. Build everything that needs no data: repo, lockfile, column constants, loader, projection, tracks, SceneWindow builder, report writer, synthetic fixtures, PACE environment script, and Slurm templates.
2. Test how I can reach PACE, because the evidence conflicts. One third-party note says ICE blocks SSH multiplexing, and others report key login without Duo. The order is key login, then multiplexing, then the app's Terminal panel, then you running scripts and pasting reports.
3. Probe quotas, default permissions, modules, and whether compute nodes can reach Google and PyPI.
4. Create `~/scratch/waymo` at mode 700 and verify it before any file lands.
5. Build the environment in a Slurm job with caches and the venv on scratch, then run the synthetic test suite.
6. Check bucket access, real sizes, and the schema.
7. Download six components for the tutorial segment `10023947602400723454_1120_000_1140_000` in a job.
8. Run FRONT and read only the sanitized report.
9. Freeze thresholds after your visual review, then add FRONT_LEFT and FRONT_RIGHT by config.
10. Run a label-only cut-in scan across the 1,000 labeled segments, about 2 to 7 GB (estimated), so you can pick the demo segment by eye.
11. Remind you to run `gcloud auth revoke` once the downloads finish, since it's a credential on a shared cluster.

## 3. Interim data boundary

Until Waymo answers in writing, no real frames, rows, box values, tracks, or context lists enter any Claude session, including this one.

The license makes no distinction between a pipeline calling an API and a coding agent reading a file, and whatever Claude Code reads goes to Anthropic. The one exception to the context-list rule is the tutorial segment name, which is already public.

- **What can come back to me:** column names, shapes, counts, timings, pass or fail, and aggregate statistics.
- **How that's enforced:** a run writes one report, to `$WOD_OUTPUTS/reports/<label>.json`, holding only those fields. It refuses unknown strings, numbers the size of a raw timestamp, and lists the length of a column. Exceptions record type and file:line, not the message. Tests never run on real data. Slurm logs stay in the private directory.
- **Reports are fetched by name, never listed:** a label is an unsalted hash of a context name that anyone with the bucket listing can invert, so a list of labels counts as a context list.
- **Visual review is yours:** open the contact sheet in OnDemand Jupyter at <https://ondemand-ice.pace.gatech.edu/> using Chrome or Firefox, and tell me pass and fail counts.
- **Stage 2 consequence:** real-data Stage 2 waits for Waymo's reply. If the answer is no, or never comes, an open-weight vision model on an ICE L40S avoids the third-party question, at the cost of reworking Stage 2. Writing Stage 2 behind a model-agnostic interface keeps that open.
- **Stage 3 consequence:** the same applies to Supabase. Incident metadata is the lowest-risk use. Clips are not cleared.

One research note proposed Claude Code deny rules and sandbox settings. I'm not applying them. The data stays on PACE, so local rules would protect nothing, and the sandbox would stop plain `ssh`.

## 4. Changes from the design doc

| Doc | Problem | What Stage 1 does instead |
|---|---|---|
| Section 3: Waymo v2 libraries | The package pins `jaxlib==0.4.13`, which PyPI no longer has (confirmed 404), and importing it pulls in TensorFlow. | Plain PyArrow, with a column-constants module checked against `pq.read_schema`. |
| Sections 4 and 6: SceneWindow as an export | Pass 2 needs any time range at 5 to 10 FPS, and Stage 4 needs full-rate data. | Full-rate `frames.parquet` and `tracks.parquet`. SceneWindow becomes `build_scene_window(segment, t_start, t_end, fps, cameras, profile)`, and renders are cached by timestamp and profile. |
| Section 4 step 3: projection | Labels are in `vehicle_pose`'s frame, but each image has its own pose. Using the frame pose is about 30 px off on FRONT_LEFT and FRONT_RIGHT at 25 m/s (simulated). Camera axes are x forward, y left, z up, so depth is x. | `project(p_vehicle, T_w_vf, T_w_vi, T_v_c, intrinsics)` in float64, with forward distortion. Images are never undistorted. |
| Section 4 step 3: clipping | The distortion polynomial folds back, so a car beside the ego can draw a ghost box inside the image. | Sample each 3D edge, keep points with positive depth and `r_d` in [0.8, 1.2], then clip. |
| Section 4: timing | The doc doesn't mention it. Global-shutter error is about 25 to 30 px at image edges at 25 m/s and 15 m (simulated). Extrapolating a box by 50 ms is wrong by about 103 px for a crossing car. | Draw `lidar_camera_synced_box` when this camera is the object's most visible one, otherwise `lidar_box` through the image pose. No extrapolation. Record `draw_source`. Overlays are visual aids, not measurements. |
| Section 4 step 4: motion | `speed` is absolute in world axes, `z` can be a filled-in zero, and acceleration is noisy. Differencing raw vehicle-frame positions invents lateral motion under ego yaw (a false 0.9 m/s in one public sample). | Rotate x and y into ego axes. Difference in a segment-local frame over at least 0.2 s, and recompute acceleration over at least 0.5 s. Field names carry frame and unit. |
| Sections 4 and 6: coordinate frame | "A declared coordinate frame" doesn't say which one. The world origin is remote, and the vehicle origin is the rear axle on the ground, so raw x overstates clearance by roughly 4 m (estimate). | Three named frames: `world`, `segment_local`, `ego_at_sample`. Gaps use box corners and a declared ego footprint. |
| Sections 4 and 5: sampling | Frames can be missing (one segment has 191), spacing varies, and 2 and 5 FPS grids don't nest. | Nearest timestamp to a grid anchored at segment start, deduplicated, reporting requested and achieved FPS. Joins use int64 timestamps only. |
| Section 5.2: windows | The list assumes exactly 20 s, but segments end near 19.7 to 19.8 s. A 1 s overlap can split cut-ins that last up to 6 s. | Windows are computed from the real last timestamp. Each window gets track-only context on both sides. Stride 2.5 s gets tested on the labeled set. |
| Section 4: filtering | Not defined, and the obvious filters remove stopped vehicles or a cut-in vehicle before it reaches FRONT. | Full-rate tables are never filtered. SceneWindow ranks by distance to the ego path, caps at 20 objects, and reports what it omitted. |
| Section 6: object IDs | 22-character IDs are unreadable on images and easy for a VLM to corrupt (inference, untested). | Per-segment aliases such as `V012`, with a map back to `laser_object_id`. |
| Section 4: outputs | Only annotated images are listed, but cones, barriers, and debris have no 3D boxes, so wireframes can hide the evidence. | Keep the original JPEG bytes plus annotated variants per render profile. |
| Section 6: absent evidence | The contract doesn't say that no box isn't the same as no object. Labels stop at 75 m, No Label Zones have no polygons in v2, and cameras cover about 230 degrees. | A fixed limits block in every SceneWindow, plus `first_t_s`, `last_t_s`, and range at first sight per track. |
| Section 4: inputs | `projected_lidar_box` and `lidar_camera_synced_box` are listed as optional. `camera_box` has no vehicle link. | Both are required. `camera_box` is skipped. |
| Section 4: acceptance checks | Too vague to fail. | The numeric checks in section 6. |
| Section 9: build order | The doc starts the M5 mesh spike alongside M1 and M2. | Deferred until Stage 1 passes, since you asked to start here. `tracks.parquet` keeps Spike A ready. |
| Section 10: fixtures | Real rows can't go in git. | Synthetic Parquet fixtures, plus the constants from Waymo's `camera_model_test.cc` as unit vectors. |

## 5. Defaults

Tell me if you'd rather change any of these.

- **Environment:** Python 3.12 through uv on both machines, with `pyarrow`, `numpy`, `pandas`, `opencv-python-headless<5`, `pydantic`, `pytest`, and `imageio-ffmpeg`. No TensorFlow.
- **Dataset:** Perception v2.0.1, training and validation only. No v1.4.3 maps. Lane position uses an ego-path proxy.
- **Projection:** NumPy global shutter, with `projected_lidar_box` as the numeric oracle. No port of Waymo's rolling-shutter solver, because its patent grant covers only Waymo's own code.
- **Motion:** x and y only, with 0.2 m/s as the stationary threshold, which is Waymo's own.
- **Ego footprint:** 4.0 m from rear axle to front bumper, labeled as an assumption in every output that uses it.
- **Repo:** local git with no remote and no push until you ask. Code reaches PACE by rsync. Section 11 layout. The `.gitignore` covers data, outputs, Parquet, TFRecord, JPEG, MP4, and Slurm logs from the first commit.
- **Attribution:** the license's sentence form in `NOTICE`, since it applies to any work made using the data, including software.
- **PACE paths:** data, outputs, and logs under `~/scratch/waymo` at mode 700, `umask 077` in every job, and nothing Dataset-derived in home, because home has daily snapshots.
- **Jobs:** CPU only, with no `-p` and no `-A`, starting at 4 cores, 16 GB, and 1 hour.
- **Download:** `gcloud storage` inside a job. No GCP project or billing account unless `400 UserProjectMissing` appears.

## 6. Stage 1 acceptance checks

The pixel thresholds are unmeasured proposals. They get calibrated on the first segment and then frozen.

1. Joins use exact int64 equality, with zero orphans and no duplicate keys.
2. A gap report replaces any fixed frame count.
3. Per-edge residuals against `projected_lidar_box`, on boxes that are unclamped with all corners in front of the camera. The limits judge what the images show: the synced box where this is its camera, otherwise the label. Each camera must meet median 5 px and p95 30 px. FRONT's label path must meet those limits too, because a synced box never reads the label columns that Stage 4 relies on, and FRONT's image pose is effectively the frame pose. The label path must also meet 1 px and 3 px on still objects at ego speed under 1 m/s with yaw rate under 0.05 rad/s, because yaw moves the image during readout at any range. Elsewhere it ignores rolling shutter, so on the side cameras it's reported by camera, speed, and column, but not judged.
4. Track integrity: gaps, type changes, and implied speed spikes.
5. The world-axes hypothesis for `speed` beats the vehicle-axes and ego-relative hypotheses across moving tracks.
6. Two runs produce identical hashes of the manifest and JSON.
7. FRONT_LEFT and FRONT_RIGHT pass check 3 by config change alone.
8. You review a random sample, the worst-residual frames, image edges, and near and fast objects.

## 7. Drafts

Both are yours to edit and send. Keep the Waymo email to data flows and purposes, because Waymo's Terms of Service treat what you send them about their products as non-confidential.

### To open-dataset@waymo.com

Subject: License questions on third-party processing of Waymo Open Dataset content

> Hi Waymo Open Dataset team,
>
> I'm a graduate student at Georgia Tech working on a personal research project with Perception v2.0.1, and I want to stay within the Waymo Dataset License Agreement for Non-Commercial Use (March 2025). The license doesn't address a few data flows, so I'd like to ask before I use them.
>
> 1. May I send annotated camera frames and label-derived JSON to a commercial LLM API that acts only for me? The provider's terms say it doesn't train on this data and deletes it within 30 days, although flagged content can be kept for up to 2 years.
> 2. Is the answer the same for an AI coding assistant that reads those files during development?
> 3. May I keep short clips in private cloud storage, such as Supabase, that only I can access?
> 4. Does a personal research project stay within Non-commercial Purposes if I also describe or show it in job applications?
> 5. Does a portfolio page, blog, or thesis count as a research publication under the De Minimis clause, and what size counts as small? May I show frames in a live screen share during an interview?
> 6. May I publicly show CARLA replay renders or a reconstructed mesh derived from the data?
> 7. Does Production Systems reach third-party infrastructure such as an API provider, or a demo limited to registered users?
> 8. Are the sample files in the waymo-open-dataset GitHub repository covered by Apache 2.0 or by the Dataset license?
>
> Thanks for your help.
>
> Best,
> Andrew

It's your call whether to mention that one of those applications is to Waymo.

### To your instructor or TA

Only needed if the project isn't part of a course.

> Hi [name],
>
> I'm in [course] and have ICE access through it. I'd like to use ICE for a small personal research project on autonomous-driving data (the Waymo Open Dataset): CPU jobs at first, under 20 GB of scratch, and possibly a little GPU time later. It isn't part of the course. PACE's docs describe ICE as being for classes, so I wanted to check whether this is okay, or whether you'd point me somewhere else. If you're not sure, could you ask PACE?
>
> Thanks,
> Andrew

## 8. November 5

The 15-minute screen doesn't need visuals, and describing the project verbally is fine.

If you do show anything, stick to code, architecture, synthetic overlays, and aggregate metrics, with no Dataset frames. Don't send clips to the recruiter, because copies may go only to people who registered and accepted the terms. Describe scene events, not the performance of any real vehicle, since Waymo's FAQ rules that out.

No PACE maintenance is scheduled in November. The next window is tentatively Jan 11 to 13, 2027. Unplanned problems do happen, though: the ICE scheduler failed on Oct 29 and the VPN had trouble on Oct 31. Don't rely on ICE being up during the call, and check <https://status.gatech.edu/> on Nov 3 and 4.

## 9. PACE quick reference

| Item | Value |
|---|---|
| Login node limits | 1 CPU-equivalent and 4 GB per user. Going over kills the process with no error message, so load no Parquet there. |
| Home | 30 GB, daily snapshots |
| Scratch | 300 GB and 1 million files, no backup. Files untouched for 120 days are deleted at semester end. |
| CPU job | Up to 512 CPU-hours and 18 h |
| GPU job | Up to 16 GPU-hours and 16 h. L40S: `--gres=gpu:L40S:1`. |
| Hardware for later | 4 nodes with 8 L40S 48 GB cards each, which matches the design doc |
| Slurm | No `-p` and no `-A`. Partitions and QOS are assigned automatically. |
| Python | uv on a compute node. Set `UV_CACHE_DIR`, `UV_PYTHON_INSTALL_DIR`, and `UV_PROJECT_ENVIRONMENT` to scratch first. |
| Support | Your instructor or TA, not PACE directly |

`pace-quota` is the authority for your account's limits. Some third-party notes still quote older figures.

### Measured on your account, 2026-11-01

These replace anything above or in the research notes that disagrees.

| Item | Measured |
|---|---|
| Account | Active, Account `coc`, QOS `coc-ice`, partition `coc-cpu` |
| Home | 2.1 of 30 GB used, mode 700 |
| Scratch | 226.2 of 300 GB and 619k of 1M files used, so 74 GB free. Mode 700, no ACL entries for others. |
| Platform | x86_64, RHEL 9.8, glibc 2.34 |
| Default memory | 4 GB per core (`DefMemPerCPU=4096`), not the 1 GB in PACE's docs |
| Modules | `uv/0.9.17`, `python/3.12.5`, `ffmpeg/7.1`. No `gcloud` or `rclone`. |
| Compute-node egress | Reaches Google Cloud Storage, Google sign-in, PyPI, GitHub, and uv's Python host |
| SSH | Reusing your master socket `~/.ssh/cm-ice` works with no password. `module`, `sbatch`, and `pace-quota` work over non-interactive SSH. |
| Other access | Your groups include Phoenix (`pace-phoenix-users`, `pace-lgan31`) and `coc-futurama-elibigle`, so there are real fallbacks if ICE stops working for this project. |
| Bucket access | Works from a Gmail account with no billing project. Splits: training 798, validation 202, testing 16, testing_location 134. 17 components. |
| Tutorial segment | 588.6 MiB in all, 329 MiB for the six Stage 1 inputs. 199 frames, 995 images, 18,633 `lidar_box` rows, 15,115 synced rows, 17,523 projected rows. |
| Real schema | All required columns present. Transforms are `fixed_size_list<double>[16]`, keys `int8`, `linear_velocity` `float`. Most files add an `index` column. |

If the master connection drops, reopen it with this command:

```bash
ssh -M -S ~/.ssh/cm-ice ali497@login-ice.pace.gatech.edu
```

## 10. First real results, 2026-11-02

Stage 1 ran on two real segments with three cameras: the slow public tutorial segment, and the fastest of the 798 training segments, a steady 29 m/s. The second one's name stays on PACE.

The tutorial segment passes all nine checks, and two separate runs gave identical output. The highway segment passes eight. Check 3 there is `no_data`, not a failure: the only objects wholly inside FRONT_LEFT were signs, which aren't drawn. So FRONT_LEFT at highway speed is still unproven.

| Measured against Waymo's `projected_lidar_box` | Tutorial (under 5 m/s) | Highway (29 m/s) |
|---|---|---|
| Drawn boxes, all cameras: median and p95 | 0.005 and 0.26 px | 0.015 and 0.22 px |
| FRONT drawn p95 | 0.12 px | 0.07 px |
| FRONT_LEFT drawn p95 | 0.43 px | no road user wholly in view |
| FRONT_RIGHT drawn p95 | 1.54 px | 3.11 px, on 79 boxes |
| Label path alone: FRONT, FRONT_LEFT, FRONT_RIGHT p95 | 0.78, 1.25, 2.23 px | 3.3, 67.9, 80.3 px |

What this settled:

- **`speed` is absolute, in world axes.** World-axes residual 0.04 m/s, against 59 and 30 m/s for the other two readings at 29 m/s.
- **The synced-box frame is right.** On stationary objects at 29 m/s, synced p95 is 0.4, 4.8, and 2.6 px, against 12, 68, and 36 px for the label.
- **Image pose is the right direction.** On the side cameras it beats frame pose at p95 (68 against 112 px, and 80 against 93 px), and the signed errors flip between them. The remainder is rolling shutter.
- **`projected_lidar_box` carries object motion.** Synced boxes match it to a median of 0.017 px on objects mostly moving at 15 m/s or more. The research had inferred the opposite.
- **Timing matches the paper.** Capture time at mid-exposure falls within about 2 ms of the nominal 37.5, 50, and 62.5 ms. Readout is 44 ms, left to right, on both segments.
- **The label path can't be exact at speed.** That's why check 3 changed. It drew 18 of the 236 overlays in the highway segment's windows. Most label-drawn boxes straddle the image border, where Waymo clamps its rectangles, so they can't be measured. The 11 that could be measured peak near 16 px.

Limits of this evidence: two segments, few side-camera boxes at speed, and none on FRONT_LEFT. A fast segment with traffic on the left would close that gap, and the cut-in scan is a natural way to find one.

Andrew reviewed both contact sheets on 2026-11-02 and reported that the boxes look right. That was an overall judgment, not a per-frame count.

Also found on ICE: `gcloud storage rsync` exits with an error on the component folders, after copying every file, because of a `_metadata` entry in the bucket. Count the Parquet files rather than trusting the exit code. ffmpeg's Linux build needs `-fps_mode passthrough`, or it pads clips to the 1 MHz input rate.
