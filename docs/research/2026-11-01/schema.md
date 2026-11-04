# TOPIC schema: Getting and reading Waymo Open Dataset Perception v2 data (access, bucket layout, sizes, Parquet schema, package, maps, stats, cut-in scanning, code licensing)

## DL-1 [high CRIT] Q1: Current Perception versions and exact GCS bucket paths
CLAIM: As of 2026-11-01 Waymo's download page lists two current Perception releases, both dated March 2024: V 2.0.1 (modular Parquet, no maps) in gs://waymo_open_dataset_v_2_0_1 and V 1.4.3 (with maps) in gs://waymo_open_dataset_v_1_4_3. Nothing newer is listed.
EVID: The page links the Cloud Storage console browser for buckets waymo_open_dataset_v_2_0_1 and waymo_open_dataset_v_1_4_3. Its March 2024 note says the bump fixed small errors in 3D semantic segmentation labels, especially for motorcyclists, so box labels are not the reason for it. The page is current, since it also lists Motion v1.3.1 from October 2025. The English /open/download/ URL redirects to Google sign-in, which I did not follow; this localized URL served the same English text without sign-in. The gs:// spelling is confirmed by community scripts such as https://github.com/willGuimont/torch_waymo.
SRC: https://waymo.com/intl/de/open/download/
IMPACT: Pin v2.0.1 as the component version in provenance config. Touch v1.4.3 only if maps are wanted.
VERDICT: confirmed

## DL-2 [high CRIT] Q1: How access is granted and which Google account
CLAIM: Access is tied to the Google account signed in at https://waymo.com/open/licensing/. gcloud must log in as that same account. Service accounts and compute default accounts do not inherit access. Inference from the endpoint name: accepting the license adds the account to a group that can read the buckets.
EVID: Issue 1014 shows the form posting fields including GoogleAccountEmailAddress and Institution to /open/api/addUserToGroup.json, and shows the storage.objects.list denial for an unregistered account. Waymo's Waymax README says to run gcloud auth login with the email used to apply (https://github.com/waymo-research/waymax). Issue 768 shows a compute service account getting 403 until gcloud init used the registered email. Issue 819 shows a user-made service account getting 403, with a maintainer pointing to the account mismatch. Issue 1014 covers both an institutional Workspace account and a personal Gmail account, and I saw nothing restricting registration to institutional accounts.
SRC: https://github.com/waymo-research/waymo-open-dataset/issues/1014
IMPACT: Andrew must register personally, since accepting terms cannot be done for them. Ask which Google account they will use and whether they have already registered. The PACE login must use that same account.
VERDICT: confirmed

## DL-3 [high CRIT] Q1: How long access takes after license acceptance
CLAIM: The latest maintainer statement (2026-10-20) says access should arrive within a few minutes of registering. A 2020 maintainer statement said within 2 days. The form was broken on 2026-10-19 with LICENSE_TERMS_NOT_ACCEPTED and a maintainer reported a fix on 2026-10-20. Issue 1015, with the same symptom, was filed before the fix and is still open with no reply.
EVID: In issue 1014 a responder wrote that a backend validation problem was identified and resolved and asked users to resubmit. Issue 109 has the older two-day statement from a dataset author (https://github.com/waymo-research/waymo-open-dataset/issues/109). Issue 1015 is at https://github.com/waymo-research/waymo-open-dataset/issues/1015. Issue 540 shows a maintainer advising re-registration when bucket access fails.
SRC: https://github.com/waymo-research/waymo-open-dataset/issues/1014
IMPACT: Andrew should register today so a failure still leaves time before the 2026-11-05 call. All Stage 1 work on real data is blocked until the bucket listing succeeds.
VERDICT: confirmed

## DL-4 [medium CRIT] Q1: Requester-pays or billing project
CLAIM: Inference: no billing project or requester-pays setup is needed. No Waymo source says so outright.
EVID: Waymo's download page mentions no billing. Community recipes for this bucket use only gcloud auth login with no billing flag (https://github.com/willGuimont/torch_waymo, https://github.com/whesense/XSIM). GitHub issue and code searches for requester pays and billing-project on these buckets found nothing. Google's docs say a requester-pays request without a billing project fails with 400 UserProjectMissing, and the flag is --billing-project (https://docs.cloud.google.com/storage/docs/using-requester-pays).
SRC: https://docs.cloud.google.com/storage/docs/requester-pays
IMPACT: Do not create a GCP project or billing account in advance. Add --billing-project only if 400 UserProjectMissing appears on the first listing.
VERDICT: partially_correct
CORRECTED: Inference, strongly supported but not stated by Waymo for billing projects specifically: no billing project or requester-pays setup is needed. A dataset author said in 2020 that a Google Cloud account would cost nothing, and the terms page calls the dataset free of charge. A community full copy of the v2.0.1 bucket set no rclone user_project, which rclone documents as needed only for requester pays. One user reported a download egress charge in 2022 with a billing account attached, setup unknown.
VERIF-EVID: The researcher's search missed issue 106, in both directions. Support: peisun1115 said no cost (2020-03-27) and RalphMao said no project is required. Against: hongvin (2022-09-21) reported an egress charge from gsutil rsync with a billing account attached, and gave no details of the setup. New evidence: waymo-download-job.yaml copies gcs:waymo_open_dataset_v_2_0_1 with a user OAuth token and no user_project (https://rclone.org/googlecloudstorage/). The 400 UserProjectMissing text is on the requester-pays page, and --billing-project is on the using-requester-pays page. The impact holds, and also suggests attaching no billing account at all. I made no bucket request, so this stays indirect.
VERIF-SRC: https://github.com/waymo-research/waymo-open-dataset/issues/106

## DL-5 [high CRIT] Q1: gcloud versus gsutil
CLAIM: Use gcloud storage, not gsutil. Google describes gsutil as legacy and minimally maintained, and says it leaves the Google Cloud CLI package after March 2027.
EVID: Google's gsutil page states this and the transition guide recommends gcloud storage (https://docs.cloud.google.com/storage/docs/gsutil-transition-to-gcloud). The guide adds that gcloud storage does not support parallel invocations from multiple terminals on one machine. gcloud auth login and gcloud auth application-default login both document --no-launch-browser, which prints a URL and accepts a pasted code. The Linux x86_64 tarball bundles its own Python (https://docs.cloud.google.com/sdk/docs/install-sdk). A maintainer warned in issue 819 that the bucket-get operation is not expected to be permitted, so Python clients should avoid client.get_bucket.
SRC: https://docs.cloud.google.com/storage/docs/gsutil
IMPACT: Write all download scripts with gcloud storage. Use the tarball install and --no-launch-browser on headless PACE nodes. Run one gcloud storage job per node at a time. Application-default login is only needed if Python reads gs:// directly.
VERDICT: confirmed

## LAY-1 [high CRIT] Q2: File layout and whether there is one Parquet per segment
CLAIM: Each component has its own folder, and each folder holds one Parquet file per segment named <context_name>.parquet. A segment is fetched by copying the same filename from each component folder.
EVID: The tutorial's folder structure section gives paths of the form <dataset>/<component_tag>/<context_name>.parquet, with example context 10023947602400723454_1120_000_1140_000. Community scripts use gs://waymo_open_dataset_v_2_0_1/training/<component>/<segment>.parquet, for example https://github.com/whesense/XSIM/blob/main/tools/data/download_waymo_scene.sh.
SRC: https://github.com/waymo-research/waymo-open-dataset/blob/master/tutorial/tutorial_v2.ipynb
IMPACT: Stage 1 can download only the components it needs, per segment, with no archive unpacking.
VERDICT: confirmed

## LAY-2 [high CRIT] Q2: Splits and segment counts
CLAIM: v2.0.1 has at least four top-level split folders: training (798 segments), validation (202), testing and testing_location. The last two are disjoint and together form the 150-segment test set. Whether v2.0.1 also has a domain_adaptation folder is unconfirmed.
EVID: The WOD paper gives 798, 202 and 150 scenes (https://ar5iv.labs.arxiv.org/html/1912.04838). In issue 648 a maintainer says testing and testing_location are separate, disjoint, and should be treated as a union. A community notebook lists the four folders (https://github.com/KE7/graid/blob/main/notebooks/download_waymo_open.ipynb). Inference: its sizes imply about 16 segments in testing and 134 in testing_location, which fits the 16 reported in issue 995. Waymo's About page counts 2,030 segments, and the download page's notes add 800 domain adaptation and 80 camera-only test segments. A domain_adaptation path appears only in one unrelated Rust unit test, which is weak evidence.
SRC: https://github.com/waymo-research/waymo-open-dataset/issues/648
IMPACT: The usable labeled pool is 1,000 segments (training plus validation). The first bucket listing settles the folder list.
VERDICT: confirmed

## LAY-3 [high CRIT] Q2: Whether the test split has labels
CLAIM: Test-split labels are not public, so any lidar_box-based scan is limited to the 1,000 training and validation segments. Whether testing/ has an empty lidar_box folder or none at all is unconfirmed for v2.
EVID: The WOD paper reports baselines on a hidden test set of 150 scenes. The panoptic paper says test annotations are not made public and are evaluated through a server (https://arxiv.org/pdf/2206.07704). I found no source describing the v2 testing folder contents.
SRC: https://ar5iv.labs.arxiv.org/html/1912.04838
IMPACT: Exclude testing and testing_location from cut-in mining and from any box-based evaluation.
VERDICT: confirmed

## LAY-4 [high] Q2: How to list context names
CLAIM: I found no official manifest file. Context names come from listing an always-present component, and community scripts list vehicle_pose.
EVID: The egolens download script lists $BUCKET/vehicle_pose/*.parquet and takes each basename. It uses gsutil, but gcloud storage ls with the same pattern is equivalent (inference).
SRC: https://github.com/egolens/egolens/blob/main/download.sh
IMPACT: Generate contexts_training.txt and contexts_validation.txt once and commit them to configs. Expect 798 and 202 lines. The command is in exact_reference.

## LAY-5 [medium] Q2: Component availability varies by segment
CLAIM: Some components are sparse. A user reported camera_segmentation files of about 10 KB with zero rows, and a community loader treats segmentation and keypoint rows as normally absent.
EVID: A comment in issue 638 reports about 20 nonempty camera_segmentation files in validation for v2.0.0. The torch_waymo README describes these components as sparse. The maintainers' note that some camera_synced_box values are missing is in the camera-only tutorial.
SRC: https://github.com/waymo-research/waymo-open-dataset/issues/638
IMPACT: The Stage 1 inspection step should record row counts per component and treat missing or zero-row files as valid. None of the core Stage 1 inputs are in the sparse group.

## SZ-1 [medium CRIT] Q3: Per-segment size by component
CLAIM: Waymo publishes no per-component sizes. Community-measured sizes for one segment are: camera_image ~330 MB, lidar ~170 MB, lidar_box ~1.9 MB, camera_box ~450 KB, vehicle_pose ~30 KB, camera_to_lidar_box_association ~26 KB, stats ~24 KB, camera_calibration ~9 KB, lidar_calibration ~5 KB.
EVID: The Perception Studio (now egolens) memory report has a table of compressed file sizes per component, and its footer says it is based on actual files in its local waymo_data folder, generated 2026-03-13. The repository's Git tree also shows a vehicle_pose fixture of 12,287 bytes for the tutorial segment, though it may be trimmed.
SRC: https://raw.githubusercontent.com/happyhj/waymo-perception-studio/main/data-pipeline-memory.html
IMPACT: Camera and LiDAR are about 99 percent of the bytes. Labels and poses for the whole labeled pool are small enough to fetch before any image download.
VERDICT: confirmed

## SZ-2 [medium CRIT] Q3: Whole-segment and split totals
CLAIM: Community figures put all components of one segment at about 0.6 GB: 589 MB for 17 components on the tutorial segment, and split totals of training 463.4 GiB, validation 119.8 GiB, testing 9.6 GiB and testing_location 79.3 GiB. Estimate by division: 0.58 to 0.59 GiB per segment.
EVID: The torch_waymo README gives about 589 MB raw for one segment. The KE7/graid notebook gives the split sizes (https://github.com/KE7/graid/blob/main/notebooks/download_waymo_open.ipynb). A separate figure of about 2.29 TB is for the whole v2.0.0 bucket, which also holds object assets, so it is not comparable (https://github.com/KushalBKusram/WaymoOpenDatasetToolKit).
SRC: https://github.com/willGuimont/torch_waymo
IMPACT: Estimates from these numbers: one segment of Stage 1 inputs ~0.33 GB; 30 segments of Stage 1 inputs ~10 GB; 30 segments of everything ~18 GB; lidar_box, vehicle_pose and stats for 1,000 segments ~2 GB. All of this fits a small scratch allocation, and the full 672 GiB is not needed.
VERDICT: partially_correct
CORRECTED: All components of one segment are about 0.6 GB: 589 MB for 17 components on the tutorial segment, and about 0.594 GiB per segment on average. The whole v2.0.1 bucket is about 682.6 GiB (733,019,484,365 bytes, 19,618 files), not 672 GiB: training about 474 GiB, validation about 120 GiB, testing about 9.6 GiB, testing_location about 79.3 GiB.
VERIF-EVID: The notebook comments do say 463.4, 119.8, 9.6 and 79.3 GiB, but they sum to 672.1 and disagree with a tallied full copy: 682.62 GiB in total, about 593.78 GiB for training plus validation. That implies training is near 474 GiB, so 463.4 looks about 10 GiB low, possibly a typo. With 474, all four splits give a uniform 0.59 GiB per segment, against 0.58 from 463.4/798. The test-side figures agree (88.9 GiB). The 589 MB and 17 components are confirmed in torch_waymo, and ~2.29 TB is confirmed for the v2.0.0 bucket. The impact estimates survive within rounding.
VERIF-SRC: https://github.com/zhangchl007/azure-lustre-vmss-metrics/blob/main/docs/lustre-pressure-test.md

## SZ-3 [low] Q3: Sizes for projected_lidar_box, lidar_camera_synced_box and lidar_pose
CLAIM: Estimate only, no published figure found: projected_lidar_box and lidar_camera_synced_box ~1 to 2 MB each, and lidar_pose in the tens of MB.
EVID: 589 MB minus the ~503 MB measured leaves about 85 MB across the eight unmeasured components, although the two figures may come from different segments. The two box components have similar row counts to lidar_box but fewer numeric columns. lidar_pose stores a per-pixel range image, so it is likely the largest of the remainder. This is my own arithmetic, not a measurement.
SRC: https://github.com/willGuimont/torch_waymo
IMPACT: Budget only. Replace with real numbers from gcloud storage ls --long --readable-sizes on one segment.

## SC-1 [high CRIT] Q4: Column naming convention
CLAIM: Key columns are named key.<field> and share names across components. Component columns are [<PythonClassName>].<dotted.path>, for example [LiDARBoxComponent].box.center.x, and repeated fields add [*]. The bracketed name is the exact class name, with casing such as LiDAR.
EVID: component.py builds the prefix from the class __name__ and appends [*] for repeated columns. The tutorial states the same convention. Code that reads real files uses identical names: tri3d (https://github.com/CEA-LIST/tri3d/blob/main/tri3d/datasets/waymo.py) and the egolens Waymo recipe.
SRC: https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/v2/component.py
IMPACT: The loader can hard-code column names with no Waymo package. The lists in exact_reference were derived by hand from the dataclasses because no fetched code was executed, and the core names were cross-checked against tri3d and egolens.
VERDICT: confirmed

## SC-2 [high] Q4: Key columns and row granularity
CLAIM: Granularity differs by component. camera_calibration and lidar_calibration have no timestamp key and hold one row per sensor per segment. vehicle_pose and stats hold one row per frame. camera_image holds one row per frame and camera. lidar_box holds one row per frame and object.
EVID: base.py defines SegmentCameraKey, SegmentLaserKey, FrameKey, CameraKey, LaserKey, LaserLabelKey and CameraLabelKey, and each component inherits one of them. The comment on LaserLabelKey explains that laser labels apply to the whole 3D scene, so they have no laser_name.
SRC: https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/v2/perception/base.py
IMPACT: Joins must use the shared key subset. v2.merge does this automatically, and a small pandas helper can replace it. The join table is in exact_reference.

## SC-3 [high CRIT] Q4: camera_to_lidar_box_association coverage
CLAIM: camera_to_lidar_box_association cannot link vehicle tracks. The code comment says it covers only pedestrian camera labels, and Waymo's About page says pedestrians and cyclists. A cut-in vehicle's 3D track must reach images through projected_lidar_box or lidar_camera_synced_box, both keyed by key.laser_object_id.
EVID: The docstring on CameraToLiDARBoxAssociationComponent states the pedestrian limit, and label.proto has the same comment. The About page lists the association for pedestrians and cyclists (https://waymo.com/open/about/). The Perception page says 2D and 3D labels are generated independently, not by projection.
SRC: https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/v2/perception/box.py
IMPACT: Do not use camera_box with the association table for vehicles. Keep the design doc's approach of drawing from 3D boxes, and treat camera_box IDs as a separate ID space from key.laser_object_id.
VERDICT: partially_correct
CORRECTED: camera_to_lidar_box_association cannot link vehicle tracks. The code comment says it covers only pedestrian camera labels, and Waymo's About page says pedestrians and cyclists. A cut-in vehicle's 3D track reaches images through projected_lidar_box or lidar_camera_synced_box, both keyed by key.laser_object_id, or by projecting lidar_box directly with camera_calibration, which is the design doc's own route.
VERIF-EVID: The box.py docstring, the label.proto comment and the About page all exclude vehicles, though they differ on cyclists. The word must is too strong, since direct projection is a third route, as the researcher's own impact note implies. tutorial_camera_only says pre-projected labels always correspond to laser labels, unlike camera labels. compat_v1 _parse_lidar_object_id strips the camera suffix, so projected_lidar_box key.laser_object_id joins to lidar_box, with key.camera_name as an extra key.
VERIF-SRC: https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/v2/perception/box.py

## SC-4 [high] Q4: projected_lidar_box keys
CLAIM: projected_lidar_box keys are key.segment_context_name, key.frame_timestamp_micros, key.camera_name and key.laser_object_id. The camera suffix on the v1 id is stripped, so it joins directly with lidar_box. The class docstring's mention of laser_name does not match the key dataclass.
EVID: ProjectedLaserLabelKey combines LaserLabelKey and CameraKey. compat_v1/box.py slices the first 22 characters as the object id and parses the camera from the rest. The docstring says the box is the smallest axis-aligned rectangle over the projected points, clamped to the image, and that fully outside projections are dropped.
SRC: https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/v2/perception/compat_v1/box.py
IMPACT: Use it as the automated acceptance check: compare the bounding rectangle of projected corners against [ProjectedLiDARBoxComponent].box.* per object and camera.

## SC-5 [high CRIT] Q4: Coordinate frames of lidar_box fields
CLAIM: lidar_box centers and headings are in the vehicle frame of that frame's pose, but [LiDARBoxComponent].speed and .acceleration are in the global frame according to Waymo staff, and are derived from label centers rather than measured. They are not ego-relative.
EVID: label.proto and the Perception page place boxes in the vehicle frame. In issue 39 a dataset author said the values can be assumed to be global and are computed from center sequences. In issue 215 the same author said they are derivatives of label centers in the global frame, with constant velocity assumed between frames (https://github.com/waymo-research/waymo-open-dataset/issues/215). OpenPCDet stores them as speed_global and rotates them with the pose (https://github.com/open-mmlab/OpenPCDet/blob/master/pcdet/datasets/waymo/waymo_utils.py). The v2 docstring does not state the frame.
SRC: https://github.com/waymo-research/waymo-open-dataset/issues/39
IMPACT: SceneWindow must convert to ego-relative velocity explicitly using world_from_vehicle, and take ego velocity from pose differences. Add a unit test comparing speed with finite differences of global centers.
VERDICT: confirmed

## SC-6 [high] Q4: Timestamp semantics
CLAIM: key.frame_timestamp_micros is the frame start time (first top-lidar scan) and is shared by every component of a frame. It does not match the vehicle pose time, which is roughly mid-frame. Each camera image carries its own pose and pose_timestamp.
EVID: dataset.proto comments on Frame.timestamp_micros and Frame.pose say this. compat_v1 extractors set the key from frame.timestamp_micros for images, boxes and stats. Issue 851 confirms timestamps are absolute and span years, and torch_waymo reports 199 frames for the tutorial segment rather than exactly 200.
SRC: https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/dataset.proto
IMPACT: Treat the timestamp as a join key and derive segment-relative seconds as (ts - ts0) / 1e6. Do not assume exactly 200 frames.

## SC-7 [high CRIT] Q4: lidar_camera_synced_box semantics
CLAIM: lidar_camera_synced_box gives, per object, a box shifted to the time its most visible camera captured it. Waymo's camera-only tutorial projects these with a global-shutter model, so aligned overlays need no rolling-shutter code. Boxes exist only for objects visible in at least one camera.
EVID: The tutorial says rolling shutter effects are already factored into camera_synced_box, projects with GLOBAL_SHUTTER metadata using the frame pose, and says cameras span about 230 degrees horizontally. It also says projected_lidar_labels come from rolling-shutter projection of the LiDAR-synced box. Issue 834 has a maintainer stating that all images are lens distorted, so distortion coefficients still apply (https://github.com/waymo-research/waymo-open-dataset/issues/834).
SRC: https://github.com/waymo-research/waymo-open-dataset/blob/master/tutorial/tutorial_camera_only.ipynb
IMPACT: Fetch lidar_camera_synced_box in the first download and project it into most_visible_camera_name. Use lidar_box for other cameras and expect small offsets during fast motion.
VERDICT: confirmed

## PK-1 [high CRIT] Q5: Official package name, version and platforms
CLAIM: The official package is waymo-open-dataset-tf-2-12-0, latest 1.6.7, uploaded 2025-04-29. It ships a single py3-none manylinux_2_24/2_28 x86_64 wheel, sets no Requires-Python, and pins tensorflow==2.13 despite its name, plus numpy==1.23.5, pandas==1.5.3, pyarrow==16.0.0, dask[dataframe]==2023.3.1 and jax==0.4.13.
EVID: PyPI JSON metadata shows the wheel and requires_dist. tutorial_v2 installs this exact version. PyPI has no tf-2-13-0, 2-14-0, 2-15-0, 2-16-0 or 2-17-0 package. The bare name waymo-open-dataset is a 2019 release, 1.0.1. Issue 976 confirms no aarch64 wheel.
SRC: https://pypi.org/pypi/waymo-open-dataset-tf-2-12-0/json
IMPACT: It cannot be installed on Andrew's Mac or on ARM nodes, only on x86_64 Linux. Its old pins also make it a poor fit for a shared environment with modern tools.
VERDICT: confirmed

## PK-2 [high CRIT] Q5: Supported Python versions
CLAIM: Python 3.10 is the only version Waymo's lock file targets. 1.6.7 fails on Python 3.12 because jaxlib==0.4.13 is unavailable. Python 3.11 is unverified: tensorflow 2.13 has cp311 wheels, but a community pin excludes it. A clean 3.10 environment may also need protobuf 3.20.x.
EVID: requirements.txt says it was compiled with Python 3.10 (https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/requirements.txt). Issue 978 reports the 3.12 failure, a 3.10 dependency warning, and a working 3.10 recipe with protobuf 3.20.x. PyPI lists tensorflow 2.13.1 wheels for cp38 to cp311. torch_waymo limits the package to python_version < 3.11 with protobuf==3.20.3.
SRC: https://github.com/waymo-research/waymo-open-dataset/issues/978
IMPACT: If the package is used at all, give it a dedicated Python 3.10 environment, kept separate from the Stage 1 environment.
VERDICT: partially_correct
CORRECTED: Python 3.10 is the only version Waymo's lock file targets. As of 2026-11-01 a plain pip install of 1.6.7 cannot resolve on any Python, because PyPI no longer carries jaxlib==0.4.13 (the oldest is 0.4.18). That wheel survives only in the JAX archive, for cp38 to cp311 on manylinux2014_x86_64, so Python 3.12 is out and the install needs the archive as an extra source. Python 3.11 is unverified: wheels exist for all compiled pins, but torch_waymo excludes it. A clean 3.10 environment also needs protobuf 3.20.x, pinned in the lock file but absent from the wheel metadata.
VERIF-EVID: PyPI's project JSON and simple index list no jaxlib below 0.4.18, and /pypi/jaxlib/0.4.13/json returns 404. In jax-ml/jax issue 34532 (Jan 2026), JAX collaborators confirm that old wheels are deliberately removed for quota reasons, and the JAX install docs say the same. jax_releases.html still lists jaxlib-0.4.13, and a HEAD on the cp310 wheel returned 200. Issue 978's working 3.10 recipe dates from October 2025, before the removal, so it cannot be assumed to work now. I did not run pip; this comes from the index listings. Per-pin check on PyPI: tensorflow 2.13.0, numpy, pandas, pyarrow, scikit-image, scikit-learn, matplotlib and pillow all have cp311 Linux x86_64 wheels. Note tensorflow==2.13 means 2.13.0, not the 2.13.1 cited. This strengthens the conclusion to keep the package out of the Stage 1 environment.
VERIF-SRC: https://github.com/jax-ml/jax/issues/34532

## PK-3 [high CRIT] Q5: Whether importing waymo_open_dataset.v2 needs TensorFlow
CLAIM: Yes. Importing waymo_open_dataset.v2 needs TensorFlow and Dask at import time. Inference: submodules such as v2.component cannot avoid this, because Python runs the package __init__ first.
EVID: v2/__init__.py imports lidar, segmentation, object_asset and lidar_utils, each of which has import tensorflow as tf at module top, and dataframe_utils, which imports dask.dataframe. component.py itself imports only pyarrow, but it sits under that package.
SRC: https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/v2/__init__.py
IMPACT: The design doc's stack entry for Waymo v2 libraries brings in TensorFlow. Recommend removing it from the Stage 1 environment.
VERDICT: confirmed

## PK-4 [high CRIT] Q5: Reading Parquet with plain pyarrow and no Waymo package
CLAIM: The files can be read with plain pyarrow and no Waymo package. This is a stated design goal of the format, and two community loaders do it.
EVID: tutorial_v2 says the format works with any Parquet library and no extra dependencies, not even protocol buffers. tri3d uses pq.read_table with filters and columns on key.camera_name and key.laser_name, and declares no TensorFlow dependency (https://github.com/CEA-LIST/tri3d/blob/main/tri3d/datasets/waymo.py). torch_waymo describes a TensorFlow-free converter built on pq.ParquetFile (https://github.com/willGuimont/torch_waymo/blob/master/src/torch_waymo/converters/parquet.py). v2.merge in dataframe_utils.py is a thin DataFrame.merge wrapper over columns prefixed key.
SRC: https://github.com/waymo-research/waymo-open-dataset/blob/master/tutorial/tutorial_v2.ipynb
IMPACT: Stage 1 needs only pyarrow, pandas, NumPy and OpenCV in a modern Python, which also runs on the Mac for development. Join keys are in exact_reference.
VERDICT: confirmed

## PK-5 [medium] Q5: Known pyarrow version and performance issues
CLAIM: I found no Waymo-specific pyarrow version bug for plain reads. Real issues are dependency-resolver conflicts when installing the official package, slow filtered reads, and row order that differs between components.
EVID: Official pins moved from pyarrow==10.0.0 to 16.0.0 between releases, and issue 623 shows resolver conflicts plus a tutorial need for pyarrow>=10. torch_waymo declares pyarrow>=14, or >=18 on Python 3.13, and tri3d leaves it unpinned. Issue 841 shows about 1.65 s per filtered lidar read on the original files. Issue 656 has a maintainer saying rows in lidar, lidar_pose and lidar_calibration do not follow the same key order. A web search for pyarrow errors on these files found nothing specific.
SRC: https://github.com/waymo-research/waymo-open-dataset/issues/841
IMPACT: Always join on keys, never on row position. Use columns= and filters= when reading. Cache per-segment results, and re-encode with tri3d's optimizer only if random frame access becomes a bottleneck.

## MP-1 [high CRIT] Q6: Whether Perception v2.0.1 includes vectorized maps
CLAIM: Perception v2.0.1 has no map data in any component. Maps are only in the v1.4.x TFRecord releases.
EVID: The Perception page says maps are the one feature not supported in the modular format. The download page labels V 2.0.1 as modular without maps and V 1.4.3 as with maps (https://waymo.com/intl/de/open/download/). LOG.md says v2.0.0 has all v1.4.2 features except maps. TAG_BY_COMPONENT in v2/__init__.py has 23 tags and none is a map.
SRC: https://waymo.com/intl/de/open/data/perception/
IMPACT: Stage 1 data has no lane geometry. Ask Andrew whether lane geometry is needed now or whether the ego-path proxy in SCAN-1 is enough.
VERDICT: partially_correct
CORRECTED: Perception v2.0.1 has no map data in any component. Within Perception, maps exist only in the v1.4.2 and v1.4.3 TFRecord releases; v1.4.0 and v1.4.1 predate them.
VERIF-EVID: The download page says v1.4.2 (March 2023) added 3D map data and that v1.4.3 is the current with-maps release. The Perception page and LOG.md say v2 lacks only maps. TAG_BY_COMPONENT has 23 tags, none a map. Impact correction: no vector lane geometry does not mean no lane information. lidar_segmentation labels include Lane Marker, Road and Curb, for 1,150 segments at 2 Hz per the About and Perception pages. camera_segmentation has Lane Marker and Road Marker on a subset.
VERIF-SRC: https://waymo.com/intl/de/open/download/

## MP-2 [high CRIT] Q6: Map data in v1.4.3
CLAIM: In v1.4.3, Frame.map_features is populated only in the first frame of a segment, and Frame.map_pose_offset is a per-frame offset added after the vehicle-to-world transform for lidar points. Segment files are at gs://waymo_open_dataset_v_1_4_3/individual_files/<split>/segment-<context_name>_with_camera_labels.tfrecord. Estimate: about 1 GB per segment.
EVID: dataset.proto comments state the first-frame rule and describe the offset as compensating for pose drift. tutorial_maps.ipynb adds the offset to world-frame points. map.proto covers lanes, road lines, road edges, stop signs, crosswalks, speed bumps and driveways. The path comes from the torch_waymo README. The size estimate scales the repository's two_frame.tfrecord fixture (9.8 MB for 2 frames) to 199 frames. A maintainer's example in issue 736 aligns label centers to the map with pose alone, so the need for the offset on boxes is unclear.
SRC: https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/dataset.proto
IMPACT: Maps would add a second format, a proto reader and about 1 GB per segment. Inference: gcloud storage cat -r could fetch only the first record, but per-frame offsets need the full file. The design doc's Stage 4 says no legacy map data, so this needs Andrew's decision.
VERDICT: confirmed

## MP-3 [medium CRIT] Q6: Alternative source of lane geometry
CLAIM: OpenLane is a third-party label set on Waymo Perception giving per-frame 3D lane lines over 1,000 segments, with no TFRecord map reading. Inference from its one-camera-per-frame format and its reputation: it is front-camera only.
EVID: The README reports 200K frames and 880K lanes on 1,000 segments. Lane JSON has xyz in camera coordinates, category, left-right attribute and track_id, in folders named segment-xxx (https://github.com/OpenDriveLab/OpenLane/blob/main/anno_criterion/Lane/README.md). Data is CC BY-NC-SA 4.0 plus the Waymo license, code is Apache 2.0. Download goes through a Google form linked from https://github.com/OpenDriveLab/OpenLane/blob/main/data/README.md.
SRC: https://github.com/OpenDriveLab/OpenLane
IMPACT: An option to offer Andrew. They would have to submit the form personally. Check the frame-to-timestamp mapping on the first file.
VERDICT: confirmed

## ST-1 [high] Q7: What the stats component provides
CLAIM: stats has one row per frame with time_of_day, location, weather, and lidar and camera object counts by type. It can filter segments cheaply after de-duplicating by segment. It has no road-type or speed-limit field.
EVID: StatsComponent is a FrameComponent with those fields. Docstrings say time of day is Day, Dawn/Dusk or Night from sun elevation, weather is Sunny or Rain, and location is human readable. TensorFlow Lingvo lists values Day, Dawn/Dusk, Night; rain, rainy, sunny, unknown; location_sf, location_phx, location_kir, location_other (https://github.com/tensorflow/lingvo/blob/master/lingvo/tasks/car/waymo/waymo_open_input_generator.py). The v1 proto comment says counts are unique objects in the segment, while the table is keyed per frame, so the count meaning needs checking.
SRC: https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/v2/perception/context.py
IMPACT: Include stats in the light scan set (about 24 KB per segment). Use it to prefer clear daytime segments for the demo. Inference: ego speed from vehicle_pose can stand in for road type.

## SCAN-1 [medium CRIT] Q8: Whether lidar_box and vehicle_pose alone can scan for cut-ins
CLAIM: Inference: lidar_box plus vehicle_pose are enough for a first-pass cut-in candidate scan of the 1,000 labeled segments, at about 2 MB per segment and about 2 GB in total, before any camera download.
EVID: lidar_box gives per-object 3D boxes in the vehicle frame with type and tracking IDs at 10 Hz, and vehicle_pose gives world_from_vehicle per frame, so lateral offset and gap relative to the ego path can be computed. The Perception page says IDs are globally unique, and that No Label Zones can cover areas such as the far side of a highway. Limits: no lane lines, so curves and ego lane changes will create false positives unless offsets are measured against the ego's own path; occlusion breaks tracks; test segments have no labels.
SRC: https://waymo.com/intl/de/open/data/perception/
IMPACT: Add a scan step to Stage 1 that ranks candidates, then fetch camera_image only for the top few and review them by eye. This fits the time before 2026-11-05 far better than downloading images first.
VERDICT: partially_correct
CORRECTED: Inference: lidar_box plus vehicle_pose are enough for a first-pass cut-in candidate scan of the 1,000 labeled segments before any camera download. About 2 MB per segment and 2 GB in total is a one-segment extrapolation; 2 to 7 GB is the safer range. v2 carries no No Label Zone polygons, so a scan cannot tell a missing track from an unlabeled area.
VERIF-EVID: The approach is sound: boxes, type and IDs at 10 Hz are in lidar_box, world_from_vehicle is in vehicle_pose, and the paper says IDs are consistent across a sequence. No source states feasibility, and no scan has been run. Size: 1.9 MB comes from one segment that the same report says has about 3,000 boxes, while the About page's 12.6M labels over 1,200 segments average about 10,500. If both hold, the mean file is about 3.5 times larger; if the box count is wrong, about 2 MB stands. Either is harmless. NLZ: grep of v2 context.py, compat_v1/context.py and lidar.py finds no NLZ fields, and torch_waymo says v2 lacks the polygons, keeping only a per-pixel range-image flag. Spec examples include the far side of a highway and parking lots, which matters most for the design doc's opposing-direction encroachment case.
VERIF-SRC: https://github.com/willGuimont/torch_waymo

## SCAN-2 [medium CRIT] Q8: Published cut-in lists or scenario mining on Perception versus Motion
CLAIM: I found no published list of cut-in segments for the Perception dataset. Cut-in mining exists on other Waymo datasets: a 2026 Motion-dataset study and WOD-E2E, whose scenario clusters include cut-ins.
EVID: arXiv 2605.01485 uses an eight-criterion detector on WOMD, reports 706 and 3,172 cut-in events, and has no data availability statement. The WOD-E2E paper lists Cut-ins among 11 clusters but does not list LiDAR or 3D boxes as released data (https://arxiv.org/html/2510.26125v1). WaymoQA is built on the E2E dataset, not Perception (https://arxiv.org/html/2511.20022). This is absence of evidence from web and issue searches, not proof.
SRC: https://arxiv.org/html/2605.01485
IMPACT: Plan to mine cut-ins in-house. WOD-E2E cannot feed Stage 1 or Stage 4 because it lacks 3D tracks and LiDAR. The detector thresholds are a usable starting template and are in exact_reference.
VERDICT: confirmed

## SCAN-3 [medium CRIT] Q8: Closest public cut-in signal on Perception data
CLAIM: OpenLane's CIPO labels are the closest public cut-in signal on Perception data. Level 2 marks objects interacting with the ego-lane lines, "typically in the process of cut-in or cut-out" (OpenLane CIPO criterion).
EVID: The criterion defines four importance levels generated from Waymo 2D boxes. Its per-frame JSON has box, level id, trackid unique within a segment, and object type. The data README shows cipo folders by segment for training and validation.
SRC: https://github.com/OpenDriveLab/OpenLane/blob/main/anno_criterion/CIPO/README.md
IMPACT: If Andrew submits the OpenLane form, Level 2 runs can prefilter or validate the rule-based scan. They are camera-box based, so linking to 3D tracks needs overlap matching against projected_lidar_box.
VERDICT: partially_correct
CORRECTED: OpenLane's CIPO Level 2 is the only public label on Perception data I found whose definition names cut-ins: objects interacting with the ego-lane lines, "typically in the process of cut-in or cut-out" (OpenLane CIPO criterion). It is not established as the closest signal, because ROAD-Waymo offers track-level action and location labels on Perception front-camera video and was not considered.
VERIF-EVID: The CIPO definition, four levels, JSON fields and cipo folders are all confirmed. Level 2 mixes cut-ins with cut-outs. ROAD-Waymo is linked from Waymo's download page (bucket waymo_open_dataset_road_plus_plus), with 198K frames from 1,000 videos and 54K tubes. Agents inherit Waymo tracks. Action classes include Move left, Move right and Overtake; location classes include AV lane and outgoing lane. A tube entering the AV lane is a plausible cut-in filter (my inference, untested). It has no explicit cut-in class. Annotations are in a train/val JSON, videos are named Train_00000 and so on, and I found no documented mapping to context names, so linking to 3D tracks needs work there too. See https://arxiv.org/html/2411.01683v2 and https://github.com/salmank255/Road-waymo-dataset.
VERIF-SRC: https://github.com/OpenDriveLab/OpenLane/blob/main/anno_criterion/CIPO/README.md

## SCAN-4 [medium] Q8: Other Perception-based behavior labels
CLAIM: ROAD-Waymo, an official community contribution in bucket waymo_open_dataset_road_plus_plus, adds front-camera agent tubes with action and location labels on 1,000 videos, including MovLft, MovRht and Ovtak. No explicit lane-change or cut-in class was found.
EVID: The paper lists action and location classes and says only the frontal view is annotated. The repository README shows videos named like Train_00000.mp4 and does not explain the mapping to Waymo segments (https://github.com/salmank255/Road-waymo-dataset). The download page lists the bucket.
SRC: https://arxiv.org/html/2411.01683v1
IMPACT: A secondary cross-check only. Do not plan around it unless the video-to-context mapping is found.

## SCAN-5 [medium] Q8: Precedent for trajectories from Perception labels
CLAIM: Hu et al. (Transportation Research Part C, 2022) extracted object and AV trajectories from the Perception dataset and released them, but the release covers car-following pairs, with no cut-in list described.
EVID: The Mendeley record describes tabular object data, top-view videos, paired car-following trajectories and processing code, under CC BY 4.0. It gives no Waymo version.
SRC: https://data.mendeley.com/datasets/wfn2c3437n/2
IMPACT: Shows the label-only approach works in practice. Its processing code may be a useful reference for the scan.

## LIC-1 [high CRIT] Q9: What the wdl_limited directories are
CLAIM: src/waymo_open_dataset/wdl_limited has three subfolders, camera, camera_segmentation and sim_agents_metrics. They are licensed apart from the Apache-2.0 repository, under BSD 3-Clause plus a limited patent grant for a stated use with the Waymo dataset under the non-commercial license.
EVID: The README states the split, and each subfolder has its own LICENSE and PATENTS file. The camera grant covers use of the Works to process Waymo Open Dataset data as authorized by the non-commercial license. py_camera_model_ops, which provides world_to_image, lives under wdl_limited/camera/ops.
SRC: https://github.com/waymo-research/waymo-open-dataset/blob/master/README.md
IMPACT: Only the camera folder matters for Stage 1, and only if Waymo's projection ops are used.
VERDICT: confirmed

## LIC-2 [medium CRIT] Q9: Whether our own NumPy projection avoids those terms
CLAIM: Inference, not legal advice: a NumPy projection written from the public docs is not the covered Works, so the BSD conditions and the limited patent grant do not attach to it. It also receives no patent grant. A plain global-shutter pinhole projection does not perform the iterative time estimation that the listed software patent claims.
EVID: PATENTS defines the Works as Waymo's distributed code and excludes claims infringed only through modification. It lists US 10,939,057, whose claim 1 concerns a rotating LIDAR with rolling-shutter cameras timed to it, and application 17/105,141, now US 11,977,167 (https://patents.google.com/patent/US11977167). Independent claims 1, 10 and 14 of that patent recite iteratively estimating sensing time and also operating the autonomous vehicle to navigate.
SRC: https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/wdl_limited/camera/PATENTS
IMPACT: Write the projection from the Perception page and proto comments. Do not port camera_model.cc, since a port keeps the BSD notice duties and modified code is outside the grant. Pair it with lidar_camera_synced_box so no rolling-shutter solver is needed. If that solver is ever wanted, use Waymo's op in the separate environment or ask counsel.
VERDICT: partially_correct
CORRECTED: Inference, not legal advice: a NumPy projection written from the public docs is not the covered Works, so the BSD conditions and the limited patent grant do not attach to it, and it receives no patent grant. US 11,977,167 requires iterative sensing-time estimation and operating the vehicle, which a global-shutter pinhole projection in a research pipeline does not do. That reasoning does not cover US 10,939,057, whose claims 1 and 14 are method and medium claims with no timing solver: receive rotating-LIDAR data, receive image data from rolling-shutter cameras coordinated with that rotation, and combine them.
VERIF-EVID: I read the claims text on Google Patents directly. For '167 (application 17/105,141), independent claims 1, 10 and 14 match the researcher's description. For '057, claim 1 ends with simply combining the image and LIDAR data; claim 14 is the medium version and claim 19 the system. So treating '167 as the only software patent is wrong, and avoiding the solver says nothing about '057. Drawing label boxes is not obviously combining LIDAR data with images, but Stage 4's LiDAR-visual PINGS fusion is much closer, and the grant covers only Waymo's Works. PATENTS' exclusion for claims infringed only through modification is confirmed. Whether WOD capture meets the coordinated-rolling-shutter limitation is a question for counsel.
VERIF-SRC: https://patents.google.com/patent/US10939057B2/en

## LIC-3 [high CRIT] Q1 and Q9: Dataset license is independent of the code used
CLAIM: The data stays under the Waymo Dataset License Agreement for Non-Commercial Use (March 2025) whichever code reads it. If the institution named at registration exercises the rights through the download, the registrant represents they are authorized to accept for it, or that an authorized person has registered.
EVID: The terms page carries the March 2025 title and the institution paragraph. The repository README's metadata still cites the August 2019 version, so it is out of date. The registration form asks for an institution, per issue 1014.
SRC: https://waymo.com/open/terms/
IMPACT: Ask Andrew what they will enter as institution, and whether this is personal research or work under Georgia Tech. Cite the March 2025 text in provenance notes.
VERDICT: confirmed

# VERIFIER MISSED
- The official package no longer installs from PyPI on any Python: it pins jaxlib==0.4.13, and PyPI's oldest jaxlib is now 0.4.18 after deliberate deletions (https://github.com/jax-ml/jax/issues/34532). Any fallback environment for Waymo's camera ops or v1 tools needs the JAX archive (https://storage.googleapis.com/jax-releases/jax_releases.html) with a Python no newer than 3.11, and is untested. This also affects the LIC-2 and MP-2 fallbacks.
- License distribution limits matter for Stage 2 and 3. Terms Section 2 allows small extracts in research publications and distribution only to people who registered and accepted the terms. Neither clearly covers uploading frames or clips to the Claude API or Supabase. Section 1 also requires a set attribution in any Derivative IP, a term that covers works made using the dataset. Decide this before the first image leaves PACE. This is my reading, not legal advice (https://waymo.com/open/terms/).
- ROAD-Waymo was overlooked. It is on Waymo's own download page and adds track-level action and location labels to 1,000 Perception front-camera videos, including Move left, Move right, Overtake and AV lane. It may cut the in-house mining effort, needs no Google form, and should be checked before building a scanner from scratch. Mapping its Train_00000 video names to context names is unresolved.
- US 10,939,057 claim 1 is a data-processing method ending in combining LIDAR and image data, so writing a clean-room projection does not address it, and Stage 4's PINGS fusion sits closer to it than box overlays do. Worth raising with Georgia Tech counsel if the work will be published or demoed. Not legal advice.
- v2.0.1 has point-level lane hints without v1.4.3: lidar_segmentation includes Lane Marker, Road and Curb, for 1,150 segments at 2 Hz. Using it means fetching lidar too and decoding range images, so it is a third option alongside v1.4.3 maps and OpenLane, not a free one. No one has evaluated its quality for lane estimation.
- v2 omits No Label Zone polygons, which v1 stores in the global frame; only a per-pixel flag remains in the range image. A scan on lidar_box cannot exclude unlabeled areas such as the far side of a highway, which matters for the taxonomy's opposing-direction encroachment case.
- A community full bucket copy gives better numbers than the sources cited: 682.62 GiB, 19,618 files, exactly 17 x 1,154, and four splits. This suggests all 17 components exist in all splits, so test folders likely hold lidar_box files, and scan code must check row counts rather than file existence. The 4 extra files per component are unidentified. Verify on the first listing (https://github.com/zhangchl007/azure-lustre-vmss-metrics/blob/main/docs/lustre-pressure-test.md).
- PACE ICE limits are unverified. docs.pace.gatech.edu did not resolve from here and the ACM page returned 403, so this comes from search summaries: 15 GB home and 100 GB scratch per user, no scratch backups, and no confirmed purge rule for ICE, while other PACE clusters delete after 60 days (https://dl.acm.org/doi/fullHtml/10.1145/3624062.3624131). The 30-segment plan fits, but full camera_image for 1,000 segments (about 330 GB) does not. Run pace-quota on first login, and keep the data in a private directory, since the license limits sharing.
- For a hand-written projection, Waymo's camera frame is x forward, y left, z up (Perception page), not OpenCV's. StandardE2E converts it explicitly. Skipping this yields wrong overlays even with correct calibration. Confirm exact_reference covers it.
- key.frame_timestamp_micros is the join key, not a per-camera capture time. Per-camera time and pose are in [CameraImageComponent].pose_timestamp and .pose, and tri3d uses these for its camera timeline. FrameManifest should keep both, or Stage 2 event timing and the timestamp-join tests can blur the two.

# UNRESOLVED
- Whether the v2.0.1 bucket is requester-pays (inferred no). -> After registering, run gcloud storage ls gs://waymo_open_dataset_v_2_0_1/. A 400 UserProjectMissing error means --billing-project is required.
- Whether registration works today; issue 1015 is still open, though it predates the reported fix. -> Andrew submits https://waymo.com/open/licensing/ and runs the listing. If it fails, comment on issue 1014 or 1015.
- Exact top-level folders in v2.0.1, including whether domain_adaptation exists. -> gcloud storage ls gs://waymo_open_dataset_v_2_0_1/
- Whether testing/ and testing_location/ contain lidar_box, empty or otherwise. -> gcloud storage ls gs://waymo_open_dataset_v_2_0_1/testing/ and the same for testing_location/.
- True per-component sizes, especially projected_lidar_box, lidar_camera_synced_box and lidar_pose. -> gcloud storage ls --long --readable-sizes "gs://waymo_open_dataset_v_2_0_1/training/*/10023947602400723454_1120_000_1140_000.parquet"
- Column lists were derived by hand from dataclasses, not read from a real file. tri3d's re-encoder also lists a column named index, so an extra index column may exist. -> Run pyarrow.parquet.read_schema on one file per component and diff it against exact_reference.
- Global-frame speed rests on maintainer comments from 2019 and 2020, not on v2 docs. -> On one segment, compare [LiDARBoxComponent].speed with finite differences of centers transformed by world_from_vehicle.
- Whether stats object counts are per frame or per segment. -> Read one stats file and check whether counts vary across frames within the segment.
- Whether 3D boxes need map_pose_offset to align with v1.4.3 maps; the maps tutorial adds it for lidar points, while a maintainer's box example in issue 736 uses pose alone. -> Only if maps are adopted: plot boxes against lanes on one segment with and without the offset.
- How many genuine cut-ins exist in the 1,000 labeled segments. -> Run the scan, rank candidates, and review the top candidates by eye. No published count was found.
- OpenLane frame filename to key.frame_timestamp_micros mapping, the Waymo version it used, and whether it is strictly front camera. -> Inspect one segment folder after Andrew obtains access through the OpenLane form.
- ROAD-Waymo video name to Waymo context name mapping. -> Inspect the db entries in the ROAD-Waymo annotation JSON, or ask its maintainers.

# EXACT REFERENCE
BUCKETS
gs://waymo_open_dataset_v_2_0_1/   Perception v2.0.1, Parquet, no maps
gs://waymo_open_dataset_v_1_4_3/   Perception v1.4.3, TFRecord, with maps
v2 path:  gs://waymo_open_dataset_v_2_0_1/<split>/<component_tag>/<context_name>.parquet
v1 path:  gs://waymo_open_dataset_v_1_4_3/individual_files/<split>/segment-<context_name>_with_camera_labels.tfrecord
splits:   training (798), validation (202), testing + testing_location (150 together, no public labels)
example context_name: 10023947602400723454_1120_000_1140_000 (training; tutorial segment)
registration: https://waymo.com/open/licensing/ (Andrew only)

GCLOUD SETUP (no root, headless)
curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz
tar -xf google-cloud-cli-linux-x86_64.tar.gz
./google-cloud-sdk/install.sh
gcloud auth login --no-launch-browser          # must be the Google account registered with Waymo
gcloud auth list
# Only if Python reads gs:// directly (gcsfs, google-cloud-storage):
gcloud auth application-default login --no-launch-browser

ACCESS CHECK AND CONTEXT LISTS
B=gs://waymo_open_dataset_v_2_0_1
gcloud storage ls $B/
gcloud storage ls $B/training/
gcloud storage ls "$B/training/vehicle_pose/*.parquet"   | xargs -n1 basename | sed 's/\.parquet$//' > contexts_training.txt
gcloud storage ls "$B/validation/vehicle_pose/*.parquet" | xargs -n1 basename | sed 's/\.parquet$//' > contexts_validation.txt
wc -l contexts_*.txt      # expect 798 and 202

SIZES WITHOUT DOWNLOADING
SEG=10023947602400723454_1120_000_1140_000
gcloud storage ls --long --readable-sizes "$B/training/*/$SEG.parquet"
gcloud storage du --summarize --readable-sizes $B/training/lidar_box $B/training/vehicle_pose $B/training/stats

ONE SEGMENT, STAGE 1 INPUTS
DATA=<scratch path>/waymo_v2_0_1
for c in camera_image camera_calibration lidar_box vehicle_pose stats projected_lidar_box lidar_camera_synced_box camera_box; do
  mkdir -p "$DATA/training/$c"
  gcloud storage cp "$B/training/$c/$SEG.parquet" "$DATA/training/$c/"
done
# Stage 4 additions: lidar lidar_calibration lidar_pose

LIGHT SCAN SET, ALL LABELED SEGMENTS (about 2 GB, estimate)
for s in training validation; do for c in lidar_box vehicle_pose stats; do
  mkdir -p "$DATA/$s/$c"
  gcloud storage rsync --recursive "$B/$s/$c" "$DATA/$s/$c"
done; done

ONLY IF 400 UserProjectMissing APPEARS
gcloud storage cp "$B/training/vehicle_pose/$SEG.parquet" . --billing-project=<PROJECT_ID>

OPTIONAL SEPARATE ENV FOR WAYMO OPS (x86_64 Linux, Python 3.10)
conda create -n waymo310 python=3.10 pip
python3 -m pip install waymo-open-dataset-tf-2-12-0==1.6.7
python3 -m pip install "protobuf==3.20.*"     # per issue 978, if needed

COMPONENT TAG -> CLASS (bracket prefix)
camera_image -> CameraImageComponent
camera_calibration -> CameraCalibrationComponent
lidar_box -> LiDARBoxComponent
vehicle_pose -> VehiclePoseComponent
projected_lidar_box -> ProjectedLiDARBoxComponent
lidar_camera_synced_box -> LiDARCameraSyncedBoxComponent
camera_to_lidar_box_association -> CameraToLiDARBoxAssociationComponent
camera_box -> CameraBoxComponent
stats -> StatsComponent
lidar -> LiDARComponent
lidar_calibration -> LiDARCalibrationComponent
lidar_pose -> LiDARPoseComponent
Others in TAG_BY_COMPONENT: camera_hkp, camera_segmentation, lidar_camera_projection, lidar_hkp, lidar_segmentation, and object_asset_* (6 tags).

COLUMNS (derived from dataclasses in src/waymo_open_dataset/v2; confirm with pq.read_schema)
camera_image
  string key.segment_context_name
  int64  key.frame_timestamp_micros
  int8   key.camera_name
  binary [CameraImageComponent].image
  fixed_size_list<double>[16] [CameraImageComponent].pose.transform
  float  [CameraImageComponent].velocity.linear_velocity.x / .y / .z
  double [CameraImageComponent].velocity.angular_velocity.x / .y / .z
  double [CameraImageComponent].pose_timestamp
  double [CameraImageComponent].rolling_shutter_params.shutter
  double [CameraImageComponent].rolling_shutter_params.camera_trigger_time
  double [CameraImageComponent].rolling_shutter_params.camera_readout_done_time
camera_calibration (one row per camera per segment; no timestamp key)
  string key.segment_context_name
  int8   key.camera_name
  double [CameraCalibrationComponent].intrinsic.f_u / .f_v / .c_u / .c_v / .k1 / .k2 / .p1 / .p2 / .k3
  fixed_size_list<double>[16] [CameraCalibrationComponent].extrinsic.transform
  int32  [CameraCalibrationComponent].width
  int32  [CameraCalibrationComponent].height
  int8   [CameraCalibrationComponent].rolling_shutter_direction
lidar_box
  string key.segment_context_name
  int64  key.frame_timestamp_micros
  string key.laser_object_id
  double [LiDARBoxComponent].box.center.x / .y / .z
  double [LiDARBoxComponent].box.size.x / .y / .z      (x length, y width, z height)
  double [LiDARBoxComponent].box.heading
  int8   [LiDARBoxComponent].type
  int64  [LiDARBoxComponent].num_lidar_points_in_box
  int64  [LiDARBoxComponent].num_top_lidar_points_in_box
  double [LiDARBoxComponent].speed.x / .y / .z         (optional)
  double [LiDARBoxComponent].acceleration.x / .y / .z  (optional)
  int8   [LiDARBoxComponent].difficulty_level.detection / .tracking (optional)
vehicle_pose
  string key.segment_context_name
  int64  key.frame_timestamp_micros
  fixed_size_list<double>[16] [VehiclePoseComponent].world_from_vehicle.transform
projected_lidar_box
  string key.segment_context_name
  int64  key.frame_timestamp_micros
  int8   key.camera_name
  string key.laser_object_id
  double [ProjectedLiDARBoxComponent].box.center.x / .y
  double [ProjectedLiDARBoxComponent].box.size.x / .y
  int8   [ProjectedLiDARBoxComponent].type
lidar_camera_synced_box
  string key.segment_context_name
  int64  key.frame_timestamp_micros
  string key.laser_object_id
  int8   [LiDARCameraSyncedBoxComponent].most_visible_camera_name
  double [LiDARCameraSyncedBoxComponent].camera_synced_box.center.x / .y / .z
  double [LiDARCameraSyncedBoxComponent].camera_synced_box.size.x / .y / .z
  double [LiDARCameraSyncedBoxComponent].camera_synced_box.heading
camera_to_lidar_box_association (key columns only; not usable for vehicles)
  string key.segment_context_name
  int64  key.frame_timestamp_micros
  int8   key.camera_name
  string key.camera_object_id
  string key.laser_object_id
camera_box
  string key.segment_context_name
  int64  key.frame_timestamp_micros
  int8   key.camera_name
  string key.camera_object_id
  double [CameraBoxComponent].box.center.x / .y
  double [CameraBoxComponent].box.size.x / .y
  int8   [CameraBoxComponent].type
  int8   [CameraBoxComponent].difficulty_level.detection / .tracking (optional)
stats
  string key.segment_context_name
  int64  key.frame_timestamp_micros
  string [StatsComponent].time_of_day
  string [StatsComponent].location
  string [StatsComponent].weather
  list<int8>  [StatsComponent].lidar_object_counts.types
  list<int32> [StatsComponent].lidar_object_counts.counts
  list<int8>  [StatsComponent].camera_object_counts.types
  list<int32> [StatsComponent].camera_object_counts.counts
lidar
  string key.segment_context_name
  int64  key.frame_timestamp_micros
  int8   key.laser_name
  list<float> [LiDARComponent].range_image_return1.values
  fixed_size_list<int32>[3] [LiDARComponent].range_image_return1.shape
  list<float> [LiDARComponent].range_image_return2.values
  fixed_size_list<int32>[3] [LiDARComponent].range_image_return2.shape
lidar_calibration (one row per laser per segment; no timestamp key)
  string key.segment_context_name
  int8   key.laser_name
  fixed_size_list<double>[16] [LiDARCalibrationComponent].extrinsic.transform
  double [LiDARCalibrationComponent].beam_inclination.min / .max
  list<double> [LiDARCalibrationComponent].beam_inclination.values (optional)
lidar_pose
  string key.segment_context_name
  int64  key.frame_timestamp_micros
  int8   key.laser_name
  list<float> [LiDARPoseComponent].range_image_return1.values
  fixed_size_list<int32>[3] [LiDARPoseComponent].range_image_return1.shape

ENUMS
key.camera_name: 0 UNKNOWN, 1 FRONT, 2 FRONT_LEFT, 3 FRONT_RIGHT, 4 SIDE_LEFT, 5 SIDE_RIGHT
key.laser_name:  0 UNKNOWN, 1 TOP, 2 FRONT, 3 SIDE_LEFT, 4 SIDE_RIGHT, 5 REAR
box type:        0 UNKNOWN, 1 VEHICLE, 2 PEDESTRIAN, 3 SIGN, 4 CYCLIST
LiDARCameraSyncedBox most_visible_camera_name uses the camera enum.

JOIN KEYS
lidar_box + vehicle_pose: key.segment_context_name, key.frame_timestamp_micros
lidar_box + stats: key.segment_context_name, key.frame_timestamp_micros
lidar_box + lidar_camera_synced_box: key.segment_context_name, key.frame_timestamp_micros, key.laser_object_id
lidar_box + projected_lidar_box: same three (projected has one row per camera)
camera_image + projected_lidar_box: key.segment_context_name, key.frame_timestamp_micros, key.camera_name
camera_image + camera_box: same three
camera_image + camera_calibration: key.segment_context_name, key.camera_name
camera_box + camera_to_lidar_box_association: key.segment_context_name, key.frame_timestamp_micros, key.camera_name, key.camera_object_id
lidar + lidar_pose: key.segment_context_name, key.frame_timestamp_micros, key.laser_name
lidar + lidar_calibration: key.segment_context_name, key.laser_name
Always join on keys. Row order differs across components (issue 656).

FRAMES AND UNITS
Transforms are 4x4 row-major: np.asarray(v).reshape(4, 4)
[VehiclePoseComponent].world_from_vehicle.transform: vehicle to global (global is ENU, origin at vehicle start)
[CameraCalibrationComponent].extrinsic.transform: camera to vehicle
Vehicle frame: x forward, y left, z up. Camera frame: x down the lens barrel, z up, right-handed.
Box center and heading: vehicle frame. Speed and acceleration: global frame (maintainer statements).
[CameraImageComponent].velocity: ego velocity in the global frame at pose_timestamp (seconds).
Images are lens distorted (maintainer, issue 834).

PLAIN PYARROW READ (not run against real data)
import numpy as np, pyarrow.parquet as pq
df = pq.read_table(path, columns=["key.frame_timestamp_micros", "key.camera_name", "[CameraImageComponent].image"], filters=[("key.camera_name", "=", 1)]).to_pandas()
T = np.stack(pose_df["[VehiclePoseComponent].world_from_vehicle.transform"]).reshape(-1, 4, 4)

CUT-IN DETECTOR TEMPLATE (arXiv 2605.01485, built for WOMD, thresholds to be retuned)
Cutting vehicle speed at least 2 m/s through the maneuver; lateral position crosses from adjacent lane into target lane; front bumper within 25 m of target at entry; lane-change duration 0.5 to 6.0 s; centroid ahead of target at completion; peak lateral speed at least 0.3 m/s; not stationary; within 1.5 m of lane center after the maneuver. AV-centric frame aligned to heading at the entry frame. For Perception without lanes, substitute lateral offset from the ego path built from world_from_vehicle (my proposal, untested).