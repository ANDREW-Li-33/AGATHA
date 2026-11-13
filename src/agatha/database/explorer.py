"""Streamlit views. Database and Storage adapters are passed in for offline UI tests."""

from __future__ import annotations

from ..contracts import SOURCE_DATASET_ATTRIBUTION
from ..vlm.labels import HumanLabels, agreement, summary
from ..vlm.schema import CATEGORIES, TAXONOMY, IncidentReport
from .search import SearchFilters


def title(row: dict) -> str:
    labels = row["subcategories"] or [a["label"] for a in row["raw_report"]["label_assessments"]]
    return ", ".join(label.replace("_", " ").capitalize() for label in labels) or "Driving incident"


def render_details(st, repository, storage, incident_id: str) -> None:
    detail = repository.detail(incident_id)
    st.header("Incident Details")
    videos = [asset for asset in detail["assets"] if asset["kind"] == "source_video"]
    if videos and storage is not None:
        try:
            st.video(storage.signed_url(videos[0]))
            meta = videos[0]["metadata"]
            st.caption(f"Plain FRONT video · scene {meta['start_s']:.2f}–{meta['end_s']:.2f} s · "
                       f"{meta['frame_count']} frames · largest gap {meta['max_gap_s']:.2f} s")
        except Exception:
            st.warning("The private video could not be loaded. Check Storage access and refresh the video link.")
        st.button("Refresh video link", key=f"video-{incident_id}")  # rerun creates a fresh five-minute URL
    else:
        st.info("Plain source video is not available yet. Import this scene's clean clips to enable playback.")
    st.subheader(title(detail))
    st.text(" · ".join(f"{label['label']} ({label['status'].lower()})" for label in detail["labels"]))
    st.subheader("VLM reasoning")
    st.text(detail["vlm_reasoning"])
    st.text(f"Driving impact: {detail['driving_impact']}")
    st.caption(f"{detail['start_s']:.2f}–{detail['end_s']:.2f} s · {detail['duration_s']:.2f} s · "
               f"{detail['verification_status']}")
    st.text(f"Source: {detail['context_name']} / {detail['split']} / {detail['dataset_version']}")
    st.text(f"Model: {detail['model_id']}\nPrompt: {detail['prompt_version']}\n"
            f"Pipeline: {detail['pipeline_version']}\nPreprocessing: {detail['preprocessing_version']}")
    if detail["duplicate_of"]:
        st.caption(f"Duplicate of incident {detail['duplicate_of']}")
    for reason in detail["raw_report"]["uncertainty_reasons"]:
        st.text(f"Uncertainty: {reason}")
    if st.button("Run Simulation", key=f"simulate-{incident_id}"):
        job = repository.enqueue(incident_id)
        st.success(f"Simulation {job['id']}: {job['status']}")
        detail = repository.detail(incident_id)
    st.caption("Stage 4 is not connected yet. New jobs remain QUEUED.")
    st.button("Refresh job status", key=f"jobs-{incident_id}")
    if detail["jobs"]:
        st.dataframe([{k: str(j[k]) if k == "id" else j[k]
                       for k in ("id", "status", "start_s", "end_s", "error", "output_clip_path")}
                      for j in detail["jobs"]], hide_index=True)
    with st.expander("Evidence and original report"):
        st.json({"labels": detail["labels"], "frames and observations": detail["evidence"],
                 "objects": detail["objects"], "report": detail["raw_report"]}, expanded=False)


def render_comparison(st, repository, segment_id: str) -> None:
    data = repository.comparison(segment_id)
    humans = data["human_labels"]
    if not humans:
        st.info("Human labels have not been imported for this scene.")
    for human in humans:
        st.subheader(human["scene"])
        labels = HumanLabels.model_validate(human["raw_labels"])
        if not labels.reviewed:
            st.info("Not labeled yet. The imported file is an unfinished review.")
            continue
        st.text("Human: no incident" if labels.no_incident else "Human annotations")
        if labels.incidents:
            st.dataframe([i.model_dump() for i in labels.incidents], hide_index=True)
        for run in data["runs"]:
            st.write(" / ".join(run["model_ids"]) or "Model not recorded")
            st.caption(f"Run {run['id']}")
            reports = [IncidentReport.model_validate(raw) for raw in run["reports"]]
            unassessed = [c for c in run["archive"]["candidates"]["merged"]
                          if c["outcome"] not in ("CONFIRMED", "UNCERTAIN", "REJECTED")]
            if not run["complete"]:
                st.warning("Stage 2 did not assess the whole scene. Missing results are not negative findings.")
            result = agreement(labels, reports, run["complete"], unassessed)
            st.text(summary(result))
            st.json(result, expanded=False)
    with st.expander("Model reports, raw responses and validation errors"):
        for run in data["runs"]:
            st.caption(f"Run {run['id']} · {' / '.join(run['model_ids'])}")
            st.json({"reports": run["reports"], "audit": run["archive"]}, expanded=False)
    st.caption("Agreement uses the review pages' overlapping-time/shared-label rule. It is not M3 precision or recall.")


def render_app(repository, storage=None) -> None:
    import streamlit as st

    st.title("AGATHA")
    st.caption("Driving incidents · Waymo Open Dataset")
    explorer, comparison = st.tabs(["Scenario Explorer", "VLM vs human"])
    with st.sidebar:
        st.header("Find incidents")
        categories = st.multiselect("Category", CATEGORIES)
        labels = st.multiselect("Subcategories", list(TAXONOMY))
        label_mode = st.radio("Match subcategories", ["any", "all"], horizontal=True)
        object_classes = st.multiselect("Object class", ["vehicle", "pedestrian", "cyclist", "sign", "unknown"])
        statuses = []
        if st.checkbox("Confirmed", value=True):
            statuses.append("CONFIRMED")
        if st.checkbox("Include uncertain"):
            statuses.append("UNCERTAIN")
        if st.checkbox("Include rejected"):
            statuses.append("REJECTED")
        use_duration = st.checkbox("Filter duration")
        minimum = st.number_input("Minimum seconds", min_value=0.0, value=0.0, disabled=not use_duration)
        maximum = st.number_input("Maximum seconds", min_value=0.0, value=20.0, disabled=not use_duration)
        combine = st.radio("Combine category, labels, class and duration", ["AND", "OR"], horizontal=True)
        st.caption("Status choices always apply, including with OR.")
        duplicates = st.checkbox("Include duplicate reports")
    with explorer:
        if use_duration and minimum > maximum:
            st.error("Minimum duration must not exceed maximum duration.")
        else:
            filters = SearchFilters(statuses=statuses, categories=categories, labels=labels, label_mode=label_mode,
                                    object_classes=object_classes, combine=combine, include_duplicates=duplicates,
                                    min_duration_s=minimum if use_duration else None,
                                    max_duration_s=maximum if use_duration else None)
            page = st.number_input("Page", min_value=1, value=1, step=1)
            rows, total = repository.search(filters, limit=25, offset=(page - 1) * 25)
            st.caption(f"{total} matching incidents · page {page} of {max(1, (total + 24) // 25)}")
            if rows:
                st.dataframe([{"Incident": title(r), "Scene": r["context_name"], "Model": r["model_id"],
                               "Status": r["verification_status"], "Start (s)": r["start_s"],
                               "Duration (s)": r["duration_s"]} for r in rows], hide_index=True)
                by_id = {str(r["id"]): r for r in rows}
                selected = st.selectbox("Open incident", list(by_id), format_func=lambda k, rows=by_id:
                                        f"{rows[k]['context_name']} · {rows[k]['source_incident_id']} · "
                                        f"{rows[k]['model_id']} · {k[:8]}")
                render_details(st, repository, storage, selected)
            else:
                st.info("No incidents on this page. Adjust the filters or return to page 1.")
    with comparison:
        scenes = repository.scenes()
        if scenes:
            by_id = {str(s["id"]): s for s in scenes}
            selected = st.selectbox("Review scene", list(by_id), format_func=lambda k:
                                    f"{by_id[k]['context_name']} · {by_id[k]['preprocessing_version']}")
            render_comparison(st, repository, selected)
        else:
            st.info("No scenes have been imported yet.")
    st.caption(SOURCE_DATASET_ATTRIBUTION)
