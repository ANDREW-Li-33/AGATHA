"""Ground-truth labeling UI for the human-tier labels (construction_zone, road_change, emergency_vehicle).

    streamlit run tools/label_app.py -- --db data/miner.duckdb

Close it before running `miner` commands that write: DuckDB allows one writer per file.
"""

from __future__ import annotations

import argparse
import sys

import duckdb
import streamlit as st

from scenario_miner.render import render_overlay
from scenario_miner.sample import load_sample
from scenario_miner.taxonomy import load_taxonomy


def _args():
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="data/miner.duckdb")
    p.add_argument("--splits", default="eval,fewshot", help="comma-separated splits to label")
    return p.parse_args(sys.argv[1:])  # streamlit forwards everything after `--`


ARGS = _args()
TAXONOMY = load_taxonomy()
HUMAN = [l for l in TAXONOMY if l.gt == "human"]
AUTO = [l.name for l in TAXONOMY if l.gt != "human"]

st.set_page_config(page_title="Scenario labeler", layout="wide")


def query(sql: str, params=(), write: bool = False):
    # Short-lived connections, so the database isn't locked between clicks.
    with duckdb.connect(ARGS.db, read_only=not write) as con:
        return con.execute(sql, list(params)).fetchall()


splits = [s.strip() for s in ARGS.splits.split(",")]
queue = query(
    f"""
    SELECT s.sample_id, s.sample_dir, sp.split,
           count(g.label) FILTER (WHERE g.source = 'human') AS n_human
    FROM samples s JOIN splits sp USING (sample_id)
    LEFT JOIN ground_truth g USING (sample_id)
    WHERE sp.split IN ({', '.join('?' * len(splits))})
    GROUP BY ALL ORDER BY sp.split DESC, s.sample_id
    """,
    splits,
)
if not queue:
    st.error("Nothing to label. Run `miner ingest` and `miner split` first.")
    st.stop()

done = sum(1 for r in queue if r[3] >= len(HUMAN))
st.progress(done / len(queue), text=f"{done} / {len(queue)} frames have human labels")

if "idx" not in st.session_state:
    st.session_state.idx = next((i for i, r in enumerate(queue) if r[3] < len(HUMAN)), 0)


def go(delta: int = 0, next_unlabeled: bool = False):
    if next_unlabeled:
        start = st.session_state.idx + 1
        order = list(range(start, len(queue))) + list(range(0, start))
        st.session_state.idx = next((i for i in order if queue[i][3] < len(HUMAN)), st.session_state.idx)
    else:
        st.session_state.idx = (st.session_state.idx + delta) % len(queue)


nav = st.columns([1, 1, 2, 6])
nav[0].button("◀ Prev", on_click=go, args=(-1,))
nav[1].button("Next ▶", on_click=go, args=(1,))
nav[2].button("Next unlabeled ⏭", on_click=go, kwargs={"next_unlabeled": True})

sample_id, sample_dir, split, _ = queue[st.session_state.idx]
sample = load_sample(sample_dir)
existing = {label: (present, source) for label, present, source in
            query("SELECT label, present, source FROM ground_truth WHERE sample_id = ?", [sample_id])}

left, right = st.columns([3, 1])
with left:
    st.caption(f"`{sample_id}` · split **{split}** · {sample.context.get('time_of_day')} · "
               f"{sample.context.get('weather')} · ego {sample.ego_speed or 0:.1f} m/s")
    overlay = st.toggle("Show HD-map + 3D-box overlay", value=False)
    st.image(render_overlay(sample) if overlay else sample.load_image(), width="stretch")

with right:
    with st.form(key=f"form-{sample_id}"):
        answers = {}
        for label in HUMAN:
            answers[label.name] = st.checkbox(label.name, value=bool(existing.get(label.name, (False,))[0]),
                                              help=label.definition, key=f"{sample_id}:{label.name}")
        if st.form_submit_button("Save & next unlabeled", type="primary"):
            for name, present in answers.items():
                query("INSERT OR REPLACE INTO ground_truth VALUES (?, ?, ?, 'human')", [sample_id, name, present],
                      write=True)
            go(next_unlabeled=True)
            st.rerun()
    st.markdown("**Auto-derived labels** (check these too; fix the SQL rule if one is wrong)")
    for name in AUTO:
        present, source = existing.get(name, (None, None))
        mark = "–" if present is None else ("✅" if present else "·")
        st.markdown(f"{mark} {name} <small>({source or 'n/a'})</small>", unsafe_allow_html=True)
