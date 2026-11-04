"""`miner` command line.

Typical flow:
    miner synth                     # or: miner extract data/raw/*.tfrecord
    miner ingest && miner split
    streamlit run tools/label_app.py -- --db data/miner.duckdb    # human labels
    miner label --backend heuristic
    miner label --backend gemini --strategy zero_shot
    miner label --backend gemini --strategy map_overlay_context
    miner eval --baseline <run_id>
    miner query sql/mining_queries.sql
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_DB = "data/miner.duckdb"
DEFAULT_EXTRACTED = "data/extracted"


def cmd_synth(a):
    from .data.synthetic import generate

    paths = generate(a.out, n_segments=a.segments, n_samples=a.samples, seed=a.seed)
    print(f"wrote {len(paths)} synthetic samples to {a.out}")


def cmd_extract(a):
    from .data.waymo_v1 import extract_segment

    total = 0
    for path in a.tfrecords:
        written = extract_segment(path, a.out, every_n=a.every_n, cameras=tuple(a.cameras), map_radius=a.map_radius)
        total += len(written)
        print(f"{Path(path).name}: {len(written)} samples")
    print(f"wrote {total} samples to {a.out}")


def cmd_ingest(a):
    from .store import connect, ingest

    con = connect(a.db)
    n = ingest(con, a.extracted)
    gt = con.execute("SELECT label, source, count(*) FILTER (WHERE present), count(*) FROM ground_truth "
                     "GROUP BY ALL ORDER BY source, label").fetchall()
    print(f"ingested {n} samples into {a.db}\nground truth (label, source, positives, total):")
    for row in gt:
        print("  ", *row)


def cmd_split(a):
    from .store import connect, make_splits

    counts = make_splits(connect(a.db), a.fewshot_segments, a.per_segment, a.min_gap_s, a.seed)
    print("split sizes:", counts)


def cmd_label(a):
    from .pipeline import run_labeling, select_samples
    from .store import connect
    from .taxonomy import load_taxonomy
    from .vlm import make_backend

    con = connect(a.db)
    samples = select_samples(con, a.split, a.limit)
    if not samples:
        sys.exit(f"no samples in split {a.split!r}; run `miner ingest` and `miner split` first")
    backend = make_backend(a.backend, a.model, a.base_url, a.thinking_budget)
    run_id = run_labeling(con, backend, a.strategy, load_taxonomy(a.taxonomy), samples, votes=a.votes,
                          temperature=a.temperature, fewshot_k=a.fewshot_k, workers=a.workers,
                          max_side=a.max_side, run_id=a.run_id)
    print(run_id)


def cmd_eval(a):
    from .evaluate import evaluate, write_report
    from .store import connect

    con = connect(a.db)
    runs = a.runs or None
    report = write_report(con, evaluate(con, runs, a.n_boot), a.baseline, a.n_boot)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(report)
        print(f"wrote {a.out}")
    print(report)


def cmd_query(a):
    from .store import connect

    con = connect(a.db)
    text = Path(a.sql).read_text() if a.sql.endswith(".sql") and Path(a.sql).exists() else a.sql
    for stmt in (s.strip() for s in text.split(";")):
        code = [l for l in stmt.splitlines() if l.strip() and not l.strip().startswith("--")]
        if not code:
            continue
        if not code[0].lstrip().upper().startswith(("SELECT", "WITH", "FROM", "SUMMARIZE", "DESCRIBE", "SHOW")):
            con.execute(stmt)  # DDL / setup statements: run silently
            continue
        comments = [l.strip()[2:].strip() for l in stmt.splitlines() if l.strip().startswith("--")]
        if comments:
            print("\n== " + " ".join(comments))
        con.sql(stmt).show(max_rows=a.max_rows, max_width=200)


def cmd_render(a):
    from .render import render_overlay
    from .sample import load_sample
    from .store import connect

    con = connect(a.db)
    if a.sample_ids:
        ids = a.sample_ids
    else:
        n = int(a.random)  # SAMPLE only accepts a literal
        ids = [r[0] for r in con.execute(f"SELECT sample_id FROM samples USING SAMPLE reservoir({n} ROWS) REPEATABLE (7)").fetchall()]
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for sid in ids:
        row = con.execute("SELECT sample_dir FROM samples WHERE sample_id = ?", [sid]).fetchone()
        if row is None:
            print(f"unknown sample {sid}")
            continue
        render_overlay(load_sample(row[0])).save(out / f"{sid}.jpg", quality=90)
    print(f"wrote {len(ids)} overlays to {out}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="miner", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("synth", help="generate synthetic samples (no download needed)")
    s.add_argument("--out", default=DEFAULT_EXTRACTED)
    s.add_argument("--segments", type=int, default=30)
    s.add_argument("--samples", type=int, default=6, help="samples per segment")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_synth)

    s = sub.add_parser("extract", help="extract samples from Waymo v1.4.x TFRecords")
    s.add_argument("tfrecords", nargs="+")
    s.add_argument("--out", default=DEFAULT_EXTRACTED)
    s.add_argument("--every-n", type=int, default=20, help="keep every Nth frame (10 Hz; 20 = every 2 s)")
    s.add_argument("--cameras", nargs="+", default=["FRONT"],
                   choices=["FRONT", "FRONT_LEFT", "FRONT_RIGHT", "SIDE_LEFT", "SIDE_RIGHT"])
    s.add_argument("--map-radius", type=float, default=60.0)
    s.set_defaults(func=cmd_extract)

    s = sub.add_parser("ingest", help="load extracted samples into DuckDB + derive ground truth")
    s.add_argument("--extracted", default=DEFAULT_EXTRACTED)
    s.add_argument("--db", default=DEFAULT_DB)
    s.set_defaults(func=cmd_ingest)

    s = sub.add_parser("split", help="choose eval frames and a segment-disjoint few-shot pool")
    s.add_argument("--db", default=DEFAULT_DB)
    s.add_argument("--fewshot-segments", type=int, default=3)
    s.add_argument("--per-segment", type=int, default=3)
    s.add_argument("--min-gap-s", type=float, default=4.0)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_split)

    from .prompts import STRATEGIES

    s = sub.add_parser("label", help="run a backend + prompt strategy and store predictions")
    s.add_argument("--db", default=DEFAULT_DB)
    s.add_argument("--backend", choices=["heuristic", "gemini", "openai"], default="heuristic")
    s.add_argument("--model", help="e.g. gemini-2.5-flash, gpt-4o-mini, qwen2.5vl:7b")
    s.add_argument("--base-url", help="OpenAI-compatible endpoint, e.g. http://localhost:11434/v1 for Ollama")
    s.add_argument("--strategy", choices=STRATEGIES, default="zero_shot")
    s.add_argument("--split", choices=["eval", "fewshot", "all"], default="eval")
    s.add_argument("--limit", type=int)
    s.add_argument("--votes", type=int, default=1, help=">1 enables self-consistency voting")
    s.add_argument("--temperature", type=float)
    s.add_argument("--fewshot-k", type=int, default=3)
    s.add_argument("--workers", type=int, default=4)
    s.add_argument("--max-side", type=int, default=1024, help="downscale images to save tokens")
    s.add_argument("--thinking-budget", type=int, help="Gemini 2.5 thinking budget (0 disables thinking on Flash)")
    s.add_argument("--taxonomy", help="path to a taxonomy.toml (default: packaged one)")
    s.add_argument("--run-id")
    s.set_defaults(func=cmd_label)

    s = sub.add_parser("eval", help="precision/recall/F1 with cluster-bootstrap CIs")
    s.add_argument("--db", default=DEFAULT_DB)
    s.add_argument("--runs", nargs="*")
    s.add_argument("--baseline", help="run_id to compare every other run against")
    s.add_argument("--n-boot", type=int, default=2000)
    s.add_argument("--out", default="reports/eval.md")
    s.set_defaults(func=cmd_eval)

    s = sub.add_parser("query", help="run SQL (a string or a .sql file) against the database")
    s.add_argument("sql")
    s.add_argument("--db", default=DEFAULT_DB)
    s.add_argument("--max-rows", type=int, default=20)
    s.set_defaults(func=cmd_query)

    s = sub.add_parser("render", help="save HD-map + 3D-box overlays to check alignment")
    s.add_argument("sample_ids", nargs="*")
    s.add_argument("--random", type=int, default=8)
    s.add_argument("--db", default=DEFAULT_DB)
    s.add_argument("--out-dir", default="data/overlays")
    s.set_defaults(func=cmd_render)

    args = p.parse_args(argv)
    if getattr(args, "db", None):
        Path(args.db).parent.mkdir(parents=True, exist_ok=True)
    args.func(args)


if __name__ == "__main__":
    main()
