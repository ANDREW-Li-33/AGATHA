"""Precision / recall / F1 with segment-level (cluster) bootstrap confidence intervals.

Frames from the same 20 s segment are strongly correlated (same street, same
weather, often the same construction zone), so resampling individual frames
would make the intervals far too narrow. We resample whole segments instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import duckdb
import numpy as np


@dataclass
class LabelEval:
    run_id: str
    label: str
    gt_source: str
    n: int
    support: int
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    ap: float
    ci: dict[str, tuple[float, float]] = field(default_factory=dict)


@dataclass
class _Rows:
    segment: np.ndarray
    pred: np.ndarray
    conf: np.ndarray
    gt: np.ndarray


def prf(tp, fp, fn):
    """Precision, recall, F1 (NaN where undefined); works on scalars or arrays."""
    tp, fp, fn = (np.asarray(x, dtype=float) for x in (tp, fp, fn))
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(tp + fp > 0, tp / (tp + fp), np.nan)
        r = np.where(tp + fn > 0, tp / (tp + fn), np.nan)
        f = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), np.nan)
    return p, r, f


def average_precision(conf: np.ndarray, gt: np.ndarray) -> float:
    """Area under the precision-recall curve from ranking by confidence."""
    gt = np.asarray(gt, dtype=bool)
    if gt.sum() == 0:
        return float("nan")
    order = np.argsort(-np.asarray(conf, dtype=float), kind="stable")
    hits = gt[order]
    precision_at_k = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float((precision_at_k * hits).sum() / hits.sum())


def _segment_counts(segments: np.ndarray, pred: np.ndarray, gt: np.ndarray, seg_index: dict[str, int]):
    idx = np.array([seg_index[s] for s in segments], dtype=int)
    n = len(seg_index)
    tp = np.bincount(idx, weights=(pred & gt), minlength=n)
    fp = np.bincount(idx, weights=(pred & ~gt), minlength=n)
    fn = np.bincount(idx, weights=(~pred & gt), minlength=n)
    return tp, fp, fn


def _bootstrap_weights(n_segments: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """(n_boot, n_segments) counts: how often each segment is drawn in each resample."""
    return rng.multinomial(n_segments, np.full(n_segments, 1.0 / n_segments), size=n_boot)


def _ci(samples: np.ndarray) -> tuple[float, float]:
    finite = samples[np.isfinite(samples)]
    if len(finite) < 0.5 * len(samples):  # metric undefined in most resamples (e.g. no positives)
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(finite, [2.5, 97.5])
    return float(lo), float(hi)


def load_rows(con: duckdb.DuckDBPyConnection, run_ids: list[str] | None = None) -> dict[tuple[str, str], tuple[str, _Rows]]:
    q = """
        SELECT p.run_id, p.label, g.source, s.segment_id, p.present, p.confidence, g.present
        FROM predictions p
        JOIN ground_truth g USING (sample_id, label)
        JOIN samples s USING (sample_id)
        LEFT JOIN splits sp USING (sample_id)
        WHERE coalesce(sp.split, 'eval') <> 'fewshot'
    """
    params: list = []
    if run_ids:
        q += f" AND p.run_id IN ({', '.join('?' * len(run_ids))})"
        params = list(run_ids)
    grouped: dict[tuple[str, str], list] = {}
    for run_id, label, source, seg, pred, conf, gt in con.execute(q + " ORDER BY p.run_id, p.label, p.sample_id", params).fetchall():
        grouped.setdefault((run_id, label), []).append((source, seg, pred, conf, gt))
    out = {}
    for key, rows in grouped.items():
        sources = sorted({r[0] for r in rows})
        out[key] = ("+".join(sources), _Rows(
            segment=np.array([r[1] for r in rows]),
            pred=np.array([r[2] for r in rows], dtype=bool),
            conf=np.array([r[3] for r in rows], dtype=float),
            gt=np.array([r[4] for r in rows], dtype=bool),
        ))
    return out


def evaluate(con: duckdb.DuckDBPyConnection, run_ids: list[str] | None = None, n_boot: int = 2000,
             seed: int = 0) -> list[LabelEval]:
    data = load_rows(con, run_ids)
    rng = np.random.default_rng(seed)
    results = []
    for (run_id, label), (source, rows) in sorted(data.items()):
        seg_index = {s: i for i, s in enumerate(sorted(set(rows.segment)))}
        tp_s, fp_s, fn_s = _segment_counts(rows.segment, rows.pred, rows.gt, seg_index)
        p, r, f = prf(tp_s.sum(), fp_s.sum(), fn_s.sum())
        W = _bootstrap_weights(len(seg_index), n_boot, rng)
        bp, br, bf = prf(W @ tp_s, W @ fp_s, W @ fn_s)
        results.append(LabelEval(
            run_id=run_id, label=label, gt_source=source, n=len(rows.gt), support=int(rows.gt.sum()),
            tp=int(tp_s.sum()), fp=int(fp_s.sum()), fn=int(fn_s.sum()),
            precision=float(p), recall=float(r), f1=float(f),
            ap=average_precision(rows.conf, rows.gt),
            ci={"precision": _ci(bp), "recall": _ci(br), "f1": _ci(bf)},
        ))
    return results


def paired_f1_difference(con: duckdb.DuckDBPyConnection, run_a: str, run_b: str, label: str,
                         n_boot: int = 2000, seed: int = 0) -> dict | None:
    """F1(run_b) - F1(run_a) on the samples both runs labeled, with a paired cluster bootstrap.

    Pairing matters: both runs see the same resampled segments, so the noise from
    "which segments happened to be in the eval set" cancels out of the difference.
    """
    rows = con.execute(
        """
        SELECT s.segment_id, a.present, b.present, g.present
        FROM predictions a
        JOIN predictions b ON b.sample_id = a.sample_id AND b.label = a.label AND b.run_id = ?
        JOIN ground_truth g ON g.sample_id = a.sample_id AND g.label = a.label
        JOIN samples s ON s.sample_id = a.sample_id
        LEFT JOIN splits sp ON sp.sample_id = a.sample_id
        WHERE a.run_id = ? AND a.label = ? AND coalesce(sp.split, 'eval') <> 'fewshot'
        """,
        [run_b, run_a, label],
    ).fetchall()
    if not rows:
        return None
    seg = np.array([r[0] for r in rows])
    pa, pb, gt = (np.array([r[i] for r in rows], dtype=bool) for i in (1, 2, 3))
    seg_index = {s: i for i, s in enumerate(sorted(set(seg)))}
    ca, cb = _segment_counts(seg, pa, gt, seg_index), _segment_counts(seg, pb, gt, seg_index)
    W = _bootstrap_weights(len(seg_index), n_boot, np.random.default_rng(seed))
    fa = prf(*(W @ c for c in ca))[2]
    fb = prf(*(W @ c for c in cb))[2]
    diff = fb - fa
    point = float(prf(*(c.sum() for c in cb))[2] - prf(*(c.sum() for c in ca))[2])
    finite = diff[np.isfinite(diff)]
    return {
        "label": label, "n": len(rows), "delta_f1": point, "ci": _ci(diff),
        "p_not_better": float((finite <= 0).mean()) if len(finite) else float("nan"),
    }


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def _fmt(x: float, digits: int = 2) -> str:
    return "–" if x is None or not np.isfinite(x) else f"{x:.{digits}f}"


def _fmt_ci(ci: tuple[float, float]) -> str:
    return "" if not all(np.isfinite(ci)) else f" [{ci[0]:.2f}, {ci[1]:.2f}]"


def run_summaries(con: duckdb.DuckDBPyConnection, run_ids: list[str]) -> list[tuple]:
    return con.execute(
        f"""
        SELECT r.run_id, r.backend, r.model, r.strategy, r.votes,
               count(*) AS n, count(*) FILTER (WHERE x.error IS NOT NULL) AS errors,
               avg(x.latency_s), avg(x.input_tokens), avg(x.output_tokens)
        FROM runs r JOIN responses x USING (run_id)
        WHERE r.run_id IN ({', '.join('?' * len(run_ids))})
        GROUP BY ALL ORDER BY min(r.created_at)
        """,
        run_ids,
    ).fetchall()


def write_report(con: duckdb.DuckDBPyConnection, results: list[LabelEval], baseline: str | None = None,
                 n_boot: int = 2000) -> str:
    run_ids = list(dict.fromkeys(r.run_id for r in results))
    if not run_ids:
        return "# Evaluation\n\nNo predictions joined to ground truth yet.\n"
    n_segments = con.execute("SELECT count(DISTINCT segment_id) FROM samples").fetchone()[0]
    lines = [
        "# Scenario-miner evaluation",
        "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M}. 95% intervals: cluster bootstrap over segments "
        f"({n_boot} resamples; {n_segments} segments in the database). Few-shot pool excluded.",
        "",
        "## Runs",
        "",
        "| run | backend:model | strategy | votes | frames | JSON/API errors | s/frame | tokens in / out per frame | macro-F1 | micro-F1 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    by_run: dict[str, list[LabelEval]] = {}
    for r in results:
        by_run.setdefault(r.run_id, []).append(r)
    for run_id, backend, model, strategy, votes, n, errors, lat, tin, tout in run_summaries(con, run_ids):
        evals = by_run.get(run_id, [])
        macro = np.nanmean([e.f1 for e in evals]) if evals and np.isfinite([e.f1 for e in evals]).any() else float("nan")
        micro = prf(sum(e.tp for e in evals), sum(e.fp for e in evals), sum(e.fn for e in evals))[2]
        tokens = f"{_fmt(tin, 0)} / {_fmt(tout, 0)}" if tin else "–"
        lines.append(f"| `{run_id}` | {backend}:{model} | {strategy} | {votes} | {n} | {errors} | "
                     f"{_fmt(lat)} | {tokens} | {_fmt(macro)} | {_fmt(float(micro))} |")

    lines += ["", "## Per label", "",
              "| label | GT | run | n | positives | precision | recall | F1 [95% CI] | AP |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(results, key=lambda r: (r.label, r.run_id)):
        lines.append(f"| {r.label} | {r.gt_source} | `{r.run_id}` | {r.n} | {r.support} | "
                     f"{_fmt(r.precision)} | {_fmt(r.recall)} | {_fmt(r.f1)}{_fmt_ci(r.ci['f1'])} | {_fmt(r.ap)} |")

    if baseline and baseline in run_ids:
        lines += ["", f"## Paired comparison vs `{baseline}`", "",
                  "ΔF1 > 0 means the run beats the baseline. P(Δ≤0) is the share of bootstrap resamples "
                  "where it doesn't.", "",
                  "| label | run | n | ΔF1 [95% CI] | P(Δ≤0) |", "|---|---|---|---|---|"]
        base_labels = {r.label for r in results if r.run_id == baseline}
        for run_id in run_ids:
            if run_id == baseline:
                continue
            for label in sorted(base_labels & {r.label for r in by_run.get(run_id, [])}):
                d = paired_f1_difference(con, baseline, run_id, label, n_boot)
                if d:
                    lines.append(f"| {label} | `{run_id}` | {d['n']} | {d['delta_f1']:+.2f}{_fmt_ci(d['ci'])} | "
                                 f"{_fmt(d['p_not_better'])} |")
    return "\n".join(lines) + "\n"
