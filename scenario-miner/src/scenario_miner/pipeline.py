"""Run a labeling backend over samples and store raw responses + parsed predictions."""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import duckdb
import numpy as np

from .prompts import build_request, parse_response, VLMRequest
from .sample import Sample, load_sample
from .taxonomy import Label, label_names
from .vlm import Backend, VLMResult, generate_with_retry


def select_samples(con: duckdb.DuckDBPyConnection, split: str = "eval", limit: int | None = None) -> list[Sample]:
    if split == "all":
        q, params = "SELECT sample_dir FROM samples ORDER BY sample_id", []
    else:
        q = "SELECT s.sample_dir FROM samples s JOIN splits USING (sample_id) WHERE split = ? ORDER BY sample_id"
        params = [split]
    rows = con.execute(q + (f" LIMIT {int(limit)}" if limit else ""), params).fetchall()
    return [load_sample(r[0]) for r in rows]


def fewshot_examples(con: duckdb.DuckDBPyConnection, labels: list[str], k: int) -> list[tuple[Sample, dict]]:
    """Up to k examples from the few-shot pool, preferring ones with positive labels."""
    rows = con.execute(
        """
        SELECT s.sample_dir, list(g.label) FILTER (WHERE g.present) AS positives, count(g.label) AS n_labels
        FROM samples s JOIN splits USING (sample_id) JOIN ground_truth g USING (sample_id)
        WHERE split = 'fewshot'
        GROUP BY s.sample_dir
        HAVING n_labels >= ?
        ORDER BY len(positives) DESC, s.sample_dir
        """,
        [len(labels)],
    ).fetchall()
    chosen, covered = [], set()
    for sample_dir, positives, _ in rows:  # greedy: cover as many positive labels as possible
        new = set(positives or []) - covered
        if new or len(chosen) < k:
            chosen.append((load_sample(sample_dir), {l: l in (positives or []) for l in labels}))
            covered |= new
        if len(chosen) >= k:
            break
    return chosen


def _cached_call(con, backend: Backend, req: VLMRequest, temperature: float, vote: int) -> tuple[str, VLMResult | None]:
    key = f"{backend.name}|{backend.model}|t={temperature}|v={vote}|{req.cache_key}"
    row = con.execute("SELECT raw_text, latency_s, input_tokens, output_tokens FROM response_cache WHERE cache_key = ?",
                      [key]).fetchone()
    return key, (VLMResult(*row) if row else None)


def run_labeling(con: duckdb.DuckDBPyConnection, backend: Backend, strategy: str, taxonomy: list[Label],
                 samples: list[Sample], votes: int = 1, temperature: float | None = None,
                 fewshot_k: int = 3, workers: int = 4, max_side: int = 1024, run_id: str | None = None,
                 log=print) -> str:
    labels = backend.supported_labels(label_names(taxonomy))
    if temperature is None:
        temperature = 0.0 if votes == 1 else 0.7  # self-consistency needs sampling diversity
    run_id = run_id or f"{strategy}__{backend.name}:{backend.model}__{datetime.now():%m%d-%H%M%S}"
    con.execute("INSERT OR REPLACE INTO runs (run_id, backend, model, strategy, votes, temperature) VALUES (?, ?, ?, ?, ?, ?)",
                [run_id, backend.name, backend.model, strategy, votes, temperature])

    fewshot = fewshot_examples(con, labels, fewshot_k) if strategy == "few_shot" else []
    if strategy == "few_shot" and not fewshot:
        raise RuntimeError("few_shot needs labeled samples in the 'fewshot' split (run `miner split`, then label them)")

    # Build requests and resolve cache hits on the main thread (DuckDB connections aren't shared across threads).
    jobs = []  # (sample index, vote, cache key, request)
    results: dict[tuple[int, int], VLMResult | Exception] = {}
    for i, sample in enumerate(samples):
        req = build_request(sample, strategy, taxonomy, labels, fewshot, max_side)
        for v in range(votes):
            key, hit = _cached_call(con, backend, req, temperature, v)
            if hit is not None and backend.cacheable:
                results[(i, v)] = hit
            else:
                jobs.append((i, v, key, req))
    log(f"[{run_id}] {len(samples)} samples x {votes} votes: {len(results)} cached, {len(jobs)} to call")

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(generate_with_retry, backend, req, temperature): (i, v, key) for i, v, key, req in jobs}
        for n, fut in enumerate(as_completed(futures), 1):
            i, v, key = futures[fut]
            try:
                res = fut.result()
                if backend.cacheable:
                    con.execute("INSERT OR REPLACE INTO response_cache (cache_key, raw_text, latency_s, input_tokens, output_tokens) "
                                "VALUES (?, ?, ?, ?, ?)", [key, res.text, res.latency_s, res.input_tokens, res.output_tokens])
                results[(i, v)] = res
            except Exception as e:  # keep going; the failure is recorded per sample
                results[(i, v)] = e
            if n % 25 == 0 or n == len(jobs):
                log(f"  {n}/{len(jobs)} calls done ({time.perf_counter() - t0:.0f}s)")

    pred_rows, resp_rows = [], []
    for i, sample in enumerate(samples):
        outs = [results[(i, v)] for v in range(votes)]
        ok = [o for o in outs if isinstance(o, VLMResult)]
        errors = [f"api: {o}" for o in outs if isinstance(o, Exception)]
        parsed = [parse_response(o.text, labels) for o in ok]
        errors += [p.error for p in parsed if p.error]
        for label in labels:
            if not parsed:
                continue
            votes_present = np.array([p.decisions[label][0] for p in parsed], dtype=float)
            if len(parsed) == 1:
                present, conf = bool(votes_present[0]), parsed[0].decisions[label][1]
            else:  # self-consistency: confidence = fraction of samples voting "present"
                conf = float(votes_present.mean())
                present = conf >= 0.5
            pred_rows.append((run_id, sample.sample_id, label, present, conf))
        resp_rows.append((
            run_id, sample.sample_id,
            ok[0].text if ok else None,
            json.dumps(parsed[0].observations) if parsed and parsed[0].observations else None,
            "; ".join(errors) or None,
            sum(o.latency_s for o in ok) if ok else None,
            sum(o.input_tokens or 0 for o in ok) if ok else None,
            sum(o.output_tokens or 0 for o in ok) if ok else None,
        ))

    con.execute("BEGIN")
    con.execute("DELETE FROM predictions WHERE run_id = ?", [run_id])
    con.execute("DELETE FROM responses WHERE run_id = ?", [run_id])
    if pred_rows:
        con.executemany("INSERT INTO predictions VALUES (?, ?, ?, ?, ?)", pred_rows)
    con.executemany("INSERT INTO responses VALUES (?, ?, ?, ?, ?, ?, ?, ?)", resp_rows)
    con.execute("COMMIT")
    n_err = sum(1 for r in resp_rows if r[4])
    log(f"[{run_id}] stored {len(pred_rows)} predictions; {n_err} samples with errors")
    return run_id
