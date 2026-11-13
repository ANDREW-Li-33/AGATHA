"""Review pages for a list of scenes: each one's video, what Stage 2 found, and Andrew's own labels beside it.

agatha-pages reads $WOD_HOME/selected/<list> and writes $WOD_HOME/review/<prefix>/: a page per scene, an index, and
labels/<scene>.json, a template for Andrew to fill in (labels.py). A label file is never overwritten, so a rerun
after Andrew labels refreshes the pages around the labels. Each page is one file with its clip inside, so it opens
in OnDemand.
"""

from __future__ import annotations

import base64
import html
import json
import os
import re
import signal
import sys
import tempfile
from collections import Counter
from pathlib import Path

from pydantic import TypeAdapter

from ..columns import CAMERA_IDS, TIME
from ..config import load_config
from ..files import write_atomic
from ..render import write_clip
from ..run_stage1 import _QuietParser
from ..store import load_segment
from .labels import NO_FILE, HumanLabels, agreement, read_labels, summary, template
from .schema import IncidentReport

_FRONT = CAMERA_IDS["FRONT"]
_REPORTS = TypeAdapter(list[IncidentReport])
_UNASSESSED = ("failed", "invalid", "no_frames")  # a candidate's outcome when Pass 2 gave no verdict on it
_EXAMPLE = '{"labels": ["SUDDEN_CUT_IN"], "start_s": 8.0, "end_s": 12.0, "notes": ""}'
_HEAD = """<!doctype html>
<meta charset="utf-8">
<style>
  body { margin: 16px; font-family: sans-serif; } .REJECTED { color: #777; }
  video { width: 100%; max-height: 55vh; background: #000; position: sticky; top: 0; }  /* in view while times seek */
  table { border-collapse: collapse; } td, th { border: 1px solid #ccc; padding: 4px 8px; vertical-align: top; }
</style>
<script>
function seek(t) {  // the clip starts at the scene time in its data-offset
  const video = document.getElementById("clip");
  if (video) { video.currentTime = Math.max(0, t - Number(video.dataset.offset)); video.play(); }
  return false;
}
</script>
"""


def list_prefix(name: str) -> str:
    """The scene prefix the PACE scripts use: cutin for cutin_top.txt, as its pages always were, else the stem."""
    return "cutin" if name == "cutin_top.txt" else name.removesuffix(".txt")


def _time(t_s: float) -> str:
    """A time that seeks the clip to it."""
    return f'<a href="#" onclick="return seek({t_s:.2f})">{t_s:.2f} s</a>'


def _clip(directory: Path, profile: str) -> str:
    """Stage 1's FRONT frames over the whole scene as one <video>, or a line saying why there's none.

    Stage 1 writes a clip per 5 s window. One clip of the same frames lets every time on the page seek into it.
    """
    if not (directory / "segment.json").exists():  # Stage 1 writes it last
        return "<p>No clip: Stage 1 hasn't run on this scene.</p>\n"
    segment = load_segment(directory)
    front = segment.frames[segment.frames["camera"] == _FRONT]
    drawn = [t_s for ts, t_s in zip(front[TIME], front["t_s"])
             if (directory / "render" / "FRONT" / f"{ts}__{profile}.jpg").exists()]
    if not drawn:
        return "<p>No clip: Stage 1 drew no FRONT frames.</p>\n"
    with tempfile.TemporaryDirectory() as scratch:  # with its ffconcat script, gone once read
        data = write_clip(segment, _FRONT, min(drawn), max(drawn), Path(scratch) / "scene.mp4", profile).read_bytes()
    # preload auto: the first frame shows before it plays. The data is in the page already, so it costs nothing.
    return (f'<video id="clip" data-offset="{min(drawn):.3f}" controls muted playsinline preload="auto" '
            f'src="data:video/mp4;base64,{base64.b64encode(data).decode()}"></video>\n')


def _verdicts(reports: list[IncidentReport]) -> str:
    status = Counter(report.verification_status for report in reports)
    return (f"{status['CONFIRMED']} confirmed, {status['REJECTED']} rejected, {status['UNCERTAIN']} uncertain, "
            f"{sum(report.duplicate_of is not None for report in reports)} marked duplicate")


def _unassessed(stage2: Path) -> tuple[str, list[dict]]:
    """What Stage 2 left unassessed, as a line, or "", and the merged candidates Pass 2 gave no verdict on.

    agatha-vlm writes its results though requests failed. Each candidate keeps Pass 1's labels and span.
    """
    path = stage2 / "candidates.json"  # written with incidents.json, so only a run older than it lacks one
    record = json.loads(path.read_bytes()) if path.exists() else {"windows": [], "merged": []}
    windows = sum(item["outcome"] != "answered" for item in record["windows"])
    candidates = [item for item in record["merged"] if item["outcome"] in _UNASSESSED]
    parts = [f"{windows} of {len(record['windows'])} windows" if windows else "",
             f"{len(candidates)} of {len(record['merged'])} candidates" if candidates else ""]
    return (f"Stage 2 didn't assess {' and '.join(filter(None, parts))}" if any(parts) else ""), candidates


def _reports_html(reports: list[IncidentReport] | None, missed: str, unassessed: list[dict]) -> str:
    if reports is None:
        return "<h2>Stage 2</h2>\n<p>No results: Stage 2 hasn't written incidents.json for this scene.</p>\n"
    models = html.escape(", ".join(sorted({report.model_id for report in reports})))
    items = "".join(f"<li>{html.escape(item['incident_id'])}: {html.escape(', '.join(item['labels']))}, "
                    f"{_time(item['start_s'])} to {_time(item['end_s'])}, {html.escape(item['outcome'])}</li>\n"
                    for item in unassessed)  # where Andrew can look for what Stage 2 didn't
    head = (f"<h2>Stage 2{f' ({models})' if models else ''}</h2>\n" + (f"<p><b>{missed}.</b></p>\n" if missed else "")
            + (f"<ul>\n{items}</ul>\n" if items else ""))
    if not reports:
        return f"{head}<p>0 reports: Stage 2 reported nothing in this scene.</p>\n"
    rows = ""
    for report in reports:
        rejected = [f"{item.label} (rejected)" for item in report.label_assessments if item.status == "REJECTED"]
        cells = [report.incident_id, report.verification_status, ", ".join([*report.subcategories, *rejected]),
                 f"{_time(report.start_s)} to {_time(report.end_s)}", html.escape(report.vlm_reasoning),
                 html.escape("; ".join(report.uncertainty_reasons)),
                 f"duplicate of {report.duplicate_of}" if report.duplicate_of else ""]
        rows += f'<tr class="{report.verification_status}"><td>{"</td><td>".join(cells)}</td></tr>\n'
    return (f"{head}<p>{len(reports)} reports: {_verdicts(reports)}.</p>\n<table><tr><th>Report</th><th>Status</th>"
            "<th>Labels</th><th>Time</th><th>Explanation</th><th>Uncertain because</th><th>Duplicate</th></tr>\n"
            f"{rows}</table>\n")


def _labels_html(labels: HumanLabels | str, scene: str) -> str:
    head = "<h2>Andrew's labels</h2>\n"
    if isinstance(labels, str) or not labels.reviewed:
        unusable = isinstance(labels, str) and labels != NO_FILE
        message = f"The label file can't be used: {labels}" if unusable else "Not labeled yet."
        return (f"{head}<p>{html.escape(message)}</p>\n<p>Edit labels/{scene}.json: set no_incident to true or false, "
                f"list incidents such as <code>{html.escape(_EXAMPLE)}</code>, set reviewed to true, then run "
                "agatha-pages again.</p>\n")
    if labels.no_incident:
        return f"{head}<p>Nothing happened.</p>\n"
    rows = "".join(f"<tr><td>{number}</td><td>{', '.join(item.labels)}</td><td>{_time(item.start_s)} to "
                   f"{_time(item.end_s)}</td><td>{html.escape(item.notes)}</td></tr>\n"
                   for number, item in enumerate(labels.incidents, 1))
    return f"{head}<table><tr><th>Incident</th><th>Labels</th><th>Time</th><th>Notes</th></tr>\n{rows}</table>\n"


def _agreement_html(result: dict) -> str:
    lines = [f"Andrew's {number}: " + ("matches " + ", ".join(ids) if ids else
                                       f"overlaps {', '.join(others)}, which Stage 2 didn't assess" if others else
                                       "Andrew only")
             for number, (ids, others) in enumerate(zip(result["matches"], result["unassessed"]), 1)]
    lines += [f"{incident_id}: Stage 2 only" for incident_id in result["model_only"]]
    items = "".join(f"<li>{html.escape(line)}</li>\n" for line in lines)
    return f"<h2>Agreement</h2>\n<p>{html.escape(summary(result))}</p>\n" + (f"<ul>\n{items}</ul>\n" if items else "")


def write_pages(config, home: Path, list_name: str, ranks=None) -> Counter:
    """A page per scene of the list (or of `ranks`), a label template for each that has none, and the index."""
    contexts = (home / "selected" / list_name).read_text().split()
    prefix, ranks = list_prefix(list_name), set(ranks or range(1, len(contexts) + 1))
    folder, counts, rows = home / "review" / prefix, Counter(), []
    for rank, context in enumerate(contexts, 1):  # every scene gets a row in the index, its page or not
        scene = f"{prefix}_{rank}"
        try:
            if not re.fullmatch(r"[A-Za-z0-9_]+", context):  # a path could reach outside the outputs
                raise ValueError("not a context name")
            label_path = folder / "labels" / f"{scene}.json"
            if rank in ranks and not label_path.exists():  # never over Andrew's
                write_atomic(label_path, (json.dumps(template(scene), indent=2) + "\n").encode())
                counts["templates"] += 1
            stage2 = config.outputs / context / "stage2"
            path = stage2 / "incidents.json"
            reports = _REPORTS.validate_json(path.read_bytes()) if path.exists() else None
            missed, unassessed = ("", []) if reports is None else _unassessed(stage2)
            labels = read_labels(label_path, scene)
            labeled = isinstance(labels, HumanLabels) and labels.reviewed
            state = "yes" if labeled else "invalid" if isinstance(labels, str) and labels != NO_FILE else "no"
            result = agreement(labels, reports, not missed, unassessed) if labeled and reports is not None else None
            if rank in ranks:
                clip = _clip(config.outputs / context, config.profile)
                write_atomic(folder / f"{scene}.html", (
                    f"{_HEAD}<title>{scene}</title>\n<h1>{scene}</h1>\n{clip}"
                    f"{_reports_html(reports, missed, unassessed)}"
                    f"{_labels_html(labels, scene)}{_agreement_html(result) if result else ''}").encode())
                for key, hit in (("pages", True), ("clips", "<video" in clip), ("stage2", reports is not None),
                                 ("incomplete", bool(missed)), ("labeled", labeled), ("invalid", state == "invalid")):
                    counts[key] += hit  # 0 + a bool is 1 or 0, where Counter.update would keep the bool itself
            verdicts = "-" if reports is None else _verdicts(reports) + (f"; {missed}" if missed else "")
            rows.append((scene, verdicts, state, summary(result) if result else "-"))
        except Exception:  # counted only: a message can echo the scene's data
            counts["failed"] += 1
            rows.append((scene, "failed", "-", "-"))
    cells = "".join(f'<tr><td><a href="{scene}.html">{scene}</a></td><td>{verdicts}</td><td>{state}</td>'
                    f"<td>{html.escape(agreed)}</td></tr>\n" for scene, verdicts, state, agreed in rows)
    write_atomic(folder / "index.html", (f"{_HEAD}<title>{prefix}</title>\n<h1>{prefix}: {len(rows)} scenes</h1>\n"
                                         "<table><tr><th>Scene</th><th>Stage 2</th><th>Labeled</th><th>Agreement</th>"
                                         f"</tr>\n{cells}</table>\n").encode())
    return counts


def main(argv=None) -> int:
    os.umask(0o077)  # before anything is written: the pages hold real frames
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))
    sys.excepthook = lambda kind, *_: print(f"agatha-pages: failed ({kind.__name__})", file=sys.stderr)
    parser = _QuietParser(prog="agatha-pages", description="Write a review page per scene of a list, and an index.")
    parser.add_argument("--config", default="configs/stage1.toml")
    parser.add_argument("--list", required=True, help="a list in $WOD_HOME/selected, such as random15.txt")
    parser.add_argument("--ranks", nargs="+", type=int, help="only these scenes' pages, by line in the list")
    args = parser.parse_args(argv)
    home = Path(os.environ["WOD_HOME"])  # env.sh sets it, as for agatha-scan
    listed = home / "selected" / args.list
    if not re.fullmatch(r"[a-z0-9_]+\.txt", args.list) or not listed.exists():
        parser.error("list")
    if not all(1 <= rank <= len(listed.read_text().split()) for rank in args.ranks or []):
        parser.error("ranks")
    counts, prefix = write_pages(load_config(args.config), home, args.list, args.ranks), list_prefix(args.list)
    print(f"{prefix}: {counts['pages']} pages, {counts['clips']} with a clip, {counts['stage2']} with Stage 2 results "
          f"({counts['incomplete']} incomplete), {counts['labeled']} labeled, {counts['invalid']} label files invalid, "
          f"{counts['templates']} templates made, {counts['failed']} failed")
    print(f"index: {home / 'review' / prefix / 'index.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
