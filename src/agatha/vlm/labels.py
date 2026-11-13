"""Andrew's own labels for a scene, and how they agree with Stage 2's reports. The review pages show both.

A label file is review/<prefix>/labels/<scene>.json, made from template() and filled in by hand:
{"scene": "random15_3", "reviewed": false, "no_incident": null, "incidents": [{"labels": [...], "start_s": ...,
"end_s": ..., "notes": ""}]}. Labels come from Stage 2's taxonomy, and times are the scene's seconds.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .schema import TAXONOMY, IncidentReport, Label

NO_FILE = "no label file"
_KEPT = ("CONFIRMED", "UNCERTAIN")  # a report that says something happened. REJECTED says it didn't.


class HumanIncident(BaseModel):
    """One event Andrew saw: labels from the taxonomy, and its span in the scene's seconds, as the reports give them."""

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    labels: list[Label] = Field(min_length=1)
    start_s: float = Field(ge=0)
    end_s: float
    notes: str = ""


class HumanLabels(BaseModel):
    """One scene's label file. Strict and closed, so a typo fails loudly instead of reading as no label."""

    model_config = ConfigDict(extra="forbid", strict=True)

    scene: str                # such as random15_3, so a file copied to another scene is caught
    reviewed: bool            # false until Andrew is done: until then the page says "not labeled yet"
    no_incident: bool | None  # true: nothing happened; false: the incidents below; null until decided
    incidents: list[HumanIncident]
    allowed_labels: list[str] = []  # the template's reminder of the taxonomy, ignored

    @model_validator(mode="after")
    def _consistent(self) -> HumanLabels:
        """Each span runs forward, and a reviewed scene says whether anything happened, as its incidents do."""
        if any(item.start_s > item.end_s for item in self.incidents):
            raise ValueError("an incident ends before it starts")
        if self.reviewed and self.no_incident is None:
            raise ValueError("a reviewed scene sets no_incident to true or false")
        if self.reviewed and self.no_incident == bool(self.incidents):
            raise ValueError("no_incident must be true with no incidents, and false with some")
        return self


def template(scene: str) -> dict:
    """A label file for Andrew to fill in. allowed_labels is a reminder, ignored when it's read."""
    return {"scene": scene, "reviewed": False, "no_incident": None, "incidents": [], "allowed_labels": list(TAXONOMY)}


def read_labels(path: Path, scene: str) -> HumanLabels | str:
    """Andrew's labels, or why they can't be used: the page shows it, so Andrew can fix the file."""
    if not path.exists():
        return NO_FILE
    try:
        labels = HumanLabels.model_validate_json(path.read_bytes())
    except ValidationError as exc:
        return "; ".join(map(_why, exc.errors()))
    return labels if labels.scene == scene else f"scene: {labels.scene}, not {scene}"


def _why(error: dict) -> str:
    """One validation error, where it is in the file and what's wrong, as Andrew reads it."""
    where = ".".join(map(str, error["loc"]))  # empty for the checks on the whole file, in _consistent
    what = error["msg"].removeprefix("Value error, ")  # pydantic's prefix on those checks' own words
    return f"{where}: {what}" if where else what


def _meets(item: HumanIncident, labels: list[str], start_s: float, end_s: float) -> bool:
    """Whether an incident shares one of these labels and overlaps this span, touching counting."""
    return bool(set(item.labels) & set(labels)) and item.start_s <= end_s and start_s <= item.end_s


def agreement(labels: HumanLabels, reports: list[IncidentReport], complete: bool = True, unassessed=()) -> dict:
    """Which reports match each of Andrew's incidents, the kept reports none match, and whether each saw nothing.

    A match is a CONFIRMED or UNCERTAIN report that shares a label it kept and overlaps in time, touching counting.
    A duplicate can keep a label the report it repeats doesn't, or run longer, so it matches too, and then that
    report counts as found: only the first report of an event can be Stage 2's only. `complete` is false when
    Stage 2 didn't assess the whole scene: if it kept nothing, whether it says nothing happened is then unknown.
    `unassessed` holds the merged candidates Pass 2 gave no verdict on, from candidates.json. An incident no report
    matches, but one of them would by the same rule, is listed as not assessed, not as Andrew's only.
    """
    kept = [report for report in reports if report.verification_status in _KEPT]
    matches = [[report.incident_id for report in kept
                if _meets(item, report.subcategories, report.start_s, report.end_s)] for item in labels.incidents]
    pending = [[] if ids else [candidate["incident_id"] for candidate in unassessed
                               if _meets(item, candidate["labels"], candidate["start_s"], candidate["end_s"])]
               for item, ids in zip(labels.incidents, matches)]  # only for an incident no report matches
    found = {incident_id for ids in matches for incident_id in ids}
    for report in reversed(kept):  # a duplicate names an earlier report, so this follows a chain back to its first
        if report.incident_id in found and report.duplicate_of:
            found.add(report.duplicate_of)
    return {"matches": matches, "unassessed": pending,
            "human_only": [number for number, (ids, others) in enumerate(zip(matches, pending), 1)
                           if not (ids or others)],
            "model_only": [report.incident_id for report in kept
                           if report.duplicate_of is None and report.incident_id not in found],
            # (Andrew says nothing happened, Stage 2 does, or None for unknown)
            "nothing": (bool(labels.no_incident), not kept if kept or complete else None)}


def summary(result: dict) -> str:
    """An agreement in one line, for the page and the index."""
    andrew, model = result["nothing"]
    nothing = ("nothing happened: unknown, as Stage 2 is incomplete" if model is None
               else "both say nothing happened" if andrew and model else "both saw something" if not (andrew or model)
               else "only Andrew says nothing happened" if andrew else "only Stage 2 says nothing happened")
    not_assessed = sum(bool(ids) for ids in result["unassessed"])
    return (f"{sum(bool(ids) for ids in result['matches'])} of Andrew's matched, {len(result['model_only'])} "
            f"Stage 2 only, {len(result['human_only'])} Andrew only"
            f"{f', {not_assessed} not assessed' if not_assessed else ''}; {nothing}")
