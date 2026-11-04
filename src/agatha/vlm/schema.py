"""Stage 2's types: the taxonomy, what a model must answer, the IncidentReport, and the referential checks.

An answer is trusted once it parses into these strict models and the checks find it in its window.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from ..contracts import SceneWindow

# Design doc 5.1, taxonomy v1, by category: label -> operational definition.
_TABLE = {
    "DYNAMIC_HAZARD": {
        "SUDDEN_CUT_IN": "Vehicle unexpectedly enters ego lane/path with limited clearance, including "
                         "opposing-direction encroachment when applicable.",
        "HARD_BRAKING": "Leading vehicle decelerates abruptly enough to create a potential response demand.",
        "PEDESTRIAN_INTRUSION": "Pedestrian unexpectedly enters/crosses the expected vehicle path.",
        "CYCLIST_INTRUSION": "Cyclist unexpectedly moves into/across the expected vehicle path.",
        "CROSS_TRAFFIC_CONFLICT": "Cross traffic enters or conflicts with the expected vehicle path.",
    },
    "CONSTRUCTION_DISRUPTION": {
        "CONSTRUCTION_LANE_CLOSURE": "Construction blocks an otherwise available travel lane.",
        "TEMPORARY_LANE_SHIFT": "Work zone redirects traffic from its usual lane alignment.",
        "TEMPORARY_TRAFFIC_CONTROL": "Temporary worker/sign/signal changes normal traffic instructions.",
        "CONFLICTING_LANE_GUIDANCE": "Temporary instructions/markings conflict with permanent guidance.",
        "WORK_ZONE_ENCROACHMENT": "Worker, equipment or machinery intrudes into expected driving space.",
    },
    "ROAD_OBSTRUCTION": {
        "STOPPED_VEHICLE_OBSTRUCTION": "Stopped/disabled vehicle blocks an expected travel path.",
        "DOUBLE_PARKED_OBSTRUCTION": "Unusually parked vehicle obstructs traffic flow.",
        "ROAD_DEBRIS": "Fallen/loose material interferes with the expected path.",
        "OTHER_PHYSICAL_OBSTRUCTION": "Other identified physical obstacle affects expected travel.",
    },
}
CATEGORIES = tuple(_TABLE)
TAXONOMY = {label: (category, text) for category, labels in _TABLE.items() for label, text in labels.items()}

Label = Literal[tuple(TAXONOMY)]
Status = Literal["CONFIRMED", "REJECTED", "UNCERTAIN"]

_TIME_SLACK_S = 0.05  # half a 10 Hz frame: jitter can put a frame just outside a window's grid bounds


class _Answer(BaseModel):
    """What a model writes: strict and closed, with every field required, as constrained decoding needs.

    Subclasses take comments, not docstrings: pydantic would send a docstring to the model in the schema.
    """

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class Candidate(_Answer):  # one of Pass 1's answers, and what Pass 2 tests: merged candidates are a candidate too
    labels: list[Label]
    start_s: float
    end_s: float
    object_aliases: list[str]
    evidence: list[str]
    driving_impact: str


class Pass1Answer(_Answer):
    candidates: list[Candidate]


class LabelAssessment(_Answer):  # evidence before status: an answer comes out in field order
    label: Label
    evidence: list[str]
    contradicting_evidence: list[str]
    status: Status


class CitedFrame(_Answer):
    camera: str
    t_s: float  # as captioned, to two decimals


class Pass2Answer(_Answer):  # the evidence first, then the verdict, as for each label
    label_assessments: list[LabelAssessment]
    source_frames: list[CitedFrame]
    explanation: str  # not "reasoning": a request for a model's reasoning can be refused as reasoning extraction
    uncertainty_reasons: list[str]
    start_s: float
    end_s: float
    object_aliases: list[str]
    status: Status
    driving_impact: str


PASS1_SCHEMA = Pass1Answer.model_json_schema()
PASS2_SCHEMA = Pass2Answer.model_json_schema()


class SourceFrame(BaseModel):
    camera: str
    timestamp_s: float
    frame_timestamp_micros: int  # the exact key, for whatever cuts a clip from it


class IncidentReport(BaseModel):
    """Stage 2 -> Stage 3, as in design doc section 6, plus the prompt and preprocessing versions, and duplicate_of."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    schema_version: str
    segment_id: str
    incident_id: str
    duplicate_of: str | None    # an earlier report's incident_id when both are one event, found from two candidates
    start_s: float
    end_s: float
    categories: list[Literal[CATEGORIES]]
    subcategories: list[Label]  # the labels Pass 2 didn't reject. label_assessments has every one.
    object_ids: list[str]       # laser_object_id
    verification_status: Status
    label_assessments: list[LabelAssessment]
    vlm_reasoning: str          # a concise evidence-based explanation, not the model's hidden reasoning
    driving_impact: str
    uncertainty_reasons: list[str]
    source_frames: list[SourceFrame]
    pipeline_version: str
    model_id: str               # the model that answered: a server-side fallback can stand in
    prompt_version: str
    preprocessing_version: str


def _known(window: SceneWindow) -> set[str]:
    """The aliases a model may cite. Omitted objects count: the images label them too."""
    return {item.alias for item in [*window.objects, *window.omitted_objects]}


def fit_candidate(candidate: Candidate, window: SceneWindow) -> tuple[Candidate, list[str], list[str]]:
    """(the candidate fitted to its window, what fitting changed, what still keeps it out of the merge)."""
    # Pass 1 is for recall: clip a span to the window, as the JSON's track context invites one past it, and drop an
    # alias the window lacks, rather than lose the event. Parsing has held the labels to the taxonomy.
    known, span = _known(window), (candidate.start_s, candidate.end_s)
    start_s, end_s = max(span[0], window.window_start_s), min(span[1], window.window_end_s)
    errors = [] if candidate.labels else ["no labels"]
    changes = [f"dropped alias {alias}, not in the window" for alias in candidate.object_aliases if alias not in known]
    if start_s > end_s:
        errors.append(f"{span[0]} to {span[1]} s misses the window")
    elif (start_s, end_s) != span:
        changes.append(f"{span[0]} to {span[1]} s: clipped to the window")
    aliases = [alias for alias in candidate.object_aliases if alias in known]
    return candidate.model_copy(update={"start_s": start_s, "end_s": end_s, "object_aliases": aliases}), changes, errors


def check_verdict(answer: Pass2Answer, window: SceneWindow, hypothesis: Candidate) -> list[str]:
    """Why a parsed Pass 2 answer can't be trusted, if it can't. A frame is cited by its caption."""
    shown, known = {(frame.camera, round(frame.t_s, 2)) for frame in window.frames}, _known(window)
    lower, upper = window.window_start_s - _TIME_SLACK_S, window.window_end_s + _TIME_SLACK_S
    labels = [item.label for item in answer.label_assessments]
    statuses = {item.status for item in answer.label_assessments}
    # The event takes its labels' strongest status, so a report's status and categories agree.
    expected = "CONFIRMED" if "CONFIRMED" in statuses else "UNCERTAIN" if "UNCERTAIN" in statuses else "REJECTED"
    return [
        *([] if lower <= answer.start_s <= answer.end_s <= upper else
          [f"{answer.start_s} to {answer.end_s} s is not a span inside the window"]),
        *[f"alias {alias} is not in the window" for alias in answer.object_aliases if alias not in known],
        *[f"{cited.camera} at {cited.t_s} s is not a frame in the window" for cited in answer.source_frames
          if (cited.camera, round(cited.t_s, 2)) not in shown],
        *[f"{label} is assessed more than once" for label in sorted(set(labels)) if labels.count(label) > 1],
        *[f"{label} from the hypothesis is not assessed" for label in hypothesis.labels if label not in labels],
        *([f"a {answer.status} event whose labels make it {expected}"] if answer.status != expected else []),
        *(["a CONFIRMED event cites no frame"] if answer.status == "CONFIRMED" and not answer.source_frames else []),
        *(["an UNCERTAIN event gives no reason"] if answer.status == "UNCERTAIN" and not answer.uncertainty_reasons
          else []),  # M3 reports abstentions, and their reasons
    ]
