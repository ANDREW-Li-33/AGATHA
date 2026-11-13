"""Parameterized structured search; status is always a gate, including in OR mode."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..vlm.schema import CATEGORIES, Label, Status


class SearchFilters(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    statuses: list[Status] = ["CONFIRMED"]
    categories: list[Literal[CATEGORIES]] = []
    labels: list[Label] = []
    label_mode: Literal["any", "all"] = "any"
    object_classes: list[Literal["vehicle", "pedestrian", "cyclist", "sign", "unknown"]] = []
    min_duration_s: float | None = Field(default=None, ge=0)
    max_duration_s: float | None = Field(default=None, ge=0)
    combine: Literal["AND", "OR"] = "AND"
    include_duplicates: bool = False

    @model_validator(mode="after")
    def ordered(self):
        if (self.min_duration_s is not None and self.max_duration_s is not None
                and self.min_duration_s > self.max_duration_s):
            raise ValueError("minimum duration exceeds maximum")
        return self


def where_clause(filters: SearchFilters) -> tuple[str, list]:
    clauses, params = ["i.verification_status = ANY(%s)"], [filters.statuses]
    if not filters.include_duplicates:
        clauses.append("i.duplicate_of IS NULL")
    options = []
    # Kept labels on confirmed/uncertain events; assessed labels on rejected events.
    # A rejected secondary label must not make a confirmed event match a label it rejected.
    visible = "(l.status <> 'REJECTED' OR i.verification_status = 'REJECTED')"
    if filters.categories:
        options.append("EXISTS (SELECT 1 FROM public.incident_labels l WHERE l.incident_id=i.id "
                       f"AND {visible} AND l.category = ANY(%s))")
        params.append(filters.categories)
    if filters.labels:
        if filters.label_mode == "any":
            options.append("EXISTS (SELECT 1 FROM public.incident_labels l WHERE l.incident_id=i.id "
                           f"AND {visible} AND l.label = ANY(%s))")
        else:
            options.append("NOT EXISTS (SELECT 1 FROM unnest(%s::text[]) wanted(label) WHERE NOT EXISTS "
                           "(SELECT 1 FROM public.incident_labels l WHERE l.incident_id=i.id "
                           f"AND {visible} AND l.label=wanted.label))")
        params.append(filters.labels)
    if filters.object_classes:
        options.append("EXISTS (SELECT 1 FROM public.incident_objects o WHERE o.incident_id=i.id "
                       "AND o.object_classes && %s::text[])")
        params.append(filters.object_classes)
    duration = []
    if filters.min_duration_s is not None:
        duration.append("i.duration_s >= %s")
        params.append(filters.min_duration_s)
    if filters.max_duration_s is not None:
        duration.append("i.duration_s <= %s")
        params.append(filters.max_duration_s)
    if duration:
        options.append("(" + " AND ".join(duration) + ")")
    if options:
        clauses.append("(" + f" {filters.combine} ".join(options) + ")")
    return " AND ".join(clauses), params
