"""Structural classification and compatibility checking for metrics.

Two jobs:
  1. Classify a metric's structural SHAPE from its raw name — measure type
     (level/rate/change/ratio) and reporting period (point-in-time, YoY, QoQ,
     etc.) — independent of what it's called. Name-matching alone can't tell
     that one firm's "growth" is a YoY percentage and another's an absolute
     change; two values can resolve to the same canonical name yet not be
     comparable. Rule-based and lightweight, never an LLM call; ambiguous
     inputs classify as UNKNOWN rather than guess. (Adopted from the uploaded
     metric_normaliser.)
  2. Check whether two claims sharing a canonical metric are STRUCTURALLY
     compatible (unit, basis, scale, measure_type, period). UNKNOWN on either
     side is treated as compatible — this catches CONFIRMED mismatches, not
     under-classified fields. Mismatches become searchable conflict chunks.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from .schema import Claim, ChunkType, RegulatoryChunk, content_hash


class MeasureType(str, Enum):
    LEVEL = "level"
    RATE = "rate"
    CHANGE = "change"
    RATIO = "ratio"
    UNKNOWN = "unknown"


class Period(str, Enum):
    POINT_IN_TIME = "point_in_time"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"
    YEAR_ON_YEAR = "year_on_year"
    QUARTER_ON_QUARTER = "quarter_on_quarter"
    YEAR_TO_DATE = "year_to_date"
    UNKNOWN = "unknown"


_CHANGE_KW = re.compile(r"\b(increase|decrease|change|movement|delta|growth|up|down|rose|fell)\b", re.IGNORECASE)
_RATE_KW = re.compile(r"%|ratio|percent|rate\b", re.IGNORECASE)
_YOY_KW = re.compile(r"\byoy\b|year[\s-]?on[\s-]?year", re.IGNORECASE)
_QOQ_KW = re.compile(r"\bqoq\b|quarter[\s-]?on[\s-]?quarter", re.IGNORECASE)
_YTD_KW = re.compile(r"\bytd\b|year[\s-]?to[\s-]?date", re.IGNORECASE)
_QUARTERLY_KW = re.compile(r"\bq[1-4]\b|quarterly", re.IGNORECASE)
_ANNUAL_KW = re.compile(r"\bannual\b|full[\s-]?year|\bfy\b", re.IGNORECASE)


@dataclass
class Classification:
    measure_type: MeasureType
    period: Period
    confidence: float


def classify_metric(raw_field_name: str, value: Any = None) -> Classification:
    """Classify measure_type + period from the raw name (and value hint)."""
    name = raw_field_name or ""
    if _CHANGE_KW.search(name):
        mt, mt_c = MeasureType.CHANGE, 0.75
    elif _RATE_KW.search(name):
        mt, mt_c = MeasureType.RATE, 0.85
    elif isinstance(value, str) and "%" in value:
        mt, mt_c = MeasureType.RATE, 0.80
    elif isinstance(value, (int, float)):
        mt, mt_c = MeasureType.LEVEL, 0.60
    else:
        mt, mt_c = MeasureType.UNKNOWN, 0.30

    if _YOY_KW.search(name):
        p, p_c = Period.YEAR_ON_YEAR, 0.90
    elif _QOQ_KW.search(name):
        p, p_c = Period.QUARTER_ON_QUARTER, 0.90
    elif _YTD_KW.search(name):
        p, p_c = Period.YEAR_TO_DATE, 0.90
    elif _QUARTERLY_KW.search(name):
        p, p_c = Period.QUARTERLY, 0.70
    elif _ANNUAL_KW.search(name):
        p, p_c = Period.ANNUAL, 0.70
    else:
        p, p_c = Period.POINT_IN_TIME, 0.40

    return Classification(mt, p, round((mt_c + p_c) / 2, 3))


@dataclass
class NormResult:
    compatible: bool
    reason: str = ""


def _mismatch(a: Optional[str], b: Optional[str], unknown: str = None) -> bool:
    """True only when both sides are known AND differ."""
    if a is None or b is None:
        return False
    if unknown is not None and (a == unknown or b == unknown):
        return False
    return a != b


def check_compatible(a: Claim, b: Claim) -> NormResult:
    if _mismatch(a.unit, b.unit):
        return NormResult(False, f"unit mismatch: {a.unit} vs {b.unit}")
    if _mismatch(a.reporting_basis, b.reporting_basis):
        return NormResult(False, f"basis mismatch: {a.reporting_basis} vs {b.reporting_basis}")
    if _mismatch(a.scale, b.scale):
        return NormResult(False, f"scale mismatch: {a.scale} vs {b.scale}")
    if _mismatch(a.measure_type, b.measure_type, MeasureType.UNKNOWN.value):
        return NormResult(False, f"measure_type mismatch: {a.measure_type} vs {b.measure_type}")
    if _mismatch(a.period, b.period, Period.UNKNOWN.value):
        return NormResult(False, f"period mismatch: {a.period} vs {b.period}")
    return NormResult(True)


def to_conflict_chunk(metric: str, a: Claim, b: Claim,
                      r: NormResult, entity_ref: str,
                      source_document_id: str) -> RegulatoryChunk:
    text = (
        f"For {metric}, values were structurally incompatible: {r.reason}. "
        f"Value A={a.value}{a.unit or ''}, B={b.value}{b.unit or ''}."
    )
    return RegulatoryChunk(
        chunk_id=content_hash(text),
        chunk_type=ChunkType.NORMALISATION_CONFLICT,
        content=text,
        entity_ref=entity_ref,
        source_document_id=source_document_id,
        content_hash=content_hash(text),
    )
