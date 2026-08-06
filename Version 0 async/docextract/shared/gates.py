"""Responsible AI governance gates.

Enforces the three CDO risk-appetite decisions as code:
1. External outputs require named human sign-off before release.
2. Production AI must be validated by 2nd-line before deployment.
3. POCs/MVPs may proceed without supervision until production-ready
   (but may never be released externally).

Aligned to EU AI Act / NIST AI RMF / ISO 42001 / OECD principles.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class Stage(str, Enum):
    POC = "poc"
    PRODUCTION = "production"


def _valid_iso8601(ts: str) -> bool:
    try:
        datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return True
    except (ValueError, AttributeError):
        return False


@dataclass
class SignOff:
    approver: str
    role: str
    timestamp: str

    def __post_init__(self):
        if not self.approver:
            raise ValueError("SignOff.approver is required")
        if not _valid_iso8601(self.timestamp):
            raise ValueError(f"SignOff.timestamp must be ISO-8601: {self.timestamp!r}")


@dataclass
class SecondLineValidation:
    validator: str
    passed: bool
    timestamp: str

    def __post_init__(self):
        if not _valid_iso8601(self.timestamp):
            raise ValueError(
                f"SecondLineValidation.timestamp must be ISO-8601: {self.timestamp!r}")


def gate_external_release(payload: dict, signoff: SignOff | None,
                          stage: Stage) -> dict:
    if stage == Stage.POC:
        if payload.get("external"):
            raise PermissionError("POC output cannot be released externally.")
        return {"released": False, "reason": "POC internal only"}

    if payload.get("external") and signoff is None:
        raise PermissionError(
            "External release requires named human sign-off "
            "(risk-appetite rule 1)."
        )
    return {"released": True,
            "signed_by": signoff.approver if signoff else None}


def gate_production_deployment(validation: SecondLineValidation | None) -> dict:
    if validation is None or not validation.passed:
        raise PermissionError(
            "Production deployment requires passing 2nd-line AI Governance "
            "validation (risk-appetite rule 2)."
        )
    return {"deploy": True, "validated_by": validation.validator}
