"""Unified LLM client over Databricks serving endpoints.

Databricks exposes Foundation Model APIs and External Models through a
single OpenAI-compatible surface, so one client handles every model tier
by switching the ``model`` (endpoint) string. No provider-specific classes.

The client speaks the OpenAI Chat Completions shape, including tool-calling
with forced ``tool_choice`` so extraction output is always structured JSON.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

from .config import CONFIG


@dataclass
class LLMResponse:
    text: str
    tool_arguments: Optional[dict] = None
    input_tokens: int = 0
    output_tokens: int = 0


class LLMClient:
    """Thin wrapper around the Databricks OpenAI-compatible endpoint.

    Locally (no Databricks host) it raises on use unless a base URL and key
    are supplied, keeping with the no-silent-fallback principle: tests use
    the stub client in ``tests/`` rather than a hidden fake here.
    """

    def __init__(self, base_url: Optional[str] = None,
                 api_key: Optional[str] = None):
        self._base_url = base_url or _default_base_url()
        self._api_key = api_key or CONFIG.openai_api_key or CONFIG.anthropic_api_key
        self._client = None  # lazily created OpenAI client

    def _ensure_client(self):
        if self._client is None:
            # openai>=1.0 supports a custom base_url pointing at Databricks.
            from openai import OpenAI  # imported lazily; optional at import time

            if not self._base_url or not self._api_key:
                raise RuntimeError(
                    "LLMClient needs a base_url and api_key. Configure the "
                    "Databricks serving endpoint host and a secret; refusing "
                    "to fall back to a stub."
                )
            self._client = OpenAI(base_url=self._base_url, api_key=self._api_key)
        return self._client

    def complete(self, model_key: str, prompt: str, content: str) -> LLMResponse:
        """Plain completion, used by the query router's classification call."""
        spec = CONFIG.model(model_key)
        client = self._ensure_client()
        resp = client.chat.completions.create(
            model=spec.endpoint,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": content},
            ],
            temperature=0,
        )
        usage = resp.usage
        return LLMResponse(
            text=resp.choices[0].message.content or "",
            input_tokens=getattr(usage, "prompt_tokens", 0),
            output_tokens=getattr(usage, "completion_tokens", 0),
        )

    def extract(self, model_key: str, prompt: str, content: str,
                tool_schema: dict, image_base64: Optional[str] = None) -> LLMResponse:
        """Forced tool-use so the model returns structured JSON, not prose.

        When ``image_base64`` is provided (a cropped-and-zoomed figure region
        from ``search.figure_preprocessor``), it is sent as a multimodal image
        block alongside the text, so the model reads the chart from a legible
        crop rather than a full page. Text-only extraction is unchanged when
        it is None.

        Not every endpoint honours forced ``tool_choice`` identically (the
        FMAPI OpenAI-compatible surface for Claude/Llama has historically
        differed from External Models). We therefore parse defensively: if
        no tool call comes back, ``tool_arguments`` is ``None`` and the
        caller escalates or quarantines rather than crashing.
        """
        spec = CONFIG.model(model_key)
        client = self._ensure_client()
        tool_name = tool_schema["name"]

        if image_base64 is not None:
            user_content = [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                {"type": "text", "text": content},
            ]
        else:
            user_content = content

        resp = client.chat.completions.create(
            model=spec.endpoint,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_content},
            ],
            tools=[{"type": "function", "function": tool_schema}],
            tool_choice={"type": "function", "function": {"name": tool_name}},
            temperature=0,
        )
        msg = resp.choices[0].message
        args = _safe_tool_args(msg)
        usage = resp.usage
        return LLMResponse(
            text=msg.content or "",
            tool_arguments=args,
            input_tokens=getattr(usage, "prompt_tokens", 0),
            output_tokens=getattr(usage, "completion_tokens", 0),
        )


def _safe_tool_args(msg) -> Optional[dict]:
    """Extract and JSON-parse tool-call arguments, tolerating malformed output.

    Returns ``None`` if there is no tool call or the arguments are not valid
    JSON — never raises. This keeps a single bad model response from killing
    a whole element's extraction.
    """
    tool_calls = getattr(msg, "tool_calls", None)
    if not tool_calls:
        return None
    raw = tool_calls[0].function.arguments
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _default_base_url() -> Optional[str]:
    import os

    host = os.environ.get("DATABRICKS_HOST")
    if host:
        return f"{host.rstrip('/')}/serving-endpoints"
    return None


# Aligned with the ``Claim`` contract: every optional structural field the
# reconciler / normaliser / Gold layer relies on is requestable here, so the
# model can actually return scale, netting and as_at_date (previously inert).
EXTRACTION_TOOL_SCHEMA: dict = {
    "name": "record_claims",
    "description": "Record structured regulatory claims found in the content.",
    "parameters": {
        "type": "object",
        "properties": {
            "claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "field_name": {
                            "type": "string",
                            "description": "Metric name exactly as written.",
                        },
                        "value": {
                            "type": "string",
                            "description": "Value exactly as written, incl. sign.",
                        },
                        "unit": {
                            "type": "string",
                            "description": "e.g. '%', 'bps', 'GBP'. Empty if none.",
                        },
                        "reporting_basis": {
                            "type": "string",
                            "description": "e.g. 'consolidated', 'solo'. Empty if unstated.",
                        },
                        "netting": {
                            "type": "string",
                            "description": "e.g. 'net of deductions', 'gross'. Empty if unstated.",
                        },
                        "scale": {
                            "type": "string",
                            "description": "e.g. 'millions', 'thousands', 'ratio'. Empty if unstated.",
                        },
                        "as_at_date": {
                            "type": "string",
                            "description": "Reporting/reference date if stated (ISO-8601 preferred).",
                        },
                        "confidence": {
                            "type": "number",
                            "description": "0-1 confidence this claim is correct.",
                        },
                    },
                    "required": ["field_name", "value", "confidence"],
                },
            }
        },
        "required": ["claims"],
    },
}
