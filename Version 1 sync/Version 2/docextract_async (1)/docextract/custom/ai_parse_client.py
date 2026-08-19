"""OCR fallback via the ``databricks-ai-parse`` serving endpoint.

The custom track parses PDFs with pdfplumber (fast, no OCR). Pages that are
scanned / image-only have no text layer, so pdfplumber returns nothing. For
those pages this module calls the SAME ``ai_parse_document`` capability the
native track uses — the ``databricks-ai-parse`` serving endpoint — so OCR
behaviour is identical across both tracks and there is a single parser to
reason about.

Design, consistent with the rest of the package:
  * No silent fallback. If the endpoint isn't reachable, the caller keeps the
    flagged-empty element (so the page still goes to quarantine) rather than
    pretending it parsed.
  * Lazy import of the SDK so the module imports anywhere.
  * The endpoint name comes from the native track's config so both tracks
    stay in lockstep.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Optional

from ..native.ai_functions_config import AI_CFG
from ..shared.config import CONFIG


@dataclass
class ParsedBlock:
    """One block of OCR'd content from a page (text or a table)."""
    content: str
    is_table: bool = False


class AIParseClient:
    """Thin wrapper over the ai_parse_document serving endpoint.

    ``available`` is a cheap guard the preprocessor checks before attempting a
    per-page OCR call, so we don't try to reach an endpoint that isn't
    configured (e.g. local dev without Databricks).
    """

    def __init__(self, endpoint: Optional[str] = None,
                 base_url: Optional[str] = None,
                 api_key: Optional[str] = None):
        self._endpoint = endpoint or AI_CFG.parse_endpoint
        self._base_url = base_url or _default_base_url()
        self._api_key = api_key or CONFIG.openai_api_key or CONFIG.anthropic_api_key
        self._client = None

    @property
    def available(self) -> bool:
        return bool(self._base_url and self._api_key)

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI  # lazy; OpenAI-compatible serving surface

            if not self.available:
                raise RuntimeError(
                    "AIParseClient needs a Databricks base_url and key; "
                    "refusing to fall back to a stub."
                )
            self._client = OpenAI(base_url=self._base_url, api_key=self._api_key)
        return self._client

    def parse_page_image(self, image_bytes: bytes,
                         mime: str = "image/png") -> list[ParsedBlock]:
        """OCR a single rendered page image into ordered blocks.

        Returns an empty list if the endpoint yields nothing usable; the
        caller treats that the same as "still unparsed" and keeps the page in
        quarantine. Never raises on empty/garbled output — only on a genuine
        transport/config error surfaced by the SDK.
        """
        client = self._ensure_client()
        b64 = base64.b64encode(image_bytes).decode("ascii")
        resp = client.chat.completions.create(
            model=self._endpoint,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text",
                     "text": "Parse this document page. Return the readable "
                             "text, and render any tables as markdown."},
                    {"type": "image_url",
                     "image_url": {"url": f"data:{mime};base64,{b64}"}},
                ],
            }],
            temperature=0,
        )
        text = (resp.choices[0].message.content or "").strip()
        if not text:
            return []
        return _split_parsed(text)


def _split_parsed(text: str) -> list[ParsedBlock]:
    """Split parsed output into blocks, tagging markdown tables as tables.

    A block is treated as a table if it contains a markdown table separator
    row (a line of pipes and dashes). Everything else is text. This keeps the
    downstream modality tagging consistent with the pdfplumber path.
    """
    blocks: list[ParsedBlock] = []
    for raw in text.split("\n\n"):
        chunk = raw.strip()
        if not chunk:
            continue
        is_table = any(_is_table_sep(line) for line in chunk.splitlines())
        blocks.append(ParsedBlock(content=chunk, is_table=is_table))
    return blocks


def _is_table_sep(line: str) -> bool:
    s = line.strip()
    return bool(s) and set(s) <= {"|", "-", ":", " "} and "-" in s and "|" in s


def _default_base_url() -> Optional[str]:
    import os

    host = os.environ.get("DATABRICKS_HOST")
    if host:
        return f"{host.rstrip('/')}/serving-endpoints"
    return None
