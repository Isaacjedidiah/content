"""Test doubles.

The production ``LLMClient`` refuses to run without real credentials (no
silent fallback). Tests inject these stubs instead of a hidden fake.
"""
from __future__ import annotations

from docextract.shared.llm_client import LLMResponse


class StubLLMClient:
    """Scripted LLM client.

    ``extract_script`` maps a substring found in the element content to the
    tool_arguments dict to return; ``complete_script`` maps a substring in the
    user content to the text to return. Falls back to a default.
    """

    def __init__(self, extract_script: dict | None = None,
                 complete_script: dict | None = None,
                 default_claims: list | None = None):
        self._extract = extract_script or {}
        self._complete = complete_script or {}
        self._default_claims = default_claims if default_claims is not None else []

    async def extract(self, model_key, prompt, content, tool_schema,
                image_base64=None) -> LLMResponse:
        claims = self._default_claims
        for needle, payload in self._extract.items():
            if needle in content:
                claims = payload
                break
        return LLMResponse(text="", tool_arguments={"claims": claims},
                           input_tokens=100, output_tokens=50)

    async def complete(self, model_key, prompt, content) -> LLMResponse:
        text = ""
        for needle, payload in self._complete.items():
            if needle in content:
                text = payload
                break
        return LLMResponse(text=text, input_tokens=20, output_tokens=5)


class FakeSearchStore:
    """In-memory stand-in for AzureAISearchStore."""

    def __init__(self):
        self.docs: dict[str, dict] = {}

    def ensure_index(self):
        pass

    def index(self, chunk):
        self.index_many([chunk])

    def index_many(self, chunks):
        for c in chunks:
            self.docs[c.chunk_id] = {
                "text": c.content, "entity_ref": c.entity_ref,
                "chunk_type": c.chunk_type.value,
                "source_document_id": c.source_document_id,
            }
        return len(chunks)

    def search(self, query, top_k=5, team_filter=None, entity_ref=None):
        out = []
        for cid, d in self.docs.items():
            if entity_ref and d["entity_ref"] != entity_ref:
                continue
            out.append({"text": d["text"], "metadata": d, "score": 1.0})
        return out[:top_k]
