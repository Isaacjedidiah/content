"""Analyst-facing Streamlit app (custom track).

Thin UI over the pipeline: upload a submission, run extraction, review
claims with confidence and provenance. Holds no business logic — calls the
same pipeline functions as the batch job.

Run: streamlit run apps/analyst_app.py
"""
from __future__ import annotations

import os
import tempfile

import streamlit as st  # provided on the cluster / app runtime

from docextract.custom.extractor import Extractor
from docextract.custom.preprocessor import preprocess
from docextract.shared.config import Team


def run() -> None:
    st.title("the pipeline Regulatory Data — Analyst")
    entity_ref = st.text_input("Firm reference (the pipeline ref or LEI)")
    team = st.selectbox("Team", Team.values())
    report_type = st.text_input("Report type", "primary_report")
    uploaded = st.file_uploader("Submission", type=["pdf", "docx", "xlsx"])

    if uploaded and st.button("Run extraction"):
        if not entity_ref:
            st.error("Firm reference is required so extracted metrics are "
                     "attributable.")
            return

        suffix = "." + uploaded.name.rsplit(".", 1)[-1]
        path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.getbuffer())
                path = tmp.name

            elements = preprocess(path, team, report_type, entity_ref=entity_ref)
            result = Extractor().extract(elements)
        finally:
            if path and os.path.exists(path):
                os.unlink(path)  # don't leak temp files on the app runtime

        st.metric("Extraction cost (USD)", f"${result.total_cost_usd:.4f}")
        if result.quarantine:
            st.warning(f"{len(result.quarantine)} row(s) quarantined.")
        st.dataframe([
            {
                "field": c.field_name,
                "canonical": c.canonical_metric,
                "value": c.value,
                "unit": c.unit,
                "basis": c.reporting_basis,
                "confidence": round(c.confidence, 2),
                "model": c.model_used,
                "review": c.needs_review,
            }
            for c in result.claims
        ])


if __name__ == "__main__":
    run()
