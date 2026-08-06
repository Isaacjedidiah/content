"""Operations dashboard (port 8502).

Separate Streamlit app from the analyst UI. Reads the monitoring data layer
and renders throughput, cost trend and review backlog.

Run: streamlit run apps/monitoring_app.py --server.port 8502
"""
from __future__ import annotations


def build_dashboard_view(runs: list[dict]) -> dict:
    docs = sum(r.get("documents", 0) for r in runs)
    cost = sum(r.get("cost", 0.0) for r in runs)
    return {
        "total_documents": docs,
        "total_cost_usd": round(cost, 4),
        "review_backlog": sum(r.get("needs_review", 0) for r in runs),
        "quarantined": sum(r.get("quarantined", 0) for r in runs),
        "avg_cost_per_doc": round(cost / max(docs, 1), 5),
        "error_rate": round(
            sum(r.get("errors", 0) for r in runs) / max(len(runs), 1), 2),
    }


def run() -> None:
    import streamlit as st

    from docextract.shared.monitoring import PipelineMonitor

    st.title("the pipeline Regulatory Data — Operations")
    view = build_dashboard_view(PipelineMonitor().load_runs())
    c1, c2, c3 = st.columns(3)
    c1.metric("Documents", view["total_documents"])
    c2.metric("Cost (USD)", f"${view['total_cost_usd']}")
    c3.metric("Review backlog", view["review_backlog"])
    c1.metric("Quarantined", view["quarantined"])
    c2.metric("Avg cost/doc", f"${view['avg_cost_per_doc']}")
    c3.metric("Error rate", view["error_rate"])


if __name__ == "__main__":
    run()
