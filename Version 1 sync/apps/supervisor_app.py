"""Supervisor query app — conversational, with memory, audit logging, feedback.

Natural-language questions -> routed answers via QueryRouter. A per-session
SupervisorSession gives the chat memory (so follow-ups like "will firm A
survive from that figure?" resolve against the prior turn), logs every Q&A to
the audit trail, and captures thumbs up/down feedback per answer.

Run: streamlit run apps/supervisor_app.py
"""
from __future__ import annotations

import streamlit as st

from docextract.search.ai_search_store import AzureAISearchStore
from docextract.shared.config import Team
from docextract.shared.llm_router import QueryRouter
from docextract.shared.supervisor_session import SupervisorSession


def run(router: QueryRouter | None = None) -> None:
    router = router or QueryRouter(search_store=AzureAISearchStore())
    st.title("the pipeline Regulatory Data — Supervisor")

    # Per-session memory: one SupervisorSession per Streamlit session, so the
    # conversation has context and every turn is logged under one session id.
    if "session" not in st.session_state:
        st.session_state.session = SupervisorSession()
    session = st.session_state.session
    if "history" not in st.session_state:
        st.session_state.history = []   # list of rendered turns for display

    team = st.selectbox("Your team", Team.values())
    report_type = st.text_input("Report type (for grounded answers)", "primary_report")
    question = st.text_area("Ask a question")

    if question and st.button("Ask"):
        answer = router.answer(question, team_filter=team,
                               report_type=report_type, session=session)
        st.session_state.history.append(answer)

    # Render the conversation so far (memory is visible to the user).
    for answer in st.session_state.history:
        st.markdown("---")
        if answer.get("resolved_question") and \
                answer["resolved_question"] != answer.get("raw_question"):
            st.caption(f"Interpreted as: {answer['resolved_question']}")
        st.write(f"**Route:** {answer['route']}")
        if answer.get("answer"):
            st.write(answer["answer"])
        if answer.get("sql"):
            st.caption("Generated SQL")
            st.code(answer["sql"], language="sql")
        if answer.get("sql_result"):
            st.dataframe(answer["sql_result"])
        if answer.get("narrative"):
            st.caption("Sources")
            for hit in answer["narrative"]:
                src = hit["metadata"].get("source_document_id", "")
                st.write(f"- {hit['text']}  ({src})")

        # Feedback controls, attached to this turn.
        tid = answer.get("turn_id")
        if tid:
            c1, c2, _ = st.columns([1, 1, 6])
            if c1.button("👍 Helpful", key=f"up_{tid}"):
                session.add_feedback(tid, "up")
                st.success("Thanks — logged.")
            if c2.button("👎 Not helpful", key=f"down_{tid}"):
                session.add_feedback(tid, "down")
                st.info("Logged — this helps improve answers.")


if __name__ == "__main__":
    run()
