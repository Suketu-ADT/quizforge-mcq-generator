"""Analytics across all stored quiz attempts (real data only)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from components.charts import accuracy_bar, score_trend
from components.context import get_repo, navigate
from components.metrics import empty_state, fmt_percent, metric_row, page_header
from services.quiz_service import aggregate_details, format_duration, performance_by, recommendations


def render() -> None:
    page_header("Analytics", "How you are doing across every quiz attempt stored on this computer.")
    attempts = get_repo().list_attempts()
    if not attempts:
        empty_state("No quiz attempts yet", "Take a quiz and your charts will appear here.")
        if st.button("Take a quiz", type="primary"):
            navigate("Take a Quiz")
        return

    df = pd.DataFrame(
        {
            "when": [pd.to_datetime(a.submitted_at).tz_convert(None) for a in attempts],
            "quiz": [a.quiz_title for a in attempts],
            "percent": [a.percent for a in attempts],
            "correct": [a.correct for a in attempts],
            "total": [a.total for a in attempts],
            "duration": [a.duration_s for a in attempts],
        }
    ).sort_values("when")
    total_q = int(df["total"].sum())
    total_correct = int(df["correct"].sum())
    metric_row(
        [
            ("Attempts", str(len(df)), f"{df['quiz'].nunique()} different quiz(es)"),
            ("Average score", fmt_percent(df["percent"].mean()), "mean of attempts"),
            ("Best score", fmt_percent(df["percent"].max()), ""),
            ("Overall accuracy", fmt_percent(100 * total_correct / total_q if total_q else None), f"{total_correct}/{total_q} answers"),
        ]
    )
    st.write("")
    st.plotly_chart(score_trend(df), use_container_width=True)
    details = aggregate_details(attempts)
    c1, c2 = st.columns(2)
    by_topic = performance_by(details, "topic")
    by_diff = performance_by(details, "difficulty")
    c1.plotly_chart(accuracy_bar(by_topic, "topic", "Accuracy by topic"), use_container_width=True)
    c2.plotly_chart(accuracy_bar(by_diff, "difficulty", "Accuracy by difficulty"), use_container_width=True)

    st.markdown("#### Recommendations")
    for tip in recommendations(details):
        st.markdown(f"- {tip}")

    st.markdown("#### Attempt history")
    table = df.sort_values("when", ascending=False).assign(
        when=lambda d: d["when"].dt.strftime("%Y-%m-%d %H:%M"),
        score=lambda d: d["correct"].astype(str) + " / " + d["total"].astype(str),
        time=lambda d: d["duration"].map(format_duration),
    )[["when", "quiz", "score", "percent", "time"]]
    st.dataframe(table.rename(columns={"when": "Date", "quiz": "Quiz", "score": "Score", "percent": "Percent", "time": "Time"}),
                 hide_index=True, use_container_width=True)
