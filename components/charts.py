"""Plotly charts built from real quiz data only."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from components.styles import is_light

ACCENT, OK, BAD, MUTED = "#818cf8", "#34d399", "#f87171", "#64748b"


def _style(fig: go.Figure, height: int = 300) -> go.Figure:
    light = is_light()
    fig.update_layout(
        template="plotly_white" if light else "plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=40, b=10),
        height=height,
        font=dict(family="Inter, Segoe UI, system-ui, sans-serif", color="#1b1f3b" if light else "#cfd6ea"),
        title_font=dict(size=15),
    )
    return fig


def outcome_donut(correct: int, incorrect: int, unanswered: int) -> go.Figure:
    fig = go.Figure(
        go.Pie(
            labels=["Correct", "Incorrect", "Unanswered"],
            values=[correct, incorrect, unanswered],
            hole=0.62,
            marker=dict(colors=[OK, BAD, MUTED]),
            sort=False,
            textinfo="value",
        )
    )
    fig.update_layout(title="Answer breakdown", showlegend=True, legend=dict(orientation="h", y=-0.1))
    return _style(fig)


def accuracy_bar(df: pd.DataFrame, key: str, title: str) -> go.Figure:
    fig = px.bar(
        df, x=key, y="accuracy", text=df.apply(lambda r: f"{int(r['correct'])}/{int(r['total'])}", axis=1),
        range_y=[0, 100], labels={"accuracy": "Accuracy (%)", key: key.title()},
    )
    fig.update_traces(marker_color=ACCENT, textposition="outside", cliponaxis=False)
    fig.update_layout(title=title)
    return _style(fig)


def score_trend(df: pd.DataFrame) -> go.Figure:
    """df columns: when (datetime), percent, quiz."""
    fig = px.line(df, x="when", y="percent", markers=True, hover_data=["quiz"], range_y=[0, 100],
                  labels={"percent": "Score (%)", "when": ""})
    fig.update_traces(line_color=ACCENT, marker=dict(size=9))
    fig.update_layout(title="Score over time")
    return _style(fig)
