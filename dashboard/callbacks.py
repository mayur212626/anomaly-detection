# dashboard/callbacks.py
import json
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from dash import Input, Output, html
import dash_bootstrap_components as dbc

from dashboard.app import app

# ── shared plotly theme ──────────────────────────────────────────────────────
_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#c9d1d9", size=11),
    margin=dict(l=40, r=20, t=20, b=40),
    xaxis=dict(gridcolor="#30363d", zeroline=False),
    yaxis=dict(gridcolor="#30363d", zeroline=False),
)

_SEV_COLOR = {
    "CRITICAL": "#ff4444",
    "HIGH":     "#ff8800",
    "MEDIUM":   "#e6b800",
    "LOW":      "#4488ff",
}


# ── data loaders ─────────────────────────────────────────────────────────────

def _load_flagged():
    path = "data/logs_flagged.csv"
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def _load_report():
    path = "docs/anomaly_report.json"
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _load_drift():
    path = "monitoring/drift_report.json"
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _load_model_meta():
    path = "docs/model_meta.json"
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _load_lstm_meta():
    path = "docs/lstm_meta.json"
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _empty_fig(msg="Run the pipeline first"):
    fig = go.Figure()
    fig.add_annotation(text=msg, xref="paper", yref="paper",
                       x=0.5, y=0.5, showarrow=False, font=dict(color="#8b949e", size=13))
    fig.update_layout(**_LAYOUT)
    return fig
