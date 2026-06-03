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


# ── header ────────────────────────────────────────────────────────────────────

@app.callback(
    Output("header-status", "children"),
    Input("refresh-interval", "n_intervals"),
)
def update_header(_):
    df = _load_flagged()
    if df.empty:
        return "No data — run src/data_pipeline.py and src/models.py"
    n       = int(df["anomaly"].sum()) if "anomaly" in df.columns else 0
    total   = len(df)
    rate    = f"{n / total:.1%}" if total else "—"
    return f"{total:,} records loaded  ·  {n:,} anomalies ({rate})"


# ── overview KPI cards ────────────────────────────────────────────────────────

@app.callback(
    Output("kpi-total",    "children"),
    Output("kpi-critical", "children"),
    Output("kpi-drift",    "children"),
    Output("kpi-lift",     "children"),
    Input("refresh-interval", "n_intervals"),
)
def update_kpis(_):
    df   = _load_flagged()
    rep  = _load_report()
    meta = _load_model_meta()

    if df.empty:
        return "—", "—", "—", "—"

    total    = int(df["anomaly"].sum()) if "anomaly" in df.columns else 0
    critical = sum(1 for a in rep.get("sample_alerts", []) if a.get("severity") == "CRITICAL")
    sev_sum  = rep.get("severity_summary", {})
    critical = sev_sum.get("CRITICAL", critical)

    drift_data = _load_drift()
    psi_status = drift_data.get("psi_status", "N/A")
    psi_val    = drift_data.get("psi", 0)

    lift = meta.get("eval", {}).get("critical_lift", 0)

    drift_label = f"{psi_status} ({psi_val:.3f})" if drift_data else "No data"
    lift_label  = f"{lift:.1f}x" if lift else "—"

    return f"{total:,}", f"{critical:,}", drift_label, lift_label


# ── overview charts ───────────────────────────────────────────────────────────

@app.callback(
    Output("overview-hourly-chart",  "figure"),
    Output("overview-severity-pie",  "figure"),
    Output("overview-method-bar",    "figure"),
    Output("overview-precision-bar", "figure"),
    Input("refresh-interval", "n_intervals"),
)
def update_overview(_):
    rep  = _load_report()
    meta = _load_model_meta()

    # hourly anomaly rate
    hourly = rep.get("by_hour", [])
    if hourly:
        hrs    = [r["hour"] for r in hourly]
        total  = [r["total"]   for r in hourly]
        flagged = [r.get("flagged", 0) for r in hourly]
        rate   = [f / (t + 1e-6) for f, t in zip(flagged, total)]
        fig_h  = go.Figure()
        fig_h.add_bar(x=hrs, y=flagged, name="Flagged",    marker_color="#4488ff")
        fig_h.add_scatter(x=hrs, y=[r * max(total) for r in rate], name="Rate (scaled)",
                          mode="lines+markers", line=dict(color="#ff8800", width=2),
                          yaxis="y")
        fig_h.update_layout(**_LAYOUT, legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)"),
                            xaxis_title="Hour", yaxis_title="Count")
    else:
        fig_h = _empty_fig()

    # severity pie
    sev_sum = rep.get("severity_summary", {})
    if sev_sum:
        labels = list(sev_sum.keys())
        vals   = [sev_sum[k] for k in labels]
        colors = [_SEV_COLOR.get(k, "#888") for k in labels]
        fig_p  = go.Figure(go.Pie(labels=labels, values=vals, hole=0.45,
                                   marker=dict(colors=colors),
                                   textinfo="percent+label"))
        fig_p.update_layout(**_LAYOUT)
    else:
        fig_p = _empty_fig()

    # method vote bar
    mc = meta.get("eval", {}).get("method_counts", {})
    if mc:
        methods = [k for k in mc if k != "ensemble"]
        counts  = [mc[k] for k in methods]
        fig_m   = go.Figure(go.Bar(x=methods, y=counts,
                                    marker_color=["#4488ff", "#ff8800", "#44cc88", "#cc44ff"]))
        fig_m.update_layout(**_LAYOUT, yaxis_title="Flagged rows")
    else:
        fig_m = _empty_fig()

    # precision@k bar
    pak = meta.get("eval", {}).get("precision_at_k", {})
    if pak:
        ks   = list(pak.keys())
        vals = [pak[k] for k in ks]
        fig_k = go.Figure(go.Bar(x=ks, y=vals, marker_color="#44cc88",
                                  text=[f"{v:.0%}" for v in vals], textposition="outside"))
        fig_k.update_layout(**_LAYOUT, yaxis=dict(range=[0, 1.05], **_LAYOUT["yaxis"]),
                             yaxis_title="Precision")
    else:
        fig_k = _empty_fig()

    return fig_h, fig_p, fig_m, fig_k
