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


# ── anomaly feed ──────────────────────────────────────────────────────────────

def _severity_badge(sev):
    color = {"CRITICAL": "danger", "HIGH": "warning", "MEDIUM": "info", "LOW": "primary"}.get(sev, "secondary")
    return dbc.Badge(sev, color=color, className="me-1")


@app.callback(
    Output("feed-table", "children"),
    Output("feed-count", "children"),
    Input("feed-severity-filter", "value"),
    Input("feed-page-size",       "value"),
    Input("refresh-interval",     "n_intervals"),
)
def update_feed(severity, page_size, _):
    df = _load_flagged()
    if df.empty or "anomaly" not in df.columns:
        return html.P("No data yet. Run the full pipeline first.", className="text-muted"), ""

    anom = df[df["anomaly"] == 1].copy()
    if "severity" not in anom.columns:
        sev_map = {
            "CRITICAL": (anom.get("is_critical", 0) == 1) & (anom.get("is_heavy", 0) == 1),
            "HIGH":     (anom.get("is_5xx", 0) == 1),
        }
        anom["severity"] = "LOW"
        anom.loc[sev_map.get("HIGH", False), "severity"] = "HIGH"
        anom.loc[sev_map.get("CRITICAL", False), "severity"] = "CRITICAL"

    if severity != "ALL":
        anom = anom[anom["severity"] == severity]

    anom = anom.sort_values("anomaly_score", ascending=False).head(int(page_size))
    count_label = f"Showing {len(anom):,} anomalies"

    cols = ["ip", "severity", "status", "endpoint", "anomaly_score", "ip_error_rate", "ip_n_requests"]
    cols = [c for c in cols if c in anom.columns]
    rows = []
    for _, r in anom.iterrows():
        cells = []
        for c in cols:
            val = r[c]
            if c == "severity":
                cells.append(html.Td(_severity_badge(str(val))))
            elif isinstance(val, float):
                cells.append(html.Td(f"{val:.3f}"))
            else:
                cells.append(html.Td(str(val)))
        rows.append(html.Tr(cells))

    header = html.Thead(html.Tr([html.Th(c.replace("_", " ").title()) for c in cols]))
    table  = dbc.Table([header, html.Tbody(rows)],
                       striped=True, bordered=False, hover=True,
                       dark=True, responsive=True, size="sm",
                       className="feed-table mb-0")
    return table, count_label


# ── ip analysis ───────────────────────────────────────────────────────────────

@app.callback(
    Output("ip-top-bar", "figure"),
    Output("ip-scatter",  "figure"),
    Input("refresh-interval", "n_intervals"),
)
def update_ip_charts(_):
    df = _load_flagged()
    if df.empty or "anomaly" not in df.columns:
        return _empty_fig(), _empty_fig()

    anom = df[df["anomaly"] == 1]

    # top 20 IPs by flagged count
    top = (anom.groupby("ip")
               .agg(flagged=("anomaly", "count"),
                    err_rate=("ip_error_rate", "mean"),
                    score=("anomaly_score", "mean"))
               .sort_values("flagged", ascending=False)
               .head(20)
               .reset_index())

    colors = [_SEV_COLOR["CRITICAL"] if e > 0.5 else
              _SEV_COLOR["HIGH"]     if e > 0.3 else
              _SEV_COLOR["LOW"]
              for e in top["err_rate"]]

    fig_bar = go.Figure(go.Bar(
        x=top["flagged"], y=top["ip"],
        orientation="h", marker_color=colors,
        customdata=top[["err_rate", "score"]].values,
        hovertemplate="<b>%{y}</b><br>Flagged: %{x}<br>Err rate: %{customdata[0]:.1%}<extra></extra>",
    ))
    fig_bar.update_layout(**_LAYOUT, xaxis_title="Flagged requests",
                          yaxis=dict(autorange="reversed", **_LAYOUT["yaxis"]))

    # scatter: error rate vs request volume, bubble = anomaly score
    ip_agg = (df.groupby("ip")
                .agg(n_req=("ip_n_requests", "first"),
                     err_rate=("ip_error_rate", "first"),
                     score=("anomaly_score", "mean"),
                     is_anom=("anomaly", "max"))
                .reset_index())
    ip_agg = ip_agg.sample(min(3000, len(ip_agg)), random_state=42)

    normal = ip_agg[ip_agg["is_anom"] == 0]
    flagged = ip_agg[ip_agg["is_anom"] == 1]

    fig_s = go.Figure()
    fig_s.add_scatter(
        x=normal["n_req"], y=normal["err_rate"],
        mode="markers", name="Normal",
        marker=dict(color="#30363d", size=5, opacity=0.6),
    )
    fig_s.add_scatter(
        x=flagged["n_req"], y=flagged["err_rate"],
        mode="markers", name="Anomaly",
        marker=dict(
            color=flagged["score"],
            colorscale="Reds",
            size=(flagged["score"] * 18 + 5).clip(5, 20),
            opacity=0.8,
            showscale=True,
            colorbar=dict(title="Score", len=0.6),
        ),
        text=flagged["ip"],
        hovertemplate="<b>%{text}</b><br>Requests: %{x:,}<br>Error rate: %{y:.1%}<extra></extra>",
    )
    fig_s.update_layout(**_LAYOUT, xaxis_title="Total requests (IP)", yaxis_title="Error rate",
                        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)"))

    return fig_bar, fig_s


@app.callback(
    Output("ip-drilldown", "children"),
    Input("ip-top-bar", "clickData"),
)
def update_ip_drilldown(click):
    if click is None:
        return html.P("Click any bar to see this IP's full behavioral profile.",
                      className="text-muted small")

    ip = click["points"][0]["y"]
    df = _load_flagged()
    if df.empty:
        return html.P("No data.", className="text-muted small")

    row = df[df["ip"] == ip].iloc[0]
    fields = [
        ("Total Requests",    f"{int(row.get('ip_n_requests', 0)):,}"),
        ("Error Rate",        f"{row.get('ip_error_rate', 0):.1%}"),
        ("Critical Rate",     f"{row.get('ip_crit_rate', 0):.1%}"),
        ("Admin Hits",        f"{int(row.get('ip_admin_hits', 0)):,}"),
        ("Avg Response Size", f"{row.get('ip_avg_bytes', 0):,.0f} bytes"),
        ("Empty Response %",  f"{row.get('ip_empty_rate', 0):.1%}"),
        ("Anomaly Score",     f"{row.get('anomaly_score', 0):.3f}"),
        ("DoS Signal",        "YES" if row.get("dos_signal", 0) else "no"),
        ("Admin Recon",       "YES" if row.get("admin_recon", 0) else "no"),
    ]
    rows = [
        html.Tr([html.Td(k, className="text-muted small pe-3"), html.Td(v, className="fw-semibold")])
        for k, v in fields
    ]
    return html.Div([
        html.P(ip, className="fw-bold mb-2 font-monospace"),
        dbc.Table([html.Tbody(rows)], borderless=True, size="sm", dark=True),
    ])


# ── traffic patterns ──────────────────────────────────────────────────────────

@app.callback(
    Output("traffic-hourly-line",  "figure"),
    Output("traffic-flag-rate",    "figure"),
    Output("traffic-endpoint-bar", "figure"),
    Output("traffic-status-pie",   "figure"),
    Input("refresh-interval", "n_intervals"),
)
def update_traffic(_):
    df  = _load_flagged()
    rep = _load_report()
    if df.empty:
        return _empty_fig(), _empty_fig(), _empty_fig(), _empty_fig()

    hourly = rep.get("by_hour", [])
    if hourly:
        hrs    = [r["hour"]          for r in hourly]
        total  = [r["total"]         for r in hourly]
        flagged = [r.get("flagged", 0) for r in hourly]
        rate   = [f / (t + 1e-6) for f, t in zip(flagged, total)]

        fig_line = go.Figure()
        fig_line.add_scatter(x=hrs, y=total, name="Total",
                             line=dict(color="#4488ff", width=2), mode="lines+markers")
        fig_line.add_scatter(x=hrs, y=flagged, name="Flagged",
                             line=dict(color="#ff4444", width=2), mode="lines+markers")
        fig_line.update_layout(**_LAYOUT, xaxis_title="Hour of day", yaxis_title="Request count",
                               legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)"))

        fig_rate = go.Figure(go.Bar(x=hrs, y=rate,
                                    marker_color=[
                                        "#ff4444" if r > 0.10 else
                                        "#ff8800" if r > 0.05 else "#4488ff"
                                        for r in rate
                                    ],
                                    text=[f"{r:.1%}" for r in rate],
                                    textposition="outside"))
        fig_rate.update_layout(**_LAYOUT, xaxis_title="Hour", yaxis_title="Flagged rate",
                               yaxis=dict(tickformat=".0%", **_LAYOUT["yaxis"]))
    else:
        fig_line = fig_rate = _empty_fig()

    endpoints = rep.get("by_endpoint", {})
    if endpoints:
        eps = list(endpoints.keys())[:15]
        cnts = [endpoints[e]["count"] for e in eps]
        fig_ep = go.Figure(go.Bar(x=cnts, y=eps, orientation="h",
                                   marker_color="#44cc88"))
        fig_ep.update_layout(**_LAYOUT, xaxis_title="Anomaly count",
                             yaxis=dict(autorange="reversed", **_LAYOUT["yaxis"]))
    else:
        fig_ep = _empty_fig()

    if "anomaly" in df.columns and "status" in df.columns:
        anom_status = df[df["anomaly"] == 1]["status"].value_counts()
        fig_st = go.Figure(go.Pie(
            labels=anom_status.index.astype(str),
            values=anom_status.values,
            hole=0.4,
        ))
        fig_st.update_layout(**_LAYOUT)
    else:
        fig_st = _empty_fig()

    return fig_line, fig_rate, fig_ep, fig_st
