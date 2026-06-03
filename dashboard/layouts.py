# dashboard/layouts.py
# Pure layout definitions — no callbacks, no data loading here.
import dash_bootstrap_components as dbc
from dash import html, dcc


SEVERITY_COLORS = {
    "CRITICAL": "#ff4444",
    "HIGH":     "#ff8800",
    "MEDIUM":   "#ffdd00",
    "LOW":      "#44aaff",
}


def _header():
    return dbc.Navbar(
        dbc.Container([
            dbc.NavbarBrand(
                [html.I(className="me-2"), "Anomaly Detection  |  HTTP Log Intelligence"],
                className="fw-bold fs-5",
            ),
            html.Div(
                id="header-status",
                className="text-muted small",
                children="Loading...",
            ),
        ], fluid=True),
        color="dark",
        dark=True,
        className="mb-0 border-bottom border-secondary",
    )


def _kpi_card(card_id, title, icon=""):
    return dbc.Card([
        dbc.CardBody([
            html.P(title, className="text-muted small mb-1"),
            html.H3(id=card_id, className="mb-0 fw-bold"),
            html.Small(icon, className="text-muted"),
        ])
    ], className="kpi-card h-100")


def _overview_tab():
    return dbc.Container([
        dbc.Row([
            dbc.Col(_kpi_card("kpi-total",    "Total Anomalies",   "all methods"), md=3),
            dbc.Col(_kpi_card("kpi-critical", "Critical Alerts",   "BLOCK_IP"),    md=3),
            dbc.Col(_kpi_card("kpi-drift",    "Drift Status",      "PSI score"),   md=3),
            dbc.Col(_kpi_card("kpi-lift",     "Critical Lift",     "vs baseline"), md=3),
        ], className="g-3 mb-4"),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Anomaly Rate by Hour", className="fw-semibold"),
                    dbc.CardBody(dcc.Graph(id="overview-hourly-chart", style={"height": "280px"},
                                          config={"displayModeBar": False})),
                ])
            ], md=8),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Severity Breakdown", className="fw-semibold"),
                    dbc.CardBody(dcc.Graph(id="overview-severity-pie", style={"height": "280px"},
                                          config={"displayModeBar": False})),
                ])
            ], md=4),
        ], className="g-3 mb-4"),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Model Vote Breakdown", className="fw-semibold"),
                    dbc.CardBody(dcc.Graph(id="overview-method-bar", style={"height": "240px"},
                                          config={"displayModeBar": False})),
                ])
            ], md=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Precision @ K", className="fw-semibold"),
                    dbc.CardBody(dcc.Graph(id="overview-precision-bar", style={"height": "240px"},
                                          config={"displayModeBar": False})),
                ])
            ], md=6),
        ], className="g-3"),
    ], fluid=True, className="pt-3")


def _feed_tab():
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                dbc.InputGroup([
                    dbc.InputGroupText("Severity"),
                    dbc.Select(
                        id="feed-severity-filter",
                        options=[
                            {"label": "All",      "value": "ALL"},
                            {"label": "Critical", "value": "CRITICAL"},
                            {"label": "High",     "value": "HIGH"},
                            {"label": "Medium",   "value": "MEDIUM"},
                            {"label": "Low",      "value": "LOW"},
                        ],
                        value="ALL",
                    ),
                ], className="me-3"),
            ], md="auto"),
            dbc.Col([
                dbc.InputGroup([
                    dbc.InputGroupText("Rows"),
                    dbc.Select(
                        id="feed-page-size",
                        options=[{"label": str(n), "value": n} for n in [25, 50, 100, 250]],
                        value=50,
                    ),
                ]),
            ], md="auto"),
            dbc.Col(html.Div(id="feed-count", className="text-muted small pt-2")),
        ], className="mb-3 align-items-center"),
        dbc.Card(
            dbc.CardBody(html.Div(id="feed-table")),
            className="border-secondary",
        ),
    ], fluid=True, className="pt-3")
