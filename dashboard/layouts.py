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
