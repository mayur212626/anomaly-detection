# dashboard/app.py
# ─────────────────────────────────────────────────────────────────────────────
# Plotly Dash dashboard for the anomaly detection pipeline.
#
# Design: dark theme (Cyborg), single-page with tab navigation.
# Data is read from pipeline output files so the dashboard stays stateless —
# no database needed. Refresh interval polls for new file writes.
#
# Run: python dashboard/app.py
#      then open http://localhost:8050
# ─────────────────────────────────────────────────────────────────────────────
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dash
import dash_bootstrap_components as dbc

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    suppress_callback_exceptions=True,
    title="Anomaly Detection",
    update_title=None,
)
server = app.server  # gunicorn entry point

# Layout and callbacks registered after app object is created
from dashboard.layouts import build_layout   # noqa: E402
from dashboard import callbacks              # noqa: E402, F401

app.layout = build_layout()

if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=8050)
