# Large Scale Log Anomaly Detection

End-to-end anomaly detection pipeline for HTTP server logs at scale.
Built to handle millions of rows using a four-model ensemble — Isolation Forest,
Local Outlier Factor, a hard-coded rule engine, and an LSTM Autoencoder for
temporal behavioral analysis — plus a PySpark pipeline for distributed processing,
a FastAPI scoring service, and a Plotly Dash monitoring dashboard.

**🔗 Live:** [Interactive demo](https://mjpatil.com/#anomaly) · [Scoring API + docs](https://anomaly-detection-z5fp.onrender.com/docs) · [Health](https://anomaly-detection-z5fp.onrender.com/health) · [Portfolio](https://mjpatil.com)

> The scoring API is deployed on Render (Docker, slim serving image — Isolation
> Forest + scaler only). Free tier sleeps when idle, so the first request after a
> pause can take ~50s to wake. Try the [interactive demo](https://mjpatil.com/#anomaly)
> on the portfolio — pick an attack scenario (normal / DoS flood / admin recon)
> and score it live.

---

## What it does

**Stage 1 — Data ingestion** (`src/data_pipeline.py`)

Generates 500K realistic HTTP server log records (calibrated to NASA log
statistics) and engineers behavioral features: per-IP error rates, response
size z-scores, admin endpoint access patterns, and traffic volume by hour.

**Stage 2 — Anomaly detection** (`src/models.py`)

Four complementary methods running in parallel:

- **Isolation Forest** — global outliers in high-dimensional feature space
- **Local Outlier Factor** — density-based local anomalies IF misses
- **Rule engine** — hard-coded policy violations (admin recon, DoS signals)
- **LSTM Autoencoder** — temporal behavioral modeling: learns what a normal
  IP's request sequence looks like, then flags IPs whose recent history
  deviates (catches slow brute force, data exfiltration, business-hour mimicry)

Ensemble vote: 3 of 4 must agree, reducing false positives significantly.
SHAP values explain which features drove each anomaly score.
MLflow tracks every experiment run for comparison.
Precision@K (50, 100, 500, 1000) measures how useful the top alerts actually are.

**Stage 3 — Distributed processing** (`src/spark_pipeline.py`)

PySpark version using window functions and approxQuantile for scale.
Outputs partitioned Parquet (S3-ready). Tested at 10M rows locally (~4 min)
and on 3-node EMR cluster (~45 seconds).

**Stage 4 — Alerting + Drift detection** (`src/alerting.py`)

Severity classification (CRITICAL/HIGH/MEDIUM/LOW), top-offending IP reports,
hourly and endpoint breakdowns. KS test + PSI-based drift detection flags when
the incoming traffic distribution shifts from what the model was trained on.

**Stage 5 — Monitoring dashboard** (`dashboard/`)

Plotly Dash dashboard with six tabs:
- **Overview** — KPI cards (total anomalies, critical count, drift status, lift), hourly chart
- **Anomaly Feed** — filterable table with severity badges, sortable by score
- **IP Analysis** — top offenders bar chart with click-to-drill-down behavioral profile
- **Traffic Patterns** — hourly total vs flagged, endpoint breakdown, status distribution
- **Model Performance** — score distributions, LSTM reconstruction error histogram, metadata
- **Drift Monitor** — PSI gauge, KS test results table, first-vs-second-half comparison

---

## Quickstart

```bash
pip install -r requirements.txt

# Full pipeline
python src/data_pipeline.py
python src/models.py
python src/alerting.py

# Interactive dashboard (open http://localhost:8050)
python dashboard/run.py

# Or launch the API instead
uvicorn api.main:app --reload --port 8000
# http://localhost:8000/docs

# MLflow experiment tracking
mlflow ui --port 5000

# Tests
pytest tests/ -v

# Spark (optional, for large-scale)
python src/spark_pipeline.py
```

---

## LSTM Autoencoder

The LSTM Autoencoder adds temporal pattern detection that the statistical models
cannot provide. It groups each IP's request history into chronological sequences
of `seq_len=10` timesteps and learns to reconstruct normal patterns. At inference,
IPs with high reconstruction error are flagged — they deviated from their
established behavioral baseline.

**Architecture:**
- Input: `(batch, 10, 20)` — 10 timesteps × 20 feature dimensions
- Encoder: 2-layer LSTM → 64-dimensional context vector
- Decoder: 2-layer LSTM → reconstructed sequence
- Loss: MSE between original and reconstructed sequence
- Threshold: 97.5th percentile of training reconstruction errors

**What it catches that the other models miss:**
- Slow brute force (1 failed attempt per minute over 6 hours — low error rate at any snapshot)
- Gradual data exfiltration (response bytes creeping upward across requests)
- Business-hour mimicry (attacker adapts traffic pattern to look like normal users)

---

## Key metrics (500K row dataset)

| Metric | Value |
|--------|-------|
| Records processed | 500,000 |
| Anomaly rate | ~2.5% |
| Critical error lift | 8-12x |
| Precision@100 | ~0.85 |
| Precision@1000 | ~0.70 |
| Pipeline runtime (pandas) | ~90 seconds |
| Spark — 10M rows, 3-node EMR | ~45 seconds |

---

## Architecture

```
logs (CSV or S3)
    │
    ▼
data_pipeline.py    ← feature engineering, QC checks
    │
    ▼
models.py           ← IF + LOF + Rules + LSTM → 3-of-4 ensemble → SHAP → MLflow
    │
    ▼
alerting.py         ← severity classification, KS + PSI drift detection
    │
    ├──▶ api/main.py         ← FastAPI real-time scoring + monitoring endpoints
    │
    └──▶ dashboard/          ← Plotly Dash monitoring dashboard (6 tabs)
```

---

**Mayur Patil** — M.S. Data Science, George Washington University
[LinkedIn](https://linkedin.com/in/mayurpatil26) | [GitHub](https://github.com/mayur212626)
