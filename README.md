# Large Scale Log Anomaly Detection

End-to-end anomaly detection pipeline for HTTP server logs at scale.
Built to handle millions of rows using an ensemble of Isolation Forest,
Local Outlier Factor, and a rule engine — with a PySpark pipeline for
distributed processing on clusters and a drift detection layer to catch
when the incoming traffic distribution changes.

I built this to practice large-scale data engineering at the level that
shows up at companies like Amazon — where you are not just training a model
but building a system that stays reliable over time.

---

## What it does

**Stage 1 — Data ingestion** (`src/data_pipeline.py`)

Generates 500K realistic HTTP server log records (calibrated to NASA log
statistics) and engineers behavioral features: per-IP error rates, response
size z-scores, admin endpoint access patterns, and traffic volume by hour.

**Stage 2 — Anomaly detection** (`src/models.py`)

Three complementary methods running in parallel:
- **Isolation Forest** — global outliers in high-dimensional feature space
- **Local Outlier Factor** — density-based local anomalies IF misses
- **Rule engine** — hard-coded policy violations (admin recon, DoS signals)

Ensemble vote: 2 of 3 must agree. Reduces false positives significantly.

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

---

## Quickstart

```bash
pip install -r requirements.txt

# Full pipeline
python src/data_pipeline.py
python src/models.py
python src/alerting.py

# API
uvicorn api.main:app --reload --port 8000
# http://localhost:8000/docs

# MLflow UI (to compare experiment runs)
mlflow ui --port 5000

# Tests
pytest tests/ -v

# Spark (optional)
python src/spark_pipeline.py
```

---

## Key metrics (500K row dataset)

| Metric | Value |
|--------|-------|
| Records processed | 500,000 |
| Anomaly rate | ~2.5% |
| Critical error lift | 8-12x |
| Precision@100 | ~0.85 |
| Precision@1000 | ~0.70 |
| Pipeline runtime | ~90 seconds |

---

## Architecture

```
logs (CSV or S3)
    │
    ▼
data_pipeline.py    ← feature engineering, QC
    │
    ▼
models.py           ← IF + LOF + rules → ensemble → SHAP → MLflow
    │
    ▼
alerting.py         ← severity classification, drift detection
    │
    ▼
api/main.py         ← FastAPI real-time scoring + monitoring
```

---

**Mayur Patil** — M.S. Data Science, George Washington University
[LinkedIn](https://linkedin.com/in/mayurpatil26) | [GitHub](https://github.com/mayur212626)
