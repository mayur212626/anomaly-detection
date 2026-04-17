from types import SimpleNamespace
cfg = SimpleNamespace(
    data=SimpleNamespace(n_records=500000, seed=42),
    features=SimpleNamespace(zscore_threshold=3.5, heavy_ip_percentile=0.99, admin_recon_threshold=5000),
    models=SimpleNamespace(
        isolation_forest=SimpleNamespace(n_estimators=200, contamination=0.025, bootstrap=True, random_state=42),
        lof=SimpleNamespace(n_neighbors=20, contamination=0.025),
        ensemble=SimpleNamespace(min_votes=2, lift_threshold=3.0)
    ),
    spark=SimpleNamespace(local_memory="4g", shuffle_partitions_local=8, shuffle_partitions_cluster=200),
    api=SimpleNamespace(port=8000, log_predictions=True),
    alerting=SimpleNamespace(max_alerts=100, critical_ip_error_threshold=0.50, critical_request_threshold=5000),
)# api/main.py
# FastAPI for real-time anomaly scoring with background prediction logging.
# /score: single entry, <5ms target latency
# /score/batch: bulk scoring
# /alerts, /report, /drift: from alerting.py outputs
# /monitoring/summary: live stats on scored traffic for drift monitoring
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List
import joblib, numpy as np, pandas as pd
import json, logging, os, sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from types import SimpleNamespace`ncfg = SimpleNamespace(data=SimpleNamespace(n_records=500000,seed=42),features=SimpleNamespace(zscore_threshold=3.5,heavy_ip_percentile=0.99,admin_recon_threshold=5000),models=SimpleNamespace(isolation_forest=SimpleNamespace(n_estimators=200,contamination=0.025,bootstrap=True,random_state=42),lof=SimpleNamespace(n_neighbors=20,contamination=0.025),ensemble=SimpleNamespace(min_votes=2,lift_threshold=3.0)),spark=SimpleNamespace(local_memory="4g",shuffle_partitions_local=8,shuffle_partitions_cluster=200),api=SimpleNamespace(port=8000,log_predictions=True),alerting=SimpleNamespace(max_alerts=100,critical_ip_error_threshold=0.50,critical_request_threshold=5000))

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")

app = FastAPI(
    title="Log Anomaly Detection API",
    description="Real-time scoring: Isolation Forest ensemble on 500K HTTP logs. See /docs.",
    version="2.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_model = _scaler = _features = None


@app.on_event("startup")
async def load():
    global _model, _scaler, _features
    try:
        _model    = joblib.load("models/isolation_forest.pkl")
        _scaler   = joblib.load("models/scaler.pkl")
        _features = json.load(open("data/features.json"))
        log.info(f"Model loaded — {len(_features)} features")
    except FileNotFoundError as e:
        log.error(f"Model not found ({e}). Run pipeline first.")


class LogEntry(BaseModel):
    ip: str; hour: int = Field(..., ge=0, le=23)
    status: int = Field(..., ge=100, le=599)
    bytes: int = Field(..., ge=0)
    is_error: int = Field(0, ge=0, le=1); is_critical: int = Field(0, ge=0, le=1)
    is_4xx: int = Field(0, ge=0, le=1);   is_5xx: int = Field(0, ge=0, le=1)
    is_admin: int = Field(0, ge=0, le=1); is_empty: int = Field(0, ge=0, le=1)
    is_large: int = Field(0, ge=0, le=1); is_heavy: int = Field(0, ge=0, le=1)
    ip_n_requests: int = Field(1, ge=1);  ip_error_rate: float = Field(0.0, ge=0, le=1)
    ip_crit_rate: float = Field(0.0, ge=0, le=1)
    ip_avg_bytes: float = Field(0.0, ge=0)
    ip_admin_hits: int = Field(0, ge=0);  ip_empty_rate: float = Field(0.0, ge=0, le=1)
    hour_avg_rps: float = Field(0.0, ge=0); bytes_z: float = 0.0
    dos_signal: int = Field(0, ge=0, le=1); admin_recon: int = Field(0, ge=0, le=1)
    class Config:
        schema_extra = {"example": {
            "ip":"10.0.1.100","hour":14,"status":500,"bytes":0,
            "is_error":1,"is_critical":1,"is_4xx":0,"is_5xx":1,
            "is_admin":0,"is_empty":1,"is_large":0,"is_heavy":1,
            "ip_n_requests":15000,"ip_error_rate":0.45,"ip_crit_rate":0.20,
            "ip_avg_bytes":128,"ip_admin_hits":0,"ip_empty_rate":0.30,
            "hour_avg_rps":6000,"bytes_z":3.8,"dos_signal":1,"admin_recon":0}}


class BatchRequest(BaseModel):
    records: List[LogEntry]


def _sev(e: LogEntry, is_anom: int):
    if e.is_critical and e.is_heavy:                                      return "CRITICAL"
    if e.is_5xx or (e.is_admin and e.ip_n_requests > cfg.alerting.critical_request_threshold): return "HIGH"
    if is_anom:                                                            return "MEDIUM"
    return "NORMAL"


def _score(entry: LogEntry):
    if _model is None:
        raise HTTPException(503, "Model not loaded")
    X        = pd.DataFrame([entry.dict()])[_features]
    X_scaled = _scaler.transform(X.fillna(0))
    score    = float(_model.decision_function(X_scaled)[0])
    is_anom  = int(_model.predict(X_scaled)[0] == -1)
    return {"anomaly": is_anom, "score": round(score, 4), "severity": _sev(entry, is_anom)}


def _log_pred(d, r):
    if not cfg.api.log_predictions: return
    os.makedirs("monitoring", exist_ok=True)
    with open("monitoring/scored.jsonl", "a") as f:
        f.write(json.dumps({"ts": datetime.utcnow().isoformat(),
                             "ip": d.get("ip"), "status": d.get("status"), **r}) + "\n")


@app.get("/health")
def health():
    return {"status": "ok" if _model else "degraded",
            "model_loaded": _model is not None,
            "n_features": len(_features) if _features else 0,
            "ts": datetime.utcnow().isoformat()}


@app.get("/ready")
def ready():
    if not _model: raise HTTPException(503, "Not ready")
    return {"ready": True}


@app.post("/score")
def score(entry: LogEntry, bg: BackgroundTasks):
    r = _score(entry)
    bg.add_task(_log_pred, entry.dict(), r)
    return {**r, "ts": datetime.utcnow().isoformat()}


@app.post("/score/batch")
def score_batch(req: BatchRequest):
    results = [_score(e) for e in req.records]
    n = sum(r["anomaly"] for r in results)
    return {"results": results, "n_scored": len(results),
            "n_anomalies": n, "anomaly_rate": round(n/len(results), 4),
            "ts": datetime.utcnow().isoformat()}


@app.get("/alerts")
def alerts():
    if not os.path.exists("monitoring/alerts.json"):
        return {"message": "Run src/alerting.py first.", "alerts": []}
    data = json.load(open("monitoring/alerts.json"))
    return {"total": len(data),
            "critical": sum(1 for a in data if a["severity"]=="CRITICAL"),
            "high":     sum(1 for a in data if a["severity"]=="HIGH"),
            "recent":   data[:10]}


@app.get("/report")
def report():
    try: return json.load(open("docs/anomaly_report.json"))
    except FileNotFoundError: raise HTTPException(404, "Run src/alerting.py first.")


@app.get("/drift")
def drift():
    try: return json.load(open("monitoring/drift_report.json"))
    except FileNotFoundError: raise HTTPException(404, "Run src/alerting.py first.")


@app.get("/stats")
def stats():
    try: return json.load(open("docs/model_meta.json"))
    except FileNotFoundError: raise HTTPException(404, "Run src/models.py first.")


@app.get("/monitoring/summary")
def summary():
    if not os.path.exists("monitoring/scored.jsonl"):
        return {"message": "No scored traffic yet."}
    recs   = [json.loads(l) for l in open("monitoring/scored.jsonl") if l.strip()]
    scores = [r["score"] for r in recs]
    return {"n_scored":     len(recs),
            "anomaly_rate": round(sum(r["anomaly"] for r in recs)/len(recs), 4),
            "mean_score":   round(sum(scores)/len(scores), 4),
            "severity_dist": {s: sum(1 for r in recs if r.get("severity")==s)
                              for s in ["CRITICAL","HIGH","MEDIUM","NORMAL"]},
            "last_seen":    recs[-1]["ts"]}
# background logging
# monitoring summary


