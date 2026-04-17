# src/alerting.py
# Turns flagged anomalies into actionable reports + drift detection.
# KS test + PSI catches when traffic distribution shifts from training data.
# In production: push to SNS, PagerDuty, or Slack.
import pandas as pd, numpy as np, json, logging, os, sys
from datetime import datetime
from scipy import stats as sp_stats
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.loader import cfg

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")


def load():
    return pd.read_csv("data/logs_flagged.csv")


def assign_severity(row):
    if row["is_critical"] == 1 and row["is_heavy"] == 1:
        return "CRITICAL"
    elif row["is_5xx"] == 1 or (row["is_admin"] == 1 and
                                  row["ip_n_requests"] > cfg.alerting.critical_request_threshold):
        return "HIGH"
    elif row["anomaly_score"] >= 2:
        return "MEDIUM"
    return "LOW"


def drift_check(df):
    """
    KS test between first and second half of data.
    In production this compares today vs a rolling 7-day baseline.
    PSI > 0.2 = major shift, > 0.1 = moderate, < 0.1 = stable.
    """
    log.info("Drift detection (KS test + PSI)...")
    n   = len(df)
    a, b = df.iloc[:n//2], df.iloc[n//2:]
    cols = ["ip_error_rate", "bytes", "ip_n_requests", "anomaly_score"]
    ks_results = {}
    for col in cols:
        stat, p = sp_stats.ks_2samp(a[col].dropna(), b[col].dropna())
        ks_results[col] = {"ks_stat": round(float(stat), 4),
                           "p_value": round(float(p), 4),
                           "drift":   p < 0.05}

    def psi(x, y, bins=10):
        edges = np.percentile(np.concatenate([x, y]), np.linspace(0, 100, bins+1))
        edges[0], edges[-1] = -np.inf, np.inf
        ap = np.histogram(x, edges)[0] / len(x) + 1e-8
        bp = np.histogram(y, edges)[0] / len(y) + 1e-8
        return float(np.sum((ap - bp) * np.log(ap / bp)))

    psi_val  = psi(a["anomaly_score"].values, b["anomaly_score"].values)
    detected = any(r["drift"] for r in ks_results.values()) or psi_val > 0.2
    result   = {
        "timestamp":    datetime.utcnow().isoformat(),
        "psi":          round(psi_val, 4),
        "psi_status":   "MAJOR_SHIFT" if psi_val > 0.2 else "MODERATE" if psi_val > 0.1 else "STABLE",
        "ks_tests":     ks_results,
        "drift":        detected,
        "action":       "Retrain model" if detected else "No action needed",
    }
    log.info(f"  PSI={psi_val:.4f} ({result['psi_status']}) | drift={detected}")
    return result


def build_alerts(df):
    anomalies = df[df["anomaly"] == 1].copy()
    anomalies["severity"] = anomalies.apply(assign_severity, axis=1)
    priority = anomalies[anomalies["severity"].isin(["CRITICAL", "HIGH"])]
    alerts = []
    for _, row in priority.head(cfg.alerting.max_alerts).iterrows():
        alerts.append({
            "ts":           datetime.utcnow().isoformat(),
            "severity":     row["severity"],
            "ip":           row["ip"],
            "status":       int(row["status"]),
            "endpoint":     row["endpoint"],
            "score":        int(row["anomaly_score"]),
            "ip_error_rate": round(float(row["ip_error_rate"]), 4),
            "action":       "BLOCK_IP" if row["severity"] == "CRITICAL" else "INVESTIGATE",
        })
    sev = anomalies["severity"].value_counts().to_dict()
    log.info(f"Severity: {sev}")
    return alerts, sev, anomalies


def save_all(ip_rpt, hourly, endpoints, alerts, sev, drift):
    os.makedirs("docs", exist_ok=True)
    os.makedirs("monitoring", exist_ok=True)
    report = {"generated_at": datetime.utcnow().isoformat(),
              "severity_summary": sev, "drift_report": drift,
              "top_ips": ip_rpt, "by_hour": hourly,
              "by_endpoint": endpoints, "sample_alerts": alerts[:5]}
    with open("docs/anomaly_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    with open("monitoring/alerts.json", "w") as f:
        json.dump(alerts, f, indent=2, default=str)
    with open("monitoring/drift_report.json", "w") as f:
        json.dump(drift, f, indent=2, default=str)
    log.info(f"Saved report ({len(alerts)} priority alerts)")


def run():
    log.info("=" * 55)
    log.info("ALERTING + DRIFT DETECTION")
    log.info("=" * 55)
    df                          = load()
    drift                       = drift_check(df)
    alerts, sev, anomalies      = build_alerts(df)
    ip_rpt = anomalies.groupby("ip").agg(
        flagged=("anomaly","count"), err_rate=("is_error","mean"),
        crit_rate=("is_critical","mean")).sort_values("flagged",ascending=False).head(20).round(4).to_dict(orient="index")
    total     = df.groupby("hour").size().rename("total")
    flagged   = anomalies.groupby("hour").size().rename("flagged")
    hourly    = pd.concat([total, flagged], axis=1).fillna(0).reset_index().to_dict(orient="records")
    endpoints = anomalies.groupby("endpoint").agg(
        count=("anomaly","count"), err_rate=("is_error","mean")).sort_values("count",ascending=False).round(4).to_dict(orient="index")
    save_all(ip_rpt, hourly, endpoints, alerts, sev, drift)
    log.info(f"{len(alerts)} alerts | drift: {'YES' if drift['drift'] else 'no'}")
    log.info("Done.\n")
    return alerts, drift


if __name__ == "__main__":
    run()
# ks test
# psi drift
