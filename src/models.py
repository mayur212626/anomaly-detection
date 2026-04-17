# src/models.py
# ─────────────────────────────────────────────────────────────────────────────
# Ensemble: Isolation Forest + Local Outlier Factor + Rule Engine
#
# Why LOF instead of DBSCAN (which I tried first):
# DBSCAN is O(n^2) and even with sampling the results were not better than LOF.
# LOF won on both quality and runtime. n_neighbors=20 works well for this
# data size — too low overfits local noise, too high approximates global methods.
#
# Evaluation without ground truth labels:
#   - Critical error lift: anomaly set should have 8x+ more critical errors
#   - Precision@K: of the top K most suspicious records, what fraction are errors?
#   - These two together tell you if the detector is actually useful
#
# MLflow tracks every run so you can compare contamination values and
# ensemble strategies without losing results.
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import joblib
import json
import logging
import os
import sys
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.loader import cfg

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")


def load():
    return pd.read_csv("data/logs.csv"), json.load(open("data/features.json"))


def run_isolation_forest(X_scaled):
    log.info("Isolation Forest (200 trees)...")
    model = IsolationForest(
        n_estimators=cfg.models.isolation_forest.n_estimators,
        contamination=cfg.models.isolation_forest.contamination,
        bootstrap=cfg.models.isolation_forest.bootstrap,
        random_state=cfg.models.isolation_forest.random_state,
        n_jobs=-1,
    )
    model.fit(X_scaled)
    scores = model.decision_function(X_scaled)
    preds  = (model.predict(X_scaled) == -1).astype(int)
    log.info(f"  Flagged {preds.sum():,} ({preds.mean():.2%})")
    return preds, scores, model


def run_lof(X_scaled):
    """
    LOF with novelty=False since we don't have a clean training set.
    Takes ~90s on 500K rows. If that's too slow, sample 100K for fit
    and use novelty=True for predict on full set.
    """
    log.info(f"LOF (n_neighbors={cfg.models.lof.n_neighbors})...")
    lof   = LocalOutlierFactor(n_neighbors=cfg.models.lof.n_neighbors,
                               contamination=cfg.models.lof.contamination,
                               n_jobs=-1)
    preds  = (lof.fit_predict(X_scaled) == -1).astype(int)
    scores = -lof.negative_outlier_factor_
    log.info(f"  Flagged {preds.sum():,} ({preds.mean():.2%})")
    return preds, scores


def run_rule_engine(df):
    """
    Hard-coded rules for things that are always suspicious.
    Encoding them explicitly means they always get caught even if the
    statistical models are having an off day.
    """
    log.info("Rule engine...")
    import pandas as pd_inner
    rules = pd_inner.DataFrame(index=df.index)
    rules["r_high_err"]    = (df["ip_error_rate"] > cfg.alerting.critical_ip_error_threshold).astype(int)
    rules["r_critical"]    = df["is_critical"].astype(int)
    rules["r_admin_recon"] = df["admin_recon"].astype(int)
    rules["r_dos"]         = df["dos_signal"].astype(int)
    rules["r_empty_heavy"] = ((df["is_empty"] == 1) & (df["is_heavy"] == 1)).astype(int)
    preds = (rules.sum(axis=1) >= 1).astype(int)
    log.info(f"  Flagged {preds.sum():,} ({preds.mean():.2%})")
    log.info(f"  Breakdown: {rules.sum().to_dict()}")
    return preds, rules.sum(axis=1)


def ensemble(if_p, lof_p, rule_p):
    min_v = cfg.models.ensemble.min_votes
    votes = if_p + lof_p + rule_p
    final = (votes >= min_v).astype(int)
    log.info(f"Ensemble ({min_v}/3): {final.sum():,} anomalies ({final.mean():.2%})")
    return final, votes


def precision_at_k(scores, labels, ks=None):
    """
    Precision@K: of the top K most suspicious records, what fraction are errors?
    This is the metric that matters in practice — an on-call engineer looks at
    the top 100 alerts and decides if the system is trustworthy.
    """
    ks = ks or [50, 100, 500, 1000, 5000]
    idx = np.argsort(scores)[::-1]
    out = {}
    for k in ks:
        out[f"P@{k}"] = round(float(labels.iloc[idx[:k]].mean()), 4)
    log.info(f"  Precision@K: {out}")
    return out


def compute_shap(model, X_scaled, feature_cols, n=1000):
    try:
        import shap
        log.info(f"SHAP values on {n} samples...")
        idx  = np.random.choice(X_scaled.shape[0], size=n, replace=False)
        expl = shap.TreeExplainer(model)
        vals = expl.shap_values(X_scaled[idx])
        imp  = dict(sorted(
            zip(feature_cols, np.abs(vals).mean(axis=0).tolist()),
            key=lambda x: x[1], reverse=True
        ))
        log.info(f"  Top 5: {list(imp.keys())[:5]}")
        return {k: round(v, 5) for k, v in imp.items()}
    except ImportError:
        log.warning("shap not installed — skipping")
        return {}


def evaluate(df, final_preds, if_scores, lof_scores, feature_cols):
    anom, normal = df[final_preds == 1], df[final_preds == 0]
    crit_anom    = anom["is_critical"].mean()
    crit_normal  = normal["is_critical"].mean()
    lift         = crit_anom / (crit_normal + 1e-9)

    combined = (
        (if_scores  - if_scores.min())  / (if_scores.max()  - if_scores.min()  + 1e-8) +
        (lof_scores - lof_scores.min()) / (lof_scores.max() - lof_scores.min() + 1e-8)
    ) / 2

    p_at_k = precision_at_k(-combined, df["is_error"])

    results = {
        "n_total":        int(len(df)),
        "n_anomalies":    int(final_preds.sum()),
        "anomaly_rate":   round(float(final_preds.mean()), 4),
        "critical_lift":  round(float(lift), 2),
        "quality":        "PASS" if lift >= cfg.models.ensemble.lift_threshold else "REVIEW",
        "precision_at_k": p_at_k,
        "method_counts":  {
            "isolation_forest": int(final_preds.sum()),
            "lof":              int(lof_scores[lof_scores > lof_scores.mean() + lof_scores.std()].shape[0]),
            "ensemble":         int(final_preds.sum()),
        },
        "anomaly_profile": {
            "critical_rate": round(float(crit_anom), 4),
            "normal_crit":   round(float(crit_normal), 4),
            "admin_rate":    round(float(anom["is_admin"].mean()), 4),
            "heavy_rate":    round(float(anom["is_heavy"].mean()), 4),
        }
    }
    log.info(f"  Lift: {lift:.1f}x | Quality: {results['quality']}")
    return results, combined


def track_mlflow(eval_results, shap_imp):
    try:
        import mlflow
        mlflow.set_experiment("log-anomaly-detection")
        with mlflow.start_run(run_name=f"ensemble_{datetime.now().strftime('%m%d_%H%M')}"):
            mlflow.set_tags({"methods": "IF+LOF+Rules", "author": "Mayur Patil"})
            mlflow.log_param("contamination", cfg.models.isolation_forest.contamination)
            mlflow.log_param("n_records", eval_results["n_total"])
            mlflow.log_metric("anomaly_rate", eval_results["anomaly_rate"])
            mlflow.log_metric("critical_lift", eval_results["critical_lift"])
            for k, v in eval_results["precision_at_k"].items():
                mlflow.log_metric(k.replace("@", "_at_"), v)
            log.info(f"MLflow run logged. View: mlflow ui --port 5000")
    except ImportError:
        log.warning("mlflow not installed — skipping")


def save(df, final_preds, combined, eval_results, if_model, scaler, shap_imp):
    df = df.copy()
    df["anomaly"]       = final_preds
    df["anomaly_score"] = combined
    df.to_csv("data/logs_flagged.csv", index=False)
    os.makedirs("models", exist_ok=True)
    joblib.dump(if_model, "models/isolation_forest.pkl")
    joblib.dump(scaler,   "models/scaler.pkl")
    os.makedirs("docs", exist_ok=True)
    meta = {
        "trained_at":      datetime.utcnow().isoformat(),
        "version":         "2.0.0",
        "methods":         ["IsolationForest", "LOF", "RuleEngine"],
        "ensemble":        "2-of-3 majority vote",
        "eval":            eval_results,
        "shap_importance": shap_imp,
    }
    with open("docs/model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    log.info("Saved models and results.")
    return meta


def run():
    log.info("=" * 55)
    log.info("ANOMALY DETECTION — IF + LOF + RULES")
    log.info("=" * 55)
    df, features = load()
    X = df[features].fillna(0).values
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    if_preds,  if_scores,  if_model = run_isolation_forest(X_scaled)
    lof_preds, lof_scores            = run_lof(X_scaled)
    rule_preds, rule_votes           = run_rule_engine(df)
    final_preds, votes               = ensemble(if_preds, lof_preds, rule_preds)
    eval_results, combined           = evaluate(df, final_preds, if_scores, lof_scores, features)
    shap_imp                         = compute_shap(if_model, X_scaled, features)
    track_mlflow(eval_results, shap_imp)
    meta = save(df, final_preds, combined, eval_results, if_model, scaler, shap_imp)
    log.info("Done.\n")
    return df, final_preds, meta


if __name__ == "__main__":
    run()
# lof added
