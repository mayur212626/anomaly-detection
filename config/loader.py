import os
from types import SimpleNamespace


def _to_ns(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in d.items()})
    return d


def load_config():
    try:
        import yaml
        for p in ["config/settings.yaml",
                  os.path.join(os.path.dirname(__file__), "settings.yaml")]:
            if os.path.exists(p):
                with open(p) as f:
                    return _to_ns(yaml.safe_load(f))
    except ImportError:
        pass

    return _to_ns({
        "data":     {"n_records": 500000, "seed": 42},
        "features": {"zscore_threshold": 3.5, "heavy_ip_percentile": 0.99,
                     "admin_recon_threshold": 5000},
        "models":   {"isolation_forest": {"n_estimators": 200,
                                           "contamination": 0.025,
                                           "bootstrap": True,
                                           "random_state": 42},
                     "lof":              {"n_neighbors": 20,
                                          "contamination": 0.025},
                     "ensemble":         {"min_votes": 2,
                                          "lift_threshold": 3.0}},
        "spark":    {"local_memory": "4g",
                     "shuffle_partitions_local": 8,
                     "shuffle_partitions_cluster": 200},
        "api":      {"port": 8000, "log_predictions": True},
        "alerting": {"max_alerts": 100,
                     "critical_ip_error_threshold": 0.50,
                     "critical_request_threshold": 5000},
    })


cfg = load_config()
