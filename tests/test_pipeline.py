# tests/test_pipeline.py
# Unit tests for data pipeline and anomaly detection logic.
# Running these in CI ensures nothing silently breaks between commits.

import pytest
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data_pipeline import generate_logs, build_features, qc


@pytest.fixture(scope="module")
def raw_logs():
    return generate_logs(n=20_000, seed=7)


@pytest.fixture(scope="module")
def featured(raw_logs):
    df, feats = build_features(raw_logs)
    return df, feats


# ── Data generation ───────────────────────────────────────────────────────────

class TestGenerateLogs:
    def test_correct_row_count(self, raw_logs):
        assert len(raw_logs) == 20_000

    def test_all_columns_present(self, raw_logs):
        for col in ["ip", "hour", "day", "method", "endpoint", "status", "bytes"]:
            assert col in raw_logs.columns

    def test_hour_valid_range(self, raw_logs):
        assert raw_logs["hour"].between(0, 23).all()

    def test_bytes_non_negative(self, raw_logs):
        assert (raw_logs["bytes"] >= 0).all()

    def test_error_rate_realistic(self, raw_logs):
        errors = {400, 401, 403, 404, 500, 502, 503, 504}
        rate   = raw_logs["status"].isin(errors).mean()
        assert 0.01 <= rate <= 0.15, f"Unexpected error rate: {rate:.2%}"

    def test_critical_rate_below_error_rate(self, raw_logs):
        errors   = {400, 401, 403, 404, 500, 502, 503, 504}
        critical = {500, 502, 503, 504}
        assert raw_logs["status"].isin(critical).mean() < raw_logs["status"].isin(errors).mean()

    def test_power_law_distribution(self, raw_logs):
        counts   = raw_logs["ip"].value_counts()
        top_1pct = counts.head(max(1, len(counts) // 100)).sum()
        assert top_1pct / len(raw_logs) >= 0.10, "IP distribution not power-law"

    def test_reproducible_seed(self):
        a = generate_logs(n=500, seed=42)
        b = generate_logs(n=500, seed=42)
        assert a["status"].tolist() == b["status"].tolist()

    def test_different_seeds_differ(self):
        a = generate_logs(n=500, seed=1)
        b = generate_logs(n=500, seed=2)
        assert a["status"].tolist() != b["status"].tolist()


# ── Feature engineering ───────────────────────────────────────────────────────

class TestBuildFeatures:
    def test_all_features_present(self, featured):
        df, feats = featured
        for f in feats:
            assert f in df.columns, f"Missing: {f}"

    def test_binary_flags(self, featured):
        df, _ = featured
        for col in ["is_error", "is_critical", "is_4xx", "is_5xx",
                    "is_admin", "is_empty", "is_large", "is_heavy",
                    "dos_signal", "admin_recon"]:
            assert df[col].isin([0, 1]).all(), f"{col} not binary"

    def test_ip_error_rate_bounded(self, featured):
        df, _ = featured
        assert (df["ip_error_rate"] >= 0).all()
        assert (df["ip_error_rate"] <= 1).all()

    def test_ip_request_count_positive(self, featured):
        df, _ = featured
        assert (df["ip_n_requests"] >= 1).all()

    def test_critical_implies_error(self, featured):
        df, _ = featured
        assert df.loc[df["is_critical"] == 1, "is_error"].all()

    def test_no_nulls_in_features(self, featured):
        df, feats = featured
        assert df[feats].isnull().sum().sum() == 0

    def test_heavy_ip_is_top_1pct(self, featured):
        df, _ = featured
        heavy_min = df[df["is_heavy"] == 1]["ip_n_requests"].min()
        p99       = df["ip_n_requests"].quantile(0.99)
        assert heavy_min >= p99 * 0.95  # allow small floating point tolerance

    def test_bytes_z_standardized(self, featured):
        df, _ = featured
        # z-scores should have mean ~0 and std ~1
        assert abs(df["bytes_z"].mean()) < 0.1
        assert 0.8 < df["bytes_z"].std() < 1.2


# ── QC ────────────────────────────────────────────────────────────────────────

class TestQC:
    def test_report_has_required_keys(self, featured):
        df, _ = featured
        report = qc(df)
        for key in ["timestamp", "n_records", "n_unique_ips",
                    "error_rate", "critical_rate", "issues"]:
            assert key in report

    def test_record_count_matches(self, featured):
        df, _ = featured
        assert qc(df)["n_records"] == len(df)

    def test_no_issues_on_clean_data(self, featured):
        df, _ = featured
        report = qc(df)
        assert len(report["issues"]) == 0, f"QC issues: {report['issues']}"

    def test_error_rate_in_range(self, featured):
        df, _ = featured
        assert 0 <= qc(df)["error_rate"] <= 1
