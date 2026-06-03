# tests/test_lstm.py
# Unit tests for the LSTM Autoencoder module.
# These run without torch if it's not installed — the fallback path is tested too.

import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import lstm_autoencoder as la


# ── sequence builder ──────────────────────────────────────────────────────────

def _make_df(n_ips=20, n_rows=200, n_feats=5, seed=0):
    rng = np.random.default_rng(seed)
    ips = [f"10.0.0.{i}" for i in range(n_ips)]
    ip_col = rng.choice(ips, size=n_rows)
    days   = rng.integers(0, 30, size=n_rows)
    hours  = rng.integers(0, 24, size=n_rows)
    feat_cols = [f"f{i}" for i in range(n_feats)]
    data  = rng.standard_normal((n_rows, n_feats))
    df = pd.DataFrame(data, columns=feat_cols)
    df["ip"]  = ip_col
    df["day"] = days
    df["hour"] = hours
    return df, feat_cols


class TestBuildSequences:
    def test_output_shape(self):
        df, cols = _make_df(n_ips=10, n_rows=150)
        seqs, ip_list = la.build_sequences(df, cols, seq_len=8)
        assert seqs.shape == (len(ip_list), 8, len(cols))
        assert len(ip_list) == 10

    def test_all_ips_represented(self):
        df, cols = _make_df(n_ips=15, n_rows=300)
        seqs, ip_list = la.build_sequences(df, cols, seq_len=5)
        assert set(ip_list) == set(df["ip"].unique())

    def test_dtype_float32(self):
        df, cols = _make_df()
        seqs, _ = la.build_sequences(df, cols, seq_len=6)
        assert seqs.dtype == np.float32

    def test_padding_for_sparse_ips(self):
        # IP with only 1 request should produce a seq padded with zeros at front
        df = pd.DataFrame({
            "ip":   ["10.0.0.1", "10.0.0.1", "10.0.0.2"],
            "day":  [0, 1, 0],
            "hour": [0, 1, 0],
            "f0":   [1.0, 2.0, 3.0],
            "f1":   [0.5, 0.5, 0.5],
        })
        seqs, ip_list = la.build_sequences(df, ["f0", "f1"], seq_len=5)
        idx_sparse = ip_list.index("10.0.0.2")
        # first 4 rows of the sparse IP's sequence should be zero-padded
        assert np.allclose(seqs[idx_sparse, :4, :], 0.0)

    def test_sequence_tail_matches_latest_request(self):
        df = pd.DataFrame({
            "ip":   ["x"] * 15,
            "day":  list(range(15)),
            "hour": [0] * 15,
            "val":  list(range(15)),
        })
        seqs, _ = la.build_sequences(df, ["val"], seq_len=5)
        # last seq step should be the latest request value (14)
        assert seqs[0, -1, 0] == pytest.approx(14.0)


# ── run() fallback when torch absent ─────────────────────────────────────────

class TestRunFallback:
    def test_returns_zeros_when_torch_unavailable(self, monkeypatch):
        monkeypatch.setattr(la, "TORCH_AVAILABLE", False)
        df, cols = _make_df(n_ips=5, n_rows=50)
        preds, errors, model, meta = la.run(df, cols)
        assert len(preds)  == len(df)
        assert len(errors) == len(df)
        assert model is None
        assert meta  == {}
        assert (preds == 0).all()

    def test_pred_length_matches_df(self, monkeypatch):
        monkeypatch.setattr(la, "TORCH_AVAILABLE", False)
        df, cols = _make_df(n_ips=8, n_rows=80)
        preds, errors, _, _ = la.run(df, cols)
        assert len(preds) == len(df)
        assert len(errors) == len(df)
