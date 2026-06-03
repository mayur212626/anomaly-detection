# src/lstm_autoencoder.py
# ─────────────────────────────────────────────────────────────────────────────
# LSTM Autoencoder for temporal behavioral anomaly detection.
#
# The core idea: each IP's request history is a time-ordered sequence.
# Normal IPs have consistent, predictable sequences — the autoencoder learns
# that normal pattern. An IP that suddenly shifts behavior (slow brute force,
# gradual data exfiltration, business-hour mimicry) produces a high
# reconstruction error because the decoder can't reproduce the unusual pattern.
#
# Why LSTM over a simpler RNN: the gradient vanishing problem makes plain RNNs
# useless for sequences longer than ~5 steps. LSTM's gating mechanism handles
# sequences of 10-30 steps without losing early context.
#
# Design choices:
#   seq_len=10   — captures the last 10 requests per IP chronologically
#   hidden_dim=64 — forces meaningful compression; 128 overfit on this data
#   train on full dataset (unsupervised) — threshold at 97.5th percentile
#   graceful fallback if PyTorch isn't installed
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import logging

log = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    log.warning("PyTorch not installed — LSTM Autoencoder will be skipped")

SEQ_LEN           = 10
HIDDEN_DIM        = 64
N_LAYERS          = 2
DROPOUT           = 0.2
BATCH_SIZE        = 256
MAX_EPOCHS        = 30
LR                = 1e-3
PATIENCE          = 5
ANOMALY_PERCENTILE = 97.5


class LSTMAutoencoder(nn.Module):
    """
    Seq2Seq LSTM Autoencoder.

    Encoder reads the input sequence and compresses it into a fixed-size
    context vector (the final hidden state). Decoder uses that context to
    reconstruct the original sequence timestep by timestep.

    High MSE reconstruction error for an IP's sequence means its recent
    request pattern deviated from what the model learned as 'normal'.
    """
    def __init__(self, input_dim, hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers   = n_layers

        self.layer_norm = nn.LayerNorm(input_dim)

        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        x = self.layer_norm(x)
        _, (h, c) = self.encoder(x)
        # broadcast last encoder hidden state as the decoder's input at every step
        ctx     = h[-1].unsqueeze(1).repeat(1, x.size(1), 1)
        dec_out, _ = self.decoder(ctx, (h, c))
        return self.output_layer(dec_out)
