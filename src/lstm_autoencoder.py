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


def build_sequences(df, feature_cols, seq_len=SEQ_LEN):
    """
    Group log rows by IP, sort chronologically by (day, hour), and take
    the most recent seq_len rows as a fixed-length sequence.

    IPs with fewer than seq_len requests are zero-padded at the front so
    real data always sits at the tail of the sequence — the decoder focuses
    on predicting the most recent behavior.

    Returns
    -------
    X        : ndarray of shape (n_ips, seq_len, n_features), float32
    ip_list  : list of IP strings, same order as X rows
    """
    seqs    = []
    ip_list = []
    n_feats = len(feature_cols)

    for ip, grp in df.sort_values(["ip", "day", "hour"]).groupby("ip", sort=False):
        vals = grp[feature_cols].values.astype(np.float32)
        if len(vals) >= seq_len:
            seq = vals[-seq_len:]
        else:
            pad = np.zeros((seq_len - len(vals), n_feats), dtype=np.float32)
            seq = np.vstack([pad, vals])
        seqs.append(seq)
        ip_list.append(ip)

    return np.array(seqs, dtype=np.float32), ip_list


def train_model(model, X_tensor, device, max_epochs=MAX_EPOCHS, patience=PATIENCE):
    """
    Standard autoencoder training: minimize MSE between input and reconstruction.
    Input == target — we're not doing classification, just compression + rebuild.
    """
    from torch.utils.data import DataLoader, TensorDataset

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5, verbose=False)
    loader    = DataLoader(TensorDataset(X_tensor), batch_size=BATCH_SIZE, shuffle=True,
                           pin_memory=(device.type == "cuda"))

    best_loss  = float("inf")
    wait       = 0
    best_state = None

    model.train()
    for epoch in range(max_epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out  = model(batch)
            loss = criterion(out, batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg = epoch_loss / len(loader)
        scheduler.step(avg)
        log.info(f"    epoch {epoch + 1:02d}/{max_epochs} — loss {avg:.6f}")

        if avg < best_loss - 1e-5:
            best_loss  = avg
            wait       = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                log.info(f"    early stopping at epoch {epoch + 1}")
                break

    if best_state:
        model.load_state_dict(best_state)
    return model, best_loss


def compute_errors(model, X_tensor, device):
    """
    Run inference on all sequences and return per-sequence mean squared
    reconstruction error. Higher error → IP's behavior pattern is unusual.
    """
    from torch.utils.data import DataLoader, TensorDataset

    model.eval()
    errors = []
    with torch.no_grad():
        for (batch,) in DataLoader(TensorDataset(X_tensor), batch_size=BATCH_SIZE * 2):
            batch = batch.to(device)
            out   = model(batch)
            err   = ((out - batch) ** 2).mean(dim=(1, 2))
            errors.extend(err.cpu().numpy().tolist())
    return np.array(errors, dtype=np.float32)
