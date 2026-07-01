"""
Shared forecasting model classes for the EEG Forecasting tutorial series.
Import from this module in Model Comparison.ipynb.
"""

from __future__ import annotations
import numpy as np
import scipy.linalg as linalg
from scipy.integrate import quad
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ── OptimalSVHT (inlined to avoid circular imports from utils.utils) ───────────
class OptimalSVHT:
    """Optimal singular value hard thresholding (Gavish & Donoho, 2014)."""

    def __init__(self, sigma_known: bool):
        self.sigma_known = sigma_known

    def compute_optimal_SVHT_coef(self, beta):
        coef = (self.optimal_SVHT_coef_sigma_known(beta)
                if self.sigma_known
                else self.optimal_SVHT_coef_sigma_unknown(beta))
        self.coef = coef
        return coef

    def optimal_SVHT_coef_sigma_known(self, beta):
        assert np.all((beta > 0) & (beta <= 1))
        w = (8 * beta) / (beta + 1 + np.sqrt(beta**2 + 14 * beta + 1))
        return np.sqrt(2 * (beta + 1) + w)

    def optimal_SVHT_coef_sigma_unknown(self, beta):
        coef = self.optimal_SVHT_coef_sigma_known(beta)
        MPmedian = np.array([self.MedianMarcenkoPastur(b) for b in np.atleast_1d(beta)])
        return coef / np.sqrt(MPmedian)

    def MedianMarcenkoPastur(self, beta):
        def MarPas(x):
            return 1 - self.incMarPas(x, beta, 0)
        lobnd = (1 - np.sqrt(beta)) ** 2
        hibnd = (1 + np.sqrt(beta)) ** 2
        while hibnd - lobnd > 0.001:
            x_vals = np.linspace(lobnd, hibnd, 5)
            y_vals = np.array([MarPas(x) for x in x_vals])
            if np.any(y_vals < 0.5):
                lobnd = np.max(x_vals[y_vals < 0.5])
            if np.any(y_vals > 0.5):
                hibnd = np.min(x_vals[y_vals > 0.5])
        return (hibnd + lobnd) / 2

    def incMarPas(self, x0, beta, gamma):
        if beta > 1:
            raise ValueError('beta must be in (0,1]')
        topSpec = (1 + np.sqrt(beta)) ** 2
        botSpec = (1 - np.sqrt(beta)) ** 2

        def MarPas(x):
            condition = (topSpec - x) * (x - botSpec) > 0
            return np.where(condition,
                            np.sqrt((topSpec - x) * (x - botSpec)) / (beta * x * 2 * np.pi), 0)

        fun = (lambda x: x ** gamma * MarPas(x)) if gamma != 0 else MarPas
        integral, _ = quad(fun, x0, topSpec)
        return integral


# ── DMD ────────────────────────────────────────────────────────────────────────
class DMD:
    """
    Dynamic Mode Decomposition with optional Hankel time-delay embedding.
    Follows the scikit-learn fit/predict/score interface.
    """

    def __init__(self, dt: float, time_delay: int = 0,
                 svd_threshold: str | float | None = 'optimal',
                 scale_modes: bool = False, clip_lambda: bool = False):
        self.dt = dt
        self.svd_threshold = svd_threshold
        self.scale_modes = scale_modes
        self.clip_lambda = clip_lambda
        self.time_delay = time_delay

    def fit(self, X: np.ndarray, y=None) -> 'DMD':
        m, k = X.shape
        self.m = m
        self.k = k

        if self.time_delay:
            X = self.time_delay_embedding(X, self.time_delay)

        X1, X2 = X[:, :-1], X[:, 1:]
        m, k = X1.shape
        U, s, Vh = linalg.svd(X1, full_matrices=False)
        V = Vh.conj().T
        S = np.diag(s)
        self.singular_values = s

        if self.svd_threshold == 'optimal':
            beta_ratio = m / k
            if beta_ratio > 1:
                beta_ratio = 1 / beta_ratio
            thresholder = OptimalSVHT(sigma_known=False)
            thresh = thresholder.compute_optimal_SVHT_coef(beta_ratio) * np.median(s)
            r = int((S > thresh).sum())
        elif isinstance(self.svd_threshold, float) and 0 < self.svd_threshold < 1:
            s_sq = s ** 2
            cumulative_energy = np.cumsum(s_sq) / s_sq.sum()
            r = int(np.argmax(cumulative_energy > self.svd_threshold)) + 1
        else:
            r = U.shape[-1]

        self.svd_rank = r
        U_r = U[:, :r]
        V_r = V[:, :r]
        S_r = S[:r, :r]

        A = U_r.conj().T @ X2 @ V_r @ linalg.inv(S_r)
        if self.scale_modes:
            A = np.diag(np.diag(S_r) ** (-1 / 2)) @ A @ np.diag(np.diag(S_r) ** (1 / 2))

        lamda, W = linalg.eig(A)
        if self.clip_lambda:
            self.lamda_unclipped = lamda.copy()
            lamda = np.where(np.abs(lamda) > 1.0, lamda / np.abs(lamda), lamda)

        if self.scale_modes:
            W = S_r ** (1 / 2) @ W

        Phi = X2 @ V_r @ linalg.inv(S_r) @ W
        b = linalg.pinv(Phi) @ X1[:, 0]
        Phi = Phi[:self.m, :]

        self.modes = Phi
        self.eigs = lamda
        self.amplitudes = b
        self.omega = np.log(lamda) / self.dt
        self.mode_hz = np.abs(np.imag(self.omega) / (2 * np.pi))
        return self

    def predict(self, start: int = 0, end: int | None = None) -> np.ndarray:
        if end is None:
            end = self.k
        t_pows = np.tile(np.arange(start, end), (self.amplitudes.shape[0], 1))
        C = self.eigs[:, np.newaxis] ** t_pows
        Z = np.diag(self.amplitudes) @ C
        self.time_dynamics = Z
        return (self.modes @ Z).real

    def score(self, X: np.ndarray, y=None) -> float:
        X_hat = self.predict(start=0, end=X.shape[-1])
        return float(np.mean((X - X_hat) ** 2))

    @staticmethod
    def time_delay_embedding(X: np.ndarray, h: int) -> np.ndarray:
        m, k = X.shape
        if h > k:
            raise ValueError('h must be <= number of time steps')
        delayed = [X[:, i:k - h + i + 1] for i in range(h)]
        return np.vstack(delayed)


# ── TCN ────────────────────────────────────────────────────────────────────────
class TCNResBlock(nn.Module):
    """Single residual block with two dilated causal convolutions."""

    def __init__(self, in_channels: int, nFilters: int, kernel_size: int,
                 dilation: int, dropout: float = 0.0):
        super().__init__()
        self.in_channels = in_channels
        padding_size = (kernel_size - 1) * dilation

        self.layer1 = nn.Sequential(
            nn.ZeroPad1d((padding_size, 0)),
            nn.Conv1d(in_channels=in_channels, out_channels=nFilters,
                      kernel_size=kernel_size, padding=0, dilation=dilation),
            nn.BatchNorm1d(nFilters),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )
        self.layer2 = nn.Sequential(
            nn.ZeroPad1d((padding_size, 0)),
            nn.Conv1d(in_channels=nFilters, out_channels=nFilters,
                      kernel_size=kernel_size, padding=0, dilation=dilation),
            nn.BatchNorm1d(nFilters),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )
        self.relu = nn.ReLU()
        # 1x1 conv to match dimensions for the residual connection if needed
        self.resample = (nn.Conv1d(in_channels=in_channels, out_channels=nFilters, kernel_size=1)
                         if in_channels != nFilters else None)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        out = self.layer2(self.layer1(X))
        res = X if self.resample is None else self.resample(X)
        return self.relu(out + res)


class TCN(nn.Module):
    """
    Temporal Convolutional Network for multivariate EEG forecasting.
    Depth is computed automatically to cover the full input sequence.
    """

    def __init__(self, input_size: tuple[int, int], horizon: int,
                 nFilters: int, kernel_size: int = 3, dropout: float = 0.0):
        super().__init__()
        self.input_size = input_size
        in_channels, seq_length = input_size
        n_blocks = math.ceil(np.log2((seq_length - 1) / (2 * (kernel_size - 1))))

        self.residual_blocks = nn.ModuleList()
        for i in range(n_blocks):
            self.residual_blocks.append(
                TCNResBlock(in_channels=in_channels, nFilters=nFilters,
                            kernel_size=kernel_size, dilation=2 ** (i + 1), dropout=dropout)
            )
            in_channels = nFilters

        self.conv_proj = nn.Linear(seq_length, horizon)
        self.spatial_proj = nn.Linear(nFilters, input_size[0])

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.spatial_proj.training:
            X = X + torch.randn_like(X)
        for block in self.residual_blocks:
            X = block(X)
        X = self.conv_proj(X)
        return self.spatial_proj(X.transpose(1, 2)).transpose(1, 2)


# ── GRU ────────────────────────────────────────────────────────────────────────
class GRUForecaster(nn.Module):
    """Stacked GRU for multivariate EEG forecasting."""

    def __init__(self, n_channels: int, hidden_size: int, num_layers: int,
                 pred_len: int, dropout: float = 0.1):
        super().__init__()
        self.pred_len = pred_len
        self.n_channels = n_channels
        self.gru = nn.GRU(input_size=n_channels, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.output_proj = nn.Linear(hidden_size, pred_len * n_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)  →  returns (B, pred_len, C)
        _, h_n = self.gru(x)
        out = self.output_proj(h_n[-1])
        return out.view(x.size(0), self.pred_len, self.n_channels)


# ── Transformer (Informer) ─────────────────────────────────────────────────────
class ProbSparseAttention(nn.Module):
    """ProbSparse self-attention from the Informer (Zhou et al., 2021)."""

    def __init__(self, factor: int = 5, dropout: float = 0.1):
        super().__init__()
        self.factor = factor
        self.dropout = nn.Dropout(dropout)

    def _sparsity_score(self, Q: torch.Tensor, K: torch.Tensor,
                        sample_k: int) -> torch.Tensor:
        L_K = K.size(2)
        idx = torch.randint(L_K, (sample_k,), device=Q.device)
        K_sample = K[:, :, idx, :]
        QK = torch.matmul(Q, K_sample.transpose(-2, -1))
        return QK.max(dim=-1).values - QK.mean(dim=-1)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                mask=None) -> tuple[torch.Tensor, torch.Tensor]:
        B, H, L_Q, D = Q.shape
        L_K = K.size(2)
        scale = D ** -0.5
        u = min(self.factor * math.ceil(math.log(L_K)), L_Q)
        M = self._sparsity_score(Q, K, sample_k=u)
        top_idx = M.topk(u, dim=-1).indices
        self.last_top_idx = top_idx.detach()
        Q_sparse = Q.gather(2, top_idx.unsqueeze(-1).expand(-1, -1, -1, D))
        scores = torch.matmul(Q_sparse, K.transpose(-2, -1)) * scale
        if mask is not None:
            scores = scores.masked_fill(mask[:, :, :u, :L_K] == 0, float('-inf'))
        attn = self.dropout(F.softmax(scores, dim=-1))
        attended = torch.matmul(attn, V)
        out = V.mean(dim=2, keepdim=True).expand(B, H, L_Q, D).clone()
        out.scatter_(2, top_idx.unsqueeze(-1).expand(-1, -1, -1, D), attended)
        return out, attn


class MultiHeadAttention(nn.Module):
    """Multi-head wrapper around ProbSparseAttention."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model)
        self.attn = ProbSparseAttention(dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        return x.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                mask=None) -> torch.Tensor:
        q = self._split_heads(self.W_Q(Q))
        k = self._split_heads(self.W_K(K))
        v = self._split_heads(self.W_V(V))
        ctx, _ = self.attn(q, k, v, mask=mask)
        ctx = ctx.transpose(1, 2).contiguous().view(Q.size(0), Q.size(1), -1)
        return self.dropout(self.W_O(ctx))


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float)
                        * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])


class EncoderLayer(nn.Module):
    """Informer encoder layer: ProbSparse self-attention + feed-forward + distilling."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(d_ff, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.distil = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, mask)))
        x = self.norm2(x + self.ff(x))
        return self.distil(x.transpose(1, 2)).transpose(1, 2)


class TimeSeriesTransformer(nn.Module):
    """
    Informer-based Transformer for multivariate EEG forecasting.
    Uses ProbSparse attention and distilling to handle long sequences efficiently.
    """

    def __init__(self, n_channels: int, d_model: int = 64, n_heads: int = 4,
                 n_enc_layers: int = 2, d_ff: int = 256, pred_len: int = 32,
                 dropout: float = 0.1):
        super().__init__()
        self.pred_len = pred_len
        self.n_channels = n_channels
        self.input_proj = nn.Linear(n_channels, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)
        self.encoder = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_enc_layers)
        ])
        self.output_proj = nn.Linear(d_model, pred_len * n_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)  →  returns (B, pred_len, C)
        z = self.pos_enc(self.input_proj(x))
        for layer in self.encoder:
            z = layer(z)
        z_ctx = z.mean(dim=1)
        return self.output_proj(z_ctx).view(x.size(0), self.pred_len, self.n_channels)


# ── S6 / Mamba ─────────────────────────────────────────────────────────────────
def hippo_init(N: int) -> torch.Tensor:
    """Diagonal of the HiPPO-LegS matrix, used to initialize A_log in S6."""
    A = np.zeros((N, N))
    for n in range(N):
        for k in range(N):
            if n > k:
                A[n, k] = -((2 * n + 1) ** 0.5) * ((2 * k + 1) ** 0.5)
            elif n == k:
                A[n, k] = -(n + 1)
    return torch.tensor(-np.diag(A), dtype=torch.float32)


class S6(nn.Module):
    """Selective State Space (S6) layer with input-dependent B, C, and Delta."""

    def __init__(self, d_model: int, d_state: int = 16, dt_rank: int | None = None):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = dt_rank or math.ceil(d_model / 16)
        A_init = hippo_init(d_state).unsqueeze(0).expand(d_model, -1).clone()
        self.A_log = nn.Parameter(torch.log(A_init + 1e-4))
        self.D = nn.Parameter(torch.ones(d_model))
        self.x_proj = nn.Linear(d_model, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_model, bias=True)

    def selective_scan(self, u: torch.Tensor, delta: torch.Tensor,
                       A: torch.Tensor, B: torch.Tensor,
                       C: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_in = u.shape
        n = A.shape[1]
        dA = torch.exp(delta.unsqueeze(-1) * A)
        dB = delta.unsqueeze(-1) * B.unsqueeze(2)
        h = torch.zeros(batch_size, d_in, n, device=u.device, dtype=u.dtype)
        ys: list[torch.Tensor] = []
        for t in range(seq_len):
            h = dA[:, t] * h + dB[:, t] * u[:, t].unsqueeze(-1)
            ys.append((h * C[:, t].unsqueeze(1)).sum(dim=-1))
        return torch.stack(ys, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        A = -torch.exp(self.A_log)
        x_dbl = self.x_proj(x)
        dt_raw, B, C = x_dbl.split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(dt_raw))
        y = self.selective_scan(x, delta, A, B, C)
        return y + x * self.D


class MambaBlock(nn.Module):
    """Single Mamba residual block: LayerNorm → dual branch → S6 scan → gate → residual."""

    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2,
                 conv_size: int = 4, dropout: float = 0.0):
        super().__init__()
        self.d_inner = expand * d_model
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=conv_size,
                                padding=conv_size - 1, groups=self.d_inner, bias=True)
        self.ssm = S6(d_model=self.d_inner, d_state=d_state)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x_branch, z = self.in_proj(x).chunk(2, dim=-1)
        x_conv = self.conv1d(x_branch.transpose(1, 2))
        x_conv = F.silu(x_conv[:, :, :x_branch.size(1)]).transpose(1, 2)
        y = self.ssm(x_conv) * F.silu(z)
        return residual + self.dropout(self.out_proj(y))


class MambaForecaster(nn.Module):
    """Stacked Mamba (S6) model for multivariate EEG forecasting."""

    def __init__(self, n_channels: int, d_model: int = 64, d_state: int = 16,
                 n_layers: int = 4, expand: int = 2, pred_len: int = 64,
                 dropout: float = 0.05):
        super().__init__()
        self.pred_len = pred_len
        self.n_channels = n_channels
        self.input_proj = nn.Linear(n_channels, d_model)
        self.layers = nn.ModuleList([
            MambaBlock(d_model=d_model, d_state=d_state, expand=expand, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, pred_len * n_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)  →  returns (B, pred_len, C)
        z = self.input_proj(x)
        for layer in self.layers:
            z = layer(z)
        out = self.output_proj(self.norm(z[:, -1, :]))
        return out.view(x.size(0), self.pred_len, self.n_channels)
