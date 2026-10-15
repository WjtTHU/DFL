import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from wip_pred.models.embed import Embedding
from wip_pred.models.pooling import (
    AttentionPooling,
    ConcatProjPooling,
    MeanPooling,
    QueryAttentivePooling,
)
from wip_pred.models.transformer import TransformerBatchNormEncoderLayer


class Phi(nn.Module):
    """
    Phi: Temporal embedding network with mask support.

    Supports three backbone types:
        - 'ffn': Two-layer feedforward network (default)
        - 'gru': GRU-based temporal encoder
        - 'conv': 1D convolutional encoder

    Input:
        x: (B, D, L, d_x)
        mask: (B, D, L) -- 1 for valid timesteps, 0 for padding

    Output:
        phi_out: (B, D, d_phi)
    """

    def __init__(
        self,
        d_x,
        L,
        d_phi,
        d_hidden=64,
        dropout=0.1,
        phi_type="ffn",
        bidirectional=False,
        kernel_size=3,
    ):
        super().__init__()
        self.phi_type = phi_type.lower()

        if self.phi_type == "ffn":
            # Feedforward (MLP) version
            self.model = nn.Sequential(
                nn.Linear(L * d_x, d_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_hidden, d_phi),
            )

        elif self.phi_type == "gru":
            # GRU version
            self.gru = nn.GRU(
                input_size=d_x,
                hidden_size=d_hidden,
                num_layers=1,
                batch_first=True,
                bidirectional=bidirectional,
            )
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(d_hidden * (2 if bidirectional else 1), d_phi)

        elif self.phi_type == "conv":
            # 1D convolution version
            self.conv = nn.Sequential(
                nn.Conv1d(d_x, d_hidden, kernel_size, padding=kernel_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(d_hidden, d_phi, 1),
            )
        else:
            raise ValueError(
                f"Unsupported phi_type '{phi_type}'. Choose from ['ffn', 'gru', 'conv']."
            )

    def forward(self, x, mask=None):
        """
        Forward pass.
        """
        B, D, L, d_x = x.shape

        # Apply mask to zero out padded timesteps
        if mask is not None:
            mask = mask.unsqueeze(-1)  # (B, D, L, 1)
            x = x * mask

        if self.phi_type == "ffn":
            # Flatten time dimension
            x = x.reshape(B, D, L * d_x)
            out = self.model(x)  # (B, D, d_phi)

        elif self.phi_type == "gru":
            # Merge B and D for GRU input
            x = x.reshape(B * D, L, d_x)
            _, h_n = self.gru(x)  # h_n: (num_layers * num_directions, B * D, d_hidden)
            h_n = h_n[-1]  # last layer's hidden state
            h_n = self.dropout(h_n)
            out = self.fc(h_n).reshape(B, D, -1)  # (B, D, d_phi)

        elif self.phi_type == "conv":
            # Prepare for Conv1D: (B*D, d_x, L)
            x = x.reshape(B * D, L, d_x).transpose(1, 2)
            out = self.conv(x)  # (B * D, d_phi, L)
            out = out.mean(dim=-1)  # global average pooling
            out = out.reshape(B, D, -1)  # (B, D, d_phi)

        return out


class Rho(nn.Module):
    """
    ρ: Set aggregation with optional mask over units
    Input:
        phi_all: (B, D, d_phi)
        mask_units: (B, D)  -- 1 for valid units, 0 for invalid
    Output:
        F_t: (B, d_rho)
    """

    def __init__(self, d_phi, d_rho, d_hidden=32, dropout=0.1, agg="mean"):
        super().__init__()
        self.agg = agg
        self.ffn = nn.Sequential(
            nn.Linear(d_phi, d_hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_hidden, d_rho)
        )

        if self.agg == "attention":
            self.pooling = AttentionPooling(d_phi)
        elif self.agg == "query":
            self.pooling = QueryAttentivePooling(d_phi)
        elif self.agg == "mean":
            self.pooling = MeanPooling()
        else:
            raise ValueError("agg must be 'attention' / 'query' / 'mean'")

    def forward(self, phi_all, mask_units=None):
        if mask_units is not None:
            mask_units = mask_units == 0  # (B, D)
            phi_agg = self.pooling(phi_all, mask=mask_units)  # (B, d_phi)
        else:
            phi_agg = self.pooling(phi_all)  # (B, d_phi)
        F_t = self.ffn(phi_agg)  # (B, r)
        return F_t


class Psi(nn.Module):
    """
    ψ: Feature fusion network
    Concatenate per-unit features X_i_t with set summary F_t
    Input: X_i_t shape = (B, D, d_x)
           F_t   shape = (B, r)
    Output: X_tilde_i_t shape = (B, D, d_psi)
    """

    def __init__(self, d_x, d_rho, d_psi=64, d_hidden=64, dropout=0.1):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_x + d_rho, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_psi),
        )

    def forward(self, X_t, F_t):
        # X_t: (B, D, d_x)
        # F_t: (B, r)
        B, D, d_x = X_t.shape
        # expand F_t to match each unit
        F_expand = F_t.unsqueeze(1).expand(B, D, F_t.size(-1))  # → (B, D, r)
        x_cat = torch.cat([X_t, F_expand], dim=-1)  # → (B, D, d_x + r)
        out = self.ffn(x_cat)  # → (B, D, d_psi)
        return out


class SetSeqLayer(nn.Module):
    def __init__(
        self,
        chunk_size: int,
        d_phi: int,
        d_summary: int,
        d_model: int,
        n_head: int,
        d_ff: int,
        device,
        activation="gelu",
        norm="BatchNorm",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.device = device

        # Temporal embedding network
        self.phi = Phi(d_model, chunk_size, d_phi, phi_type="ffn")

        # Set aggregation network
        self.rho = Rho(d_phi, d_summary)

        # Feature fusion network
        self.psi = Psi(d_model, d_summary, d_model)

        # Sequence layer
        if norm == "LayerNorm":
            self.seq_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_head,
                dim_feedforward=d_ff,
                dropout=dropout,
                activation=activation,
                batch_first=True,
            )
        else:
            self.seq_layer = TransformerBatchNormEncoderLayer(
                d_model=d_model,
                nhead=n_head,
                dim_feedforward=d_ff,
                dropout=dropout,
                activation=activation,
                batch_first=True,
            )

    def forward(self, X, attention_mask):
        # X: (D, S, d_x)
        # attention_mask: (D, S) -- 1 for valid, 0 for mask

        # pad tensors so the first token can have summary feature
        X = F.pad(X, (0, 0, self.chunk_size - 1, 0, 0, 0))
        attention_mask = F.pad(attention_mask, (self.chunk_size - 1, 0, 0, 0))

        D, Sp, d_x = X.shape
        # S = Sp - self.chunk_size + 1

        # Construct windows for phi: (D, Sp, d_x) -> (D, S, L, d_x)
        windows = X.unfold(dimension=1, size=self.chunk_size, step=1).transpose(
            2, 3
        )  # (D, S, L, d_x)
        window_mask = attention_mask.unfold(dimension=1, size=self.chunk_size, step=1)  # (D, S, L)

        # Apply phi to each window
        h = self.phi(windows.transpose(0, 1), window_mask.transpose(0, 1))  # (S, D, d_phi)

        # rho to get F_seq: (S, r)
        mask_units = (window_mask.sum(dim=-1) == 0).bool()
        F_seq = self.rho(h, mask_units.transpose(0, 1))  # (S, r)

        # Build augmented per-unit sequence to feed SeqLayer:
        base_seq = X[:, self.chunk_size - 1 :, :].transpose(
            0, 1
        )  # last S steps: (D, S, d_x), permute to (S, D, d_x)

        # Concatenate X and F, merge them
        X_tilde = self.psi(base_seq, F_seq).transpose(
            0, 1
        )  # (S, D, d_model), permute to (D, S, d_model)

        # Put into sequence
        attention_mask = (attention_mask == 0)[:, self.chunk_size - 1 :]
        out = self.seq_layer(X_tilde, src_key_padding_mask=attention_mask)

        return out


class SetSeqModel(nn.Module):
    def __init__(
        self,
        embedding: Embedding,
        d_phi: int,
        d_summary: int,
        d_model: int,
        n_head: int,
        n_layers: int,
        d_ff: int,
        max_len: int | None = None,
        dropout: float = 0.1,
        chunk_size: int = 3,
        agg: str = "last",
        activation="gelu",
        norm="BatchNorm",
        device: str = "cpu",
    ):
        super().__init__()

        self.device = device

        assert agg in ["attention", "query", "mean", "last"]
        assert norm in ["LayerNorm", "BatchNorm"]
        self.agg = agg

        # Process features
        self.embedding = embedding

        self.feat_proj = nn.Linear(embedding.feat_dim, d_model)

        setseq_layer = SetSeqLayer(
            chunk_size=chunk_size,
            d_phi=d_phi,
            d_summary=d_summary,
            d_model=d_model,
            n_head=n_head,
            d_ff=d_ff,
            device=device,
            activation=activation,
            norm=norm,
            dropout=dropout,
        )

        self.layers = nn.ModuleList([copy.deepcopy(setseq_layer) for _ in range(n_layers)])

        # Pooling
        if self.agg == "attention":
            self.pooling = AttentionPooling(d_model)
        elif self.agg == "query":
            self.pooling = QueryAttentivePooling(d_model)
        elif self.agg == "mean":
            self.pooling = MeanPooling()
        elif self.agg == "concat":
            assert max_len is not None
            self.pooling = ConcatProjPooling(d_model, max_len)
        elif self.agg == "last":
            self.pooling = None

        self.dropout = nn.Dropout(dropout)

        # Head for prediction
        self.pred_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1)
        )

    def forward(self, x: dict):
        """
        Args:
            x (dict): a dict with keys and values as followed
                - numeric (optional): a tensor of shape (batch_size, seq_len, numeric_feat_num)
                - categorical: a dict with column names as keys and tensors of shape (batch_size, seq_len) as values
                - temporal: a tensor of shape (batch_size, seq_len, temporal_feat_num)
                - position (optional): a tensor of shape (batch_size, seq_len)
                - is_given (optional): a tensor of shape (batch_size, seq_len)
                - length / mask: for constructing padding mask. `length` is a tensor of shape (batch_size, ) and `mask` is a tensor of shape (batch_size, seq_len)

        """
        # feat_embed: (D, S, d_x)
        # pos_embed: (D, S, d_x) or 0.0
        # given_embed: (D, S, d_x) or 0.0
        # attention_mask: (D, S)
        feat_embed, pos_embed, given_embed, attention_mask = self.embedding(x)

        # project feat_embed to d_model dimension
        feat_embed = self.feat_proj(feat_embed)

        # put into encoder
        enc = feat_embed + pos_embed + given_embed
        for mod in self.layers:
            enc = mod(enc, attention_mask)

        # reverse 0 and 1 in attention_mask
        attention_mask = attention_mask == 0

        # pooling
        if self.pooling is not None:
            rep = self.pooling(enc, mask=attention_mask)
        else:
            batch_idx = torch.arange(enc.size(0), device=self.device)
            valid_mask = ~attention_mask  # 1 for valid, 0 for pad (same as embedding output)
            end_idx = enc.size(1) - torch.argmax(valid_mask.float().flip(dims=[1]), dim=1)
            last_one_positions = end_idx - 1
            rep = enc[batch_idx, last_one_positions]

        # output
        rep = self.dropout(rep)
        out = self.pred_head(rep)

        return out
