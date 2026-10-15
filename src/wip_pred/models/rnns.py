import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from wip_pred.models.embed import Embedding
from wip_pred.models.pooling import (
    AttentionPooling,
    MeanPooling,
    QueryAttentivePooling,
)


class TSRNNEncoderRegressor(nn.Module):
    """
    RNN encoder for regression.

    Valid tokens must form a contiguous segment (leading/trailing padding allowed).
    pos_enc must be null or t2v. Normalization uses LayerNorm only.
    """

    def __init__(
        self,
        embedding: Embedding,
        d_model: int,
        n_layers: int = 2,
        max_len: int | None = None,
        dropout: float = 0.1,
        agg: str = "last",
        model_type: str = "lstm",  # "lstm" | "gru" | "rnn"
        bidirectional: bool = False,
        norm: str = "LayerNorm",
        device: str = "cpu",
        **kwargs,
    ) -> None:
        super().__init__()
        self.device = device
        self.model_type = model_type.lower()
        self.bidirectional = bidirectional
        assert self.model_type in ["lstm", "gru", "rnn"]
        assert agg in ["attention", "mean", "query", "concat", "last"]
        if agg == "concat":
            raise ValueError(
                "concat pooling is not supported for RNN with dynamic sequence crop; "
                "use mean, attention, query, or last instead."
            )
        self.agg = agg

        if embedding.pos_enc not in (None, "t2v"):
            raise ValueError(
                f"RNN only supports pos_enc=null or pos_enc='t2v', got {embedding.pos_enc!r}. "
                "Use temporal features for time, or t2v for continuous timestamps."
            )
        if norm == "BatchNorm":
            raise ValueError(
                "RNN does not support BatchNorm on variable-length sequences; use norm='LayerNorm'."
            )
        if norm != "LayerNorm":
            raise ValueError(f"RNN only supports norm='LayerNorm', got {norm!r}.")

        self.embedding = embedding
        self.feat_proj = nn.Linear(embedding.feat_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)

        rnn_kwargs = dict(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )

        if self.model_type == "lstm":
            self.encoder = nn.LSTM(**rnn_kwargs)
        elif self.model_type == "gru":
            self.encoder = nn.GRU(**rnn_kwargs)
        else:  # "rnn"
            self.encoder = nn.RNN(**rnn_kwargs, nonlinearity="tanh")

        enc_dim = 2 * d_model if bidirectional else d_model
        if self.agg == "attention":
            self.pooling = AttentionPooling(enc_dim)
        elif self.agg == "query":
            self.pooling = QueryAttentivePooling(enc_dim)
        elif self.agg == "mean":
            self.pooling = MeanPooling()
        else:
            self.pooling = None

        self.output_norm = nn.LayerNorm(enc_dim)
        self.dropout = nn.Dropout(dropout)
        self.pred_head = nn.Sequential(
            nn.Linear(enc_dim, d_model), nn.GELU(), nn.Linear(d_model, 1)
        )

    def forward(self, x: dict):
        feat_embed, pos_embed, given_embed, attention_mask = self.embedding(x)
        feat_embed = self.feat_proj(feat_embed)
        embed_x = self.input_norm(feat_embed + pos_embed + given_embed)  # (B, L, d_model)
        B, L, _ = embed_x.shape
        device = embed_x.device

        # attention_mask: (B, L)  1 for valid, 0 for pad
        attention_mask = attention_mask.float()
        start_idx = torch.argmax(attention_mask, dim=1)  # first valid token
        end_idx = L - torch.argmax(attention_mask.flip(dims=[1]), dim=1)  # last valid token
        lengths = (end_idx - start_idx).long()  # valid length of each sequence

        max_len = max(int(lengths.max().item()), 1)
        new_embed = torch.zeros((B, max_len, embed_x.size(2)), device=device)
        for i in range(B):
            s, e = start_idx[i].item(), end_idx[i].item()
            new_embed[i, : e - s] = embed_x[i, s:e]

        # pack_padded_sequence requires CPU lengths; clamp(min=1) avoids empty-seq errors
        pack_lengths = lengths.clamp(min=1).cpu()
        packed = pack_padded_sequence(
            new_embed, pack_lengths, batch_first=True, enforce_sorted=False
        )
        packed_output, hidden = self.encoder(packed)
        enc, _ = pad_packed_sequence(packed_output, batch_first=True)
        enc = self.output_norm(enc)

        if self.pooling is not None:
            new_mask = torch.arange(max_len, device=device).unsqueeze(0) < lengths.unsqueeze(1)
            rep = self.pooling(enc, mask=(~new_mask))
        else:
            h_n = hidden[0] if self.model_type == "lstm" else hidden
            if self.bidirectional:
                rep = torch.cat([h_n[-2], h_n[-1]], dim=-1)
            else:
                rep = h_n[-1]

        rep = self.dropout(rep)
        out = self.pred_head(rep)
        return out
