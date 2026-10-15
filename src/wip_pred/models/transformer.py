import copy

import torch
import torch.nn as nn
from torch.nn import functional as F

from wip_pred.models.embed import Embedding
from wip_pred.models.pooling import (
    AttentionPooling,
    ConcatProjPooling,
    MeanPooling,
    QueryAttentivePooling,
)


# reference: https://github.com/gzerveas/mvts_transformer/blob/master/src/models/ts_transformer.py
class TransformerBatchNormEncoderLayer(nn.Module):
    """Transformer Encoder Layer with BatchNorm instead of LayerNorm.

    Args:
        d_model: number of expected features in the input.
        nhead: number of heads in the multiheadattention models.
        dim_feedforward: dimension of the feedforward network (default=128).
        dropout: dropout value (default=0.1).
        activation: activation function ("relu" or "gelu").
        batch_first: if True, input and output tensors are (batch, seq, feature)
                     else (seq, batch, feature).
    """

    def __init__(
        self, d_model, nhead, dim_feedforward=128, dropout=0.1, activation="relu", batch_first=False
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )

        # Feedforward layers
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # BatchNorm layers (BatchNorm1d expects (N, C, L))
        self.norm1 = nn.BatchNorm1d(d_model, eps=1e-5)
        self.norm2 = nn.BatchNorm1d(d_model, eps=1e-5)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.batch_first = batch_first

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super().__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        Args:
            src: input sequence (shape depends on batch_first)
            src_mask: attention mask
            src_key_padding_mask: mask for padding positions
        """
        if not self.batch_first:
            # convert to (batch, seq, feature) for easier processing
            src = src.transpose(0, 1)

        # Self-attention
        src2 = self.self_attn(
            src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)

        # BatchNorm expects (batch, feature, seq)
        src = src.permute(0, 2, 1)
        src = self.norm1(src)
        src = src.permute(0, 2, 1)

        # Feedforward
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)

        # BatchNorm again
        src = src.permute(0, 2, 1)
        src = self.norm2(src)
        src = src.permute(0, 2, 1)

        if not self.batch_first:
            # restore to (seq, batch, feature)
            src = src.transpose(0, 1)

        return src


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise ValueError("activation should be relu/gelu, not {}".format(activation))


class TransformerBatchNormEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        return output


class TSTransformerEncoder(nn.Module):
    """
    Transformer model for pretraining.

    """

    def __init__(
        self,
        embedding: Embedding,
        d_model: int,
        n_head: int,
        n_layers: int,
        d_ff: int,
        num_class_dict: dict,
        dropout: float = 0.1,
        activation="gelu",
        norm="BatchNorm",
        device: str = "cpu",
        **kwargs,
    ) -> None:
        super().__init__()
        self.device = device

        assert norm in ["LayerNorm", "BatchNorm"]

        # Process features
        self.embedding = embedding

        self.feat_proj = nn.Linear(embedding.feat_dim, d_model)

        if norm == "LayerNorm":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_head,
                dim_feedforward=d_ff,
                dropout=dropout,
                activation=activation,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        else:
            encoder_layer = TransformerBatchNormEncoderLayer(
                d_model=d_model,
                nhead=n_head,
                dim_feedforward=d_ff,
                dropout=dropout,
                activation=activation,
                batch_first=True,
            )
            self.encoder = TransformerBatchNormEncoder(encoder_layer, num_layers=n_layers)

        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        self.classi_head = nn.ModuleDict()

        for feat, num_class in sorted(num_class_dict.items(), key=lambda x: x[0]):
            self.classi_head[feat] = nn.Linear(d_model, num_class)

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
        feat_embed, pos_embed, given_embed, attention_mask = self.embedding(x)

        # project feat_embed to d_model dimension
        feat_embed = self.feat_proj(feat_embed)

        # add all
        embed_x = feat_embed + pos_embed + given_embed

        # reverse 0 and 1 in attention_mask
        attention_mask = attention_mask == 0

        # put into encoder
        enc = self.encoder(embed_x, src_key_padding_mask=attention_mask)

        # activate
        out = self.act(enc)

        # dropout
        out = self.dropout(out)

        output = dict()

        # support multi-task
        for feat, head in self.classi_head.items():
            output[feat] = head(out)

        return output


class TSTransformerEncoderRegressor(nn.Module):
    """
    Transformer model for regression task (downstream task).

    """

    def __init__(
        self,
        embedding: Embedding,
        d_model: int,
        n_head: int,
        n_layers: int,
        d_ff: int,
        max_len: int | None = None,
        dropout: float = 0.1,
        agg: str = "last",
        activation="gelu",
        norm="BatchNorm",
        device: str = "cpu",
        **kwargs,
    ) -> None:
        super().__init__()
        self.device = device

        assert agg in ["attention", "mean", "query", "concat", "last"]
        assert norm in ["LayerNorm", "BatchNorm"]
        self.agg = agg

        # Process features
        self.embedding = embedding

        self.feat_proj = nn.Linear(embedding.feat_dim, d_model)

        if norm == "LayerNorm":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_head,
                dim_feedforward=d_ff,
                dropout=dropout,
                activation=activation,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        else:
            encoder_layer = TransformerBatchNormEncoderLayer(
                d_model=d_model,
                nhead=n_head,
                dim_feedforward=d_ff,
                dropout=dropout,
                activation=activation,
                batch_first=True,
            )
            self.encoder = TransformerBatchNormEncoder(encoder_layer, num_layers=n_layers)

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

        self.dropout = nn.Dropout(dropout)  # baseline: only this

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
        feat_embed, pos_embed, given_embed, attention_mask = self.embedding(x)

        # project feat_embed to d_model dimension
        feat_embed = self.feat_proj(feat_embed)

        # add all
        embed_x = feat_embed + pos_embed + given_embed

        # reverse 0 and 1 in attention_mask
        attention_mask = attention_mask == 0

        # put into encoder
        enc = self.encoder(embed_x, src_key_padding_mask=attention_mask)

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
