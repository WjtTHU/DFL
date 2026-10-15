import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    def __init__(self, d_model: int, d_ff: int = 16) -> None:
        super().__init__()
        self.attn_mlp = nn.Sequential(nn.Linear(d_model, d_ff), nn.Tanh(), nn.Linear(d_ff, 1))

    def forward(self, H, mask=None):
        """
        H: (batch_size, seq_len, d_model)
        mask: (batch_size, seq_len) -> 1 for padding, 0 for valid tokens (optional)
        """
        # Compute raw attention scores: (batch_size, seq_len)
        scores = self.attn_mlp(H).squeeze(2)
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))

            # Avoid samples which are all masked cause NaN
            all_masked = mask.all(dim=1)
            if all_masked.any():
                scores[all_masked] = 0.0

        # Softmax over time dimension
        weights = F.softmax(scores, dim=1)  # (batch_size, seq_len)

        # Weighted sum: (batch_size, d_model)
        weights = weights.unsqueeze(-1)  # (batch_size, seq_len, 1)
        pooled = torch.sum(H * weights, dim=1)

        return pooled


class QueryAttentivePooling(nn.Module):
    def __init__(self, d_model: int, simplified=True) -> None:
        super().__init__()
        self.simplified = simplified
        # leanable vector to score each timestep
        self.query = nn.Parameter(torch.randn(d_model))
        if not simplified:
            self.W = nn.Linear(d_model, d_model)

    def forward(self, H, mask=None):
        """
        H: (batch_size, seq_len, d_model)
        mask: (batch_size, seq_len) -> 1 for padding, 0 for valid tokens (optional)
        """

        # Compute raw attention scores: (batch_size, seq_len)
        if self.simplified:
            scores = torch.tensordot(H, self.query, dims=((2), (0)))  # type: ignore
        else:
            scores = torch.tensordot(self.W(H), self.query, dims=((2), (0)))  # type: ignore # dim 2 of H * dim 0 of attention_vector

        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))

            # Avoid samples which are all masked cause NaN
            all_masked = mask.all(dim=1)
            if all_masked.any():
                scores[all_masked] = 0.0

        # Softmax over time dimension
        weights = F.softmax(scores, dim=1)  # (batch_size, seq_len)

        # Weighted sum: (batch_size, d_model)
        weights = weights.unsqueeze(-1)  # (batch_size, seq_len, 1)
        pooled = torch.sum(H * weights, dim=1)

        return pooled


class MeanPooling(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, H, mask=None):
        """
        H: (batch_size, seq_len, d_model)
        mask: (batch_size, seq_len) -> 1 for padding, 0 for valid tokens (optional)
        """
        if mask is None:
            return H.mean(dim=1)  # (batch_size, d_model)
        else:
            # Masked mean pooling (ignores padding)
            mask = (mask == 0).unsqueeze(-1).float()  # (batch_size, seq_len, 1)
            H_masked = H * mask
            sum_H = H_masked.sum(dim=1)  # (batch_size, d_model)
            denom = mask.sum(dim=1).clamp(min=1e-8)  # avoid division by 0
            return sum_H / denom


class ConcatProjPooling(nn.Module):
    """This pooling method is not recommended when the order in the sequence doesn't matter and when the sequences are
    of different length (because there might be a difference between different batches even after padding )"""

    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model * max_len, d_model)

    def forward(self, H, mask=None):
        """
        H: (batch_size, seq_len, d_model)
        mask: (batch_size, seq_len) -> 1 for padding, 0 for valid tokens (optional)
        """
        if mask is not None:
            mask = mask.unsqueeze(-1)  # (batch_size, seq_len, 1)
            H = H * mask

        H = H.reshape(H.shape(0), -1)  # (batch_size, seq_len * d_model)
        pooled = self.proj(H)

        return pooled
