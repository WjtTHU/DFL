import torch
import torch.nn as nn

from wip_pred.models.helper import FixedPositionalEncoding, OneHot

from .time_encoding import Time2Vec


class Embedding(nn.Module):
    """
    Embed each feature for a given dataframe.
    Positional embedding reference: https://github.com/QData/spacetimeformer/blob/bcf953fdac56f6c174041f86ecaecc8c010710e3/spacetimeformer/spacetimeformer_model/nn/embed.py#L158

    """

    def __init__(
        self,
        d_model: int,
        numeric_feat_cols: list[str],
        categorical_feat_cols: list[dict],
        temporal_feat_cols: list[str] | None = None,
        temporal_embed_dim: int = 256,
        pos_enc: str | None = None,
        max_seq_len: int = 500,
        use_given_emb: bool = False,
        device: str = "cpu",
        **kwargs,
    ):
        """
        Embedding features.
        For categorical features: use Embedding or Onehot.
        For temporal feature: use Time2Vec. Specifying other encoders is not supported yet.
        For position (optional): use Time2Vec (t2v) / fully learnable embedding (abs).
        For is_given (optional): use nn.Embedding of 2 entries.

        Note: when pos_enc is not None, you should make sure the time series is regular (aligned)!

        Args:
            d_model (int): dimension of position_embed (optional), given_embed (optional)
            numeric_feat_cols (list[str]): column names of numeric features
            categorical_feat_cols (list[dict]): each element is a dict with following keys:
                - name: column name
                - encoder: `embed` or `onehot`
                - num_classes: number of categories of this category variable
            temporal_feat_cols (list[str]): column names of temporal features
            temporal_embed_dim (int): total length of all temporal features after embedding (each temporal feature will have equal embeded length, so make sure temporal_embed_dim % len(temporal_feat_cols) == 0)
            pos_enc (str|None): 't2v' (time2Vec), 'abs' (absolute pos), 'fixed' (fixed encoding using sine and cosine functions) or None (do not use positional encoding)
            max_seq_len (int): max possible position id. This param only matters when pos_enc is not None.
            use_given_emb (bool): whether use an Embedding of 2 entries to indicate the missing value
            device (str): 'cuda' or 'cpu'

        """
        super().__init__()
        self.d_model = d_model
        temporal_feat_cols = temporal_feat_cols or [
            "year",
            "month",
            "day",
            "weekday",
            "hour",
            "minute",
        ]
        self.numeric_feat_cols = numeric_feat_cols
        self.categorical_feat_cols = categorical_feat_cols
        self.temporal_feat_cols = temporal_feat_cols
        self.pos_enc = pos_enc
        self.max_seq_len = max_seq_len
        self.device = device

        self.feat_dim = len(numeric_feat_cols)

        # category columns
        self.cat_encoders = nn.ModuleDict()

        for col in categorical_feat_cols:
            if col["encoder"] == "embed":
                self.cat_encoders[col["name"]] = nn.Embedding(col["num_classes"], col["dim"])
                self.feat_dim += col["dim"]
            elif col["encoder"] == "onehot":
                self.cat_encoders[col["name"]] = OneHot(col["num_classes"])
                self.feat_dim += col["num_classes"]
            else:
                raise ValueError(
                    "Only 'embed' and 'onehot' can be used as encoder of category columns!"
                )

        # datetime column
        self.time_encoder = Time2Vec(len(self.temporal_feat_cols), temporal_embed_dim)
        self.feat_dim += temporal_embed_dim

        # positional encoding (optional)
        if pos_enc is None:
            self.pos_encoder = None
        elif pos_enc == "t2v":
            self.pos_encoder = Time2Vec(1, d_model + 1)
        elif pos_enc == "abs":
            self.pos_encoder = nn.Embedding(max_seq_len, d_model)
        elif pos_enc == "fixed":
            self.pos_encoder = FixedPositionalEncoding(d_model=d_model, max_len=self.max_seq_len)
        else:
            raise ValueError("`pos_enc` can only be 't2v', 'abs' or None!")

        # given encoding (optional)
        if use_given_emb:
            self.given_encoder = nn.Embedding(2, d_model)
        else:
            self.given_encoder = None

    def forward(self, x: dict):
        """
        Process a batch of sequences.

        Args:
            x (dict): a dict with keys and values as followed
                - numeric (optional): a tensor of shape (batch_size, seq_len, numeric_feat_num)
                - categorical: a dict with column names as keys and tensors of shape (batch_size, seq_len) as values
                - temporal: a tensor of shape (batch_size, seq_len, temporal_feat_num)
                - position (optional): a tensor of shape (batch_size, seq_len)
                - is_given (optional): a tensor of shape (batch_size, seq_len)
                - length / mask: for constructing padding mask. `length` is a tensor of shape (batch_size, ) and `mask` is a tensor of shape (batch_size, seq_len)

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            - A tensor of combined embeddings for the batch, shape (batch_size, max_seq_len, d_model).
            - An attention mask for the batch, shape (batch_size, max_seq_len).
        """
        batch_num = x.get("numeric", None)
        batch_cats = x["categorical"]
        batch_time = x["temporal"]
        batch_pos = x.get("position", None)
        batch_given_mask = x.get("is_given", None)

        feat_parts = []

        # ensure tensors are on the right device
        if batch_num is not None:
            batch_num = batch_num.to(self.device)
            feat_parts.append(batch_num)

        batch_cats = {col: batch_cat.to(self.device) for col, batch_cat in batch_cats.items()}
        batch_time = batch_time.to(self.device)

        # embedding for category features and temporal features
        cat_embed = {col: encoder(batch_cats[col]) for col, encoder in self.cat_encoders.items()}
        time_embed = self.time_encoder(batch_time)

        # concatenate all features
        for cat in cat_embed.values():
            feat_parts.append(cat)
        feat_parts.append(time_embed)
        # feat_embed = self.val_time_encoder(torch.cat(feat_parts, dim=-1))
        feat_embed = torch.cat(feat_parts, dim=-1)

        # positional encoding
        # If pos_enc is disabled (None) we always use 0.0, even if `position` is present in x.
        if batch_pos is None or self.pos_enc is None:
            pos_embed = 0.0
        else:
            if self.pos_enc == "t2v":
                batch_pos = batch_pos.to(self.device)
                batch_pos = batch_pos.float().unsqueeze(-1)
                pos_embed = self.pos_encoder(batch_pos)[:, :, 1:]  # type: ignore
            elif self.pos_enc == "abs":
                assert feat_embed.size(1) <= self.max_seq_len
                batch_pos = batch_pos.to(self.device)
                pos_embed = self.pos_encoder(batch_pos)  # type: ignore
            elif self.pos_enc == "fixed":
                # `FixedPositionalEncoding` expects (B, L, d_model) embeddings.
                batch_size, seq_len = feat_embed.shape[:2]
                assert seq_len <= self.max_seq_len
                zeros = torch.zeros(
                    (batch_size, seq_len, self.d_model), device=self.device, dtype=feat_embed.dtype
                )
                pos_embed = self.pos_encoder(zeros)  # type: ignore
            else:
                # safety fallback for unexpected configs
                pos_embed = 0.0

        attention_mask = None
        # is_given encoding
        if batch_given_mask is not None:
            batch_given_mask = batch_given_mask.to(self.device)
            feat_embed = feat_embed * batch_given_mask.unsqueeze(-1)

            if self.given_encoder is not None:
                given_embed = self.given_encoder(batch_given_mask)
            else:
                given_embed = 0.0
                attention_mask = batch_given_mask
        else:
            given_embed = 0.0

        if attention_mask is None:
            if "length" in x:
                batch_lengths = x["length"].to(self.device)
                max_len = feat_embed.size(1)
                attention_mask = (
                    torch.arange(max_len, device=self.device)[None, :] < batch_lengths[:, None]
                )
            elif "mask" in x:
                attention_mask = x["mask"].to(self.device)

        feat_embed = feat_embed * attention_mask.unsqueeze(-1)  # type: ignore

        return feat_embed, pos_embed, given_embed, attention_mask
