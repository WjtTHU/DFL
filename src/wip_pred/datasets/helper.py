import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence


def default_collate(batch, fixed_length=True):
    elem = batch[0]

    if isinstance(elem, torch.Tensor):
        if fixed_length:
            return torch.stack(batch, dim=0)
        return pad_sequence(batch, batch_first=True)

    elif isinstance(elem, (int, float, str)):
        return list(batch)

    elif isinstance(elem, dict):
        return {key: default_collate([d[key] for d in batch], fixed_length) for key in elem}

    else:
        raise TypeError(f"Unsupported data type: {type(elem)}")


# reference: https://github.com/gzerveas/mvts_transformer/blob/3f2e378bc77d02e82a44671f20cf15bc7761671a/src/datasets/dataset.py#L237, but this version is processed by batch, which will be faster


def noise_mask(
    mask_feats,
    batch_size: int,
    seq_len: int,
    masking_ratio: float,
    lm=3,
    mode="separate",
    distribution="geometric",
):
    """
    Generate random masks for a batch of samples, grouped by feature name.

    Args:
        mask_feats: list of features to apply masking (subset of feat_names)
        batch_size: number of samples in the batch
        seq_len: sequence length
        masking_ratio: average proportion of masked timesteps
        lm: average length of masked subsequences (only for geometric)
        mode: 'separate' (independent per feature) or 'concurrent' (shared mask)
        distribution: 'geometric' or 'bernoulli'

    Returns:
        mask_dict: dict where each key is a feature name and value is a
                   (batch_size, seq_len) int tensor with 1=keep, 0=mask
    """
    mask_feats = set(mask_feats)
    mask_dict = {}

    if distribution == "bernoulli":
        if mode == "separate":
            # independent Bernoulli for each feature and sample
            masks = np.random.rand(batch_size, seq_len, len(mask_feats)) > masking_ratio
        else:  # concurrent: same pattern across features to be masked
            shared = np.random.rand(batch_size, seq_len, 1) > masking_ratio
            masks = np.repeat(shared, len(mask_feats), axis=2)

    elif distribution == "geometric":
        if mode == "concurrent":
            # shared geometric mask across masked features
            shared = np.stack(
                [_geom_mask_fast(seq_len, lm, masking_ratio) for _ in range(batch_size)], axis=0
            )  # (batch_size, seq_len)
            masks = np.repeat(shared[:, :, None], len(mask_feats), axis=2)
        else:
            # separate geometric mask per feature
            masks = np.ones((batch_size, seq_len, len(mask_feats)), dtype=bool)
            for f_idx, _feat in enumerate(mask_feats):
                for b in range(batch_size):
                    masks[b, :, f_idx] = _geom_mask_fast(seq_len, lm, masking_ratio)

    # assign per-feature masks to dict
    for f_idx, feat in enumerate(mask_feats):
        mask_dict[feat] = torch.from_numpy(masks[:, :, f_idx].astype(np.int32))

    return mask_dict


def _geom_mask_fast(L, lm, masking_ratio):
    """Efficient geometric (Markov) mask for one sequence."""
    p_m = 1 / lm
    p_u = p_m * masking_ratio / (1 - masking_ratio)
    p = [p_m, p_u]
    mask = np.ones(L, dtype=bool)
    state = int(np.random.rand() > masking_ratio)
    i = 0
    while i < L:
        length = np.random.geometric(p[state])
        mask[i : i + length] = state
        i += length
        state = 1 - state
    return mask[:L]
