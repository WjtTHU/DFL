"""Reuse the enterprise Transformer backbone; keep point and NLL models independent."""
from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from wip_pred.daily.data import CONTEXT_COLUMNS
from wip_pred.models.embed import Embedding
from wip_pred.models.transformer import TSTransformerEncoderRegressor


def fit_target_spec(training_labels, quantile=0.95, fixed_horizon=None):
    values = np.asarray(training_labels)
    if not len(values) or not np.isfinite(values).all() or np.any(values < 1) or np.any(values != np.floor(values)):
        raise ValueError('Unfinished daily samples require positive integer labels.')
    lo, hi = int(values.min()), int(values.max())
    if fixed_horizon is not None and (type(fixed_horizon) is not int or fixed_horizon < 1):
        raise ValueError('fixed_horizon must be a positive integer or None.')
    horizon = (fixed_horizon if fixed_horizon is not None
               else max(1, int(np.ceil(np.quantile(values, quantile)))))
    result = {'min': lo, 'max': hi, 'scale': max(1, hi-lo), 'horizon': horizon,
            'horizon_quantile': quantile, 'train_tail_count': int((values > horizon).sum()),
            'classes': [*range(1, horizon+1), f'>{horizon}'],
            'tail_semantics': 'Unbounded aggregate event R > H; no within-tail time distribution.'}
    if fixed_horizon is not None:
        result.update(horizon_source='fixed_experiment_design', horizon_quantile=None)
    return result


def encode_classes(targets, horizon):
    if targets.dtype not in (torch.int32, torch.int64) or torch.any(targets < 1):
        raise ValueError('NLL labels must be positive integers, not normalized targets.')
    # H is a distinct tail class, NOT the class for day H (which has index H-1).
    return torch.where(targets > horizon, horizon, targets-1)


def build_daily_model(core, mode, spec, device='cpu'):
    if mode not in ('point', 'distribution'):
        raise ValueError(f'Unknown daily mode: {mode}')
    if core['model']['type'] != 'tsf' or core['model']['params']['norm'] != 'LayerNorm':
        raise ValueError('The daily DK path requires the validated Transformer/LayerNorm backbone.')
    embed = deepcopy(core['embed'])
    embed['numeric_feat_cols'] = [*embed['numeric_feat_cols'], *CONTEXT_COLUMNS]
    embed['use_given_emb'] = False  # Missing observation context is not an attention mask.
    embedding = Embedding(core['model']['params']['d_model'], device=device, **embed)
    model = TSTransformerEncoderRegressor(embedding=embedding, device=device, **core['model']['params'])
    if mode == 'distribution':
        model.pred_head[-1] = nn.Linear(core['model']['params']['d_model'], spec['horizon']+1)
    return model.to(device)


def loss_per_sample(outputs, targets, mode, spec):
    if targets is None:
        raise ValueError('Training/evaluation requires labels; prediction does not.')
    targets = targets.to(outputs.device)
    if mode == 'point':
        scaled = (targets.float()-spec['min'])/spec['scale']
        return F.l1_loss(outputs.flatten(), scaled, reduction='none')
    return F.cross_entropy(outputs, encode_classes(targets, spec['horizon']), reduction='none')


def point_days(outputs, spec):
    raw = outputs.flatten()*spec['scale']+spec['min']
    return raw, raw.clamp_min(0)
