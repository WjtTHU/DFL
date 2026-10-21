"""Calendar-day samples with an explicit forecast origin and separate labels."""

from copy import deepcopy
import hashlib
import json

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from wip_pred.prep_data.normalize_data import transform_df
from wip_pred.prep_data.prepare_data import get_df_of_completed_lots

CONTEXT_COLUMNS = ['snapshot_age_log_days', 'lot_age_log_days', 'no_snapshot_today']
INDEX_COLUMNS = ['sample_id', 'lot_id', 'product_id', 'forecast_date', 'forecast_at',
                 'last_snapshot_at', 'first_snapshot_at', 'row_start', 'row_stop',
                 'sequence_length', 'days_since_last_snapshot', 'no_snapshot_today',
                 'stale_fallback', 'known_quantity', 'completion_at', 'label_available_at',
                 'remaining_days', 'split']


def midnight(value):
    value = pd.Timestamp(value)
    if pd.isna(value) or value.tz is not None or value != value.normalize():
        raise ValueError('Use a local Asia/Shanghai calendar date, without an intraday time or offset.')
    return value


def validate_protocol(config):
    if config['protocol_version'] != 'daily_v1' or config['dataset'] != 'dk':
        raise ValueError('daily_v1 currently supports DK only.')
    if config['timezone'] != 'Asia/Shanghai':
        raise ValueError('DK timestamps are interpreted in Asia/Shanghai.')
    if midnight(config['train_end']) >= midnight(config['validation_end']):
        raise ValueError('train_end must precede validation_end.')
    for key in ('forecast_every_days', 'window_days', 'batch_size', 'epochs', 'patience', 'threads'):
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f'{key} must be a positive integer.')
    if config['completion_margin_days'] < 0:
        raise ValueError('completion_margin_days cannot be negative.')
    if not 0 < config['distribution_horizon_quantile'] <= 1:
        raise ValueError('distribution_horizon_quantile must lie in (0, 1].')
    fixed = config.get('distribution_horizon_days')
    if fixed is not None and (type(fixed) is not int or fixed < 1):
        raise ValueError('distribution_horizon_days must be a positive integer or null.')
    for key in ('forecast_start', 'forecast_end_exclusive'):
        if config.get(key) is not None:
            midnight(config[key])


def load_dk(config, *, cutoff_exclusive=None, with_labels=True):
    """Reuse enterprise cohort filtering and completion labels, retain raw intraday order.

    The online branch is truncated BEFORE any filtering, and never builds labels.
    Training uses the user's accepted offline whole-history quality cohort.
    """
    from local_paths import dataset_paths
    from wip_pred.prep_data.normalize_data import normalize_df

    path = dataset_paths('dk')['csv']
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    raw = pd.read_csv(path)
    raw['stage'] = raw['stage'].fillna('NULL')
    raw['file_timestamp'] = pd.to_datetime(raw['file_timestamp'], errors='raise')
    if cutoff_exclusive is not None:
        raw = raw.loc[raw.file_timestamp < pd.Timestamp(cutoff_exclusive)].copy()
    if raw.empty:
        raise ValueError('No observations available before the forecast cutoff.')
    # Exactly the DK enterprise normalization options used by the legacy loader.
    enterprise = normalize_df(raw.copy(), 'LotId', 'parentLotId', 'woItem', 'stage',
                              'tranCode', 'waferQty', 'file_timestamp',
                              do_time_binning=True, normalize_by_date=False)
    labels = (get_df_of_completed_lots(enterprise, config['completion_margin_days'])
              if with_labels else None)
    mapping = {'LotId': 'lot_id', 'parentLotId': 'parent_lot_id', 'woItem': 'product_id',
               'stage': 'stage', 'tranCode': 'lot_status', 'waferQty': 'current_wafer_qty',
               'file_timestamp': 'created_at', 'oper': 'oper'}
    observed = raw[list(mapping)].rename(columns=mapping)
    observed = observed.loc[observed.lot_id.isin(enterprise.lot_id.unique())].copy()
    # Preserve distinct stages even at the same timestamp; ties retain source row order.
    observed = observed.drop_duplicates(keep='last').sort_values(
        ['lot_id', 'created_at'], kind='stable').reset_index(drop=True)
    info = {'csv_sha256': digest, 'raw_rows': len(raw), 'enterprise_rows': len(enterprise),
            'retained_observation_rows': len(observed),
            'root_lots': int(observed.loc[observed.lot_id == observed.parent_lot_id, 'lot_id'].nunique()),
            'observation_min': str(observed.created_at.min()),
            'observation_max': str(observed.created_at.max()),
            'completion_rule': 'Enterprise completed root Lot last observation, boundary margin filtering',
            'labelled_lots': len(labels) if labels is not None else None}
    return observed, labels, info


def forecast_calendar(snapshots, config):
    start = midnight(config.get('forecast_start') or snapshots.created_at.min().normalize())
    available_end = snapshots.created_at.max().normalize()
    end = midnight(config.get('forecast_end_exclusive') or available_end)
    if end > available_end:
        raise ValueError('Forecast calendar cannot include the final partially observed day.')
    return pd.date_range(start, end, inclusive='left', freq=f"{config['forecast_every_days']}D")


def build_index(snapshots, dates, config, labels=None, *, requested_lots=None,
                known_completed_lots=()):
    """A sample is (root Lot, calendar date at day end). No labels enter inputs."""
    if not snapshots.index.equals(pd.RangeIndex(len(snapshots))):
        raise ValueError('Snapshot rows must have a contiguous RangeIndex.')
    dates = pd.DatetimeIndex([midnight(d) for d in dates]).unique().sort_values()
    train_end, val_end = midnight(config['train_end']), midnight(config['validation_end'])
    data_end = snapshots.created_at.max()
    ready_delta = pd.Timedelta(days=config['completion_margin_days'])
    requested = set(requested_lots) if requested_lots is not None else None
    excluded = set(known_completed_lots)
    rows, encountered = [], set()
    for lot, group in snapshots.groupby('lot_id', sort=False):
        if requested is not None and lot not in requested:
            continue
        if lot in excluded:
            encountered.add(lot)
            continue
        if not (group.parent_lot_id == lot).all():
            if requested is not None:
                raise ValueError(f'{lot} is not a root Lot.')
            continue
        encountered.add(lot)
        if labels is not None and lot not in labels.index:
            continue  # Unknown/boundary-censored labels are not treated as completions.
        times = group.created_at.to_numpy(dtype='datetime64[ns]')
        if np.any(times[1:] < times[:-1]) or np.any(np.diff(group.index) != 1):
            raise ValueError('Snapshots must be contiguous per Lot and sorted by observation time.')
        first = pd.Timestamp(times[0])
        completion = pd.Timestamp(labels.at[lot, 'end_time']) if labels is not None else None
        ready = completion + ready_delta if completion is not None else None
        eligible = dates[dates >= first.normalize()]
        if completion is not None:
            eligible = eligible[eligible < completion.normalize()]
        for day in eligible:
            next_day = day + pd.Timedelta(days=1)
            split = 'train' if next_day <= train_end else ('validation' if next_day <= val_end else 'test')
            label_cutoff = {'train': train_end, 'validation': val_end, 'test': data_end}[split]
            if ready is not None and ready > label_cutoff:
                continue
            stop = int(np.searchsorted(times, next_day.to_datetime64(), side='left'))
            if stop == 0:
                continue
            start_time = next_day - pd.Timedelta(days=config['window_days'])
            start = int(np.searchsorted(times, start_time.to_datetime64(), side='left'))
            fallback = start >= stop
            if fallback:
                start = stop - 1  # Last known state, with its REAL timestamp and explicit staleness.
            last = group.iloc[stop - 1]
            gap = int((day - last.created_at.normalize()).days)
            origin = day + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
            row = dict(sample_id=json.dumps([str(lot), str(day.date())], ensure_ascii=False),
                       lot_id=lot, product_id=last.product_id, forecast_date=day, forecast_at=origin,
                       last_snapshot_at=last.created_at, first_snapshot_at=first,
                       row_start=int(group.index[0]) + start, row_stop=int(group.index[0]) + stop,
                       sequence_length=stop-start, days_since_last_snapshot=gap,
                       no_snapshot_today=bool(gap), stale_fallback=fallback,
                       known_quantity=last.current_wafer_qty, completion_at=completion,
                       label_available_at=ready, remaining_days=(completion.normalize()-day).days
                       if completion is not None else None, split=split if labels is not None else 'inference')
            rows.append(row)
    if requested is not None and requested - encountered:
        raise ValueError(f'No available root history for requested Lots: {sorted(requested-encountered)}')
    return pd.DataFrame(rows, columns=INDEX_COLUMNS)


def fit_preprocessor(snapshots, core, config):
    core = deepcopy(core)
    train_rows = snapshots.loc[snapshots.created_at < midnight(config['train_end'])].copy()
    if train_rows.empty:
        raise ValueError('No observations before training cutoff.')
    _, preprocessor = transform_df(train_rows, core['embed'])
    return preprocessor, core


def transform_snapshots(snapshots, core, preprocessor):
    return transform_df(snapshots, core['embed'], preprocessor=preprocessor)


class DailyDataset(Dataset):
    def __init__(self, transformed, index, core):
        self.index = index.reset_index(drop=True)
        self.numeric = torch.tensor(transformed[core['embed']['numeric_feat_cols']].to_numpy(), dtype=torch.float32)
        self.temporal = torch.tensor(transformed[core['embed']['temporal_feat_cols']].to_numpy(), dtype=torch.float32)
        self.categorical = {c['name']: torch.tensor(transformed[c['name']].to_numpy(), dtype=torch.long)
                            for c in core['embed']['categorical_feat_cols']}
        self.days = transformed.created_at.to_numpy().astype('datetime64[D]').astype(np.int64)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        row = self.index.iloc[i]
        a, b = int(row.row_start), int(row.row_stop)
        day = np.datetime64(row.forecast_date, 'D').astype(np.int64)
        age = (row.forecast_date - row.first_snapshot_at.normalize()).days
        context = np.column_stack((np.log1p(day-self.days[a:b]),
                                   np.full(b-a, np.log1p(age)),
                                   np.full(b-a, float(row.no_snapshot_today))))
        inputs = {'numeric': torch.cat((self.numeric[a:b], torch.tensor(context, dtype=torch.float32)), dim=1),
                  'temporal': self.temporal[a:b],
                  'categorical': {name: values[a:b] for name, values in self.categorical.items()},
                  'length': b-a}
        target = None if pd.isna(row.remaining_days) else int(row.remaining_days)
        return i, inputs, target

    def loader(self, batch_size, shuffle=False, seed=42):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_daily,
                          generator=torch.Generator().manual_seed(seed), num_workers=0)


def collate_daily(batch):
    ids, items, labels = zip(*batch)
    inputs = {key: pad_sequence([item[key] for item in items], batch_first=True)
              for key in ('numeric', 'temporal')}
    inputs['categorical'] = {key: pad_sequence([item['categorical'][key] for item in items], batch_first=True)
                             for key in items[0]['categorical']}
    inputs['length'] = torch.tensor([item['length'] for item in items], dtype=torch.long)
    if any(y is None for y in labels) and not all(y is None for y in labels):
        raise ValueError('Cannot mix labelled and unlabelled samples in one batch.')
    targets = None if labels[0] is None else torch.tensor(labels, dtype=torch.long)
    return np.asarray(ids), inputs, targets


def index_summary(index):
    return {name: {'windows': len(g), 'lots': int(g.lot_id.nunique()),
                   'first_forecast_date': str(g.forecast_date.min()),
                   'last_forecast_date': str(g.forecast_date.max()),
                   'latest_label_available_at': str(g.label_available_at.max()),
                   'label_min': int(g.remaining_days.min()), 'label_max': int(g.remaining_days.max()),
                   'missing_today_windows': int(g.no_snapshot_today.sum()),
                   'stale_fallback_windows': int(g.stale_fallback.sum())}
            for name, g in index.groupby('split')}
