"""Build deterministic APP pilot inputs, separate replay outcomes, and audit checks.

Run from local_wip: .venv/Scripts/python.exe scripts/app_dataset/generate.py
No model inference, optimization, or rolling execution occurs here.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'src'))
sys.dont_write_bytecode = True


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2,
                               allow_nan=False) + '\n', encoding='utf-8')


def write_csv(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding='utf-8-sig', float_format='%.12g',
                 date_format='%Y-%m-%d %H:%M:%S.%f', lineterminator='\n')


def validate_config(c):
    require(c['schema_version'] == 'app_dataset_v1', 'Unknown schema')
    require(c['timezone'] == 'Asia/Shanghai', 'Source timestamps are Asia/Shanghai')
    require(c['quantity_column'] == 'dieQty' and c['quantity_unit'] == 'die',
            'This pilot fixes the FIRST dieQty, without wafer substitution')
    require(c['left_boundary_policy'] == 'exclude_and_record_incomplete_history', 'Unknown left policy')
    require(c['unknown_right_boundary_policy'] == 'retain_orders_and_mark_outcome_unknown', 'Unknown right policy')
    for key in ('period_days', 'planning_periods', 'history_window_days'):
        require(isinstance(c[key], int) and c[key] > 0, f'Invalid {key}')
    materials = [x['material_id'] for x in c['materials']]
    items = [x['woItem'] for x in c['materials']]
    require(len(set(materials)) == len(materials) == len(set(items)), 'Mapping must be one-to-one')
    a = np.asarray(c['bom'], dtype=float)
    require(a.shape == (len(materials), len(c['products'])), 'BOM shape')
    require(np.isfinite(a).all() and (a >= 0).all() and (a.sum(0) > 0).all()
            and (a.sum(1) > 0).all(), 'BOM contains invalid or isolated entries')
    previous = None
    for spec in c['splits'].values():
        start, end = pd.Timestamp(spec['start']), pd.Timestamp(spec['end_exclusive'])
        require(start == start.normalize() and end == end.normalize() and end > start, 'Split dates')
        require((end - start).days % c['period_days'] == 0, 'Incomplete macro period')
        require(previous is None or start == previous, 'Splits must be contiguous')
        previous = end
    require(len({s['rng_stream'] for s in c['splits'].values()}) == len(c['splits']), 'RNG streams must differ')
    require(0 < c['demand']['multiplier_low'] <= c['demand']['multiplier_high'], 'Demand bounds')
    require(0 < c['capacity']['target_mean_utilization'] <= 1, 'Utilization')
    require(c['secondary']['effective_lead_periods'] >= 1, 'Secondary lead must be positive')
    require(c['demand']['visibility'] == 'full_planning_window', 'Unsupported demand visibility')
    require(c['costs']['raw_holding_per_die_period'] == 0, 'Revisit unknown terminal outcomes before charging raw holding')
    require(c['initial_state']['secondary_pipeline'] == 'empty', 'Unsupported initial pipeline')
    require(c['capacity']['resource_per_product_unit'] == 1, 'Pilot capacity is measured in product units')


def read_upstream(c):
    """Enterprise normalization is called unchanged on the entire supplied CSV."""
    from loguru import logger
    from wip_pred.prep_data.normalize_data import normalize_df
    from wip_pred.prep_data.prepare_data import get_df_of_completed_lots
    logger.remove()
    source = (PROJECT / c['source_csv']).resolve()
    require(sha(source) == c['expected_source_sha256'], 'Source changed; review config before using it')
    raw = pd.read_csv(source)
    raw['file_timestamp'] = pd.to_datetime(raw.file_timestamp, errors='raise')
    raw['stage'] = raw.stage.fillna('NULL')  # same as daily predictor
    enterprise = normalize_df(raw.copy(), 'LotId', 'parentLotId', 'woItem', 'stage',
                              'tranCode', 'waferQty', 'file_timestamp',
                              do_time_binning=True, normalize_by_date=False)
    labels = get_df_of_completed_lots(enterprise, c['completion_margin_days']).reset_index(drop=True)
    margin = pd.Timedelta(days=c['completion_margin_days'])
    bounds = enterprise.groupby('lot_id').created_at.agg(['min', 'max'])
    left, right = enterprise.created_at.min() + margin, enterprise.created_at.max() - margin
    mapping = {x['woItem']: x['material_id'] for x in c['materials']}
    selected = raw.loc[raw.woItem.isin(mapping)].copy()
    selected['source_row_id'] = selected.index  # original CSV data row, zero-based (physical line +2)
    selected = selected.sort_values(['LotId', 'file_timestamp', 'source_row_id'], kind='stable')
    require(selected.groupby('LotId').woItem.nunique().max() == 1, 'Lot maps to multiple materials')
    require(selected.groupby('LotId').parentLotId.nunique().max() == 1, 'Inconsistent parent')
    require((selected.LotId == selected.parentLotId).all(), 'Pilot requires root Lots')
    require(selected.dieQty.notna().all() and (selected.dieQty > 0).all(), 'Missing/nonpositive dieQty')
    require((selected.waferQty == 0).all(), 'Quantity unit conflict requires review')
    first = selected.drop_duplicates('LotId', keep='first')
    lots = first[['LotId', 'parentLotId', 'woItem', 'file_timestamp', 'dieQty', 'source_row_id']].rename(
        columns={'LotId': 'lot_id', 'parentLotId': 'parent_lot_id', 'file_timestamp': 'first_seen_at',
                 'dieQty': 'fixed_quantity', 'source_row_id': 'first_source_row_id'})
    lots['material_id'] = lots.woItem.map(mapping)
    lots['supplier_id'] = c['supplier_id']
    lots['quantity_unit'] = c['quantity_unit']
    lots['last_seen_at'] = lots.lot_id.map(selected.groupby('LotId').file_timestamp.max())
    lots = lots.merge(labels[['lot_id', 'end_time']].rename(columns={'end_time': 'completion_at'}),
                      on='lot_id', how='left', validate='one_to_one')
    lots['label_available_at'] = lots.completion_at + margin
    lots['enterprise_quality_retained'] = lots.lot_id.isin(bounds.index)
    lots['left_boundary'] = lots.lot_id.map(bounds['min']) < left
    lots['right_boundary'] = lots.lot_id.map(bounds['max']) > right
    lots['included'] = lots.enterprise_quality_retained & ~lots.left_boundary
    lots['exclusion_reason'] = np.where(~lots.enterprise_quality_retained, 'enterprise_quality_filter',
                                        np.where(lots.left_boundary, 'left_boundary_incomplete_history', ''))
    require((lots.loc[lots.included, 'completion_at'].notna() |
             lots.loc[lots.included, 'right_boundary']).all(), 'Unexpected missing completion reason')
    # No later quantity, final stage, or future outcome is present in the observation table.
    snapshots = selected.loc[selected.LotId.isin(lots.loc[lots.included, 'lot_id']),
        ['source_row_id', 'LotId', 'woItem', 'file_timestamp', 'stage', 'tranCode', 'oper', 'waferQty']].rename(
        columns={'LotId': 'lot_id', 'woItem': 'product_id', 'file_timestamp': 'created_at',
                 'tranCode': 'lot_status', 'waferQty': 'current_wafer_qty'})
    duplicate_cols = [x for x in snapshots.columns if x != 'source_row_id']
    before = len(snapshots)
    snapshots = snapshots.drop_duplicates(duplicate_cols, keep='last').reset_index(drop=True)
    snapshots.insert(0, 'snapshot_id', snapshots.index)
    summary = {
        'supplier_id_meaning': c['supplier_id_origin'],
        'raw': {'snapshots': len(raw), 'lots': raw.LotId.nunique(), 'woItems': raw.woItem.nunique(),
                'first_at': raw.file_timestamp.min(), 'last_at': raw.file_timestamp.max(),
                'supplier_namespaces': 1},
        'selected_before_boundary_screen': {'snapshots': len(selected), 'lots': len(lots), 'materials': len(mapping)},
        'included': {'snapshots': len(snapshots), 'lots': int(lots.included.sum()), 'materials': len(mapping),
                     'first_at': snapshots.created_at.min(), 'last_at': snapshots.created_at.max()},
        'excluded_left_lots': int(lots.left_boundary.sum()),
        'excluded_enterprise_quality_lots': int((~lots.enterprise_quality_retained).sum()),
        'right_unknown_retained_lots': int((lots.included & lots.right_boundary).sum()),
        'exact_predictor_input_duplicates_removed': before - len(snapshots),
        'raw_same_lot_timestamp_duplicate_rows': int(selected.duplicated(['LotId', 'file_timestamp']).sum()),
        'enterprise_left_margin': left, 'enterprise_right_margin': right,
        'quantity_changes_ignored_lots': int((selected.groupby('LotId').dieQty.nunique() > 1).sum()),
        'all_selected_root': True, 'all_selected_quantities_positive_die': True,
        'lot_material_mapping_stable': True,
    }
    return lots.sort_values('lot_id').reset_index(drop=True), snapshots, summary


def calibrate(lots, c):
    spec = c['calibration']
    start, end, cutoff = (pd.Timestamp(spec[k]) for k in ('start', 'end_exclusive', 'label_cutoff_exclusive'))
    require(cutoff <= pd.Timestamp(c['splits']['train']['end_exclusive']), 'Calibration crosses train cutoff')
    require(end <= cutoff and (end-start).days % c['period_days'] == 0, 'Calibration dates')
    weeks = (end - start).days // c['period_days']
    eligible = lots.loc[lots.included & (lots.completion_at >= start) & (lots.completion_at < end)
                        & (lots.label_available_at < cutoff)].copy()
    ids = [x['material_id'] for x in c['materials']]
    supply = eligible.groupby('material_id').fixed_quantity.sum().reindex(ids, fill_value=0).to_numpy() / weeks
    consumption = supply * spec['material_consumption_to_supply']
    a = np.asarray(c['bom'], dtype=float)
    require(a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == len(ids),
            'This generator requires an invertible BOM to determine product means')
    means = np.linalg.solve(a, consumption)
    require((means > 0).all(), 'BOM yields nonpositive product demand; do not tune from held-out supply')
    require(np.allclose(a @ means, consumption), 'Per-material calibration mismatch')
    params = {'reference_weeks': weeks, 'calibration_cutoff_exclusive': str(cutoff),
        'mean_weekly_primary_supply_die': dict(zip(ids, supply)),
        'mean_weekly_material_consumption_die': dict(zip(ids, consumption)),
        'mean_weekly_product_demand': dict(zip(c['products'], means)),
        'capacity_product_units_per_period': float(means.sum() * c['capacity']['resource_per_product_unit'] /
                                                    c['capacity']['target_mean_utilization']),
        'secondary_capacity_die_per_period': dict(zip(ids, consumption * c['secondary']['capacity_to_mean_material_consumption'])),
        'initial_raw_inventory_die': dict(zip(ids, consumption * c['initial_state']['raw_stock_mean_consumption_periods'])),
        'costs': c['costs'], 'seed': c['seed'],
        'terminal_backlog_penalty_per_product_unit': c['costs']['backlog_per_product_unit_period'] * c['costs']['terminal_backlog_extra_periods'],
        'origin': 'Supply scale from training only; BOM, costs, capacity ratios and demand law are synthetic experiment settings'}
    return params, eligible


def make_demand(c, params, split):
    spec = c['splits'][split]
    days, n = c['period_days'], c['planning_periods']
    start, end = pd.Timestamp(spec['start']), pd.Timestamp(spec['end_exclusive'])
    t = (end-start).days // days
    rng = np.random.default_rng(np.random.SeedSequence(c['seed'], spawn_key=(spec['rng_stream'],)))
    means = np.asarray([params['mean_weekly_product_demand'][p] for p in c['products']])
    multipliers = rng.uniform(c['demand']['multiplier_low'], c['demand']['multiplier_high'], (t+n-1, len(means)))
    rows = []
    for k, factor in enumerate(multipliers, 1):
        for j, product in enumerate(c['products']):
            rows.append({'period': k, 'period_start': start+pd.Timedelta(days=(k-1)*days),
                'period_end_exclusive': start+pd.Timedelta(days=k*days), 'product_id': product,
                'demand': means[j]*factor[j], 'demand_unit': 'product_unit',
                'is_evaluation_period': k <= t, 'origin': 'synthetic_train_calibrated'})
    return pd.DataFrame(rows)


def eligible_orders(lots, decision_at):
    """Consume only already-occurring replay receipts, never future completion values.

    In the simulator a receipt, once it occurs, is assumed observable. The 3-day
    margin governs offline label certification, not physical transportation.
    """
    occurred = lots.completion_at.notna() & (lots.completion_at < decision_at)
    active = lots.included & (lots.first_seen_at < decision_at) & ~occurred
    cols = ['lot_id', 'parent_lot_id', 'woItem', 'material_id', 'supplier_id', 'first_seen_at',
            'first_source_row_id', 'fixed_quantity', 'quantity_unit']
    return lots.loc[active, cols].copy()


def decision_panel(lots, snapshots, dates, c):
    groups = {lot: g for lot, g in snapshots.groupby('lot_id', sort=False)}
    rows = []
    for period, at in enumerate(dates, 1):
        for order in eligible_orders(lots, at).to_dict('records'):
            g = groups[order['lot_id']]
            times = g.created_at.to_numpy(dtype='datetime64[ns]')
            stop = int(np.searchsorted(times, at.to_datetime64(), side='left'))
            require(stop > 0, 'Visible order missing history')
            begin = int(np.searchsorted(times, (at-pd.Timedelta(days=c['history_window_days'])).to_datetime64(), side='left'))
            stale = begin >= stop
            if stale:
                begin = stop - 1
            last = g.iloc[stop-1]
            forecast_day = at - pd.Timedelta(days=1)
            rows.append({**order, 'period': period, 'decision_at': at,
                'prediction_date': forecast_day, 'history_start_row': int(g.index[0])+begin,
                'history_stop_row_exclusive': int(g.index[0])+stop,
                'last_snapshot_at': last.created_at, 'history_length': stop-begin,
                'elapsed_calendar_days': int((forecast_day-order['first_seen_at'].normalize()).days),
                'snapshot_age_days': int((forecast_day-last.created_at.normalize()).days),
                'stale_fallback': stale, 'last_stage': last.stage, 'last_status': last.lot_status,
                'last_oper': last.oper, 'prediction_days': np.nan})
    columns = ['lot_id','parent_lot_id','woItem','material_id','supplier_id','first_seen_at','first_source_row_id',
               'fixed_quantity','quantity_unit','period','decision_at','prediction_date','history_start_row',
               'history_stop_row_exclusive','last_snapshot_at','history_length','elapsed_calendar_days',
               'snapshot_age_days','stale_fallback','last_stage','last_status','last_oper','prediction_days']
    return pd.DataFrame(rows, columns=columns)


def arrival_events(lots, anchor, days):
    events = lots.loc[lots.included & lots.completion_at.notna(),
        ['lot_id', 'material_id', 'fixed_quantity', 'quantity_unit', 'completion_at', 'label_available_at']].copy()
    events['arrival_day'] = events.completion_at.dt.normalize()
    events['usable_from'] = anchor + pd.to_timedelta(((events.arrival_day-anchor).dt.days // days + 1)*days, unit='D')
    events['origin'] = 'enterprise_inferred_completion_whole_lot_no_transport_assumption'
    return events.sort_values(['completion_at', 'lot_id']).reset_index(drop=True)


def split_tables(lots, snapshots, events, c, params, split, raw_end):
    spec = c['splits'][split]
    start, end = pd.Timestamp(spec['start']), pd.Timestamp(spec['end_exclusive'])
    days, n = c['period_days'], c['planning_periods']
    dates = pd.date_range(start, end, inclusive='left', freq=f'{days}D')
    t = len(dates)
    panel = decision_panel(lots, snapshots, dates, c)
    demand = make_demand(c, params, split)
    horizons, secondary = [], []
    decisions = pd.DataFrame({'period': range(1, t+1), 'decision_at': dates,
                             'information_cutoff_exclusive': dates,
                             'prediction_date': dates-pd.Timedelta(days=1),
                             'execution_end_exclusive': dates+pd.Timedelta(days=days)})
    capacity = []
    for k in range(1, t+n):
        capacity.append({'period': k, 'capacity_product_units': params['capacity_product_units_per_period'],
                         'origin': 'synthetic_train_calibrated'})
        for mat in c['materials']:
            secondary.append({'period': k, 'material_id': mat['material_id'],
                'capacity_die': params['secondary_capacity_die_per_period'][mat['material_id']],
                'effective_lead_periods': c['secondary']['effective_lead_periods'],
                'usable_at': start+pd.Timedelta(days=(k-1+c['secondary']['effective_lead_periods'])*days),
                'purchase_cost_per_die': c['costs']['secondary_purchase_per_die'], 'origin': 'synthetic'})
    for p, at in enumerate(dates, 1):
        for k in range(1, n+1):
            horizons.append({'decision_period': p, 'relative_period': k, 'calendar_period': p+k-1,
                'start_at': at+pd.Timedelta(days=(k-1)*days), 'end_exclusive': at+pd.Timedelta(days=k*days),
                'execute_now': k == 1,
                'secondary_order_arrives_in_horizon': k+c['secondary']['effective_lead_periods'] <= n})
    receipts = events.loc[(events.completion_at >= start) & (events.completion_at < end)].copy()
    label_cutoff = raw_end if split == 'test' else end
    labels = lots.loc[lots.included & (lots.first_seen_at < end) & (lots.label_available_at < label_cutoff),
        ['lot_id', 'material_id', 'completion_at', 'label_available_at']].copy()
    labels['cutoff_exclusive'] = label_cutoff
    unknown = lots.loc[lots.included & lots.completion_at.isna() & (lots.first_seen_at < end)].copy()
    new_orders = lots.loc[lots.included & (lots.first_seen_at >= start) & (lots.first_seen_at < end),
        ['lot_id', 'material_id', 'first_seen_at', 'fixed_quantity', 'quantity_unit', 'first_source_row_id']].copy()
    boundary = eligible_orders(lots, end)
    initial = {'trajectory_id': split, 'at': str(start), 'initialization': 'independent_synthetic_state_once_at_origin',
        'raw_inventory_die': params['initial_raw_inventory_die'],
        'finished_inventory_product_units': {p: c['initial_state']['finished_stock'] for p in c['products']},
        'customer_backlog_product_units': {p: c['initial_state']['backlog'] for p in c['products']},
        'secondary_in_transit_orders': [], 'primary_pipeline_file': 'initial_primary_orders.csv',
        'past_primary_receipts_not_added_again': True}
    weekly, cumulative = [], {m['material_id']: 0 for m in c['materials']}
    for p, at in enumerate(dates, 1):
        until = at + pd.Timedelta(days=days)
        arrivals = receipts.loc[(receipts.completion_at >= at) & (receipts.completion_at < until)]
        for mat in cumulative:
            group = arrivals.loc[arrivals.material_id == mat]
            # Latest observed time supplies a lower bound, not an invented completion.
            pending = unknown.loc[(unknown.material_id == mat) & (unknown.last_seen_at < until)]
            quantity = int(group.fixed_quantity.sum())
            cumulative[mat] += quantity
            weekly.append({'period': p, 'material_id': mat, 'period_start': at,
                'period_end_exclusive': until, 'confirmed_lots': len(group),
                'known_arrival_quantity_die': quantity, 'cumulative_known_arrivals_die': cumulative[mat],
                'unresolved_possible_arrival_lots': len(pending),
                'unresolved_possible_arrival_quantity_die': int(pending.fixed_quantity.sum()),
                'receipt_total_complete_for_selected_cohort': pending.empty})
    tables = {'inputs/decision_calendar.csv': decisions, 'inputs/planning_horizons.csv': pd.DataFrame(horizons),
        'inputs/orders_at_decision.csv': panel, 'inputs/demand.csv': demand,
        'inputs/capacity.csv': pd.DataFrame(capacity), 'inputs/secondary_supply.csv': pd.DataFrame(secondary),
        'inputs/initial_primary_orders.csv': panel.loc[panel.period == 1].copy(),
        'replay/actual_receipts.csv': receipts, 'replay/new_primary_order_events.csv': new_orders,
        'replay/unknown_outcomes.csv': unknown, 'replay/period_arrivals.csv': pd.DataFrame(weekly),
        'replay/terminal_primary_orders_pending_or_unknown.csv': boundary,
        'labels/available_completion_labels.csv': labels}
    return tables, initial


def check_tables(lots, snapshots, events, datasets, c, params, protected_before):
    checks = []
    def check(name, condition, detail=''):
        require(bool(condition), f'{name}: {detail}')
        checks.append({'check': name, 'passed': True, 'detail': detail})
    check('one_actual_arrival_per_lot', events.lot_id.is_unique)
    qty = lots.set_index('lot_id').fixed_quantity
    check('whole_lot_arrival_equals_first_quantity', np.array_equal(events.fixed_quantity.to_numpy(), events.lot_id.map(qty).to_numpy()))
    check('positive_arrivals', (events.fixed_quantity > 0).all())
    check('cumulative_arrivals_do_not_exceed_total_batch_quantity', events.fixed_quantity.sum() <= lots.loc[lots.included, 'fixed_quantity'].sum())
    check('common_units_die_and_product_units', c['quantity_unit'] == 'die' and c['demand']['unit'] == 'product_unit')
    check('arrivals_available_next_period_only', (events.usable_from > events.completion_at).all())
    check('no_completion_before_first_visibility_day', (events.arrival_day >= events.lot_id.map(lots.set_index('lot_id').first_seen_at).dt.normalize()).all())
    check('material_balance_of_means', np.allclose(np.asarray(c['bom']) @ np.asarray(list(params['mean_weekly_product_demand'].values())), list(params['mean_weekly_material_consumption_die'].values())))
    all_event_ids = []
    for split, (tables, initial) in datasets.items():
        p = tables['inputs/orders_at_decision.csv']
        check(f'{split}/unique_batch_per_decision', not p.duplicated(['period', 'lot_id']).any())
        check(f'{split}/orders_visible_before_decision', (p.first_seen_at < p.decision_at).all())
        check(f'{split}/latest_snapshot_before_decision', (p.last_snapshot_at < p.decision_at).all())
        check(f'{split}/no_future_outcome_columns_in_planning_orders',
              not set(p.columns) & {'completion_at', 'label_available_at', 'last_seen_at', 'right_boundary', 'remaining_days'})
        check(f'{split}/predictions_not_fabricated', p.prediction_days.isna().all())
        check(f'{split}/fixed_known_quantity', np.array_equal(p.fixed_quantity.to_numpy(), p.lot_id.map(qty).to_numpy()))
        check(f'{split}/stable_material_mapping', (p.material_id == p.woItem.map({m['woItem']:m['material_id'] for m in c['materials']})).all())
        history_ok = True
        for row in p.itertuples():
            history = snapshots.iloc[row.history_start_row:row.history_stop_row_exclusive]
            history_ok &= (len(history) == row.history_length and history.lot_id.eq(row.lot_id).all()
                           and (history.created_at < row.decision_at).all()
                           and history.created_at.is_monotonic_increasing)
        check(f'{split}/all_history_windows_causal_and_ordered', history_ok)
        labels = tables['labels/available_completion_labels.csv']
        check(f'{split}/labels_available_before_cutoff', (labels.label_available_at < labels.cutoff_exclusive).all())
        check(f'{split}/no_random_state_reset_after_origin', len(initial['secondary_in_transit_orders']) == 0 and
              initial['initialization'] == 'independent_synthetic_state_once_at_origin')
        d = tables['inputs/demand.csv']
        t = len(tables['inputs/decision_calendar.csv'])
        check(f'{split}/demand_covers_last_full_planning_window', d.period.nunique() == t+c['planning_periods']-1)
        check(f'{split}/deterministic_demand_rng', d.equals(make_demand(c, params, split)))
        arrivals = tables['replay/period_arrivals.csv']
        check(f'{split}/cumulative_arrivals_non_decreasing', all(g.cumulative_known_arrivals_die.diff().dropna().ge(0).all() for _,g in arrivals.groupby('material_id')))
        check(f'{split}/weekly_receipts_conserve_quantity', arrivals.known_arrival_quantity_die.sum() == tables['replay/actual_receipts.csv'].fixed_quantity.sum())
        all_event_ids.extend(tables['replay/actual_receipts.csv'].lot_id.tolist())
    check('no_duplicate_receipts_across_splits', len(all_event_ids) == len(set(all_event_ids)))
    # Changing future completions cannot change the order set at an earlier decision.
    at = pd.Timestamp(c['splits']['test']['start'])
    altered = lots.copy()
    future = altered.completion_at >= at
    altered.loc[future, 'completion_at'] += pd.Timedelta(days=100)
    check('future_outcome_perturbation_leaves_decision_orders_identical', eligible_orders(lots, at).equals(eligible_orders(altered, at)))
    altered_calibration = lots.copy()
    late = altered_calibration.label_available_at >= pd.Timestamp(c['calibration']['label_cutoff_exclusive'])
    altered_calibration.loc[late, 'fixed_quantity'] *= 999
    check('held_out_outcomes_do_not_calibrate_downstream_parameters', jsonable(calibrate(altered_calibration, c)[0]) == jsonable(params))
    check('protected_source_and_predictor_files_unchanged', all(sha(p) == digest for p,digest in protected_before.items()))
    return checks


def load_decision(dataset_dir, split, period):
    """Safe input accessor. Does NOT read replay/ or labels/. Only one horizon is returned."""
    root = Path(dataset_dir)
    base = root / split / 'inputs'
    calendar = pd.read_csv(base / 'decision_calendar.csv', parse_dates=['decision_at'])
    require(period in calendar.period.values, 'Unknown decision period')
    at = calendar.set_index('period').at[period, 'decision_at']
    orders = pd.read_csv(base / 'orders_at_decision.csv')
    orders = orders.loc[orders.period == period].copy()
    observations = pd.read_csv(root / 'shared' / 'observations.csv', parse_dates=['created_at'])
    rows = []
    for order in orders.itertuples():
        history = observations.iloc[order.history_start_row:order.history_stop_row_exclusive].copy()
        require((history.created_at < at).all(), 'Future snapshot')
        require(history.lot_id.eq(order.lot_id).all(), 'Mismatched history')
        rows.append(history)
    histories = pd.concat(rows, ignore_index=True) if rows else observations.iloc[:0].copy()
    h = pd.read_csv(base / 'planning_horizons.csv')
    h = h.loc[h.decision_period == period]
    future_periods = h.calendar_period
    return {'decision_at': at, 'orders': orders, 'snapshots': histories, 'planning_horizons': h,
            'demand': pd.read_csv(base / 'demand.csv').query('period in @future_periods').copy(),
            'capacity': pd.read_csv(base / 'capacity.csv').query('period in @future_periods').copy(),
            'secondary_supply': pd.read_csv(base / 'secondary_supply.csv').query('period in @future_periods').copy(),
            'initial_state': json.loads((base / 'initial_state.json').read_text(encoding='utf-8')) if period == 1 else None,
            'parameters': json.loads((root / 'shared' / 'parameters.json').read_text(encoding='utf-8'))}


def material_diagnostics(tables, initial, c):
    demand = tables['inputs/demand.csv']
    demand = demand.loc[demand.is_evaluation_period].pivot(index='period', columns='product_id', values='demand').reindex(columns=c['products'])
    a = np.asarray(c['bom'], dtype=float)
    consumption = demand.to_numpy() @ a.T
    arrivals = tables['replay/period_arrivals.csv']
    rows = []
    for j, mat in enumerate(c['materials']):
        mid = mat['material_id']
        w = arrivals.loc[arrivals.material_id == mid]
        total = float(consumption[:,j].sum())
        secondary_cap = tables['inputs/secondary_supply.csv'].query('material_id == @mid').iloc[0].capacity_die
        usable = float(w.iloc[:-1].known_arrival_quantity_die.sum())
        upper = initial['raw_inventory_die'][mid] + usable + secondary_cap*max(0,len(demand)-c['secondary']['effective_lead_periods'])
        rows.append({'material_id': mid, 'bom_implied_demand_die': total,
            'initial_raw_inventory_die': initial['raw_inventory_die'][mid],
            'known_primary_arrivals_die': int(w.known_arrival_quantity_die.sum()),
            'known_primary_usable_within_evaluation_die': usable,
            'maximum_secondary_receipts_within_evaluation_die': secondary_cap*max(0,len(demand)-c['secondary']['effective_lead_periods']),
            'quantity_upper_bound_to_demand_ratio': upper/total,
            'aggregate_material_shortfall_even_at_max_secondary_die': max(0,total-upper),
            'zero_known_arrival_periods': int(w.known_arrival_quantity_die.eq(0).sum()),
            'last_period_unknown_quantity_die': int(w.iloc[-1].unresolved_possible_arrival_quantity_die),
            'interpretation': 'Quantity diagnostic only; no optimized execution; capacity/timing can further restrict production'})
    return pd.DataFrame(rows)


def render_report(path, manifest, stats, diagnostics):
    c = manifest['config']
    def table(frame):
        return frame.to_html(index=False, border=0, float_format=lambda x:f'{x:,.2f}', escape=True)
    summary = pd.DataFrame(stats)[['split','periods','snapshot_rows_in_period','snapshot_lots_in_period',
                                  'decision_order_rows','initial_primary_lots','known_arrival_lots']].rename(columns={
        'split':'集合','periods':'评价期数','snapshot_rows_in_period':'期间快照数','snapshot_lots_in_period':'期间观测Lot数',
        'decision_order_rows':'决策时点订单行数','initial_primary_lots':'初始在途Lot数','known_arrival_lots':'已标注到货Lot数'})
    left_n = manifest['upstream']['excluded_left_lots']
    included = manifest['upstream']['included']
    parts = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>DK APP 小规模数据集检查报告</title>',
      '<style>body{font:16px/1.65 "Microsoft YaHei",sans-serif;max-width:1200px;margin:40px auto;padding:0 24px;color:#203048}h1,h2{color:#183c57}table{border-collapse:collapse;font-size:14px;width:100%;margin:18px 0}td,th{border-bottom:1px solid #d9e1e8;padding:8px;text-align:left}th{background:#edf4f8}.note{background:#fff4dc;padding:16px}code{overflow-wrap:anywhere}</style>',
      '<h1>DK APP 小规模数据集检查报告</h1>',
      '<p>3 种材料、3 种产品；7 天一期、规划 4 期；训练/验证/测试为独立初始化的 13/4/7 期片段。数据构造与一致性检查已完成，本轮没有运行预测、APP 求解或滚动执行。</p>',
      '<h2>数据范围</h2>',
      f'<p>所选材料原有 3,873 个 Lot；排除 {left_n} 个起始边界批次后，保留 {included["lots"]:,} 个 Lot、{included["snapshots"]:,} 条快照，其中含 29 个末尾未确认 Lot。各集合可重复观察跨边界 Lot，但每个 Lot 仅有一笔全局到货事件。</p>',
      table(summary),
      '<p>DK_dataset 是数据集命名空间，不是已核实的企业供应商编码。woItem 一对一映射为实验材料；批次数量固定为首次 dieQty，忽略后续数量变化。企业快照推定完工视为整批到货，无运输延迟。</p>',
      '<h2>冻结参数</h2>', table(pd.DataFrame([{'参数':k,'值':str(v)} for k,v in manifest['parameters'].items() if not isinstance(v,dict)])),
      table(pd.DataFrame({'材料':[m['material_id'] for m in c['materials']],
                         '训练周均供给':list(manifest['parameters']['mean_weekly_primary_supply_die'].values()),
                         '平均材料消耗':list(manifest['parameters']['mean_weekly_material_consumption_die'].values()),
                         '第二货源每期上限':list(manifest['parameters']['secondary_capacity_die_per_period'].values())})),
      '<p>BOM、需求、成本、产能及初始库存为合成实验设定。仅供给数量级来自 2024-10-08 至 12-23 的 11 个训练参考周，标签在训练截止前可用；验证/测试结果没有参与参数校准。</p>',
      '<h2>各材料供需规模</h2>']
    for split, frame in diagnostics.items():
        view = frame[['material_id','bom_implied_demand_die','known_primary_usable_within_evaluation_die',
                      'maximum_secondary_receipts_within_evaluation_die','quantity_upper_bound_to_demand_ratio',
                      'aggregate_material_shortfall_even_at_max_secondary_die']].rename(columns={
            'material_id':'材料','bom_implied_demand_die':'需求隐含消耗（die）',
            'known_primary_usable_within_evaluation_die':'期内可用第一货源（die）',
            'maximum_secondary_receipts_within_evaluation_die':'第二货源到货上限（die）',
            'quantity_upper_bound_to_demand_ratio':'含初始库存的供应上限/需求',
            'aggregate_material_shortfall_even_at_max_secondary_die':'仍存在的总量缺口（die）'})
        parts += [f'<h3>{html.escape(split)}</h3>', table(view)]
    parts += ['<p>上述上限含初始库存、评价期内可用的第一货源及最大可到货第二货源，不是优化结果。比值低于 1 表示即使第二货源用满也存在总量缺口；不据此调整测试需求。最后一期第一货源到货不能用于当期生产。</p>',
      '<h2>信息与结局分离</h2><p>inputs 仅含当时可见订单与快照索引；replay 保存事后到货及未来订单出现事件；labels 单独保存截止时已可用标签。请通过 load_decision 按决策时点读取输入，不能直接把所有快照和回放事件喂给预测器。已发生到货在模拟中视为可观测，3 天是离线标签确认缓冲，不是运输或收货延迟。</p>',
      f'<h2>边界与限制</h2><div class="note"><p>{left_n} 个左边界 Lot 因历史不完整被单列排除，29 个右边界未确认 Lot 保留。测试最后一期的 22 个 Lot、128,194 die 可能到货量仍未知；另外 7 个 Lot、49,037 die 在 3 月 18 日仍有观测。不把未知记为零实际到货。当前不计原材料库存成本及期末残值，最后一期到货只能下期使用，因此这一末尾结局问题不直接改变前 7 期已执行成本口径，但期末原材料/在途状态不完整。</p><p>本数据集以企业历史质量筛选后的完整起点批次为范围，训练初期不代表全厂真实供给；三种材料属于此前探索后由用户固定的小规模试验集合。独立初始化的片段不是连续运营，尚未验证优化可行性、成本或方法优劣。NLL 当前 H=14，未来接 28 天规划须另行扩展。</p></div>',
      '<h2>检查结果</h2>', table(pd.DataFrame(manifest['checks'])),
      '<p>同配置二次构造逐文件 SHA256 比对见 reproducibility.json；文件完整性清单见 manifest.json。字段与使用规则见项目 doc/app_dataset.md。</p></html>']
    path.write_text('\n'.join(parts), encoding='utf-8')


def generate(config_path, out):
    config_path, out = Path(config_path).resolve(), Path(out).resolve()
    require(not out.exists(), f'Output exists, refusing overwrite: {out}')
    c = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    validate_config(c)
    protected = [PROJECT / c['source_csv'], PROJECT / 'config/daily_dk.yaml']
    protected += sorted((PROJECT / 'src/wip_pred').rglob('*.py'))
    for mode in ('point', 'distribution'):
        checkpoint = PROJECT / 'outputs/dk/daily/full_runs/20261017_1908_pto_split/training' / mode
        protected += [checkpoint / name for name in ('best_model.pth','daily_config.json','preprocessor.pkl')]
    before = {str(p.resolve()): sha(p) for p in protected}
    print('Reading source and reconstructing enterprise labels...', flush=True)
    lots, snapshots, upstream = read_upstream(c)
    params, reference = calibrate(lots, c)
    events = arrival_events(lots, pd.Timestamp(c['splits']['train']['start']), c['period_days'])
    datasets, diagnostics, stats = {}, {}, []
    for split, spec in c['splits'].items():
        print(f'Building {split}...', flush=True)
        tables, initial = split_tables(lots, snapshots, events, c, params, split, upstream['raw']['last_at'])
        datasets[split] = tables, initial
        diagnostics[split] = material_diagnostics(tables, initial, c)
        panel = tables['inputs/orders_at_decision.csv']
        start, end = pd.Timestamp(spec['start']), pd.Timestamp(spec['end_exclusive'])
        obs = snapshots.loc[(snapshots.created_at >= start) & (snapshots.created_at < end)]
        stats.append({'split': split, 'periods': len(tables['inputs/decision_calendar.csv']),
            'materials': len(c['materials']), 'products': len(c['products']), 'supplier_namespaces': 1,
            'start': str(start.date()), 'end_exclusive': str(end.date()),
            'snapshot_rows_in_period': len(obs), 'snapshot_lots_in_period': obs.lot_id.nunique(),
            'decision_order_rows': len(panel), 'distinct_decision_lots': panel.lot_id.nunique(),
            'initial_primary_lots': len(tables['inputs/initial_primary_orders.csv']),
            'new_primary_lots': len(tables['replay/new_primary_order_events.csv']),
            'known_arrival_lots': len(tables['replay/actual_receipts.csv']),
            'known_arrival_quantity_die': int(tables['replay/actual_receipts.csv'].fixed_quantity.sum()),
            'demand_periods_with_lookahead': tables['inputs/demand.csv'].period.nunique()})
    checks = check_tables(lots, snapshots, events, datasets, c, params, before)
    # Build content hashes over deterministic data; wall-clock generation time is intentionally excluded.
    out.mkdir(parents=True)
    write_json(out/'config_frozen.json', c)
    write_json(out/'shared/parameters.json', params)
    write_csv(out/'shared/materials.csv', pd.DataFrame(c['materials']).assign(
        supplier_id=c['supplier_id'], unit='die', mapping_origin='user_fixed_experimental_one_to_one'))
    write_csv(out/'shared/products.csv', pd.DataFrame({'product_id': c['products'], 'unit': 'product_unit', 'origin': 'synthetic'}))
    write_csv(out/'shared/bom.csv', pd.DataFrame([{'material_id':m['material_id'], 'product_id':p,
        'die_per_product_unit':c['bom'][i][j], 'origin':'synthetic'} for i,m in enumerate(c['materials']) for j,p in enumerate(c['products'])]))
    write_csv(out/'shared/observations.csv', snapshots)
    write_csv(out/'audit/selected_lot_catalog.csv', lots)
    write_csv(out/'audit/excluded_lots.csv', lots.loc[~lots.included])
    write_csv(out/'audit/calibration_lots.csv', reference)
    write_csv(out/'audit/split_statistics.csv', pd.DataFrame(stats))
    screening = lots.groupby(['material_id','woItem'], sort=False).agg(
        selected_lots=('lot_id','size'), included_lots=('included','sum'),
        left_boundary_excluded=('left_boundary','sum'), right_boundary_retained=('right_boundary','sum'),
        first_seen_at=('first_seen_at','min'), last_seen_at=('last_seen_at','max')).reset_index()
    write_csv(out/'audit/material_screening.csv', screening)
    write_json(out/'audit/upstream_screening.json', upstream)
    for split, (tables, initial) in datasets.items():
        for name, frame in tables.items():
            write_csv(out/split/name, frame)
        write_json(out/split/'inputs/initial_state.json', initial)
        write_csv(out/split/'diagnostics/material_balance.csv', diagnostics[split])
    checks.append({'check':'saved_decision_accessor_has_no_future_rows', 'passed':True, 'detail':''})
    for split, (tables, _) in datasets.items():
        for period in tables['inputs/decision_calendar.csv'].period:
            view = load_decision(out, split, period)
            require((view['snapshots'].created_at < view['decision_at']).all(), 'Saved future rows')
            require(view['demand'].period.nunique() == c['planning_periods'], 'Saved horizon incomplete')
    manifest = {'schema_version':c['schema_version'], 'dataset_id':c['dataset_id'], 'config':c,
        'parameters':params, 'upstream':upstream, 'split_statistics':stats, 'checks':checks,
        'protected_source_sha256':before, 'builder_sha256':sha(__file__),
        'python':platform.python_version(), 'pandas':pd.__version__, 'numpy':np.__version__,
        'scope':'Dataset only; no inference, optimizer, rolling execution or cost results',
        'outcome_visibility_assumption':'Simulated receipts observable once occurred; 3-day delay only for offline label certification',
        'selection_disclosure':'Three materials fixed after exploratory supply diagnostics; full-history enterprise quality filter; left-history censoring excluded',
        'file_sha256':{str(p.relative_to(out)).replace('\\','/'):sha(p) for p in sorted(out.rglob('*')) if p.is_file()}}
    write_json(out/'manifest.json', manifest)
    render_report(out/'dataset_report.html', manifest, stats, diagnostics)
    write_json(out/'checks.json', checks)
    manifest['file_sha256'] = {str(p.relative_to(out)).replace('\\','/'):sha(p)
                               for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'manifest.json'}
    write_json(out/'manifest.json', manifest)
    print(json.dumps(jsonable({'output':str(out), 'upstream':upstream, 'parameters':params,
        'splits':stats, 'checks_passed':len(checks)}), ensure_ascii=False, indent=2), flush=True)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=PROJECT/'config/app_dk_small.yaml')
    parser.add_argument('--output', type=Path, default=PROJECT/'outputs/app/dk_3x3_v1_20261019')
    args = parser.parse_args()
    generate(args.config, args.output)
