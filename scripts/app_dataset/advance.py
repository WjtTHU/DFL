"""Build a new APP dataset and attach the frozen advance-order policy to TEST.

Training/validation tables remain calibration/reference data; no rolling results
are claimed there. The completed predictor is deployed only on the test trajectory.
"""
from pathlib import Path
import sys
import json
import argparse

import numpy as np
import pandas as pd
import yaml

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from scripts.app_dataset.generate import (generate, sha, require, write_csv,
    write_json, load_decision)
from scripts.app_dataset.order_information import visible_orders


def advance_panel(book, observed_panel, dates, occurred_receipts):
    """Only already-occurred receipts may remove announced pending orders.

    Observed rows provide the snapshot history. Unseen orders have empty history
    and no actual first/last observation time; planned entry is an assumption.
    """
    rows = []
    shared = set(book.columns)
    history_columns = [k for k in observed_panel.columns if k not in shared and k != 'period']
    for period, at in enumerate(dates, 1):
        completed = occurred_receipts.loc[occurred_receipts.completion_at < at, 'lot_id']
        known = visible_orders(book, at, completed)
        observed = observed_panel.loc[observed_panel.period == period]
        require(set(observed.lot_id) <= set(known.lot_id), 'Observed order absent from announcement book')
        frame = known.merge(observed[['lot_id', *history_columns]], on='lot_id', how='left', validate='one_to_one')
        frame['has_snapshot'] = frame.history_length.notna()
        frame['period'], frame['decision_at'] = period, at
        frame['prediction_date'] = at - pd.Timedelta(days=1)
        for key in ('history_start_row', 'history_stop_row_exclusive', 'history_length'):
            frame[key] = frame[key].fillna(0).astype(int)
        frame['prediction_days'] = np.nan
        require((frame.announced_at <= at).all(), 'Unannounced order')
        require((pd.to_datetime(frame.loc[frame.has_snapshot, 'last_snapshot_at']) < at).all(), 'Future snapshot')
        require(frame.loc[~frame.has_snapshot, ['first_seen_at','last_snapshot_at']].isna().all().all(),
                'Future observation metadata exposed')
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def build(config_path, out):
    config_path, out = Path(config_path).resolve(), Path(out).resolve()
    c = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    require(c['advance_order_splits'] == ['test'], 'Only test deployment is implemented')
    preflight = PROJECT/'outputs/dk/daily/preflight/20261020_train10w_h28'
    book_path = preflight/'advance_order_book.csv'
    policy_path = PROJECT/c['order_information_policy']
    policy = yaml.safe_load(policy_path.read_text(encoding='utf-8'))
    require(policy['training_label_cutoff'] == c['splits']['train']['end_exclusive'], 'Promise cutoff mismatch')
    require(policy['test_start'] == c['splits']['test']['start'] and
            policy['test_end_exclusive'] == c['splits']['test']['end_exclusive'], 'Test dates mismatch')
    initial_hashes = {str(p):sha(p) for p in [book_path, policy_path, config_path,
        preflight/'promise_training_calibration.json']}
    checkpoint = PROJECT/'outputs/dk/daily/full_runs/20261020_train10w_h28/training'
    for mode in ('point','distribution'):
        for name in ('best_model.pth','daily_config.json','preprocessor.pkl'):
            p = checkpoint/mode/name
            initial_hashes[str(p)] = sha(p)
    manifest = generate(config_path, out)
    catalog = pd.read_csv(out/'audit/selected_lot_catalog.csv',
                          parse_dates=['completion_at','first_seen_at'])
    book = pd.read_csv(book_path, parse_dates=['announced_at','planned_entry_date','promised_completion_date'])
    included = catalog.loc[catalog.included]
    require(set(book.lot_id) == set(included.lot_id), 'Order cohort mismatch')
    require(np.array_equal(book.fixed_quantity.to_numpy(),
                           book.lot_id.map(included.set_index('lot_id').fixed_quantity).to_numpy()), 'Order quantity mismatch')
    base = out/'test/inputs'
    calendar = pd.read_csv(base/'decision_calendar.csv', parse_dates=['decision_at'])
    observed = pd.read_csv(base/'orders_at_decision.csv', parse_dates=['first_seen_at','last_snapshot_at'])
    occurrences = included.loc[included.completion_at.notna(), ['lot_id','completion_at']]
    panel = advance_panel(book, observed, calendar.decision_at, occurrences)
    write_csv(base/'orders_at_decision.csv', panel)
    write_csv(base/'initial_primary_orders.csv', panel.loc[panel.period == 1])
    # Actual receipt records remain separate and unchanged.
    end = pd.Timestamp(c['splits']['test']['end_exclusive'])
    completed = occurrences.loc[occurrences.completion_at < end, 'lot_id']
    write_csv(out/'test/replay/terminal_primary_orders_pending_or_unknown.csv', visible_orders(book, end, completed))
    write_csv(out/'shared/advance_order_book.csv', book)
    write_json(out/'shared/order_information_policy.json', policy)
    write_json(out/'shared/promise_training_calibration.json',
               json.loads((preflight/'promise_training_calibration.json').read_text(encoding='utf-8')))
    # Future labels may change, but neither current order visibility nor inputs may.
    changed = occurrences.copy()
    at = calendar.decision_at.iloc[0]
    changed.loc[changed.completion_at >= at, 'completion_at'] += pd.Timedelta(days=100)
    require(advance_panel(book, observed, [at], occurrences).equals(
        advance_panel(book, observed, [at], changed)), 'Future outcomes changed inputs')
    checks = []
    for step in calendar.itertuples():
        view = load_decision(out, 'test', step.period)
        require(len(view['demand'].period.unique()) == c['planning_periods'], 'Incomplete lookahead demand')
        require((view['snapshots'].created_at < step.decision_at).all(), 'Future history')
        require(not view['orders'].lot_id.duplicated().any(), 'Duplicate Lot')
        known = view['orders']
        require((pd.to_datetime(known.announced_at) <= step.decision_at).all(), 'Unannounced Lot')
        unseen = known.loc[~known.has_snapshot]
        require((unseen.history_length == 0).all(), 'Unseen Lot has fabricated snapshots')
        checks.append({'period':step.period,'orders':len(known),'observed':int(known.has_snapshot.sum()),
                       'promise_only':len(unseen),'time_and_quantity_checks':True})
    final_horizon = calendar.decision_at.iloc[-1] + pd.Timedelta(days=c['period_days']*c['planning_periods'])
    require(final_horizon == pd.Timestamp(policy['planning_extension_end_exclusive']), 'Planning extension mismatch')
    require(final_horizon <= pd.Timestamp(manifest['upstream']['raw']['last_at']), 'Planning exceeds observation coverage')
    require(all(sha(p) == h for p,h in initial_hashes.items()), 'Protected inputs changed')
    write_json(out/'advance_order_checks.json', {
        'periods':checks, 'future_outcome_perturbation_passed':True,
        'fixed_quantity_and_cohort_match':True, 'last_horizon_end_exclusive':str(final_horizon),
        'base_checks_scope':'checks.json applies to original snapshot-only reference generation; test order panel then extended and checked here',
        'deployment_scope':'Test only; train and validation are reference datasets, not rolling backtests'})
    manifest['protected_source_sha256'].update(initial_hashes)
    manifest['advance_order_checks'] = checks
    manifest['information_protocol'] = 'Advance order input only for test; after first snapshot use selected frozen predictor'
    summary = pd.DataFrame(checks).to_html(index=False, border=0)
    params = manifest['parameters']
    (out/'dataset_report.html').write_text(
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>新版 APP 测试数据</title>'
        '<style>body{max-width:1100px;margin:40px auto;font:16px/1.8 sans-serif}td,th{padding:8px;border-bottom:1px solid #ddd}</style>'
        '<h1>新版 3×3 APP 测试数据</h1><p>测试 2025-01-07 至 02-24，7 天一期，N=4、T=7；最后一次规划截至 03-17。</p>'
        '<p>订单公告提前首次快照日 28 天；数量固定首次 dieQty；尚无快照时使用合成承诺日期。未来完工只在 replay 中回放。</p>'
        '<p>下游尺度用 2024-10-08 至 12-02 共 8 个完整参考周、12-10 前已确认的训练标签校准。10 周训练跨度的两端不作为完整供给参考周。'
        '产能目标利用率 85%、第二货源上限为平均材料消耗的 30%，其他原设定不变。训练、验证为参考数据，本次只执行测试轨迹。</p>'
        + summary + '<h2>训练期校准参数</h2><pre>' +
        json.dumps(params,ensure_ascii=False,indent=2) +
        '</pre><p>原始 CSV、预测模型、旧实验均保留。数据生成不等于已完成滚动求解。</p></html>',encoding='utf-8')
    finalize(out, manifest)
    return manifest


def finalize(out, manifest=None):
    """Record final test population after the advance-order extension."""
    out=Path(out)
    manifest=manifest or json.loads((out/'manifest.json').read_text(encoding='utf-8'))
    panel=pd.read_csv(out/'test/inputs/orders_at_decision.csv')
    stats=manifest['split_statistics']
    for row in stats:
        if row['split']=='test':
            row['snapshot_only_reference_order_rows']=int(panel.has_snapshot.sum())
            row['decision_order_rows']=len(panel)
            row['distinct_decision_lots']=int(panel.lot_id.nunique())
            row['initial_primary_lots']=int((panel.period==1).sum())
            row['promise_only_rows']=int((~panel.has_snapshot).sum())
            row['snapshot_observed_rows']=int(panel.has_snapshot.sum())
    write_csv(out/'audit/split_statistics.csv',pd.DataFrame(stats))
    book=pd.read_csv(out/'shared/advance_order_book.csv',parse_dates=['announced_at'])
    spec=manifest['config']['splits']['test']
    write_csv(out/'test/replay/order_announcement_events.csv',
              book.loc[(book.announced_at>=pd.Timestamp(spec['start']))&
                       (book.announced_at<pd.Timestamp(spec['end_exclusive']))])
    manifest['advance_builder_sha256']=sha(__file__)
    manifest['file_sha256']={str(p.relative_to(out)).replace('\\','/'):sha(p)
                             for p in sorted(out.rglob('*')) if p.is_file() and p.name!='manifest.json'}
    write_json(out/'manifest.json',manifest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=PROJECT/'config/app_dk_10w_advance28.yaml')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.config, args.output)
