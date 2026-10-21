"""Weekly point inference and 7-period rolling APP, using frozen data and CPLEX.

No predictor training or test-based parameter tuning. Future actual receipts are
read only by the executor AFTER the current production/order decisions are fixed.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import platform
import sys
import time
import traceback
import zipfile
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT/'src'))
sys.dont_write_bytecode = True

from scripts.app_dataset.generate import load_decision, require, sha, write_csv, write_json
from scripts.app_experiment.lp import build_app_lp, solve_cplex


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def timestamp():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def arrival_periods(predicted_days, period_days):
    """Previous-day-end integer label: days 1..7 arrive in current 7-day period.

    A pending order with a 0-day (or earlier) forecast is deferred to the current
    period, never credited to the already-known opening inventory.
    """
    days = np.asarray(predicted_days, dtype=float)
    require(np.isfinite(days).all() and (days >= 0).all(), 'Invalid point prediction')
    rounded = np.ceil(days).astype(np.int64)
    return np.maximum(rounded-1,0)//period_days + 1


def fit_history_baseline(index_path, cutoff):
    index = pd.read_csv(index_path, parse_dates=['completion_at','first_snapshot_at','label_available_at'])
    training = index.loc[index.split == 'train'].drop_duplicates('lot_id').copy()
    require(len(training)>0 and (training.label_available_at <= cutoff).all(), 'Baseline labels unavailable at training cutoff')
    training['cycle_days'] = (training.completion_at.dt.normalize()-training.first_snapshot_at.dt.normalize()).dt.days
    require((training.cycle_days >= 0).all(), 'Negative total cycle')
    means = training.groupby('product_id').cycle_days.mean()
    fitted = {'rule':'training unique-Lot mean total calendar cycle minus elapsed, clipped at zero',
              'cutoff':str(cutoff), 'unique_training_lots':len(training),
              'product_cycle_mean_days':means.to_dict(), 'global_fallback_days':float(training.cycle_days.mean()),
              'uses_stage':False, 'index_sha256':sha(index_path)}
    return fitted


def baseline_predict(orders, fitted):
    cycle = orders.woItem.map(fitted['product_cycle_mean_days']).fillna(fitted['global_fallback_days'])
    return np.maximum(cycle.to_numpy()-orders.elapsed_calendar_days.to_numpy(),0)


def predict_weeks(dataset_dir, split, checkpoint, baseline_index, out, experiment, phase_callback):
    from wip_pred.daily.runner import predict_checkpoint, code_hashes
    checkpoint_config = read_json(checkpoint/'daily_config.json')
    require(checkpoint_config['mode']=='point', 'Point checkpoint required')
    current = code_hashes()
    require(all(current.get(p)==h for p,h in checkpoint_config['data_manifest']['source_hashes'].items()),
            'Predictor source differs from the selected checkpoint')
    protocol = checkpoint_config['protocol']
    baseline = fit_history_baseline(baseline_index, pd.Timestamp(protocol['train_end']))
    write_json(out/'baseline_fit.json', baseline)
    dataset_config = read_json(dataset_dir/'config_frozen.json')
    require(protocol['window_days']==dataset_config['history_window_days'], 'History windows differ')
    require(protocol['train_end']==dataset_config['splits']['train']['end_exclusive'], 'Training split differs')
    calendar = pd.read_csv(dataset_dir/split/'inputs/decision_calendar.csv', parse_dates=['decision_at','prediction_date'])
    frames, verifications = [], []
    for row in calendar.itertuples():
        phase_callback(f'predicting_period_{row.period}')
        view = load_decision(dataset_dir, split, row.period)
        orders = view['orders']
        folder = out/'inference'/f'period_{row.period:02d}'
        folder.mkdir(parents=True, exist_ok=False)
        print(f'Weekly inference {row.period}/{len(calendar)}: {len(orders)} Lots', flush=True)
        if len(orders):
            predictions = predict_checkpoint(checkpoint, folder, row.prediction_date.date().isoformat(), orders.lot_id.tolist())
            require(set(predictions.lot_id)==set(orders.lot_id), 'Prediction order set differs')
            pred = predictions.set_index('lot_id').loc[orders.lot_id]
            require(np.array_equal(pd.to_datetime(pred.last_snapshot_at).to_numpy(), pd.to_datetime(orders.last_snapshot_at).to_numpy()),
                    'Inference history end differs from APP visible history')
            require(np.array_equal(pd.to_datetime(pred.first_snapshot_at).to_numpy(), pd.to_datetime(orders.first_seen_at).to_numpy()),
                    'Inference history origin differs')
            require(np.array_equal(pred.sequence_length.to_numpy(), orders.history_length.to_numpy()), 'History sequence lengths differ')
            values = pred.predicted_remaining_days.to_numpy()
            raw = pred.raw_predicted_remaining_days.to_numpy()
        else:
            values, raw = np.empty(0),np.empty(0)
        for method, predicted in [('point_transformer',values),('historical_product_cycle',baseline_predict(orders, baseline))]:
            frame = orders.drop(columns='prediction_days').copy()
            frame['method'] = method
            frame['predicted_remaining_days'] = predicted
            frame['raw_predicted_remaining_days'] = raw if method=='point_transformer' else predicted
            frame['arrival_relative_period'] = arrival_periods(predicted,dataset_config['period_days'])
            frame['predicted_calendar_completion_day'] = row.prediction_date + pd.to_timedelta(np.ceil(predicted),unit='D')
            frame['earliest_arrival_deferred_to_current_period'] = predicted <= 0
            frame['beyond_planning_horizon'] = frame.arrival_relative_period > dataset_config['planning_periods']
            require(not set(frame.columns)&{'completion_at','remaining_days','label_available_at'}, 'Future labels in forecasts')
            frames.append(frame)
        verifications.append({'period':row.period,'lots':len(orders),'history_alignment':True,'truth_labels_used':False})
    result = pd.concat(frames,ignore_index=True)
    write_csv(out/'weekly_predictions.csv',result)
    write_json(out/'inference_checks.json',verifications)
    return result, baseline, dataset_config


def forecast_supply(predictions, material_ids, horizon):
    arrivals = np.zeros((horizon,len(material_ids)))
    for row in predictions.itertuples():
        if row.arrival_relative_period <= horizon:
            arrivals[int(row.arrival_relative_period)-1, material_ids.index(row.material_id)] += row.fixed_quantity
    require((np.diff(arrivals.cumsum(axis=0),axis=0)>=0).all(), 'Non-monotonic cumulative forecast')
    require(arrivals.sum() <= predictions.fixed_quantity.sum()+1e-6, 'Forecast double counts quantity')
    return arrivals


def carry_finished_state(opening_finished, opening_backlog, production, demand):
    net = np.asarray(opening_finished)-np.asarray(opening_backlog)+np.asarray(production)-np.asarray(demand)
    return np.maximum(net,0),np.maximum(-net,0)


def period_cost(production, finished, backlog, secondary_received, costs, terminal=False, terminal_penalty=0):
    components = {'production_cost':float(np.sum(production)*costs['production_per_product_unit']),
                  'finished_holding_cost':float(np.sum(finished)*costs['finished_holding_per_product_unit_period']),
                  'backlog_cost':float(np.sum(backlog)*costs['backlog_per_product_unit_period']),
                  'secondary_receipt_cost':float(np.sum(secondary_received)*costs['secondary_purchase_per_die']),
                  'terminal_backlog_cost':float(np.sum(backlog)*terminal_penalty) if terminal else 0.}
    components['total_cost'] = sum(components.values())
    return components


def run_method(method, predictions, dataset_dir, split, config, solver_config, out, arrival_builder=None):
    ds_config = read_json(dataset_dir/'config_frozen.json')
    params = read_json(dataset_dir/'shared/parameters.json')
    costs = params['costs']
    material_ids = [m['material_id'] for m in ds_config['materials']]
    products = ds_config['products']
    bom = np.asarray(ds_config['bom'],dtype=float)
    n, lead, days = ds_config['planning_periods'], ds_config['secondary']['effective_lead_periods'], ds_config['period_days']
    calendar = pd.read_csv(dataset_dir/split/'inputs/decision_calendar.csv',parse_dates=['decision_at','execution_end_exclusive'])
    t = len(calendar)
    initial = read_json(dataset_dir/split/'inputs/initial_state.json')
    raw = np.asarray([initial['raw_inventory_die'][m] for m in material_ids],dtype=float)
    finished = np.asarray([initial['finished_inventory_product_units'][p] for p in products],dtype=float)
    backlog = np.asarray([initial['customer_backlog_product_units'][p] for p in products],dtype=float)
    initial_raw,initial_finished,initial_backlog = raw.copy(),finished.copy(),backlog.copy()
    require(not initial['secondary_in_transit_orders'], 'Pilot initializes empty secondary pipeline')
    pipeline, received_lots = [], set()
    period_rows, material_rows, product_rows, plans, solves, receipts_seen, primary_forecasts, purchase_rows = [],[],[],[],[],[],[],[]
    total_secondary_received,total_primary_received,total_used = np.zeros(len(raw)),np.zeros(len(raw)),np.zeros(len(raw))
    max_execution_violation = 0.
    for step in calendar.itertuples():
        p,at = step.period,step.decision_at
        view = load_decision(dataset_dir,split,p)
        # Fixed outstanding orders from earlier executed actions arrive at this opening.
        due = np.zeros(len(raw))
        for order in pipeline:
            if order['due_period']==p:
                due[material_ids.index(order['material_id'])] += order['quantity']
        pipeline = [o for o in pipeline if o['due_period']>p]
        raw = raw+due
        opening_raw,opening_finished,opening_backlog = raw.copy(),finished.copy(),backlog.copy()
        committed = np.zeros((n,len(raw)))
        for order in pipeline:
            k=order['due_period']-p
            if 0 <= k < n: committed[k,material_ids.index(order['material_id'])] += order['quantity']
        selected = predictions.loc[(predictions.method==method)&(predictions.period==p)].copy()
        require(set(selected.lot_id)==set(view['orders'].lot_id), 'Forecasts not aligned with current visible orders')
        require(not selected.lot_id.isin(received_lots).any(), 'Already received batch forecast again')
        # Both point and distribution adapters use this same LP and actual executor.
        arrivals = (forecast_supply if arrival_builder is None else arrival_builder)(selected,material_ids,n)
        require(arrivals.shape == (n,len(material_ids)) and np.isfinite(arrivals).all()
                and (arrivals >= 0).all(), 'Invalid planning supply panel')
        demand = view['demand'].pivot(index='period',columns='product_id',values='demand').reindex(columns=products).sort_index().to_numpy()
        capacity = view['capacity'].sort_values('period').capacity_product_units.to_numpy()
        caps = view['secondary_supply'].pivot(index='period',columns='material_id',values='capacity_die').reindex(columns=material_ids).sort_index().to_numpy()
        model = build_app_lp(demand,capacity,bom,opening_raw,opening_finished,opening_backlog,
                             arrivals,committed,caps,lead,costs,params['terminal_backlog_penalty_per_product_unit'])
        plan,audit = solve_cplex(model,out/method/'cplex'/f'period_{p:02d}',solver_config['executable'],
            config['solver_threads'],config['solver_time_limit_seconds'],config['feasibility_tolerance'])
        audit['period']=p; solves.append(audit)
        print(f'{method} period {p}/{t}: optimal, planned objective {audit["objective"]:.2f}',flush=True)
        production, placed = plan['p'][0],plan['q'][0]
        used = bom@production
        finished,backlog = carry_finished_state(opening_finished,opening_backlog,production,demand[0])
        violations = [max(0,float(np.max(used-opening_raw))),max(0,float(production.sum()-capacity[0])),
                      float(np.max(np.abs(finished-plan['i'][0]))),float(np.max(np.abs(backlog-plan['b'][0])))]
        max_execution_violation = max(max_execution_violation,*violations)
        require(max(violations)<=config['feasibility_tolerance'],'Executed first period differs from feasible plan')
        # Commit only this period's q; q for future periods in the plan is discarded.
        for i,mat in enumerate(material_ids):
            order={'order_id':f'{method}_p{p}_{mat}','placed_period':p,'due_period':p+lead,
                   'material_id':mat,'quantity':float(placed[i]),'unit_cost':costs['secondary_purchase_per_die']}
            purchase_rows.append(order)
            if placed[i]>0: pipeline.append(order)
        # Replay actual outcomes ONLY AFTER the first-period plan/actions are fixed.
        all_receipts = pd.read_csv(dataset_dir/split/'replay/actual_receipts.csv',parse_dates=['completion_at','usable_from'])
        realized = all_receipts.loc[(all_receipts.completion_at>=at)&(all_receipts.completion_at<step.execution_end_exclusive)]
        require(not realized.lot_id.isin(received_lots).any(),'Repeated realized arrival')
        require(realized.usable_from.eq(step.execution_end_exclusive).all(),'Primary availability boundary differs')
        received_lots.update(realized.lot_id)
        actual_primary = realized.groupby('material_id').fixed_quantity.sum().reindex(material_ids,fill_value=0).to_numpy(dtype=float)
        receipts_seen.append(realized.assign(execution_period=p))
        completeness = pd.read_csv(dataset_dir/split/'replay/period_arrivals.csv')
        last_info = completeness.loc[completeness.period==p].set_index('material_id').loc[material_ids]
        if p<t:
            require(last_info.receipt_total_complete_for_selected_cohort.all(), 'Unknown receipts affect a later executed period')
        # raw now denotes known receipts included at the closing boundary, before next-period secondary receipts.
        raw = opening_raw-used+actual_primary
        require(raw.min()>=-config['feasibility_tolerance'],'Negative carried raw material')
        raw=np.maximum(raw,0)
        total_secondary_received += due; total_primary_received += actual_primary; total_used += used
        cost = period_cost(production,finished,backlog,due,costs,terminal=p==t,
                           terminal_penalty=params['terminal_backlog_penalty_per_product_unit'])
        period_rows.append({'period':p,'decision_at':at,'execution_end_exclusive':step.execution_end_exclusive,
            'production_total':float(production.sum()),'demand_total':float(demand[0].sum()),
            'finished_inventory_end':float(finished.sum()),'customer_backlog_end':float(backlog.sum()),
            'secondary_ordered_die':float(placed.sum()),'secondary_received_die':float(due.sum()),
            'known_primary_received_die':float(actual_primary.sum()),'planning_objective_not_realized_cost':audit['objective'],
            **cost})
        for i,mat in enumerate(material_ids):
            material_rows.append({'period':p,'material_id':mat,'opening_raw_after_secondary_receipts':opening_raw[i],
                'material_consumed':used[i],'primary_arrived_during_period_known':actual_primary[i],
                'secondary_received_at_opening':due[i],'secondary_new_order':placed[i],
                'closing_raw_known_lower_bound':raw[i],
                'unresolved_arrival_quantity':int(last_info.iloc[i].unresolved_possible_arrival_quantity_die)})
        for j,product in enumerate(products):
            product_rows.append({'period':p,'product_id':product,'opening_finished':opening_finished[j],
                'opening_backlog':opening_backlog[j],'demand':demand[0,j],'production':production[j],
                'shipments':opening_finished[j]+production[j]-finished[j],
                'closing_finished':finished[j],'closing_backlog':backlog[j]})
        for k in range(n):
            for i,mat in enumerate(material_ids):
                primary_forecasts.append({'decision_period':p,'relative_period':k+1,'material_id':mat,
                    'predicted_primary_arrival_die':arrivals[k,i],
                    'predicted_cumulative_primary_die':arrivals[:k+1,i].sum(),
                    'committed_secondary_receipts_die':committed[k,i]})
                plans.append({'decision_period':p,'relative_period':k+1,'variable':'secondary_order',
                              'entity':mat,'value':plan['q'][k,i],'executed':k==0})
            for j,product in enumerate(products):
                for variable,family in [('production','p'),('finished_stock','i'),('backlog','b')]:
                    plans.append({'decision_period':p,'relative_period':k+1,'variable':variable,
                                  'entity':product,'value':plan[family][k,j],'executed':k==0})
    period_frame = pd.DataFrame(period_rows)
    period_frame['cumulative_realized_cost']=period_frame.total_cost.cumsum()
    material_frame,product_frame = pd.DataFrame(material_rows),pd.DataFrame(product_rows)
    require(np.allclose(raw,initial_raw+total_primary_received+total_secondary_received-total_used,atol=1e-5,rtol=0),'Global raw balance')
    product_demand=product_frame.groupby('product_id').demand.sum().reindex(products).to_numpy()
    product_made=product_frame.groupby('product_id').production.sum().reindex(products).to_numpy()
    require(np.allclose(finished-backlog,initial_finished-initial_backlog+product_made-product_demand,atol=1e-5,rtol=0),'Global finished/backlog balance')
    pending = np.asarray([sum(o['quantity'] for o in pipeline if o['material_id']==m) for m in material_ids])
    purchased = pd.DataFrame(purchase_rows).groupby('material_id').quantity.sum().reindex(material_ids).to_numpy()
    require(np.allclose(purchased,total_secondary_received+pending,atol=1e-5,rtol=0),'Secondary pipeline conservation')
    unknown_qty = last_info.unresolved_possible_arrival_quantity_die.to_numpy(dtype=float)
    terminal = {'at':str(calendar.execution_end_exclusive.iloc[-1]),
        'raw_stock_timing':'After last-period known primary arrivals, before next-period secondary receipts',
        'raw_inventory_exact':bool(last_info.receipt_total_complete_for_selected_cohort.all()),
        'raw_inventory_lower_bound_die':dict(zip(material_ids,raw)),
        'raw_inventory_upper_bound_die':dict(zip(material_ids,raw+unknown_qty)),
        'finished_inventory':dict(zip(products,finished)),'customer_backlog':dict(zip(products,backlog)),
        'secondary_pipeline':pipeline,'secondary_pipeline_unpaid_cost_outside_evaluation':float(pending.sum()*costs['secondary_purchase_per_die']),
        'primary_pipeline_file':'terminal_primary_orders_pending_or_unknown.csv',
        'unresolved_final_receipts_not_set_to_zero':True}
    folder=out/method
    for name,frame in [('period_costs',period_frame),('material_states',material_frame),('product_states',product_frame),
                       ('all_plans',pd.DataFrame(plans)),('primary_forecasts',pd.DataFrame(primary_forecasts)),
                       ('secondary_orders',pd.DataFrame(purchase_rows)),('realized_primary_receipts',pd.concat(receipts_seen,ignore_index=True))]:
        write_csv(folder/f'{name}.csv',frame)
    # This table may contain uncertain outcomes; preserve their status from the frozen dataset.
    boundary=pd.read_csv(dataset_dir/split/'replay/terminal_primary_orders_pending_or_unknown.csv')
    write_csv(folder/'terminal_primary_orders_pending_or_unknown.csv',boundary)
    write_json(folder/'terminal_state.json',terminal)
    write_json(folder/'solver_checks.json',solves)
    components=[k for k in period_frame.columns if k.endswith('_cost') and k not in ['planning_objective_not_realized_cost','cumulative_realized_cost']]
    summary={'method':method,'periods':t,'costs':{k:float(period_frame[k].sum()) for k in components},
        'terminal_backlog':float(backlog.sum()),'terminal_finished_stock':float(finished.sum()),
        'secondary_received_die':float(total_secondary_received.sum()),'secondary_ordered_die':float(purchased.sum()),
        'secondary_pending_die':float(pending.sum()),'primary_known_arrivals_lots':len(received_lots),
        'primary_known_arrivals_die':float(total_primary_received.sum()),
        'cplex_optimal_solves':len(solves),'cplex_seconds':sum(x['seconds'] for x in solves),
        'max_planning_constraint_residual':max(max(x[k] for k in ['max_equality_residual','max_inequality_violation','max_nonnegativity_violation','max_upper_bound_violation']) for x in solves),
        'max_execution_violation':max_execution_violation,'raw_balance_passed':True,'finished_balance_passed':True,
        'secondary_pipeline_balance_passed':True,'unknown_terminal_outcome_does_not_enter_cost':True,
        'terminal_raw_inventory_exact':terminal['raw_inventory_exact']}
    write_json(folder/'summary.json',summary)
    return summary


def evaluate_forecasts_after_execution(predictions,dataset_dir,split,out,period_days):
    receipts=pd.read_csv(dataset_dir/split/'replay/actual_receipts.csv',parse_dates=['completion_at'])
    # Test receipts cover labelled orders in this pilot; unknown Lots stay unscored.
    frame=predictions.merge(receipts[['lot_id','completion_at']],on='lot_id',how='left',validate='many_to_one')
    frame['true_remaining_days']=(frame.completion_at.dt.normalize()-pd.to_datetime(frame.prediction_date)).dt.days
    frame['absolute_error_days']=(frame.predicted_remaining_days-frame.true_remaining_days).abs()
    rows=[]
    for method,g in frame.groupby('method',sort=False):
        labelled=g.loc[g.true_remaining_days.notna()]
        require((labelled.true_remaining_days>=1).all(),'Scored an already-completed order')
        error=labelled.predicted_remaining_days-labelled.true_remaining_days
        actual_period=arrival_periods(labelled.true_remaining_days,period_days)
        rows.append({'method':method,'forecast_rows':len(g),'labelled_rows':len(labelled),
            'unknown_rows':int(g.true_remaining_days.isna().sum()),'mae_days':float(error.abs().mean()),
            'rmse_days':float(np.sqrt(np.mean(error**2))),
            'same_arrival_week_fraction':float(np.mean(actual_period==labelled.arrival_relative_period.to_numpy())),
            'quantity_weighted_same_week_fraction':float(np.average(actual_period==labelled.arrival_relative_period.to_numpy(),weights=labelled.fixed_quantity))})
    write_csv(out/'forecast_evaluation_only.csv',frame)
    write_json(out/'forecast_metrics.json',rows)
    return rows


def run(config_path,output=None):
    start=time.perf_counter()
    config_path=Path(config_path).resolve()
    config=yaml.safe_load(config_path.read_text(encoding='utf-8'))
    out=Path(output).resolve() if output else PROJECT/'outputs/app/pto_runs'/datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_point_v1')
    out.mkdir(parents=True,exist_ok=False)
    state={'status':'starting','started_at':timestamp(),'config':config,'output':str(out),'training_performed':False}
    def phase(value):
        state.update(status=value,updated_at=timestamp())
        write_json(out/'run_manifest.json',state)
    try:
        phase('verifying_frozen_inputs')
        dataset_dir=(PROJECT/config['dataset_dir']).resolve()
        manifest=read_json(dataset_dir/'manifest.json')
        protected={str(dataset_dir/p):h for p,h in manifest['file_sha256'].items()}
        protected.update(manifest['protected_source_sha256'])
        protected[str(dataset_dir/'manifest.json')]=sha(dataset_dir/'manifest.json')
        checkpoint=(PROJECT/config['point_checkpoint']).resolve()
        baseline_index=(PROJECT/config['training_sample_index']).resolve()
        protected[str(baseline_index)]=sha(baseline_index)
        protected[str(config_path)]=sha(config_path)
        solver_path=PROJECT/config['solver_config']
        protected[str(solver_path.resolve())]=sha(solver_path)
        for path in Path(__file__).parent.glob('*.py'): protected[str(path)]=sha(path)
        protected[str(PROJECT/'scripts/app_dataset/generate.py')]=sha(PROJECT/'scripts/app_dataset/generate.py')
        require(all(Path(p).is_file() and sha(p)==h for p,h in protected.items()),'Input hash changed')
        solver_config=yaml.safe_load(solver_path.read_text(encoding='utf-8'))
        require(solver_config['solver']=='cplex' and Path(solver_config['executable']).is_file(),'CPLEX unavailable')
        write_json(out/'protected_files.json',protected)
        write_json(out/'experiment_config.json',config)
        write_json(out/'solver_config.json',solver_config)
        write_json(out/'runtime_environment.json',{'python':platform.python_version(),'platform':platform.platform(),
            'numpy':np.__version__,'pandas':pd.__version__,'cplex_executable_sha256':sha(solver_config['executable'])})
        with zipfile.ZipFile(out/'app_source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as archive:
            for source in sorted(Path(__file__).parent.glob('*.py')):
                archive.write(source,source.relative_to(PROJECT).as_posix())
            archive.write(PROJECT/'scripts/app_dataset/generate.py','scripts/app_dataset/generate.py')
            archive.write(config_path,'config/app_pto_small.yaml')
            archive.write(solver_path,'config/app_solver.yaml')
        predictions,baseline,ds_config=predict_weeks(dataset_dir,config['split'],checkpoint,baseline_index,out,config,phase)
        results=[]
        for method in config['methods']:
            phase(f'rolling_{method}')
            results.append(run_method(method,predictions,dataset_dir,config['split'],config,solver_config,out))
        phase('reporting')
        metrics=evaluate_forecasts_after_execution(predictions,dataset_dir,config['split'],out,ds_config['period_days'])
        require(all(sha(p)==h for p,h in protected.items()),'Source, frozen data or model changed during run')
        require(len({r['primary_known_arrivals_die'] for r in results})==1,'Methods used different actual supply')
        # Same demand/BOM/capacity/costs/initial state and model are protected by hashes.
        summary={'experiment_version':config['experiment_version'],'split':config['split'],'methods':results,
            'forecast_metrics':metrics,'baseline_fit':baseline,'dataset':str(dataset_dir),'point_checkpoint':str(checkpoint),
            'point_best_epoch':read_json(checkpoint/'training_state.json')['best_epoch'],
            'config':config,'dataset_config':ds_config,'elapsed_seconds':time.perf_counter()-start,
            'completed_at':timestamp(),'checks':{'frozen_inputs_and_models_unchanged':True,
                'same_exogenous_data_across_methods':True,'all_cplex_solves_optimal':True,
                'only_first_period_actions_executed':True,'no_inventory_reset_between_periods':True,
                'secondary_cost_charged_on_receipt':True,'terminal_backlog_charged_once':True,
                'future_receipts_read_only_after_actions_fixed':True,'training_or_test_tuning_performed':False},
            'limitations':['Single test trajectory and synthetic downstream parameters; not evidence of general cost superiority.',
                'Enterprise inferred completion, fixed first dieQty, no transport, whole-Lot arrivals, experimental material mapping.',
                'Offline quality selection and excluded left-boundary Lots; known future orders are not announced before first observation.',
                'Unknown final receipts affect terminal raw/pipeline precision, not first-seven-period realized costs under current policy.',
                'NLL not run; previous point checkpoint reused without training.']}
        write_json(out/'experiment_summary.json',summary)
        from scripts.app_experiment.report import render_report
        render_report(out,summary)
        write_json(out/'artifact_hashes.json',{str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*'))
            if p.is_file() and p.name not in {'artifact_hashes.json','run_manifest.json'}})
        state.update(status='completed',completed_at=timestamp(),elapsed_seconds=time.perf_counter()-start,
                     summary_file='experiment_summary.json',report_file='experiment_report.html')
        write_json(out/'run_manifest.json',state)
        print(json.dumps({'output':str(out),'results':results,'forecast_metrics':metrics},ensure_ascii=False,indent=2),flush=True)
        return summary
    except Exception as error:
        state.update(status='failed',error=f'{type(error).__name__}: {error}',updated_at=timestamp())
        write_json(out/'run_manifest.json',state)
        (out/'failure_traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=PROJECT/'config/app_pto_small.yaml')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    run(args.config,args.output)
