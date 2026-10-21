"""Post-experiment audit: independently replay saved actions, not another solve."""
import argparse
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT))
from scripts.app_dataset.generate import require, sha, write_csv, write_json


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def audit_run(out):
    out=Path(out).resolve()
    summary=read(out/'experiment_summary.json')
    root=Path(summary['dataset']); config=read(root/'config_frozen.json')
    parameters=read(root/'shared/parameters.json'); costs=parameters['costs']
    mats=[m['material_id'] for m in config['materials']]; products=config['products']
    bom=np.asarray(config['bom']); n=config['planning_periods']; lead=config['secondary']['effective_lead_periods']
    split=summary['split']; inputs=root/split/'inputs'; replay=root/split/'replay'
    calendar=pd.read_csv(inputs/'decision_calendar.csv',parse_dates=['decision_at','execution_end_exclusive'])
    demand=pd.read_csv(inputs/'demand.csv').pivot(index='period',columns='product_id',values='demand').reindex(columns=products)
    caps=pd.read_csv(inputs/'secondary_supply.csv').pivot(index='period',columns='material_id',values='capacity_die').reindex(columns=mats)
    capacity=pd.read_csv(inputs/'capacity.csv').set_index('period').capacity_product_units
    initial=read(inputs/'initial_state.json')
    receipts=pd.read_csv(replay/'actual_receipts.csv',parse_dates=['completion_at'])
    predictions=pd.read_csv(out/'weekly_predictions.csv',parse_dates=['prediction_date','decision_at','last_snapshot_at','first_seen_at'])
    require(not {'completion_at','remaining_days','label_available_at'}&set(predictions.columns),'Truth in forecast file')
    if 'announced_at' in predictions:
        require((pd.to_datetime(predictions.announced_at)<=predictions.decision_at).all(),'Unannounced order in forecast')
        observed=predictions.loc[predictions.has_snapshot]
        unseen=predictions.loc[~predictions.has_snapshot]
        require((observed.last_snapshot_at<observed.decision_at).all(),'Future snapshot in forecast')
        require((observed.first_seen_at<observed.decision_at).all(),'Future first snapshot')
        require(unseen[['first_seen_at','last_snapshot_at']].isna().all().all(),'Unseen observation metadata')
        require((unseen.prediction_source=='synthetic_promise').all(),'Unseen order not using promise')
        require((unseen.history_length==0).all(),'Unseen order fabricated history')
    else:
        require((predictions.last_snapshot_at<predictions.decision_at).all(),'Future snapshot in forecast')
        require((predictions.first_seen_at<predictions.decision_at).all(),'Unseen order in forecast')
    require(predictions[['method','period','lot_id']].duplicated().sum()==0,'Duplicate forecast')
    hashes=read(out/'artifact_hashes.json')
    require(all(sha(out/p)==h for p,h in hashes.items()),'Output artifacts changed')
    frozen=read(root/'manifest.json')
    require(all(sha(root/p)==h for p,h in frozen['file_sha256'].items()),'Frozen dataset changed')
    require(all(sha(p)==h for p,h in frozen['protected_source_sha256'].items()),'Predictor or CSV changed')
    max_error=0.; checks=[]
    def close(a,b,label):
        nonlocal max_error
        err=float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
        max_error=max(max_error,err)
        require(err<2e-5,label+f': {err}')
    for result in summary['methods']:
        method=result['method']; folder=out/method
        raw=np.array([initial['raw_inventory_die'][m] for m in mats])
        fg=np.array([initial['finished_inventory_product_units'][p] for p in products])
        back=np.array([initial['customer_backlog_product_units'][p] for p in products])
        placed={}; totals=dict.fromkeys(result['costs'],0.); received=set()
        material_saved=pd.read_csv(folder/'material_states.csv')
        product_saved=pd.read_csv(folder/'product_states.csv')
        period_saved=pd.read_csv(folder/'period_costs.csv').set_index('period')
        for step in calendar.itertuples():
            t=step.period
            due=placed.get(t-lead,np.zeros(len(mats)))
            raw=raw+due
            root_sol=ET.parse(folder/'cplex'/f'period_{t:02d}'/'solution.sol').getroot()
            require(root_sol.find('header').get('solutionStatusValue')=='1','Not optimal')
            values={v.get('name'):float(v.get('value')) for v in root_sol.find('variables')}
            arrays={key:np.array([[values[f'{key}_{k+1}_{j+1}'] for j in range(width)] for k in range(n)])
                    for key,width in [('p',len(products)),('i',len(products)),('b',len(products)),('q',len(mats))]}
            pred=predictions.query('method == @method and period == @t')
            arrivals=np.zeros((n,len(mats)))
            if method == 'nll_expected_supply':
                probability_columns=[f'p_week_{k+1}' for k in range(n)]+['p_beyond_horizon']
                probabilities=pred[probability_columns].to_numpy(dtype=float)
                require(np.isfinite(probabilities).all() and (probabilities>=0).all(), 'Invalid weekly probabilities')
                close(probabilities.sum(axis=1),np.ones(len(pred)),'Weekly probability normalization')
                # Independently rebuild Q * P(week), without using the runtime adapter.
                for row in pred.itertuples():
                    for k in range(n):
                        arrivals[k,mats.index(row.material_id)] += row.fixed_quantity*getattr(row,f'p_week_{k+1}')
                require(arrivals.sum()<=pred.fixed_quantity.sum()+1e-6,'Expected quantity duplicated')
            else:
                for row in pred.itertuples():
                    if row.arrival_relative_period<=n: arrivals[int(row.arrival_relative_period)-1,mats.index(row.material_id)]+=row.fixed_quantity
            r,f,b=raw.copy(),fg.copy(),back.copy(); objective=0.
            for k in range(n):
                if k:
                    r=r+arrivals[k-1]
                    q_index=k-lead
                    r=r+(arrays['q'][q_index] if q_index>=0 else placed.get(t+q_index,np.zeros(len(mats))))
                production=arrays['p'][k]
                require(production.min()>-1e-5 and arrays['q'][k].min()>-1e-5,'Negative action')
                require(production.sum()<=capacity[t+k]+1e-5,'Plan capacity violation')
                require(np.all(bom@production<=r+1e-5),'Plan used not-yet-available raw material')
                require(np.all(arrays['q'][k]<=caps.loc[t+k].to_numpy()+1e-5),'Procurement cap violation')
                if k+lead>=n: close(arrays['q'][k],0.,'Order outside usable planning horizon')
                net=f-b+production-demand.loc[t+k].to_numpy()
                close(arrays['i'][k]-arrays['b'][k],net,'Plan product balance')
                f=np.maximum(net,0);b=np.maximum(-net,0);r=r-bom@production
                objective+=production.sum()*costs['production_per_product_unit']+f.sum()*costs['finished_holding_per_product_unit_period']
                objective+=b.sum()*costs['backlog_per_product_unit_period']+arrays['q'][k].sum()*costs['secondary_purchase_per_die']
                if k==n-1: objective+=b.sum()*parameters['terminal_backlog_penalty_per_product_unit']
            close(objective,float(root_sol.find('header').get('objectiveValue')),'Independent plan objective')
            production=arrays['p'][0]; placed[t]=arrays['q'][0]
            net=fg-back+production-demand.loc[t].to_numpy();fg=np.maximum(net,0);back=np.maximum(-net,0)
            actual=receipts[(receipts.completion_at>=step.decision_at)&(receipts.completion_at<step.execution_end_exclusive)]
            require(not set(actual.lot_id)&received,'Repeated receipt');received.update(actual.lot_id)
            actual_qty=actual.groupby('material_id').fixed_quantity.sum().reindex(mats,fill_value=0).to_numpy()
            material=material_saved.query('period == @t').set_index('material_id').loc[mats]
            product=product_saved.query('period == @t').set_index('product_id').loc[products]
            close(material.opening_raw_after_secondary_receipts,raw,'Opening raw state')
            close(material.material_consumed,bom@production,'Consumption')
            raw=raw-bom@production+actual_qty
            close(material.closing_raw_known_lower_bound,raw,'Closing raw state')
            close(product.production,production,'Executed production')
            close(product.closing_finished,fg,'Finished state');close(product.closing_backlog,back,'Backlog state')
            close(material.secondary_new_order,placed[t],'Executed procurement')
            computed={'production_cost':production.sum()*costs['production_per_product_unit'],
                'finished_holding_cost':fg.sum()*costs['finished_holding_per_product_unit_period'],
                'backlog_cost':back.sum()*costs['backlog_per_product_unit_period'],
                'secondary_receipt_cost':due.sum()*costs['secondary_purchase_per_die'],
                'terminal_backlog_cost':back.sum()*parameters['terminal_backlog_penalty_per_product_unit'] if t==len(calendar) else 0.}
            computed['total_cost']=sum(computed.values())
            for key,value in computed.items():
                close(period_saved.at[t,key],value,'Executed cost '+key);totals[key]+=value
        for key in totals: close(result['costs'][key],totals[key],'Total cost '+key)
        close(len(received),len(receipts),'All true receipts replayed once')
        checks.append({'method':method,'independent_sequential_plan_and_execution_checks':True,'periods':len(calendar)})
    # Diagnose decision sensitivity, without changing actions or tuning parameters.
    names=['point_transformer','historical_product_cycle']
    pred=[predictions[predictions.method==m].sort_values(['period','lot_id']).reset_index(drop=True) for m in names]
    require(pred[0][['period','lot_id']].equals(pred[1][['period','lot_id']]),'Different prediction populations')
    diff=pred[0][['period','lot_id','material_id','fixed_quantity']].copy()
    diff['point_days']=pred[0].predicted_remaining_days;diff['baseline_days']=pred[1].predicted_remaining_days
    diff['point_week']=pred[0].arrival_relative_period;diff['baseline_week']=pred[1].arrival_relative_period
    write_csv(out/'prediction_week_differences.csv',diff[diff.point_week!=diff.baseline_week])
    a=pd.read_csv(out/names[0]/'all_plans.csv');b=pd.read_csv(out/names[1]/'all_plans.csv')
    require(a.drop(columns='value').equals(b.drop(columns='value')),'Different plan axes')
    delta=a.copy();delta['baseline_value']=b.value;delta['point_minus_baseline']=a.value-b.value
    write_csv(out/'plan_differences.csv',delta[abs(delta.point_minus_baseline)>1e-5])
    invisible=[]
    for step in calendar.itertuples():
        t=step.period; actual=receipts[(receipts.completion_at>=step.decision_at)&(receipts.completion_at<step.execution_end_exclusive)]
        known=set(pred[0].loc[pred[0].period==t,'lot_id'])
        missing=actual[~actual.lot_id.isin(known)]
        invisible.append({'period':t,'receipt_lots':len(actual),'receipt_die':float(actual.fixed_quantity.sum()),
                          'not_known_at_week_start_lots':len(missing),'not_known_at_week_start_die':float(missing.fixed_quantity.sum())})
    write_csv(out/'receipt_visibility_diagnostic.csv',pd.DataFrame(invisible))
    material=pd.read_csv(out/names[0]/'material_states.csv')
    saturation=[]
    for mat,g in material.groupby('material_id'):
        limits=caps.loc[g.period,mat].to_numpy()
        saturation.append({'material':mat,'procurement_at_cap_periods':g.loc[np.isclose(g.secondary_new_order,limits,rtol=0,atol=1e-5),'period'].tolist(),
            'zero_procurement_periods':g.loc[g.secondary_new_order.abs()<1e-5,'period'].tolist(),
            'opening_stock_fully_consumed_periods':g.loc[(g.opening_raw_after_secondary_receipts-g.material_consumed).abs()<1e-5,'period'].tolist()})
    report={'frozen_dataset_and_predictor_hashes_passed':True,'output_hashes_passed':True,
        'independent_replay':checks,'max_audit_numeric_difference':max_error,
        'forecast_rows_per_method':len(pred[0]),'distinct_forecast_lots':int(pred[0].lot_id.nunique()),
        'different_day_predictions':int((abs(diff.point_days-diff.baseline_days)>1e-8).sum()),
        'different_week_predictions':int((diff.point_week!=diff.baseline_week).sum()),
        'max_executed_action_difference':float(delta.loc[delta.relative_period==1,'point_minus_baseline'].abs().max()),
        'max_future_plan_difference':float(delta.loc[delta.relative_period>1,'point_minus_baseline'].abs().max()),
        'future_plan_difference_decision_periods':sorted(delta.loc[delta.point_minus_baseline.abs()>1e-5,'decision_period'].unique().tolist()),
        'procurement_saturation':saturation,'primary_visibility':invisible,
        'purpose':'Post-run verification and explanation only; no parameter change, training, re-optimization or policy tuning.'}
    write_json(out/'post_run_audit.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run_dir',type=Path)
    audit_run(parser.parse_args().run_dir)
