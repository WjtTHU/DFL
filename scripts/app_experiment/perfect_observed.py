"""Privileged-information diagnostic: perfect times ONLY for already observed Lots.

Unseen orders keep their existing synthetic promises. The same N=4 rolling LP,
execution, receipts, costs and initial state are used. Not a deployable predictor.
"""
from pathlib import Path
import argparse
import shutil
import sys
import time
import traceback
import zipfile

import numpy as np
import pandas as pd
import yaml

PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT))
from scripts.app_dataset.generate import load_decision, require, sha, write_csv, write_json
from scripts.app_experiment.run import read_json, timestamp, arrival_periods, run_method
from scripts.app_experiment.advance import output_hashes

METHOD='perfect_observed_diagnostic'


def forecast_orders(orders, truth, period_days, planning_periods):
    """Explicit oracle access is gated by the already-frozen has_snapshot field."""
    frame=orders.drop(columns='prediction_days').copy()
    require(not frame.lot_id.duplicated().any(),'Duplicate decision Lot')
    observed=frame.has_snapshot.to_numpy(dtype=bool)
    require(not truth.lot_id.duplicated().any(),'Ambiguous completion label')
    days=(pd.to_datetime(frame.promised_completion_date)-pd.to_datetime(frame.prediction_date)).dt.days.clip(lower=0).to_numpy(dtype=float)
    audit=frame.loc[observed,['period','lot_id','prediction_date','decision_at','fixed_quantity']].copy()
    # Do not join or expose completion times of unseen orders at all.
    completion=pd.to_datetime(truth.set_index('lot_id').completion_at.reindex(audit.lot_id),format='ISO8601').to_numpy()
    require(not pd.isna(completion).any(),'Observed Lot lacks completion; diagnostic cannot be called perfect')
    audit['privileged_completion_at']=completion
    remaining=(pd.to_datetime(audit.privileged_completion_at).dt.normalize()-pd.to_datetime(audit.prediction_date)).dt.days
    require((remaining>=1).all(),'Received or invalid-time order included')
    require((pd.to_datetime(audit.privileged_completion_at)>=pd.to_datetime(audit.decision_at)).all(),
            'A received order is still in the active set')
    days[observed]=remaining.to_numpy(dtype=float)
    frame['method']=METHOD
    frame['prediction_source']=np.where(observed,'diagnostic_true_completion','synthetic_promise')
    frame['diagnostic_truth_used']=observed
    frame['predicted_remaining_days']=days
    frame['raw_predicted_remaining_days']=days
    frame['arrival_relative_period']=arrival_periods(days,period_days)
    frame['predicted_calendar_completion_day']=pd.to_datetime(frame.prediction_date)+pd.to_timedelta(np.ceil(days),unit='D')
    frame['earliest_arrival_deferred_to_current_period']=days<=0
    frame['beyond_planning_horizon']=frame.arrival_relative_period>planning_periods
    audit['exact_remaining_calendar_days']=remaining
    audit['exact_arrival_relative_period']=arrival_periods(remaining,period_days)
    require(not {'completion_at','remaining_days','label_available_at'}&set(frame),
            'Raw truth fields must remain in the separate diagnostic audit')
    return frame,audit


def prepare_forecasts(dataset,reference,out):
    c=read_json(dataset/'config_frozen.json')
    truth=pd.read_csv(dataset/'audit/selected_lot_catalog.csv',usecols=['lot_id','completion_at'])
    previous=pd.read_csv(reference/'weekly_predictions.csv',low_memory=False)
    point=previous.loc[previous.method=='point_transformer']
    calendar=pd.read_csv(dataset/'test/inputs/decision_calendar.csv')
    frames,audits,checks=[],[],[]
    for step in calendar.itertuples():
        orders=load_decision(dataset,'test',step.period)['orders']
        old=point.loc[point.period==step.period].set_index('lot_id').loc[orders.lot_id]
        require(set(old.index)==set(orders.lot_id),'Changed order set')
        for col in ('fixed_quantity','has_snapshot','history_length'):
            require(np.array_equal(old[col].to_numpy(),orders[col].to_numpy()),'Changed input '+col)
        frame,audit=forecast_orders(orders,truth,c['period_days'],c['planning_periods'])
        unseen=~frame.has_snapshot
        require(np.array_equal(frame.loc[unseen,'predicted_remaining_days'].to_numpy(),
                               old.loc[frame.loc[unseen,'lot_id'],'predicted_remaining_days'].to_numpy()),'Unseen promise changed')
        require(np.array_equal(frame.loc[unseen,'arrival_relative_period'].to_numpy(),
                               old.loc[frame.loc[unseen,'lot_id'],'arrival_relative_period'].to_numpy()),'Unseen arrival week changed')
        aligned=old.loc[audit.lot_id]
        audit['point_predicted_remaining_days']=aligned.predicted_remaining_days.to_numpy()
        audit['point_arrival_relative_period']=aligned.arrival_relative_period.to_numpy()
        checks.append({'period':step.period,'orders':len(frame),'observed_exact':len(audit),'promise_only':int(unseen.sum()),
                       'orders_quantities_information_and_unseen_promises_unchanged':True,
                       'unknown_observed_completions':0})
        frames.append(frame);audits.append(audit)
    forecasts=pd.concat(frames,ignore_index=True);audit=pd.concat(audits,ignore_index=True)
    write_csv(out/'diagnostic_forecasts.csv',forecasts)
    write_csv(out/'privileged_completion_audit.csv',audit)
    write_json(out/'information_scope_checks.json',checks)
    return forecasts,audit,previous


def run(config_path,out):
    config_path,out=Path(config_path).resolve(),Path(out).resolve()
    out.mkdir(parents=True,exist_ok=False)
    cfg=yaml.safe_load(config_path.read_text(encoding='utf-8'))
    require(cfg['method']==METHOD and cfg['diagnostic_only'] and not cfg['deployable'],'Diagnostic must be explicit')
    dataset=(PROJECT/cfg['dataset_dir']).resolve();reference=(PROJECT/cfg['reference_run']).resolve()
    require(cfg['split']=='test' and cfg['planning_periods']==4,'Only approved test N=4 diagnostic')
    require(read_json(dataset/'config_frozen.json')['planning_periods']==4,'Dataset N differs')
    solver=yaml.safe_load((PROJECT/cfg['solver_config']).read_text(encoding='utf-8'))
    prior=read_json(reference/'experiment_summary.json')
    prior_cfg=read_json(reference/'experiment_config.json')
    require(prior['status']=='completed' and Path(prior['dataset'])==dataset,'Reference uses different data')
    for key in ('solver_config','solver_threads','solver_time_limit_seconds','feasibility_tolerance','split'):
        require(prior_cfg[key]==cfg[key],'Comparison parameter changed: '+key)
    start=time.perf_counter()
    state={'status':'starting','started_at':timestamp(),'training_requested':False,'diagnostic_only':True}
    def phase(status):
        state.update(status=status,updated_at=timestamp());write_json(out/'run_manifest.json',state)
    try:
        hashes=read_json(reference/'artifact_hashes.json')
        require(all(sha(reference/p)==h for p,h in hashes.items()),'Reference output changed')
        manifest=read_json(dataset/'manifest.json')
        protected={str(dataset/p):h for p,h in manifest['file_sha256'].items()}
        protected.update(manifest['protected_source_sha256'])
        require(all(sha(p)==h for p,h in protected.items()),'Frozen data/predictor changed')
        for directory in ('outputs/dk/daily/full_runs','outputs/app/pto_runs','outputs/archives'):
            for p in (PROJECT/directory).rglob('*'):
                if p.is_file() and not p.is_relative_to(out):protected[str(p)]=sha(p)
        sources=[]
        for directory in ('scripts/app_dataset','scripts/app_experiment'):
            sources.extend((PROJECT/directory).glob('*.py'))
        sources += [config_path,PROJECT/cfg['solver_config']]
        for p in sources:protected[str(p)]=sha(p)
        write_json(out/'protected_files.json',protected);write_json(out/'experiment_config.json',cfg)
        with zipfile.ZipFile(out/'app_source_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
            for p in sorted(set(sources)):z.write(p,str(p.relative_to(PROJECT)))
        phase('preparing_privileged_observed_forecasts')
        forecasts,truth_audit,previous=prepare_forecasts(dataset,reference,out)
        for result in prior['methods']:shutil.copytree(reference/result['method'],out/result['method'])
        combined=pd.concat([previous,forecasts],ignore_index=True)
        write_csv(out/'weekly_predictions.csv',combined)
        phase('rolling_perfect_observed_diagnostic')
        result=run_method(METHOD,combined,dataset,'test',cfg,solver,out)
        summary={'status':'completed','dataset':str(dataset),'split':'test','methods':prior['methods']+[result],
                 'reference_run':str(reference),'reference_trajectories_reused_without_resolving':True,
                 'policy':'Perfect completion labels only for already-observed Lots; unchanged promises for all others.',
                 'diagnostic_only':True,'deployable':False,'privileged_completion_used':True,
                 'new_predictor_training':False,'test_parameter_tuning':False,
                 'truth_scope':{'exact_observed_rows':len(truth_audit),'exact_observed_lots':int(truth_audit.lot_id.nunique()),
                     'promise_only_rows':int((~forecasts.has_snapshot).sum()),'unknown_observed':0,
                     'point_vs_exact_different_week_rows':int((truth_audit.point_arrival_relative_period!=truth_audit.exact_arrival_relative_period).sum()),
                     'point_mae_on_exact_rows':float((truth_audit.point_predicted_remaining_days-truth_audit.exact_remaining_calendar_days).abs().mean())}}
        write_json(out/'experiment_summary.json',summary);write_json(out/'artifact_hashes.json',output_hashes(out))
        phase('independent_replay')
        from scripts.app_experiment.verify import audit_run
        audit=audit_run(out)
        phase('full_T_oracle_comparison')
        from scripts.app_experiment.oracle import calculate
        oracle=calculate(out,out/'oracle')
        require(abs(oracle['costs']['total_cost']-prior['oracle']['costs']['total_cost'])<2e-5,'Oracle no longer matches')
        diagnostic_cost=result['costs']['total_cost'];point=next(m for m in prior['methods'] if m['method']=='point_transformer')
        point_cost=point['costs']['total_cost'];oracle_cost=oracle['costs']['total_cost']
        summary.update(oracle=oracle,post_run_audit=audit,elapsed_seconds=time.perf_counter()-start,
          diagnostic_comparison={'point_minus_diagnostic_cost':point_cost-diagnostic_cost,
            'diagnostic_regret_absolute':diagnostic_cost-oracle_cost,
            'diagnostic_regret_relative':(diagnostic_cost-oracle_cost)/oracle_cost,
            'fraction_of_point_oracle_gap_closed_on_this_trajectory':(point_cost-diagnostic_cost)/(point_cost-oracle_cost),
            'interpretation':'An observed counterfactual cost difference, not an attainable guarantee or a formal lower bound.'})
        require(all(sha(p)==h for p,h in protected.items()),'Protected file changed')
        summary['all_protected_files_unchanged']=True
        write_json(out/'experiment_summary.json',summary)
        from scripts.app_experiment.perfect_observed_report import render
        render(out,summary)
        write_json(out/'artifact_hashes.json',output_hashes(out))
        state.update(status='completed',completed_at=timestamp(),elapsed_seconds=time.perf_counter()-start)
        write_json(out/'run_manifest.json',state)
        print(summary['diagnostic_comparison'],flush=True)
        return summary
    except Exception as error:
        state.update(status='failed',error=str(error));write_json(out/'run_manifest.json',state)
        (out/'failure_traceback.txt').write_text(traceback.format_exc(),encoding='utf-8');raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=PROJECT/'config/app_perfect_observed.yaml')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.config,args.output)
