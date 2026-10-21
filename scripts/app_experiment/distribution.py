"""NLL daily probabilities -> expected weekly arrivals -> unchanged rolling APP.

Only the planning forecast is fractional. Replay still receives each real Lot once,
in full. This is an expected-supply PTO policy, not a stochastic-programming policy.
"""
from pathlib import Path
import argparse
import json
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
sys.path.insert(0,str(PROJECT/'src'))
from scripts.app_dataset.generate import load_decision, require, sha, write_csv, write_json
from scripts.app_experiment.run import read_json, timestamp, arrival_periods, run_method
from scripts.app_experiment.advance import output_hashes

METHOD='nll_expected_supply'


def daily_to_weekly(probabilities, period_days, planning_periods):
    p=np.asarray(probabilities,dtype=float)
    horizon=period_days*planning_periods
    require(p.ndim==2 and p.shape[1]==horizon+1,'Daily horizon must match APP planning horizon')
    require(np.isfinite(p).all() and (p>=0).all(),'Invalid probability')
    require(np.allclose(p.sum(1),1.,atol=1e-7,rtol=0),'Probability mass differs from one')
    # Open tail stays outside; no conditioning on arrival within the planning window.
    return np.column_stack((p[:,:horizon].reshape(-1,planning_periods,period_days).sum(2),p[:,-1]))


def forecast_orders(orders, predictions, period_days, planning_periods):
    frame=orders.drop(columns='prediction_days').copy()
    require(not frame.lot_id.duplicated().any(),'Duplicate Lot at one decision')
    observed=frame.has_snapshot.to_numpy(dtype=bool)
    horizon=period_days*planning_periods
    columns=[*[f'p_day_{d}' for d in range(1,horizon+1)],'p_tail']
    daily=np.full((len(frame),horizon+1),np.nan)
    weekly=np.zeros((len(frame),planning_periods+1))
    if observed.any():
        known=frame.loc[observed]
        require(set(predictions.lot_id)==set(known.lot_id) and not predictions.lot_id.duplicated().any(),
                'Distribution cohort differs')
        p=predictions.set_index('lot_id').loc[known.lot_id,columns].to_numpy(dtype=float)
        weekly[observed]=daily_to_weekly(p,period_days,planning_periods)
        daily[observed]=p
    unseen=np.flatnonzero(~observed)
    if len(unseen):
        pending=frame.iloc[unseen]
        promised=(pd.to_datetime(pending.promised_completion_date)-pd.to_datetime(pending.prediction_date)).dt.days.clip(lower=0)
        buckets=arrival_periods(promised,period_days)
        weekly[unseen,np.minimum(buckets,planning_periods+1)-1]=1.
    frame['method']=METHOD
    frame['prediction_source']=np.where(observed,'nll_distribution','synthetic_promise')
    for j,col in enumerate(columns):frame[col]=daily[:,j]
    for k in range(planning_periods):frame[f'p_week_{k+1}']=weekly[:,k]
    frame['p_beyond_horizon']=weekly[:,-1]
    frame['expected_quantity_within_horizon']=frame.fixed_quantity*weekly[:,:-1].sum(1)
    frame['expected_quantity_beyond_horizon']=frame.fixed_quantity*weekly[:,-1]
    require(not {'completion_at','remaining_days','label_available_at','last_seen_at'}&set(frame), 'Truth in planning forecast')
    require((frame.fixed_quantity>0).all(),'Nonpositive quantity')
    return frame


def expected_supply(predictions, material_ids, horizon):
    cols=[*[f'p_week_{k+1}' for k in range(horizon)],'p_beyond_horizon']
    p=predictions[cols].to_numpy(dtype=float)
    require(np.isfinite(p).all() and (p>=0).all() and np.allclose(p.sum(1),1,atol=1e-7,rtol=0),
            'Invalid weekly probability mass')
    require(not predictions.lot_id.duplicated().any(),'Double-counted expected Lot')
    arrivals=np.zeros((horizon,len(material_ids)))
    for i,mat in enumerate(material_ids):
        mask=predictions.material_id.eq(mat).to_numpy()
        arrivals[:,i]=(p[mask,:horizon]*predictions.loc[mask,'fixed_quantity'].to_numpy()[:,None]).sum(0)
    require(np.allclose(arrivals.sum()+(p[:,-1]*predictions.fixed_quantity.to_numpy()).sum(),
                        predictions.fixed_quantity.sum(),atol=1e-6,rtol=0),'Lost or duplicated expected quantity')
    return arrivals


def predict_weeks(dataset, checkpoint, reference, out, phase):
    from wip_pred.daily.runner import predict_checkpoint, code_hashes, load_checkpoint
    _,_,cfg=load_checkpoint(checkpoint)
    data_cfg=read_json(dataset/'config_frozen.json')
    require(cfg['mode']=='distribution','Distribution checkpoint required')
    protocol=cfg['protocol']
    require(protocol['train_end']==data_cfg['splits']['train']['end_exclusive'],'Wrong training boundary')
    require(protocol['validation_end']==data_cfg['splits']['test']['start'],'Wrong validation boundary')
    require(protocol['window_days']==data_cfg['history_window_days'],'Wrong history window')
    require(cfg['target_spec']['horizon']==data_cfg['period_days']*data_cfg['planning_periods'],'Unsupported open-tail horizon')
    require(all(code_hashes().get(p)==h for p,h in cfg['data_manifest']['source_hashes'].items()),'Predictor source changed')
    require(cfg['data_manifest']['data']['csv_sha256']==data_cfg['expected_source_sha256'],'CSV differs')
    previous=pd.read_csv(reference/'weekly_predictions.csv')
    previous=previous.loc[previous.method=='point_transformer']
    calendar=pd.read_csv(dataset/'test/inputs/decision_calendar.csv',parse_dates=['prediction_date'])
    frames,checks=[],[]
    for step in calendar.itertuples():
        phase(f'nll_inference_period_{step.period}')
        orders=load_decision(dataset,'test',step.period)['orders']
        known=orders.loc[orders.has_snapshot]
        old=previous.loc[previous.period==step.period].set_index('lot_id').loc[orders.lot_id]
        require(set(old.index)==set(orders.lot_id),'Reference order set differs')
        require(np.array_equal(old.fixed_quantity.to_numpy(),orders.fixed_quantity.to_numpy()),'Reference quantities differ')
        require(np.array_equal(old.has_snapshot.to_numpy(),orders.has_snapshot.to_numpy()),'Reference information differs')
        folder=out/'inference'/f'period_{step.period:02d}'
        folder.mkdir(parents=True,exist_ok=False)
        pred=pd.DataFrame()
        if len(known):
            pred=predict_checkpoint(checkpoint,folder,step.prediction_date.date().isoformat(),known.lot_id.tolist())
            aligned=pred.set_index('lot_id').loc[known.lot_id]
            for a,b in [('first_snapshot_at','first_seen_at'),('last_snapshot_at','last_snapshot_at')]:
                require(np.array_equal(pd.to_datetime(aligned[a]).to_numpy(),pd.to_datetime(known[b]).to_numpy()),'History alignment mismatch')
            require(np.array_equal(aligned.sequence_length.to_numpy(),known.history_length.to_numpy()),'Sequence length mismatch')
        frame=forecast_orders(orders,pred,data_cfg['period_days'],data_cfg['planning_periods'])
        unseen=frame.loc[~frame.has_snapshot]
        expected_bucket=np.minimum(old.loc[unseen.lot_id].arrival_relative_period.to_numpy(),data_cfg['planning_periods']+1)
        cols=[*[f'p_week_{k+1}' for k in range(data_cfg['planning_periods'])],'p_beyond_horizon']
        require(np.array_equal(unseen[cols].to_numpy().argmax(1)+1,expected_bucket),'Promise rule differs from point/baseline')
        checks.append({'period':step.period,'orders':len(orders),'observed':len(known),'promise_only':len(unseen),
                       'same_orders_quantities_histories_and_promises_as_reference':True,'truth_used':False})
        frames.append(frame)
        print(f'NLL week {step.period}: {len(known)} observed + {len(unseen)} promise-only',flush=True)
    result=pd.concat(frames,ignore_index=True)
    write_csv(out/'distribution_forecasts.csv',result)
    write_json(out/'inference_checks.json',checks)
    return result,cfg


def score_after_execution(forecasts,dataset,out,horizon):
    from wip_pred.daily.metrics import distribution_metrics
    truth=pd.read_csv(dataset/'audit/selected_lot_catalog.csv',parse_dates=['completion_at'])
    frame=forecasts.loc[forecasts.has_snapshot].merge(truth[['lot_id','completion_at']],on='lot_id',how='left',validate='many_to_one')
    frame['remaining_days']=(frame.completion_at.dt.normalize()-pd.to_datetime(frame.prediction_date)).dt.days
    labelled=frame.loc[frame.remaining_days.notna()].copy()
    require(len(labelled)>0 and (labelled.remaining_days>=1).all(),'Invalid remaining time')
    p=labelled[[*[f'p_day_{d}' for d in range(1,horizon+1)],'p_tail']].to_numpy(dtype=float)
    category=np.minimum(labelled.remaining_days.to_numpy(dtype=int),horizon+1)-1
    likelihood=p[np.arange(len(p)),category]
    require((likelihood>0).all(),'Zero probability on observed class')
    losses=-np.log(likelihood)
    labelled['coarsened_nll']=losses
    metrics=distribution_metrics(labelled,p,horizon,losses)
    metrics.update(scope='snapshot_observed_only',unknown_rows=int(frame.remaining_days.isna().sum()),
                   used_for_policy_selection=False)
    write_csv(out/'distribution_evaluation_only.csv',labelled)
    write_json(out/'distribution_metrics.json',metrics)
    return metrics


def run(config_path,out):
    config_path,out=Path(config_path).resolve(),Path(out).resolve()
    out.mkdir(parents=True,exist_ok=False)
    cfg=yaml.safe_load(config_path.read_text(encoding='utf-8'))
    require(cfg['method']==METHOD and cfg['split']=='test','Unsupported policy')
    dataset=(PROJECT/cfg['dataset_dir']).resolve()
    checkpoint=(PROJECT/cfg['distribution_checkpoint']).resolve()
    reference=(PROJECT/cfg['reference_run']).resolve()
    solver=yaml.safe_load((PROJECT/cfg['solver_config']).read_text(encoding='utf-8'))
    state={'status':'starting','started_at':timestamp(),'training_requested':False,'run_dir':str(out)}
    start=time.perf_counter()
    def phase(status):
        state.update(status=status,updated_at=timestamp());write_json(out/'run_manifest.json',state)
    try:
        prior=read_json(reference/'experiment_summary.json')
        require(prior['status']=='completed' and Path(prior['dataset'])==dataset,'Reference comparison uses different data')
        reference_hashes=read_json(reference/'artifact_hashes.json')
        require(all(sha(reference/p)==h for p,h in reference_hashes.items()),'Reference artifacts changed')
        manifest=read_json(dataset/'manifest.json')
        protected={str(dataset/p):h for p,h in manifest['file_sha256'].items()}
        protected.update(manifest['protected_source_sha256'])
        require(all(sha(p)==h for p,h in protected.items()),'Frozen data or model source changed')
        # Old reports/models are read-only; save exact hashes before inference.
        for directory in ('outputs/dk/daily/full_runs','outputs/app/pto_runs','outputs/archives'):
            for p in (PROJECT/directory).rglob('*'):
                if p.is_file() and not p.is_relative_to(out):protected[str(p)]=sha(p)
        source_files=[]
        for directory in ('scripts/app_dataset','scripts/app_experiment'):
            source_files.extend((PROJECT/directory).glob('*.py'))
        source_files += [config_path,PROJECT/cfg['solver_config']]
        for p in source_files:protected[str(p)]=sha(p)
        write_json(out/'protected_files.json',protected)
        write_json(out/'experiment_config.json',cfg)
        with zipfile.ZipFile(out/'app_source_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
            for p in sorted(set(source_files)):z.write(p,str(p.relative_to(PROJECT)))
        forecasts,model_cfg=predict_weeks(dataset,checkpoint,reference,out,phase)
        # Reuse earlier completed trajectories under exactly the same data/solver.
        for old_result in prior['methods']:
            shutil.copytree(reference/old_result['method'],out/old_result['method'])
        previous=pd.read_csv(reference/'weekly_predictions.csv')
        combined=pd.concat([previous,forecasts],ignore_index=True)
        write_csv(out/'weekly_predictions.csv',combined)
        phase('rolling_nll_expected_supply')
        result=run_method(METHOD,combined,dataset,'test',cfg,solver,out,arrival_builder=expected_supply)
        metrics=score_after_execution(forecasts,dataset,out,model_cfg['target_spec']['horizon'])
        summary={'status':'completed','dataset':str(dataset),'split':'test','methods':prior['methods']+[result],
            'reference_run':str(reference),'reference_trajectories_reused_without_resolving':True,
            'distribution_checkpoint':str(checkpoint),'distribution_best_epoch':read_json(checkpoint/'training_state.json')['best_epoch'],
            'point_checkpoint':prior['point_checkpoint'],'point_best_epoch':prior['point_best_epoch'],
            'distribution_metrics':metrics,'reference_forecast_metrics':prior['forecast_metrics'],
            'policy':'Expected weekly quantity in the same deterministic APP, not a stochastic program.',
            'new_predictor_training':False,'test_parameter_tuning':False}
        write_json(out/'experiment_summary.json',summary)
        write_json(out/'artifact_hashes.json',output_hashes(out))
        phase('independent_replay_audit')
        from scripts.app_experiment.verify import audit_run
        audit=audit_run(out)
        phase('oracle_same_objective_check')
        from scripts.app_experiment.oracle import calculate
        oracle=calculate(out,out/'oracle')
        require(abs(oracle['costs']['total_cost']-prior['oracle']['costs']['total_cost'])<2e-5,'Oracle changed under same data')
        summary.update(oracle=oracle,post_run_audit=audit,elapsed_seconds=time.perf_counter()-start)
        require(all(sha(p)==h for p,h in protected.items()),'Protected files changed')
        summary['all_protected_files_unchanged']=True
        write_json(out/'experiment_summary.json',summary)
        from scripts.app_experiment.distribution_report import render
        render(out,summary)
        write_json(out/'artifact_hashes.json',output_hashes(out))
        state.update(status='completed',completed_at=timestamp(),elapsed_seconds=time.perf_counter()-start)
        write_json(out/'run_manifest.json',state)
        print(json.dumps({'run':str(out),'comparison':oracle['comparison']},ensure_ascii=False,indent=2))
        return summary
    except Exception as error:
        state.update(status='failed',error=str(error));write_json(out/'run_manifest.json',state)
        (out/'failure_traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=PROJECT/'config/app_nll_expected_28d.yaml')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.config,args.output)
