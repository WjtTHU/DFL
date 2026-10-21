"""Frozen point predictor + advance orders + CPLEX rolling APP and full-T oracle."""
from pathlib import Path
import argparse
import json
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
from scripts.app_experiment.run import (read_json, timestamp, arrival_periods,
    fit_history_baseline, baseline_predict, run_method)


def forecast_orders(orders, predictions, baseline, period_days, planning_periods=4):
    """No outcomes: both methods share promises until snapshots become visible."""
    observed = orders.has_snapshot.to_numpy(dtype=bool)
    promised = (pd.to_datetime(orders.promised_completion_date) -
                pd.to_datetime(orders.prediction_date)).dt.days.clip(lower=0).to_numpy(dtype=float)
    frames = []
    for method in ('point_transformer','historical_product_cycle'):
        frame = orders.drop(columns='prediction_days').copy()
        days, raw = promised.copy(), promised.copy()
        if observed.any():
            subset = orders.loc[observed]
            if method == 'point_transformer':
                pred = predictions.set_index('lot_id').loc[subset.lot_id]
                days[observed] = pred.predicted_remaining_days.to_numpy()
                raw[observed] = pred.raw_predicted_remaining_days.to_numpy()
            else:
                days[observed] = baseline_predict(subset, baseline)
                raw[observed] = days[observed]
        frame['method'] = method
        frame['prediction_source'] = np.where(observed, method, 'synthetic_promise')
        frame['predicted_remaining_days'] = days
        frame['raw_predicted_remaining_days'] = raw
        frame['arrival_relative_period'] = arrival_periods(days, period_days)
        frame['predicted_calendar_completion_day'] = pd.to_datetime(frame.prediction_date) + pd.to_timedelta(np.ceil(days),unit='D')
        frame['earliest_arrival_deferred_to_current_period'] = days <= 0
        frame['beyond_planning_horizon'] = frame.arrival_relative_period > planning_periods
        require(not {'completion_at','remaining_days','label_available_at','last_seen_at'} & set(frame), 'Outcomes in forecast')
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def predict(dataset, checkpoint, index, out, phase):
    from wip_pred.daily.runner import predict_checkpoint, code_hashes
    cfg=read_json(checkpoint/'daily_config.json')
    data_cfg=read_json(dataset/'config_frozen.json')
    require(cfg['mode']=='point','Point model required')
    require(cfg['protocol']['train_end']==data_cfg['splits']['train']['end_exclusive'],'Wrong training split')
    require(cfg['protocol']['validation_end']==data_cfg['splits']['test']['start'],'Test starts before model selection')
    require(cfg['protocol']['window_days']==data_cfg['history_window_days'],'Wrong history window')
    require(all(code_hashes().get(p)==h for p,h in cfg['data_manifest']['source_hashes'].items()),'Predictor source changed')
    baseline=fit_history_baseline(index,pd.Timestamp(cfg['protocol']['train_end']))
    write_json(out/'baseline_fit.json',baseline)
    calendar=pd.read_csv(dataset/'test/inputs/decision_calendar.csv',parse_dates=['prediction_date'])
    frames, checks = [],[]
    for row in calendar.itertuples():
        phase(f'inference_period_{row.period}')
        view=load_decision(dataset,'test',row.period)
        orders=view['orders']
        known=orders.loc[orders.has_snapshot]
        print(f'Week {row.period}: {len(known)} snapshot orders, {len(orders)-len(known)} promise-only orders',flush=True)
        folder=out/'inference'/f'period_{row.period:02d}'
        folder.mkdir(parents=True,exist_ok=False)
        if len(known):
            predictions=predict_checkpoint(checkpoint,folder,row.prediction_date.date().isoformat(),known.lot_id.tolist())
            require(set(predictions.lot_id)==set(known.lot_id),'Inference cohort mismatch')
            aligned=predictions.set_index('lot_id').loc[known.lot_id]
            require(np.array_equal(pd.to_datetime(aligned.last_snapshot_at).to_numpy(),pd.to_datetime(known.last_snapshot_at).to_numpy()),'History end mismatch')
            require(np.array_equal(pd.to_datetime(aligned.first_snapshot_at).to_numpy(),pd.to_datetime(known.first_seen_at).to_numpy()),'History origin mismatch')
            require(np.array_equal(aligned.sequence_length.to_numpy(),known.history_length.to_numpy()),'History length mismatch')
        else:
            predictions=pd.DataFrame()
        frames.append(forecast_orders(orders,predictions,baseline,data_cfg['period_days'],data_cfg['planning_periods']))
        checks.append({'period':row.period,'snapshot_orders':len(known),'promise_only_orders':len(orders)-len(known),
                       'snapshot_history_aligned':True,'true_labels_used':False})
    result=pd.concat(frames,ignore_index=True)
    write_csv(out/'weekly_predictions.csv',result)
    write_json(out/'inference_checks.json',checks)
    return result,baseline


def evaluate(predictions,dataset,out):
    """Evaluation only, after rolling actions; include outcomes beyond T if known."""
    catalog=pd.read_csv(dataset/'audit/selected_lot_catalog.csv',parse_dates=['completion_at'])
    days=read_json(dataset/'config_frozen.json')['period_days']
    frame=predictions.merge(catalog[['lot_id','completion_at']],on='lot_id',how='left',validate='many_to_one')
    frame['true_remaining_days']=(frame.completion_at.dt.normalize()-pd.to_datetime(frame.prediction_date)).dt.days
    frame['absolute_error_days']=(frame.predicted_remaining_days-frame.true_remaining_days).abs()
    result=[]
    for method,g in frame.groupby('method',sort=False):
        for scope in ('all_orders','snapshot_observed','promise_only'):
            sub=g if scope=='all_orders' else g.loc[g.has_snapshot == (scope=='snapshot_observed')]
            labelled=sub.loc[sub.true_remaining_days.notna()]
            require((labelled.true_remaining_days>=1).all(),'Already received order scored')
            err=labelled.predicted_remaining_days-labelled.true_remaining_days
            match=arrival_periods(labelled.true_remaining_days,days)==labelled.arrival_relative_period.to_numpy()
            result.append({'method':method,'scope':scope,'forecast_rows':len(sub),'labelled_rows':len(labelled),
                'unknown_rows':len(sub)-len(labelled),'mae_days':float(err.abs().mean()) if len(err) else None,
                'rmse_days':float(np.sqrt(np.mean(err**2))) if len(err) else None,
                'same_arrival_week_fraction':float(np.mean(match)) if len(err) else None,
                'quantity_weighted_same_week_fraction':float(np.average(match,weights=labelled.fixed_quantity)) if len(err) else None})
    write_csv(out/'forecast_evaluation_only.csv',frame)
    write_json(out/'forecast_metrics.json',result)
    return result


def output_hashes(out):
    return {str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*'))
            if p.is_file() and p.name not in ('artifact_hashes.json','run_manifest.json')}


def run(config_path,out):
    config_path,out=Path(config_path).resolve(),Path(out).resolve()
    out.mkdir(parents=True,exist_ok=False)
    cfg=yaml.safe_load(config_path.read_text(encoding='utf-8'))
    require(cfg['split']=='test','Frozen model deployed only on test')
    dataset=(PROJECT/cfg['dataset_dir']).resolve()
    checkpoint=(PROJECT/cfg['point_checkpoint']).resolve()
    index=(PROJECT/cfg['training_sample_index']).resolve()
    solver_path=PROJECT/cfg['solver_config']
    solver=yaml.safe_load(solver_path.read_text(encoding='utf-8'))
    start=time.perf_counter()
    state={'status':'starting','started_at':timestamp(),'run_dir':str(out),'training_requested':False}
    def phase(name):
        state.update(status=name,updated_at=timestamp())
        write_json(out/'run_manifest.json',state)
    try:
        manifest=read_json(dataset/'manifest.json')
        protected={str(dataset/p):h for p,h in manifest['file_sha256'].items()}
        protected.update(manifest['protected_source_sha256'])
        for p in [dataset/'manifest.json',config_path,index,solver_path,Path(__file__)]:
            protected[str(p)]=sha(p)
        for folder in ('scripts/app_dataset','scripts/app_experiment'):
            for p in (PROJECT/folder).glob('*.py'): protected[str(p)]=sha(p)
        # Preserve every previously saved experiment/report/model, not just checkpoints.
        for directory in ('outputs/dk/daily/full_runs','outputs/app/pto_runs','outputs/app/oracle_runs','outputs/archives'):
            for p in (PROJECT/directory).rglob('*'):
                if p.is_file() and not p.is_relative_to(out): protected[str(p)]=sha(p)
        require(all(sha(p)==h for p,h in protected.items()),'Frozen input changed')
        write_json(out/'protected_files.json',protected)
        write_json(out/'experiment_config.json',cfg)
        with zipfile.ZipFile(out/'app_source_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
            for p in sorted({*list((PROJECT/'scripts/app_dataset').glob('*.py')),
                             *list((PROJECT/'scripts/app_experiment').glob('*.py')),config_path}):
                z.write(p,str(p.relative_to(PROJECT)))
        forecasts,baseline=predict(dataset,checkpoint,index,out,phase)
        results=[]
        for method in cfg['methods']:
            phase('rolling_'+method)
            results.append(run_method(method,forecasts,dataset,'test',cfg,solver,out))
        metrics=evaluate(forecasts,dataset,out)
        summary={'status':'completed','completed_at':timestamp(),'dataset':str(dataset),'split':'test',
                 'point_checkpoint':str(checkpoint),'point_best_epoch':read_json(checkpoint/'training_state.json')['best_epoch'],
                 'methods':results,'forecast_metrics':metrics,'baseline_fit':baseline,
                 'elapsed_seconds':time.perf_counter()-start,'information_policy':'advance28_then_snapshot',
                 'new_predictor_training':False,'test_parameter_tuning':False}
        write_json(out/'experiment_summary.json',summary)
        write_json(out/'artifact_hashes.json',output_hashes(out))
        phase('independent_audit')
        from scripts.app_experiment.verify import audit_run
        audit=audit_run(out)
        phase('full_T_oracle')
        from scripts.app_experiment.oracle import calculate
        oracle=calculate(out,out/'oracle')
        summary['oracle']=oracle
        summary['post_run_audit']=audit
        summary['elapsed_seconds']=time.perf_counter()-start
        require(all(sha(p)==h for p,h in protected.items()),'Protected model, source or old artifacts changed')
        summary['all_protected_files_unchanged']=True
        write_json(out/'experiment_summary.json',summary)
        from scripts.app_experiment.advance_report import render
        render(out,summary)
        write_json(out/'artifact_hashes.json',output_hashes(out))
        state.update(status='completed',completed_at=timestamp(),elapsed_seconds=time.perf_counter()-start)
        write_json(out/'run_manifest.json',state)
        print(json.dumps({'out':str(out),'comparison':oracle['comparison'],'forecast_metrics':metrics},ensure_ascii=False,indent=2),flush=True)
        return summary
    except Exception as error:
        state.update(status='failed',error=str(error))
        write_json(out/'run_manifest.json',state)
        (out/'failure_traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=PROJECT/'config/app_pto_10w_advance28.yaml')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    run(args.config,args.output)
