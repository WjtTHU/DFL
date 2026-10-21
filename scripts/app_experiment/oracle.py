"""Full-T perfect-information APP oracle with policy feasibility and replay checks.

Extracted from the verified earlier oracle; no fixed experiment paths or results.
"""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json
import sys

import numpy as np
import pandas as pd
import yaml


PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT))
from scripts.app_dataset.generate import sha, require, write_csv, write_json
from scripts.app_experiment.lp import build_app_lp, solve_cplex, residuals


def calculate(policy_run, output):
    POLICY=Path(policy_run).resolve()
    OUT=Path(output).resolve()
    OUT.mkdir(parents=True,exist_ok=False)
    old=json.loads((POLICY/'experiment_summary.json').read_text(encoding='utf-8'))
    # Keep an immutable Oracle input; the outer controller later finalizes its summary.
    write_json(OUT/'policy_summary_at_oracle.json',old)
    DATA=Path(old['dataset']); split=old['split']; inputs=DATA/split/'inputs'
    read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
    c=read(DATA/'config_frozen.json'); parameters=read(DATA/'shared/parameters.json'); costs=parameters['costs']
    solver=yaml.safe_load((PROJECT/'config/app_solver.yaml').read_text(encoding='utf-8'))
    manifest=read(DATA/'manifest.json')
    protected={str(DATA/p):h for p,h in manifest['file_sha256'].items()}
    protected.update(manifest['protected_source_sha256'])
    for p in POLICY.rglob('*'):
        if (p.is_file() and not p.is_relative_to(OUT)
                and p.name not in ('run_manifest.json','artifact_hashes.json','experiment_summary.json')):
            protected[str(p)]=sha(p)
    protected[str(OUT/'policy_summary_at_oracle.json')]=sha(OUT/'policy_summary_at_oracle.json')
    for p in [PROJECT/'scripts/app_experiment/lp.py',PROJECT/'config/app_solver.yaml',Path(__file__)]:
        protected[str(p)]=sha(p)
    require(all(sha(p)==h for p,h in protected.items()),'Frozen input changed')
    write_json(OUT/'protected_files.json',protected)
    
    mats=[m['material_id'] for m in c['materials']]; products=c['products']; a=np.asarray(c['bom'])
    calendar=pd.read_csv(inputs/'decision_calendar.csv',parse_dates=['decision_at','execution_end_exclusive'])
    T=len(calendar); lead=c['secondary']['effective_lead_periods']
    demand=pd.read_csv(inputs/'demand.csv').pivot(index='period',columns='product_id',values='demand').reindex(index=range(1,T+1),columns=products).to_numpy()
    caps=pd.read_csv(inputs/'secondary_supply.csv').pivot(index='period',columns='material_id',values='capacity_die').reindex(index=range(1,T+1),columns=mats).to_numpy()
    capacity=pd.read_csv(inputs/'capacity.csv').set_index('period').capacity_product_units.reindex(range(1,T+1)).to_numpy()
    initial=read(inputs/'initial_state.json')
    raw0=np.array([initial['raw_inventory_die'][m] for m in mats]);fg0=np.array([initial['finished_inventory_product_units'][p] for p in products]);back0=np.array([initial['customer_backlog_product_units'][p] for p in products])
    require(not initial['secondary_in_transit_orders'],'Initial pipeline needs explicit handling')
    receipts=pd.read_csv(DATA/split/'replay/period_arrivals.csv')
    require(receipts.loc[receipts.period<T,'receipt_total_complete_for_selected_cohort'].all(),'Unknown arrivals can affect executed periods')
    require(costs['raw_holding_per_die_period']==0 and costs['terminal_inventory_salvage']==0,'Unconfirmed final arrivals may affect this objective')
    primary=receipts.pivot(index='period',columns='material_id',values='known_arrival_quantity_die').reindex(index=range(1,T+1),columns=mats).to_numpy()
    model=build_app_lp(demand,capacity,a,raw0,fg0,back0,primary,np.zeros_like(primary),caps,lead,costs,parameters['terminal_backlog_penalty_per_product_unit'])
    plan,solver_check=solve_cplex(model,OUT/'cplex',solver['executable'],threads=1,time_limit=120)
    
    # Each existing rolling path, restricted to decisions relevant to T, must remain
    # feasible for the hindsight LP. Last-period q is omitted: it cannot arrive by T.
    candidate_checks=[]
    for result in old['methods']:
        method=result['method'];folder=POLICY/method
        product=pd.read_csv(folder/'product_states.csv');orders=pd.read_csv(folder/'secondary_orders.csv')
        x=np.zeros(len(model.names))
        for k in range(T):
            g=product.loc[product.period==k+1].set_index('product_id').loc[products]
            for f,col in [('p','production'),('i','closing_finished'),('b','closing_backlog')]:
                for j,val in enumerate(g[col]):x[model.indices[f,k,j]]=val
            q=orders.loc[orders.placed_period==k+1].set_index('material_id').loc[mats]
            for i,val in enumerate(q.quantity):
                x[model.indices['q',k,i]]=val if k+lead<T else 0.
        r=residuals(model,x)
        require(max(r.values())<2e-5,'Existing policy not feasible for same-horizon oracle')
        cost=float(model.objective@x)
        require(abs(cost-result['costs']['total_cost'])<2e-5,'Oracle and policy cost definitions differ')
        require(solver_check['objective']<=cost+2e-5,'Oracle worse than a feasible policy')
        candidate_checks.append({'method':method,'policy_cost_recomputed_in_oracle_lp':cost,
            'policy_cost_matches_saved_result':True,**r})
    
    # Independent sequential replay of the oracle, using original daily/weekly timing.
    raw,fg,back=raw0.copy(),fg0.copy(),back0.copy();period_rows=[];material_rows=[];product_rows=[]
    for k in range(T):
        due=plan['q'][k-lead] if k>=lead else np.zeros(len(mats))
        raw=raw+due;opening=raw.copy();used=a@plan['p'][k]
        require(np.all(used<=opening+1e-5),'Current production used future receipts')
        require(plan['p'][k].sum()<=capacity[k]+1e-5,'Capacity violation')
        net=fg-back+plan['p'][k]-demand[k];fg=np.maximum(net,0);back=np.maximum(-net,0)
        require(np.max(abs(fg-plan['i'][k]))<1e-5 and np.max(abs(back-plan['b'][k]))<1e-5,'Oracle inventory/backlog mismatch')
        raw=opening-used+primary[k]
        components={'production_cost':float(plan['p'][k].sum()*costs['production_per_product_unit']),
            'finished_holding_cost':float(fg.sum()*costs['finished_holding_per_product_unit_period']),
            'backlog_cost':float(back.sum()*costs['backlog_per_product_unit_period']),
            'secondary_receipt_cost':float(due.sum()*costs['secondary_purchase_per_die']),
            'terminal_backlog_cost':float(back.sum()*parameters['terminal_backlog_penalty_per_product_unit']) if k==T-1 else 0.}
        components['total_cost']=sum(components.values())
        period_rows.append({'period':k+1,'decision_at':calendar.decision_at.iloc[k],
            'production_total':float(plan['p'][k].sum()),'demand_total':float(demand[k].sum()),
            'finished_inventory_end':float(fg.sum()),'customer_backlog_end':float(back.sum()),
            'secondary_ordered_die':float(plan['q'][k].sum()),'secondary_received_die':float(due.sum()),**components})
        for i,m in enumerate(mats):
            material_rows.append({'period':k+1,'material_id':m,'opening_raw':opening[i],'consumption':used[i],
                'known_primary_arrival':primary[k,i],'secondary_received':due[i],'secondary_ordered':plan['q'][k,i],
                'closing_raw_known_lower_bound':raw[i]})
        for j,p in enumerate(products):
            product_rows.append({'period':k+1,'product_id':p,'production':plan['p'][k,j],
                'demand':demand[k,j],'closing_finished':fg[j],'closing_backlog':back[j]})
    periods=pd.DataFrame(period_rows)
    component_names=[x for x in periods if x.endswith('_cost')]
    totals={x:float(periods[x].sum()) for x in component_names}
    require(abs(totals['total_cost']-solver_check['objective'])<1e-5,'Oracle replay cost mismatch')
    regrets=[]
    for result in old['methods']:
        J=result['costs']['total_cost'];Jstar=totals['total_cost']
        regrets.append({'method':result['method'],'policy_cost':J,'oracle_cost':Jstar,
            'absolute_regret':J-Jstar,'relative_regret_over_oracle':(J-Jstar)/Jstar,
            'fraction_of_policy_cost_above_oracle':(J-Jstar)/J})
    require(all(sha(p)==h for p,h in protected.items()),'Protected data/code/policy output changed')
    summary={'status':'completed','calculated_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        'oracle_definition':'Offline full-T perfect-information optimum; not N-period rolling perfect-forecast policy.',
        'period_days':c['period_days'],'T':T,'policy_N':c['planning_periods'],
        'dataset':str(DATA),'policy_run':str(POLICY),'costs':totals,'solver':solver_check,
        'comparison':regrets,'existing_policy_feasible_and_same_objective':candidate_checks,
        'all_protected_files_unchanged':True,'independent_oracle_replay_passed':True,
        'preterminal_period_arrivals_complete':True,'any_final_unknown_arrivals_irrelevant_to_T_objective':True,
        'last_period_secondary_orders':'Canonical zero: receipt after T has no cost or benefit inside this objective; no terminal pipeline target.',
        'regret_scope':'Includes information and finite-rolling-horizon effects; not an isolated prediction loss.',
        'units':'Synthetic experiment cost units, not enterprise currency.'}
    write_csv(OUT/'oracle_period_costs.csv',periods);write_csv(OUT/'oracle_material_states.csv',pd.DataFrame(material_rows));write_csv(OUT/'oracle_product_states.csv',pd.DataFrame(product_rows))
    write_csv(OUT/'regret_comparison.csv',pd.DataFrame(regrets));write_json(OUT/'oracle_summary.json',summary)
    return summary
