"""Behavioral checks of APP timing, state carry, costs and CPLEX LP export."""
from pathlib import Path
import sys
import unittest
import uuid

import numpy as np
import pandas as pd
from scipy.optimize import linprog
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.app_experiment.lp import build_app_lp, residuals, solve_cplex
from scripts.app_experiment.run import (arrival_periods, baseline_predict,
    carry_finished_state, forecast_supply, period_cost)

COSTS = {'production_per_product_unit':.2, 'finished_holding_per_product_unit_period':.05,
         'backlog_per_product_unit_period':2., 'secondary_purchase_per_die':1.,
         'raw_holding_per_die_period':0., 'terminal_inventory_salvage':0.}


def toy(demand, raw=0., arrivals=None, committed=None, caps=None, lead=1):
    n = len(demand)
    return build_app_lp(np.asarray(demand).reshape(n,1), np.full(n,100.), [[1.]],
        [raw],[0.],[0.], np.zeros((n,1)) if arrivals is None else np.asarray(arrivals).reshape(n,1),
        np.zeros((n,1)) if committed is None else np.asarray(committed).reshape(n,1),
        np.zeros((n,1)) if caps is None else np.asarray(caps).reshape(n,1), lead,COSTS,4.)


def oracle(model):
    result = linprog(model.objective, A_ub=model.a_ub,b_ub=model.b_ub,
                    A_eq=model.a_eq,b_eq=model.b_eq,bounds=list(zip(np.zeros(len(model.names)),model.upper)),method='highs')
    if not result.success: raise AssertionError(result.message)
    return result


class AppExperimentTests(unittest.TestCase):
    def test_primary_current_arrival_cannot_support_current_production(self):
        model=toy([5.,5.], arrivals=[100.,0.])
        result=oracle(model)
        self.assertAlmostEqual(result.x[model.indices['p',0,0]],0.)
        self.assertAlmostEqual(result.x[model.indices['b',0,0]],5.)
        self.assertAlmostEqual(result.x[model.indices['p',1,0]],10.)

    def test_secondary_order_usable_exactly_next_opening(self):
        model=toy([0.,5.],caps=[10.,10.])
        result=oracle(model)
        self.assertAlmostEqual(result.x[model.indices['q',0,0]],5.)
        self.assertAlmostEqual(result.x[model.indices['p',0,0]],0.)
        self.assertAlmostEqual(result.x[model.indices['p',1,0]],5.)
        self.assertEqual(model.upper[model.indices['q',1,0]],0.)

    def test_positive_longer_lead_not_rounded_down(self):
        model=toy([0.,5.,0.],caps=[10.,10.,10.],lead=2)
        result=oracle(model)
        self.assertAlmostEqual(result.x[model.indices['p',1,0]],0.)
        self.assertAlmostEqual(result.x[model.indices['b',1,0]],5.)
        self.assertAlmostEqual(result.x[model.indices['p',2,0]],5.)

    def test_committed_orders_not_purchased_twice(self):
        model=toy([0.,5.],committed=[0.,5.],caps=[10.,10.])
        result=oracle(model)
        self.assertAlmostEqual(result.x[model.indices['q',0,0]],0.)
        self.assertAlmostEqual(result.x[model.indices['p',1,0]],5.)
        with self.assertRaises(ValueError): toy([1.],committed=[5.])

    def test_inventory_backlog_carry_and_clearing(self):
        finished,backlog=carry_finished_state([0.,8.],[3.,0.],[1.,0.],[4.,3.])
        np.testing.assert_allclose(finished,[0.,5.]); np.testing.assert_allclose(backlog,[6.,0.])
        finished,backlog=carry_finished_state(finished,backlog,[10.,0.],[2.,6.])
        np.testing.assert_allclose(finished,[2.,0.]); np.testing.assert_allclose(backlog,[0.,1.])

    def test_procurement_charge_on_receipt_and_terminal_once(self):
        ordinary=period_cost([10.],[2.],[3.],[5.],COSTS)
        terminal=period_cost([10.],[2.],[3.],[0.],COSTS,True,4.)
        self.assertAlmostEqual(ordinary['total_cost'],13.1)
        self.assertAlmostEqual(terminal['total_cost'],20.1)
        self.assertEqual(ordinary['terminal_backlog_cost'],0.)
        self.assertEqual(terminal['secondary_receipt_cost'],0.)

    def test_day_to_week_boundary_and_whole_lot_forecasts(self):
        np.testing.assert_array_equal(arrival_periods([0.,.1,1.,7.,7.01,8.,14.,15.,28.,29.],7),[1,1,1,1,2,2,2,3,4,5])
        predictions=pd.DataFrame({'material_id':['A','A','B'], 'arrival_relative_period':[1,5,2], 'fixed_quantity':[10.,20.,30.]})
        np.testing.assert_array_equal(forecast_supply(predictions,['A','B'],4),[[10,0],[0,30],[0,0],[0,0]])
        with self.assertRaises(ValueError): arrival_periods([-1],7)

    def test_baseline_uses_elapsed_days_and_known_product_only(self):
        fit={'product_cycle_mean_days':{'a':10.},'global_fallback_days':8.}
        orders=pd.DataFrame({'woItem':['a','a','unseen'],'elapsed_calendar_days':[3,12,2], 'future_completion':[99,99,99]})
        np.testing.assert_array_equal(baseline_predict(orders,fit),[7,0,6])
        orders.future_completion=0
        np.testing.assert_array_equal(baseline_predict(orders,fit),[7,0,6])

    def test_cplex_export_matches_independent_highs_solution(self):
        solver=yaml.safe_load((ROOT/'config/app_solver.yaml').read_text(encoding='utf-8'))
        if not Path(solver['executable']).is_file(): self.skipTest('CPLEX executable unavailable')
        model=build_app_lp([[3.,7.,4.],[9.,8.,11.],[10.,6.,15.],[8.,5.,12.]],
            [20.,22.,24.,24.],[[1,0,1],[1,1,0],[0,1,1]],[12.,10.,15.],[1.,0.,2.],[0.,2.,0.],
            [[5.,4.,9.],[7.,3.,8.],[5.,8.,7.],[40.,40.,40.]],np.zeros((4,3)),np.full((4,3),5.),1,COSTS,4.)
        result=oracle(model)
        directory=ROOT/'outputs/app/solver_tests'/uuid.uuid4().hex
        plan,audit=solve_cplex(model,directory,solver['executable'])
        self.assertAlmostEqual(audit['objective'],result.fun,places=6)
        self.assertLess(max(residuals(model,result.x).values()),1e-7)
        self.assertEqual(audit['status'],'optimal')
        self.assertLessEqual((np.array([[1,0,1],[1,1,0],[0,1,1]])@plan['p'][0]-[12,10,15]).max(),1e-7)


if __name__=='__main__': unittest.main()
