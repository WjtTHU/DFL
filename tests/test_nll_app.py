"""Behavioral tests for expected-supply PTO, including tails and physical timing."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
from scipy.optimize import linprog

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.app_experiment.distribution import daily_to_weekly, forecast_orders, expected_supply
from scripts.app_experiment.run import forecast_supply
from scripts.app_experiment.lp import build_app_lp


def orders():
    return pd.DataFrame({'lot_id':['observed','future','overdue'], 'material_id':['A','A','B'],
        'fixed_quantity':[100.,80.,50.], 'has_snapshot':[True,False,False],
        'prediction_days':[0,0,0], 'prediction_date':pd.to_datetime(['2025-01-06']*3),
        'promised_completion_date':pd.to_datetime(['2025-01-07','2025-02-10','2025-01-01'])})


class NLLAppTests(unittest.TestCase):
    def test_exact_week_boundaries_and_open_tail(self):
        days=[1,7,8,14,15,21,22,28,29]
        p=np.zeros((len(days),29));p[np.arange(len(days)),np.array(days)-1]=1.
        w=daily_to_weekly(p,7,4)
        np.testing.assert_array_equal(w.argmax(1),[0,0,1,1,2,2,3,3,4])
        np.testing.assert_array_equal(w.sum(1),np.ones(len(days)))

    def test_tail_is_not_renormalized_into_four_weeks(self):
        p=np.zeros((1,29));p[0,0]=.4;p[0,7]=.1;p[0,-1]=.5
        frame=pd.DataFrame({'lot_id':['x'],'material_id':['A'],'fixed_quantity':[100.]})
        for i,col in enumerate(['p_week_1','p_week_2','p_week_3','p_week_4','p_beyond_horizon']):
            frame[col]=daily_to_weekly(p,7,4)[:,i]
        np.testing.assert_allclose(expected_supply(frame,['A'],4).ravel(),[40,10,0,0])

    def test_degenerate_probability_reduces_to_point_supply(self):
        rows=pd.DataFrame({'lot_id':['a','b','c'],'material_id':['A','B','A'],
                          'fixed_quantity':[5.,7.,9.],'arrival_relative_period':[1,3,5]})
        for k in range(1,5):rows[f'p_week_{k}']=(rows.arrival_relative_period==k).astype(float)
        rows['p_beyond_horizon']=(rows.arrival_relative_period>4).astype(float)
        np.testing.assert_array_equal(expected_supply(rows,['A','B'],4),forecast_supply(rows,['A','B'],4))

    def test_unseen_orders_share_promise_and_do_not_fabricate_daily_distribution(self):
        p=pd.DataFrame({'lot_id':['observed'],**{f'p_day_{d}':[1. if d==8 else 0.] for d in range(1,29)},'p_tail':[0.]})
        f=forecast_orders(orders(),p,7,4)
        self.assertEqual(f.loc[0,'p_week_2'],1.)
        self.assertEqual(f.loc[1,'p_beyond_horizon'],1.)
        self.assertEqual(f.loc[2,'p_week_1'],1.)
        self.assertTrue(f.loc[1:,['p_day_1','p_tail']].isna().all().all())
        self.assertTrue(f.loc[1:,'prediction_source'].eq('synthetic_promise').all())
        np.testing.assert_array_equal(f.fixed_quantity,orders().fixed_quantity)

    def test_invalid_probabilities_horizon_and_duplicate_lots_rejected(self):
        valid=np.zeros((1,29));valid[0,0]=1.
        invalid=[valid[:,:28],valid*.9,valid.copy(),valid.copy()]
        invalid[2][0,0]=np.nan;invalid[3][0,1]=-.1;invalid[3][0,0]=1.1
        for p in invalid:
            with self.assertRaises((ValueError,AssertionError)):daily_to_weekly(p,7,4)
        frame=pd.DataFrame({'lot_id':['x','x'],'material_id':['A','A'],'fixed_quantity':[1.,1.],
            'p_week_1':[1.,1.],'p_week_2':[0.,0.],'p_week_3':[0.,0.],'p_week_4':[0.,0.],'p_beyond_horizon':[0.,0.]})
        with self.assertRaises((ValueError,AssertionError)):expected_supply(frame,['A'],4)

    def test_expected_receipts_do_not_make_current_production_feasible(self):
        costs={'production_per_product_unit':.2,'finished_holding_per_product_unit_period':.05,
            'backlog_per_product_unit_period':2.,'secondary_purchase_per_die':1.,
            'raw_holding_per_die_period':0.,'terminal_inventory_salvage':0.}
        model=build_app_lp([[5.],[5.]],[100.,100.],[[1.]],[0.],[0.],[0.],
                           [[50.],[0.]],np.zeros((2,1)),np.zeros((2,1)),1,costs,4.)
        r=linprog(model.objective,A_eq=model.a_eq,b_eq=model.b_eq,A_ub=model.a_ub,b_ub=model.b_ub,
                  bounds=list(zip(np.zeros(len(model.names)),model.upper)),method='highs')
        self.assertTrue(r.success)
        self.assertAlmostEqual(r.x[model.indices['p',0,0]],0.)
        self.assertAlmostEqual(r.x[model.indices['p',1,0]],10.)

    def test_input_with_future_truth_is_rejected(self):
        frame=orders().iloc[1:].copy();frame['completion_at']='2025-03-01'
        with self.assertRaises((ValueError,AssertionError)):
            forecast_orders(frame,pd.DataFrame(),7,4)


if __name__=='__main__':unittest.main()
