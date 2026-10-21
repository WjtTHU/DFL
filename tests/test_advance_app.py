"""Causal order announcement, promise fallback and switch-to-snapshot checks."""
import unittest
import pandas as pd
import numpy as np
from scripts.app_dataset.advance import advance_panel
from scripts.app_experiment.advance import forecast_orders
from scripts.app_experiment.run import arrival_periods, forecast_supply


class AdvanceAppTests(unittest.TestCase):
    def fixture(self):
        book=pd.DataFrame({'lot_id':['seen','future','unannounced','received'],
            'woItem':['a']*4,'material_id':['A']*4,'supplier_id':['DK']*4,
            'fixed_quantity':[10.,20.,30.,40.],'quantity_unit':['die']*4,
            'announced_at':pd.to_datetime(['2025-01-01','2025-01-07','2025-01-08','2025-01-01']),
            'planned_entry_date':pd.to_datetime(['2025-01-02','2025-02-04','2025-02-05','2025-01-02']),
            'promised_completion_date':pd.to_datetime(['2025-01-09','2025-02-10','2025-02-11','2025-01-04'])})
        observed=pd.DataFrame({'lot_id':['seen'],'period':[1],'first_seen_at':pd.to_datetime(['2025-01-02 10:00']),
            'last_snapshot_at':pd.to_datetime(['2025-01-06 20:00']),'history_start_row':[2],
            'history_stop_row_exclusive':[4],'history_length':[2],'elapsed_calendar_days':[4],
            'prediction_days':[np.nan]})
        receipts=pd.DataFrame({'lot_id':['seen','future','received'],
            'completion_at':pd.to_datetime(['2025-01-10','2025-02-12','2025-01-05'])})
        return book,observed,receipts

    def test_announcement_boundary_and_unseen_history(self):
        b,o,r=self.fixture()
        p=advance_panel(b,o,[pd.Timestamp('2025-01-07')],r).set_index('lot_id')
        self.assertEqual(set(p.index),{'seen','future'})
        self.assertEqual(p.loc['future','history_length'],0)
        self.assertTrue(pd.isna(p.loc['future','first_seen_at']))
        self.assertFalse(p.loc['future','has_snapshot'])
        self.assertEqual(p.loc['future','fixed_quantity'],20)

    def test_future_outcome_perturbation_is_not_an_input(self):
        b,o,r=self.fixture(); at=pd.Timestamp('2025-01-07')
        first=advance_panel(b,o,[at],r)
        r.loc[r.completion_at>=at,'completion_at']+=pd.Timedelta(days=50)
        pd.testing.assert_frame_equal(first,advance_panel(b,o,[at],r))

    def test_shared_promise_and_observed_predictor_switch(self):
        b,o,r=self.fixture()
        p=advance_panel(b,o,[pd.Timestamp('2025-01-07')],r)
        pred=pd.DataFrame({'lot_id':['seen'],'predicted_remaining_days':[2.],
                           'raw_predicted_remaining_days':[2.]})
        f=forecast_orders(p,pred,{'product_cycle_mean_days':{'a':10},'global_fallback_days':8},7)
        seen=f[f.lot_id=='seen'].set_index('method')
        self.assertEqual(seen.loc['point_transformer','predicted_remaining_days'],2)
        self.assertEqual(seen.loc['historical_product_cycle','predicted_remaining_days'],6)
        unseen=f[f.lot_id=='future']
        self.assertEqual(unseen.predicted_remaining_days.nunique(),1)
        self.assertTrue((unseen.prediction_source=='synthetic_promise').all())
        self.assertTrue((unseen.arrival_relative_period==5).all())
        for _,g in f.groupby('method'):
            self.assertEqual(forecast_supply(g,['A'],4).sum(),10)

    def test_overdue_promise_retained_and_not_in_opening_stock(self):
        b,o,r=self.fixture()
        b.loc[b.lot_id=='future','promised_completion_date']=pd.Timestamp('2025-01-01')
        p=advance_panel(b,o,[pd.Timestamp('2025-01-07')],r)
        f=forecast_orders(p,pd.DataFrame({'lot_id':['seen'],'predicted_remaining_days':[2.],
            'raw_predicted_remaining_days':[2.]}),{'product_cycle_mean_days':{'a':10},'global_fallback_days':8},7)
        self.assertTrue((f.loc[f.lot_id=='future','arrival_relative_period']==1).all())
        np.testing.assert_equal(arrival_periods([7,8,14,15,21,22,28,29],7),[1,2,2,3,3,4,4,5])


if __name__=='__main__': unittest.main()
