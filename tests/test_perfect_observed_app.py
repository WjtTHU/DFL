"""The diagnostic is perfect only for observed Lots, with no extra population access."""
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.app_experiment.perfect_observed import forecast_orders


def fixture():
    orders=pd.DataFrame({'period':[1]*3,'lot_id':['seen','unseen','overdue'],
        'fixed_quantity':[100.,200.,300.],'has_snapshot':[True,False,False],
        'prediction_days':[np.nan]*3,'prediction_date':pd.to_datetime(['2025-01-06']*3),
        'decision_at':pd.to_datetime(['2025-01-07']*3),
        'promised_completion_date':pd.to_datetime(['2025-01-09','2025-02-10','2025-01-05'])})
    truth=pd.DataFrame({'lot_id':['seen','unseen','overdue'],
                       'completion_at':['2025-01-14 15:30:00','2025-01-12',None]})
    return orders,truth


class PerfectObservedTests(unittest.TestCase):
    def test_only_observed_truth_used_and_promises_preserved(self):
        o,t=fixture();f,a=forecast_orders(o,t,7,4)
        np.testing.assert_array_equal(f.predicted_remaining_days,[8,35,0])
        np.testing.assert_array_equal(f.arrival_relative_period,[2,5,1])
        self.assertEqual(a.lot_id.tolist(),['seen'])
        self.assertTrue(f.loc[1:,'prediction_source'].eq('synthetic_promise').all())
        np.testing.assert_array_equal(f.fixed_quantity,o.fixed_quantity)

    def test_future_truth_of_unseen_orders_cannot_change_forecast(self):
        o,t=fixture();a,_=forecast_orders(o,t,7,4)
        t.loc[t.lot_id!='seen','completion_at']=['2040-01-01','2025-01-07']
        b,_=forecast_orders(o,t,7,4)
        pd.testing.assert_frame_equal(a,b)

    def test_missing_observed_truth_fails_instead_of_dropping_or_filling(self):
        o,t=fixture();t.loc[t.lot_id=='seen','completion_at']=None
        with self.assertRaises(ValueError):forecast_orders(o,t,7,4)

    def test_past_completion_fails_and_whole_calendar_days_used(self):
        o,t=fixture();t.loc[t.lot_id=='seen','completion_at']='2025-01-06 23:59:59'
        with self.assertRaises(ValueError):forecast_orders(o,t,7,4)
        t.loc[t.lot_id=='seen','completion_at']='2025-01-07 23:59:59'
        f,_=forecast_orders(o,t,7,4)
        self.assertEqual(f.predicted_remaining_days.iloc[0],1.)

    def test_information_switches_only_when_snapshot_appears(self):
        o,t=fixture();a,_=forecast_orders(o,t,7,4)
        o.loc[o.lot_id=='unseen','has_snapshot']=True
        b,audit=forecast_orders(o,t,7,4)
        self.assertEqual(a.predicted_remaining_days.iloc[1],35.)
        self.assertEqual(b.predicted_remaining_days.iloc[1],6.)
        self.assertEqual(audit.lot_id.tolist(),['seen','unseen'])

    def test_no_observed_orders_requires_no_completion_labels(self):
        o,t=fixture();o.has_snapshot=False;t.completion_at=None
        f,a=forecast_orders(o,t,7,4)
        self.assertTrue(a.empty)
        np.testing.assert_array_equal(f.predicted_remaining_days,[3,35,0])


if __name__=='__main__':unittest.main()
