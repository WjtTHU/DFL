"""Behavioral boundary tests for the independent APP dataset generator."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.app_dataset.generate import (arrival_events, calibrate, decision_panel,
                                         eligible_orders, make_demand, validate_config)


def config():
    return yaml.safe_load((ROOT/'config/app_dk_small.yaml').read_text(encoding='utf-8'))


def sample_lots():
    rows = []
    # At Jan28 00:00: old receipt is gone; exact-boundary receipt is not yet usable;
    # future and unknown are pending; not-yet-seen order is not known.
    for lot, first, complete in [
        ('old', '2025-01-20', '2025-01-27 22:00'),
        ('boundary', '2025-01-20', '2025-01-28 00:00'),
        ('future', '2025-01-26', '2025-02-03 12:00'),
        ('unknown', '2025-01-26', None),
        ('not_yet_seen', '2025-01-28 00:00', '2025-02-05'),
    ]:
        rows.append({'lot_id':lot, 'parent_lot_id':lot, 'woItem':'x', 'material_id':'A',
            'supplier_id':'DK_dataset', 'first_seen_at':pd.Timestamp(first), 'first_source_row_id':0,
            'fixed_quantity':100, 'quantity_unit':'die', 'completion_at':pd.Timestamp(complete),
            'included':True})
    result = pd.DataFrame(rows)
    result['label_available_at'] = result.completion_at + pd.Timedelta(days=3)
    return result


class AppDatasetTests(unittest.TestCase):
    def test_decision_boundary_and_unlabelled_lot(self):
        active = eligible_orders(sample_lots(), pd.Timestamp('2025-01-28'))
        self.assertEqual(set(active.lot_id), {'boundary','future','unknown'})
        self.assertFalse({'completion_at','label_available_at'} & set(active.columns))

    def test_future_completion_never_changes_present_order_input(self):
        lots = sample_lots()
        before = eligible_orders(lots, pd.Timestamp('2025-01-28'))
        lots.loc[lots.lot_id == 'future', 'completion_at'] = pd.Timestamp('2026-01-01')
        pd.testing.assert_frame_equal(before, eligible_orders(lots, pd.Timestamp('2025-01-28')))

    def test_arrival_during_period_available_next_period(self):
        receipts = arrival_events(sample_lots(), pd.Timestamp('2025-01-28'), 7).set_index('lot_id')
        self.assertEqual(receipts.at['old','usable_from'], pd.Timestamp('2025-01-28'))
        self.assertEqual(receipts.at['boundary','usable_from'], pd.Timestamp('2025-02-04'))
        self.assertEqual(receipts.at['future','usable_from'], pd.Timestamp('2025-02-04'))
        self.assertEqual(receipts.at['not_yet_seen','usable_from'], pd.Timestamp('2025-02-11'))
        self.assertNotIn('unknown', receipts.index)
        self.assertEqual(receipts.fixed_quantity.sum(), 400)

    def test_history_keeps_intraday_order_but_excludes_midnight(self):
        lots = sample_lots().query("lot_id == 'future'").copy()
        snapshots = pd.DataFrame({'lot_id':['future']*3,
            'created_at':pd.to_datetime(['2025-01-27 08:00','2025-01-27 16:00','2025-01-28 00:00']),
            'stage':['A','B','future'], 'lot_status':['Run']*3, 'oper':['x']*3})
        panel = decision_panel(lots, snapshots, [pd.Timestamp('2025-01-28')], config()).iloc[0]
        self.assertEqual(panel.history_start_row, 0)
        self.assertEqual(panel.history_stop_row_exclusive, 2)
        self.assertEqual(panel.last_stage, 'B')
        self.assertEqual(panel.prediction_date, pd.Timestamp('2025-01-27'))
        self.assertTrue(pd.isna(panel.prediction_days))

    def test_missing_recent_snapshots_uses_real_stale_timestamp(self):
        lots = sample_lots().query("lot_id == 'unknown'").copy()
        lots['first_seen_at'] = pd.Timestamp('2025-01-01')
        snapshots = pd.DataFrame({'lot_id':['unknown'], 'created_at':[pd.Timestamp('2025-01-01 08:00')],
                                 'stage':['A'], 'lot_status':['Run'], 'oper':['x']})
        panel = decision_panel(lots, snapshots, [pd.Timestamp('2025-01-28')], config()).iloc[0]
        self.assertTrue(panel.stale_fallback)
        self.assertEqual(panel.snapshot_age_days, 26)
        self.assertEqual(panel.last_snapshot_at, pd.Timestamp('2025-01-01 08:00'))

    def test_per_material_calibration_and_late_label_exclusion(self):
        c = config()
        lots = pd.DataFrame({'lot_id':['a','b','c','late'], 'material_id':['A','B','C','A'],
            'fixed_quantity':[2200,2200,2200,999999], 'included':[True]*4,
            'completion_at':pd.to_datetime(['2024-10-10']*4),
            'label_available_at':pd.to_datetime(['2024-10-13']*3+['2025-01-01'])})
        params, reference = calibrate(lots, c)
        self.assertEqual(len(reference), 3)
        self.assertEqual(params['mean_weekly_primary_supply_die'], {'A':200.,'B':200.,'C':200.})
        self.assertEqual(params['mean_weekly_product_demand'], {'P1':90.,'P2':90.,'P3':90.})
        self.assertEqual(params['secondary_capacity_die_per_period']['A'],54.)

    def test_demand_has_independent_reproducible_streams_and_lookahead(self):
        c = config()
        params = {'mean_weekly_product_demand':{'P1':100,'P2':200,'P3':300}}
        test = make_demand(c, params, 'test')
        pd.testing.assert_frame_equal(test, make_demand(c, params, 'test'))
        self.assertEqual(test.period.nunique(),10)
        self.assertEqual(test.loc[test.is_evaluation_period].period.nunique(),7)
        validation = make_demand(c, params, 'validation')
        self.assertFalse(np.array_equal(test.demand.iloc[:21].to_numpy(), validation.demand.to_numpy()))
        for product, mean in params['mean_weekly_product_demand'].items():
            self.assertTrue(test.loc[test.product_id == product,'demand'].between(.8*mean,1.2*mean).all())

    def test_zero_lead_and_incomplete_period_are_rejected(self):
        c = config()
        validate_config(c)
        altered = deepcopy(c); altered['secondary']['effective_lead_periods'] = 0
        with self.assertRaises(ValueError): validate_config(altered)
        altered = deepcopy(c); altered['splits']['test']['end_exclusive'] = '2025-03-17'
        with self.assertRaises(ValueError): validate_config(altered)


if __name__ == '__main__':
    unittest.main()
