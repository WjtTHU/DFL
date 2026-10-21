"""Behavioral checks for the post-training conditional distribution baseline."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluation.report_daily_experiment import fit_conditional_cycle, predict_conditional_cycle


def training_index():
    origin = pd.Timestamp('2025-01-01 09:00')
    return pd.DataFrame([{'lot_id': str(i), 'product_id': product, 'split': 'train',
                          'first_snapshot_at': origin,
                          'completion_at': origin+pd.Timedelta(days=cycle),
                          'remaining_days': cycle}
                         for i,(product,cycle) in enumerate([('A',6),('A',8),('A',10),('B',2),('B',3),('B',4)])])


def features(product, age):
    return pd.DataFrame({'product_id':[product],
                         'first_snapshot_at':[pd.Timestamp('2025-01-01 09:00')],
                         'forecast_date':[pd.Timestamp('2025-01-01')+pd.Timedelta(days=age)]})


class ConditionalBaselineTests(unittest.TestCase):
    def test_elapsed_conditioning_hand_calculation(self):
        fitted=fit_conditional_cycle(training_index(),14)
        p3,s3=predict_conditional_cycle(fitted,features('A',3))
        p7,s7=predict_conditional_cycle(fitted,features('A',7))
        # At age 7 only total cycles 8 and 10 survive: residual days 1 and 3.
        global_prior=np.ones(15)/17
        global_prior[[0,2]]=2/17
        expected=global_prior/3
        expected[[0,2]]+=1/3
        np.testing.assert_allclose(p7[0],expected)
        self.assertEqual(s3.conditional_product_support_lots.iloc[0],3)
        self.assertEqual(s7.conditional_product_support_lots.iloc[0],2)
        self.assertFalse(np.allclose(p3,p7))
        _,s6=predict_conditional_cycle(fitted,features('A',6))
        self.assertEqual(s6.conditional_product_support_lots.iloc[0],2)  # T=e is already finished.

    def test_products_and_unique_lot_weighting(self):
        training=training_index()
        first=fit_conditional_cycle(training,14)
        duplicated=fit_conditional_cycle(pd.concat([training,training.iloc[[0]*30]],ignore_index=True),14)
        a,_=predict_conditional_cycle(first,features('A',0))
        b,_=predict_conditional_cycle(first,features('B',0))
        again,_=predict_conditional_cycle(duplicated,features('A',0))
        self.assertFalse(np.allclose(a,b))
        np.testing.assert_allclose(a,again)

    def test_held_out_labels_do_not_affect_fit_or_prediction(self):
        training=training_index()
        held=training.copy(); held['split']='test'; held['remaining_days']=999
        held['completion_at']=pd.Timestamp('2099-01-01')
        first=fit_conditional_cycle(training,14)
        second=fit_conditional_cycle(pd.concat([training,held],ignore_index=True),14)
        bare=features('A',7)
        with_truth=bare.assign(remaining_days=-100,completion_at=pd.Timestamp('2100-01-01'))
        p,_=predict_conditional_cycle(first,bare)
        q,_=predict_conditional_cycle(second,with_truth)
        np.testing.assert_array_equal(p,q)

    def test_unknown_product_and_exhausted_history_fallbacks(self):
        fitted=fit_conditional_cycle(training_index(),14)
        unknown,meta=predict_conditional_cycle(fitted,features('unknown',7))
        expected=np.ones(15)/17; expected[[0,2]]=2/17
        np.testing.assert_allclose(unknown[0],expected)
        self.assertEqual(meta.conditional_source.iloc[0],'global_conditional_fallback')
        exhausted,meta=predict_conditional_cycle(fitted,features('A',100))
        np.testing.assert_allclose(exhausted[0],fitted['frequency_fallback'])
        self.assertEqual(meta.conditional_source.iloc[0],'global_frequency_fallback')

    def test_tail_is_separate_and_all_nll_values_finite(self):
        fitted=fit_conditional_cycle(training_index(),4)
        p,_=predict_conditional_cycle(fitted,features('A',0))
        self.assertGreater(p[0,-1],p[0,3])  # 6,8,10 are tail, never day 4.
        self.assertTrue((p>0).all())
        self.assertTrue(np.isfinite(-np.log(p)).all())
        np.testing.assert_allclose(p.sum(1),1)


if __name__=='__main__':
    unittest.main()
