"""Behavioral checks for daily origins, leakage boundaries, masks and categorical tails."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
TEST_OUTPUT = ROOT/'outputs'
TEST_OUTPUT.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT))
from local_paths import configure_imports, load_configs
configure_imports()

from wip_pred.daily.data import (DailyDataset, build_index, collate_daily, fit_preprocessor,
    forecast_calendar, load_dk, transform_snapshots, validate_protocol)
from wip_pred.daily.metrics import distribution_metrics, distribution_summary, evaluate_predictions
from wip_pred.daily.model import build_daily_model, encode_classes, fit_target_spec, loss_per_sample
from wip_pred.daily.runner import infer_loader, load_checkpoint, train_mode


def protocol():
    config = yaml.safe_load((ROOT/'config/daily_dk.yaml').read_text(encoding='utf-8'))
    config.update(train_end='2025-02-10', validation_end='2025-03-01', epochs=1, batch_size=2, threads=1)
    return config


def snapshots(rows=None):
    rows = rows or [('A', '2025-01-01 09:00', 's1'), ('A', '2025-01-01 23:59', 's2'),
                    ('A', '2025-01-03 10:00', 's3'), ('A', '2025-01-06 10:00', 's4'),
                    ('B', '2025-01-02 10:00', 's1'), ('B', '2025-01-08 10:00', 's4')]
    frame = pd.DataFrame([dict(lot_id=lot, parent_lot_id=lot, product_id='product',
                              created_at=pd.Timestamp(time), stage=stage, lot_status='run',
                              current_wafer_qty=10., oper='operation') for lot,time,stage in rows])
    return frame.sort_values(['lot_id', 'created_at'], kind='stable').reset_index(drop=True)


def labels(frame):
    result = frame.groupby('lot_id').created_at.agg(start_time='min', end_time='max')
    result['lot_id'] = result.index
    return result


class DailyDataTests(unittest.TestCase):
    def test_fixed_origin_integer_label_and_no_future_inputs(self):
        frame = snapshots()
        index = build_index(frame, pd.to_datetime(['2025-01-01','2025-01-02','2025-01-03']), protocol(), labels(frame))
        sample = index.loc[(index.lot_id=='A') & (index.forecast_date==pd.Timestamp('2025-01-02'))].iloc[0]
        self.assertEqual(sample.remaining_days, 4)  # Jan 6 - Jan 2, regardless of Jan 1 last snapshot.
        self.assertEqual(sample.sequence_length, 2)
        self.assertEqual(sample.days_since_last_snapshot, 1)
        self.assertTrue(sample.no_snapshot_today)
        selected = frame.iloc[int(sample.row_start):int(sample.row_stop)]
        self.assertEqual(selected.stage.tolist(), ['s1', 's2'])
        self.assertTrue((selected.created_at < pd.Timestamp('2025-01-03')).all())
        self.assertTrue((index.remaining_days >= 1).all())
        self.assertFalse(index.sample_id.duplicated().any())

    def test_missing_history_still_predicts_and_finished_lots_excluded(self):
        frame = snapshots([('A','2025-01-01 09:00','s1'), ('A','2025-02-01 09:00','s2')])
        index = build_index(frame, pd.to_datetime(['2025-01-25','2025-02-01','2025-02-02']), protocol(), labels(frame))
        self.assertEqual(len(index), 1)
        self.assertEqual(index.iloc[0].sequence_length, 1)
        self.assertTrue(index.iloc[0].stale_fallback)
        self.assertEqual(index.iloc[0].days_since_last_snapshot, 24)
        online = build_index(frame, [pd.Timestamp('2025-01-25')], protocol(), requested_lots=['A'])
        self.assertTrue(online.remaining_days.isna().all())
        self.assertTrue(build_index(frame, [pd.Timestamp('2025-01-25')], protocol(), requested_lots=['A'], known_completed_lots=['A']).empty)

    def test_split_boundaries_and_label_availability(self):
        frame = snapshots([('A','2025-02-01 09:00','s1'), ('A','2025-02-15 09:00','s2'),
                           ('B','2025-02-20 09:00','s1'), ('B','2025-03-07 09:00','s2'),
                           ('C','2025-03-14 09:00','s1')])
        index = build_index(frame, pd.date_range('2025-02-01','2025-03-14'), protocol(), labels(frame))
        self.assertFalse((index.split=='train').any())  # A's label wasn't available by Feb 10.
        self.assertTrue((index.loc[index.lot_id=='A','split']=='validation').all())
        self.assertTrue((index.loc[index.lot_id=='B','split']=='test').all())
        self.assertTrue((index.loc[index.split=='validation','forecast_date'] >= pd.Timestamp('2025-02-10')).all())
        self.assertTrue((index.loc[index.split=='validation','label_available_at'] <= pd.Timestamp('2025-03-01')).all())
        self.assertTrue((index.loc[index.split=='test','forecast_date'] >= pd.Timestamp('2025-03-01')).all())

    def test_calendar_ignores_final_partial_day_and_rejects_intraday_origin(self):
        frame = snapshots()
        dates = forecast_calendar(frame, protocol())
        self.assertEqual(dates.max(), pd.Timestamp('2025-01-07'))
        with self.assertRaises(ValueError):
            build_index(frame, [pd.Timestamp('2025-01-01 12:00')], protocol())
        config = protocol(); config['train_end']='2025-03-02'
        with self.assertRaises(ValueError): validate_protocol(config)

    def test_preprocessing_fit_excludes_future_and_unknown_is_supported(self):
        frame = snapshots([('A','2025-01-01','old'), ('A','2025-01-02','old'),
                           ('A','2025-01-03','old'), ('A','2025-03-05','new')])
        frame.loc[3, 'current_wafer_qty'] = 100000
        core,_ = load_configs('dk')
        pre, fitted = fit_preprocessor(frame, core, protocol())
        transformed = transform_snapshots(frame, fitted, pre)
        self.assertEqual(pre.scaler.mean_[0], 10.)
        self.assertEqual(transformed.iloc[3].stage, pre.label_encoders['stage'].stoi['<UNK>'])
        index = build_index(frame, [pd.Timestamp('2025-01-02')], protocol(), requested_lots=['A'])
        data = DailyDataset(transformed, index, fitted)
        _, features, target = data[0]
        self.assertIsNone(target)
        self.assertNotIn('remaining_days', features)
        self.assertNotIn('completion_at', features)
        self.assertEqual(features['numeric'].shape[-1], 4)

    def test_raw_intraday_stage_retention_and_online_future_independence(self):
        raw = pd.DataFrame({'LotId':['A']*5,'parentLotId':['A']*5,'woItem':['p']*4+['future-product'],
            'stage':['s1','s2','s3','s3','future'],'tranCode':['run']*5,'waferQty':[10]*5,'oper':['op']*5,
            'file_timestamp':['2025-01-01 23:56','2025-01-01 23:57','2025-01-01 23:57','2025-01-01 23:57','2025-01-03 12:00']})
        with tempfile.TemporaryDirectory(dir=TEST_OUTPUT) as tmp:
            self.assertTrue(Path(tmp).resolve().is_relative_to(TEST_OUTPUT.resolve()))
            path=Path(tmp)/'input.csv'; raw.to_csv(path,index=False)
            with patch('local_paths.dataset_paths', return_value={'csv':path}):
                first, truth, _ = load_dk(protocol(), cutoff_exclusive='2025-01-02', with_labels=False)
                self.assertIsNone(truth)
                self.assertEqual(first.stage.tolist(), ['s1','s2','s3'])
                self.assertEqual(first.created_at.max(),pd.Timestamp('2025-01-01 23:57'))
                raw.iloc[:4].to_csv(path,index=False)
                second,_,_ = load_dk(protocol(), cutoff_exclusive='2025-01-02', with_labels=False)
                pd.testing.assert_frame_equal(first,second)


class DistributionTests(unittest.TestCase):
    def test_tail_is_separate_class_and_nll_matches_hand_calculation(self):
        spec=fit_target_spec([1,2,2,3,30],.5)
        self.assertEqual(spec['horizon'],2)
        self.assertEqual(spec['train_tail_count'],2)
        target=torch.tensor([1,2,30])
        self.assertEqual(encode_classes(target,2).tolist(),[0,1,2])
        p=torch.tensor([[.2,.3,.5],[.2,.3,.5],[.2,.3,.5]],dtype=torch.float64)
        loss=loss_per_sample(p.log(),target,'distribution',spec)
        np.testing.assert_allclose(loss.numpy(),-np.log([.2,.3,.5]))
        with self.assertRaises(ValueError): encode_classes(torch.tensor([0]),2)
        with self.assertRaises(ValueError): fit_target_spec([1,2.5])

    def test_open_tail_quantiles_and_calibration(self):
        p=np.array([[.1,.2,.7],[.6,.3,.1]])
        summaries=distribution_summary(p,2)
        self.assertTrue(np.isnan(summaries['median_upper_days'][0]))
        self.assertEqual(summaries['median_lower_days'][0],3)
        self.assertEqual(summaries['median_upper_days'][1],1)
        self.assertAlmostEqual(summaries['mean_days_lower_bound'][0],2.6)
        index=pd.DataFrame({'lot_id':['A','B'],'remaining_days':[100,1]})
        result=distribution_metrics(index,p,2,-np.log([.7,.6]))
        self.assertEqual(result['central_90_class_interval_coverage'],1.)
        self.assertEqual(result['central_90_finite_upper_fraction'],0.)
        self.assertIn('2',result['calibration_by_horizon_days'])

    def test_constant_target_scaler_and_point_loss_roundtrip(self):
        spec=fit_target_spec([3,3,3])
        self.assertEqual(spec['scale'],1)
        self.assertEqual(loss_per_sample(torch.zeros(3,1),torch.tensor([3,3,3]),'point',spec).sum().item(),0.)


class DailyModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def fixture(self):
        frame=snapshots()
        core,training=load_configs('dk')
        pre,core=fit_preprocessor(frame,core,protocol())
        transformed=transform_snapshots(frame,core,pre)
        index=build_index(frame,pd.to_datetime(['2025-01-01','2025-01-02','2025-01-03']),protocol(),labels(frame))
        return transformed,index,pre,core,training

    def test_padding_invariance_for_both_modes(self):
        transformed,index,_,core,_=self.fixture()
        data=DailyDataset(transformed,index,core)
        spec=fit_target_spec(index.remaining_days)
        short=int(index.sequence_length.argmin()); long=int(index.sequence_length.argmax())
        for mode in ('point','distribution'):
            model=build_daily_model(core,mode,spec).eval()
            _,alone,_=collate_daily([data[short]])
            _,padded,_=collate_daily([data[short],data[long]])
            with torch.inference_mode():
                a=model(alone)[0]; b=model(padded)[0]
                torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)
                n=int(padded['length'][0])
                padded['numeric'][0,n:]=1000
                padded['temporal'][0,n:]=-1000
                c=model(padded)[0]
                torch.testing.assert_close(b,c,atol=1e-6,rtol=1e-5)

    def test_training_checkpoint_reload_and_no_test_selection(self):
        transformed,index,pre,core,training=self.fixture()
        # Disjoint rows for this numerical smoke test; real chronology is checked above.
        index.loc[index.index[-2:], 'split']='validation'
        index.loc[index.index[-1:], 'split']='test'
        spec=fit_target_spec(index.loc[index.split=='train','remaining_days'])
        with tempfile.TemporaryDirectory(dir=TEST_OUTPUT) as tmp:
            self.assertTrue(Path(tmp).resolve().is_relative_to(TEST_OUTPUT.resolve()))
            out=Path(tmp)
            for mode in ('point','distribution'):
                train_mode(mode,transformed,index,pre,core,training,spec,protocol(),{'smoke_only':True},out)
                model,_,cfg=load_checkpoint(out/mode)
                self.assertFalse(model.training)
                state=json.loads((out/mode/'training_state.json').read_text())
                self.assertTrue(state['checkpoint_validation_verified'])
                self.assertFalse(state['test_evaluated_during_training'])
                data=DailyDataset(transformed,index.loc[index.split=='test'],core)
                outputs,losses=infer_loader(model,data,2,mode,spec)
                _,metrics=evaluate_predictions(data.index,outputs,mode,spec,losses)
                self.assertEqual(metrics['windows'],1)
                self.assertEqual(cfg['mode'],mode)

    def test_future_other_lot_does_not_change_forecast_features(self):
        frame=snapshots()
        core,_=load_configs('dk')
        pre,core=fit_preprocessor(frame,core,protocol())
        first=build_index(frame,[pd.Timestamp('2025-01-02')],protocol(),requested_lots=['A'])
        changed=snapshots([('A','2025-01-01 09:00','s1'),('A','2025-01-01 23:59','s2'),
                           ('A','2025-01-03 10:00','s3'),('A','2025-01-06 10:00','s4'),
                           ('B','2025-01-02 10:00','s1'),('B','2025-06-01 10:00','s4')])
        second=build_index(changed,[pd.Timestamp('2025-01-02')],protocol(),requested_lots=['A'])
        a=DailyDataset(transform_snapshots(frame,core,pre),first,core)[0][1]
        b=DailyDataset(transform_snapshots(changed,core,pre),second,core)[0][1]
        torch.testing.assert_close(a['numeric'],b['numeric'])
        torch.testing.assert_close(a['temporal'],b['temporal'])

    def test_held_out_test_targets_cannot_change_training_or_checkpoint_selection(self):
        transformed,index,pre,core,training=self.fixture()
        index.loc[index.index[-2:], 'split']='validation'
        index.loc[index.index[-1:], 'split']='test'
        changed=index.copy()
        changed.loc[changed.split=='test','remaining_days']=10000
        spec=fit_target_spec(index.loc[index.split=='train','remaining_days'])
        with tempfile.TemporaryDirectory(dir=TEST_OUTPUT) as tmp:
            self.assertTrue(Path(tmp).resolve().is_relative_to(TEST_OUTPUT.resolve()))
            out=Path(tmp); (out/'a').mkdir(); (out/'b').mkdir()
            for name,samples in [('a',index),('b',changed)]:
                train_mode('point',transformed,samples,pre,core,training,spec,protocol(),{'smoke_only':True},out/name)
            a=torch.load(out/'a/point/best_model.pth',weights_only=True)
            b=torch.load(out/'b/point/best_model.pth',weights_only=True)
            for key in a: torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
            sa=json.loads((out/'a/point/training_state.json').read_text())
            sb=json.loads((out/'b/point/training_state.json').read_text())
            self.assertEqual(sa['best_validation_score'],sb['best_validation_score'])

    def test_unlabelled_prediction_has_no_truth_columns_and_clips_negative_days(self):
        frame=snapshots()
        index=build_index(frame,[pd.Timestamp('2025-01-02')],protocol(),requested_lots=['A'])
        spec=fit_target_spec([1,5])
        result,metrics=evaluate_predictions(index,np.array([[-2.]]),'point',spec)
        self.assertIsNone(metrics)
        self.assertEqual(result.iloc[0].predicted_remaining_days,0.)
        self.assertEqual(result.iloc[0].raw_predicted_remaining_days,-7.)
        for name in ['remaining_days','completion_at','label_available_at']:
            self.assertNotIn(name,result.columns)


if __name__=='__main__': unittest.main()
