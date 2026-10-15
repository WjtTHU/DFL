"""Regression coverage for shared train/validation/inference target units."""

import sys
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wip_pred.datasets.get_dataset import get_dataset
from wip_pred.training.trainers.reg_trainer import RegTrainer


def make_fixture(remaining_days):
    completion = pd.Timestamp("2025-06-01 12:00:00")
    rows, labels = [], []
    for number, days in enumerate(remaining_days):
        lot_id = f"lot-{number}"
        reference = completion - pd.Timedelta(days=days)
        for timestamp in (reference, completion):
            rows.append({"lot_id": lot_id, "created_at": timestamp, "day": timestamp.day})
        labels.append({
            "lot_id": lot_id, "start_time": reference, "end_time": completion,
            "min_pred_time": reference, "max_pred_time": reference,
        })
    return pd.DataFrame(rows), pd.DataFrame(labels).set_index("lot_id", drop=False)


def core_config():
    return {
        "dataset": {"type": "lotwise", "params": {"window_days": 15, "step_days": 1}},
        "embed": {"numeric_feat_cols": [], "categorical_feat_cols": [],
                  "temporal_feat_cols": ["day"], "position_col": None},
    }


class ConstantPrediction(torch.nn.Module):
    def forward(self, inputs):
        # Seven days expressed in the training scale [1, 62].
        return torch.full((len(inputs["length"]),), (7.0 - 1.0) / 61.0)


class TargetScalingTests(unittest.TestCase):
    def build(self, targets, scaler=None):
        data, labels = make_fixture(targets)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return get_dataset(data, core_config(), df_of_completed_lots=labels,
                               target_scaler=scaler)

    def test_training_still_fits_the_observed_range(self):
        dataset = self.build([1.0, 62.0])
        self.assertTrue(dataset.fit_scaler)
        self.assertEqual((dataset.min_target, dataset.max_target), (1.0, 62.0))
        _, _, targets, _ = next(iter(dataset.to_dataloader(batch_size=2, shuffle=False)))
        torch.testing.assert_close(targets, torch.tensor([0.0, 1.0]))

    def test_validation_preserves_range_and_out_of_range_labels(self):
        dataset = self.build([0.5, 108.0], scaler=(1.0, 62.0))
        self.assertFalse(dataset.fit_scaler)
        self.assertEqual((dataset.min_target, dataset.max_target), (1.0, 62.0))
        self.assertEqual([s["RCT_in_days"] for s in dataset.samples], [0.5, 108.0])
        _, _, targets, _ = next(iter(dataset.to_dataloader(batch_size=2, shuffle=False)))
        self.assertLess(targets[0].item(), 0.0)
        self.assertGreater(targets[1].item(), 1.0)
        torch.testing.assert_close(targets * 61.0 + 1.0, torch.tensor([0.5, 108.0]))

    def test_validation_metric_matches_original_days(self):
        dataset = self.build([0.5, 108.0], scaler=(1.0, 62.0))
        trainer = RegTrainer(
            ConstantPrediction(), criterion=torch.nn.L1Loss(reduction="none"),
            optimizer=None, device="cpu", target_scaler=(1.0, 62.0),
            metrics={"MAE": lambda actual, pred: float(np.abs(actual - pred).mean())},
            use_logger=False, use_tensorboard=False,
        )
        loss, result = trainer.validate(
            dataset.to_dataloader(batch_size=2, shuffle=False), collect_details=True)
        expected_mae = (abs(7.0 - 0.5) + abs(7.0 - 108.0)) / 2
        self.assertAlmostEqual(result["MAE"], expected_mae, delta=1e-4)
        self.assertAlmostEqual(loss * 61.0, expected_mae, delta=1e-4)

    def test_prediction_reuses_supplied_range(self):
        data, _ = make_fixture([108.0])
        dataset = get_dataset(data, core_config(), predict_mode=True,
                              lots_for_test=["lot-0"], target_scaler=(1.0, 62.0))
        self.assertFalse(dataset.fit_scaler)
        self.assertEqual((dataset.min_target, dataset.max_target), (1.0, 62.0))

    def test_dataset_factory_rejects_scaler_drift(self):
        data, labels = make_fixture([108.0])
        with patch("wip_pred.datasets.lotwise_dataset.LotwiseTensorRegDataset",
                   return_value=SimpleNamespace(min_target=1.0, max_target=108.0)):
            with self.assertRaisesRegex(ValueError, "target scaler"):
                get_dataset(data, core_config(), df_of_completed_lots=labels,
                            target_scaler=(1.0, 62.0))


if __name__ == "__main__":
    unittest.main()
