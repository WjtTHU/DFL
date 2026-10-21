"""Boundary and leakage checks for the authorized 28-day research protocol."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts.app_dataset.order_information import build_order_book, fit_promised_cycles, visible_orders
from wip_pred.daily.data import DailyDataset, validate_protocol
from wip_pred.daily.metrics import distribution_metrics
from wip_pred.daily.model import encode_classes, fit_target_spec, loss_per_sample
from wip_pred.daily.runner import infer_loader, load_checkpoint, train_mode
import test_daily_prediction as daily_tests
from test_daily_prediction import protocol


class FixedHorizonTests(unittest.TestCase):
    def test_fixed_range_and_day28_not_tail(self):
        spec = fit_target_spec([1, 2, 28, 29, 43], fixed_horizon=28)
        self.assertEqual(spec["horizon"], 28)
        self.assertEqual(spec["classes"], [*range(1, 29), ">28"])
        self.assertEqual(spec["train_tail_count"], 2)
        targets = torch.tensor([1, 28, 29, 43])
        self.assertEqual(encode_classes(targets, 28).tolist(), [0, 27, 28, 28])
        probability = torch.full((4, 29), 1 / 29, dtype=torch.float64)
        np.testing.assert_allclose(loss_per_sample(probability.log(), targets, "distribution", spec),
                                   np.full(4, np.log(29)))
        short = fit_target_spec([1, 2, 3], fixed_horizon=28)
        self.assertEqual(short["horizon"], 28)
        self.assertEqual(short["train_tail_count"], 0)
        self.assertEqual(fit_target_spec([1, 2, 28, 29, 43], .5)["horizon"], 28)
        for invalid in (0, -1, 28.5, True, "28"):
            with self.assertRaises(ValueError):
                fit_target_spec([1, 2], fixed_horizon=invalid)
            cfg = dict(protocol(), distribution_horizon_days=invalid)
            with self.assertRaises(ValueError):
                validate_protocol(cfg)

    def test_weekly_cdf_and_open_tail(self):
        p = np.full((3, 29), 1 / 29)
        idx = pd.DataFrame({"lot_id": ["a", "b", "c"], "remaining_days": [7, 28, 43]})
        metrics = distribution_metrics(idx, p, 28, np.full(3, np.log(29)))
        self.assertTrue({"7", "14", "21", "28"} <= set(metrics["calibration_by_horizon_days"]))
        self.assertAlmostEqual(metrics["actual_tail_fraction"], 1 / 3)

    def test_train_and_reload_29_class_checkpoint(self):
        torch.set_num_threads(1)
        transformed, index, pre, core, training = daily_tests.DailyModelTests().fixture()
        index.loc[index.index[-2:], "split"] = "validation"
        index.loc[index.index[-1:], "split"] = "test"
        cfg = dict(protocol(), distribution_horizon_days=28)
        spec = fit_target_spec(index.loc[index.split == "train", "remaining_days"], fixed_horizon=28)
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as directory:
            out = Path(directory)
            train_mode("distribution", transformed, index, pre, core, training, spec, cfg,
                       {"smoke_only": True}, out)
            model, _, saved = load_checkpoint(out / "distribution")
            data = DailyDataset(transformed, index.loc[index.split == "test"], core)
            predictions, loss = infer_loader(model, data, 2, "distribution", spec)
            self.assertEqual(predictions.shape, (1, 29))
            self.assertTrue(np.isfinite(loss).all())
            self.assertEqual(saved["target_spec"]["horizon_source"], "fixed_experiment_design")


class OrderInformationTests(unittest.TestCase):
    def fixture(self):
        train = pd.DataFrame([
            ["a", "X", "2024-11-01", "2024-11-01", "2024-11-06", "2024-11-09", "train"],
            ["a", "X", "2024-11-02", "2024-11-01", "2024-11-06", "2024-11-09", "train"],
            ["b", "X", "2024-11-01", "2024-11-01", "2024-11-08", "2024-11-11", "train"],
            ["c", "Y", "2025-01-10", "2025-01-10", "2025-02-20", "2025-02-23", "test"],
        ], columns=["lot_id", "product_id", "forecast_date", "first_snapshot_at",
                    "completion_at", "label_available_at", "split"])
        catalog = pd.DataFrame([
            ["future", "X", "A", "DK", 100, "die", "2025-01-29 11:00", "2025-02-20"],
            ["unknown", "Z", "B", "DK", 200, "die", "2025-01-30 11:00", "2025-03-20"],
        ], columns=["lot_id", "woItem", "material_id", "supplier_id", "fixed_quantity",
                    "quantity_unit", "first_seen_at", "completion_at"])
        return train, catalog

    def test_announcement_boundary_and_unknown_future_outcomes(self):
        train, catalog = self.fixture()
        fitted = fit_promised_cycles(train, "2024-12-10")
        self.assertEqual(fitted["product_mean_days"]["X"], 6)
        book = build_order_book(catalog, fitted)
        self.assertEqual(len(visible_orders(book, "2024-12-31 23:59:59")), 0)
        self.assertEqual(visible_orders(book, "2025-01-01").lot_id.tolist(), ["future"])
        self.assertEqual(book.iloc[0].promised_completion_date, pd.Timestamp("2025-02-04"))
        self.assertEqual(book.iloc[1].promise_cycle_source, "training_global_mean")
        changed = catalog.copy()
        changed["completion_at"] = "2099-01-01"
        pd.testing.assert_frame_equal(book, build_order_book(changed, fitted))
        changed_train = train.copy()
        changed_train.loc[changed_train.split == "test", "completion_at"] = "2099-01-01"
        self.assertEqual(fitted, fit_promised_cycles(changed_train, "2024-12-10"))
        self.assertNotIn("completion_at", book)
        self.assertEqual(len(visible_orders(book, "2025-03-01")), 2)
        self.assertEqual(visible_orders(book, "2025-03-01", ["future"]).lot_id.tolist(), ["unknown"])

    def test_invalid_quantities_duplicate_ids_and_unavailable_train_labels(self):
        train, catalog = self.fixture()
        fitted = fit_promised_cycles(train, "2024-12-10")
        changed = catalog.copy()
        changed.loc[0, "fixed_quantity"] = 0
        with self.assertRaises(ValueError):
            build_order_book(changed, fitted)
        with self.assertRaises(ValueError):
            build_order_book(pd.concat([catalog, catalog]), fitted)
        train.loc[0, "label_available_at"] = "2024-12-11"
        with self.assertRaises(ValueError):
            fit_promised_cycles(train, "2024-12-10")


if __name__ == "__main__":
    unittest.main()
