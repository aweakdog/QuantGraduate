import importlib.util
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from scripts.falcon_data import MarketPanel, fixed_entity_batch
from scripts.falcon_trial import block_bootstrap_interval, context_shift_metrics, summarize_return_group

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch

    from scripts.falcon_trial import evaluate


class FalconEvaluationTests(unittest.TestCase):
    def panel(self):
        dates = pd.bdate_range("2024-01-01", periods=30)
        return MarketPanel(np.ones((30, 5, 2), dtype=np.float32), np.ones((30, 5), dtype=bool),
                           dates, tuple(str(i) for i in range(5)), ("log_close", "other"))

    def test_full_pool_is_sorted_and_excludes_only_asof_ineligible_stocks(self):
        panel = self.panel()
        panel.members[10, 1] = False
        panel.values[11:, 2, 0] = np.nan
        batch = fixed_entity_batch(panel, 10, 4, 5)
        np.testing.assert_array_equal(batch["entities"], [[0, 2, 3, 4]])
        self.assertEqual(batch["context"].shape, (1, 4, 2, 4))
        self.assertTrue(np.isnan(batch["targets"][0, 1, 0]).all())

    def test_explicit_groups_reject_duplicates_and_ineligible_entities(self):
        panel = self.panel()
        panel.members[10, 1] = False
        for selected in [[0, 0], [1, 2], [-1], [1.5]]:
            with self.subTest(selected=selected), self.assertRaises(ValueError):
                fixed_entity_batch(panel, 10, 4, 5, selected)
        batch = fixed_entity_batch(panel, 10, 4, 5, [4, 0, 2])
        np.testing.assert_array_equal(batch["entities"], [[4, 0, 2]])

    def test_missing_top_pick_outcome_is_not_replaced_by_next_stock(self):
        result = summarize_return_group([4, 3, 2, 1], [np.nan, 0.1, 0.2, 0.3])
        self.assertEqual(result["n_predictions"], 4)
        self.assertEqual(result["n_labeled"], 3)
        self.assertIsNone(result["top3_excess_5d"])
        self.assertEqual(result["top3_missing_labels"], 1)

    def test_rank_and_top_pick_diagnostics(self):
        result = summarize_return_group([1, 2, 3, 4], [0.01, 0.02, 0.03, 0.04])
        self.assertAlmostEqual(result["rank_ic"], 1)
        self.assertAlmostEqual(result["top3_excess_5d"], 0.005)
        self.assertEqual(result["top3_missing_labels"], 0)
        self.assertIsNone(summarize_return_group([1, 1, 1], [1, 2, 3])["rank_ic"])

    def test_block_bootstrap_is_deterministic_and_preserves_constant_mean(self):
        a = block_bootstrap_interval([0.2] * 40, samples=100)
        self.assertAlmostEqual(a["low"], 0.2)
        self.assertAlmostEqual(a["high"], 0.2)
        x = np.linspace(-0.1, 0.2, 40)
        self.assertEqual(block_bootstrap_interval(x, samples=100), block_bootstrap_interval(x, samples=100))
        self.assertIsNone(block_bootstrap_interval([0.2] * 8)["low"])

    def test_context_shift_is_measured_on_same_targets(self):
        same = context_shift_metrics([0.01, 0.02, 0.03, 0.04], [0.01, 0.02, 0.03, 0.04])
        self.assertEqual(same["median_abs_change_bp"], 0)
        self.assertAlmostEqual(same["rank_agreement"], 1)
        reverse = context_shift_metrics([0.01, 0.02, 0.03, 0.04], [0.04, 0.03, 0.02, 0.01])
        self.assertAlmostEqual(reverse["rank_agreement"], -1)
        self.assertAlmostEqual(reverse["top3_overlap"], 2 / 3)


@unittest.skipUnless(HAS_TORCH, "PyTorch tests run in GPU environment")
class FalconCalibrationTests(unittest.TestCase):
    def test_coverage_and_daily_metrics_use_observed_targets(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(quantiles=(0.1, 0.5, 0.9))
                self.quantiles = torch.tensor(self.config.quantiles)

            def forward(self, context, horizon):
                shape = (*context.shape[:-1], horizon, 3)
                prediction = torch.tensor([-1., 0., 1.]).expand(shape)
                return {"prediction": prediction, "normalized_prediction": torch.asinh(prediction),
                        "location": torch.zeros((*context.shape[:-1], 1)),
                        "scale": torch.ones((*context.shape[:-1], 1)),
                        "active": torch.ones(context.shape[:-1], dtype=torch.bool)}

        panel = MarketPanel(np.zeros((20, 4, 2), dtype=np.float32), np.ones((20, 4), dtype=bool),
                            pd.bdate_range("2024-01-01", periods=20), ("a", "b", "c", "d"), ("log_close", "other"))
        batch = fixed_entity_batch(panel, 10, 4, 5)
        result = evaluate(Model(), [batch], panel, torch.device("cpu"), False)
        self.assertEqual(result["endpoint_quantile_coverage"], [0.0, 1.0, 1.0])
        self.assertEqual(result["n_endpoint_labels"], 4)
        self.assertEqual(result["daily"][0]["n_predictions"], 4)
        self.assertEqual(result["quantile_crossing_fraction"], 0)


if __name__ == "__main__":
    unittest.main()
