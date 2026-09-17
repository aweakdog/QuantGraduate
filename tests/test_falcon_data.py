import unittest

import numpy as np
import pandas as pd

from scripts.falcon_data import CHANNELS, MarketPanel, make_batch, pit_membership, split_positions


class FalconDataTests(unittest.TestCase):
    def panel(self):
        dates = pd.bdate_range("2024-01-01", periods=60)
        values = np.arange(60 * 5 * 2, dtype=np.float32).reshape(60, 5, 2) / 100
        return MarketPanel(values, np.ones((60, 5), dtype=bool), dates,
                           ("000001", "000002", "000003", "000004", "000005"), ("log_close", "other"))

    def test_training_and_validation_targets_close_before_next_split(self):
        panel = self.panel()
        split = split_positions(panel.dates, context=4, horizon=5,
                                validation_start=panel.dates[30], test_start=panel.dates[45])
        self.assertTrue((split["train"] + 5 < 30).all())
        self.assertTrue((split["validation"] >= 30).all())
        self.assertTrue((split["validation"] + 5 < 45).all())
        self.assertTrue((split["test"] >= 45).all())
        self.assertTrue((split["test"] + 5 < 60).all())
        self.assertFalse(set(split["train"]) & set(split["test"]))

    def test_future_tampering_does_not_change_context_or_group_selection(self):
        panel = self.panel()
        first = make_batch(panel, [20], 8, 4, 5, np.random.default_rng(19))
        changed = MarketPanel(panel.values.copy(), panel.members.copy(), panel.dates, panel.codes, panel.channels)
        changed.values[21:] = -999
        second = make_batch(changed, [20], 8, 4, 5, np.random.default_rng(19))
        np.testing.assert_array_equal(first["context"], second["context"])
        np.testing.assert_array_equal(first["entities"], second["entities"])
        self.assertFalse(np.array_equal(first["targets"], second["targets"]))
        self.assertEqual(first["context"].shape, (1, 4, 2, 8))
        self.assertEqual(first["targets"].shape, (1, 4, 2, 5))

    def test_pit_membership_uses_effective_dates_not_final_universe(self):
        dates = pd.bdate_range("2024-01-01", periods=10)
        universe = pd.DataFrame({"effective_date": [dates[2], dates[2], dates[6]],
                                 "code": ["000001.SZ", "000002.SZ", "000003.SZ"]})
        members = pit_membership(dates, ["000001", "000002", "000003"], universe)
        self.assertFalse(members[:2].any())
        self.assertTrue(members[2:6, :2].all())
        self.assertFalse(members[:6, 2].any())
        self.assertTrue(members[6:, 2].all())
        self.assertFalse(members[6:, :2].any())

    def test_entities_are_selected_from_current_date_membership(self):
        panel = self.panel()
        panel.members[20] = [True, False, True, False, False]
        batch = make_batch(panel, [20], 8, 4, 5, np.random.default_rng(2))
        self.assertEqual(set(batch["entities"][0, :2]), {0, 2})
        self.assertTrue((batch["entities"][0, 2:] == -1).all())
        self.assertTrue(np.isnan(batch["context"][0, 2:]).all())

    def test_each_batch_item_is_one_forecast_date(self):
        panel = self.panel()
        batch = make_batch(panel, [20, 40], 8, 3, 5, np.random.default_rng(2))
        for b, t in enumerate([20, 40]):
            for e, stock in enumerate(batch["entities"][b]):
                np.testing.assert_array_equal(batch["context"][b, e], panel.values[t - 7:t + 1, stock].T)
                np.testing.assert_array_equal(batch["targets"][b, e], panel.values[t + 1:t + 6, stock].T)

    def test_unavailable_future_is_nan_not_fabricated_zero(self):
        panel = self.panel()
        batch = make_batch(panel, [58], 8, 3, 5, np.random.default_rng(2))
        self.assertTrue(np.isfinite(batch["targets"][..., :1]).all())
        self.assertTrue(np.isnan(batch["targets"][..., 1:]).all())

    def test_missing_current_price_excludes_entity_without_looking_forward(self):
        panel = self.panel()
        panel.values[20, 0, 0] = np.nan
        self.assertNotIn(0, panel.eligible(20, 8))
        panel.values[21:, 1, 0] = np.nan
        self.assertIn(1, panel.eligible(20, 8))

    def test_invalid_split_and_future_feature_names_rejected(self):
        panel = self.panel()
        with self.assertRaises(ValueError):
            split_positions(panel.dates, 4, 5, panel.dates[40], panel.dates[30])
        with self.assertRaises(ValueError):
            make_batch(panel, [2], 8, 3, 5, np.random.default_rng(2))
        self.assertFalse(any(name.startswith(("fwd_", "lab1_", "y_target")) for name in CHANNELS))


if __name__ == "__main__":
    unittest.main()
