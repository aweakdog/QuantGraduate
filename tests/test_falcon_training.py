import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from scripts.falcon_data import MarketPanel
from scripts.falcon_objectives import OBJECTIVES
from scripts.falcon_study import study_commands, study_name, summarize_objectives, summarize_study
from scripts.falcon_trial import milestone_steps, summarize_return_group

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch

    from scripts.falcon_trial import train_variant


class FalconTrainingPlanTests(unittest.TestCase):
    def test_milestones_are_fixed_increasing_and_include_last_update(self):
        self.assertEqual(milestone_steps(2048, [128, 512, 2048]), (128, 512, 2048))
        for bad in [[512, 128, 2048], [128, 128, 2048], [0, 2048], [128, 512], [128, 4096]]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                milestone_steps(2048, bad)

    def test_study_commands_force_budget_mode_and_same_inputs(self):
        tasks = study_commands(Path('/code'), Path('/output'), Path('/data'), [42, 1, 123], [128, 512, 2048])
        self.assertEqual(len(tasks), 3)
        self.assertEqual(len({t['output'] for t in tasks}), 3)
        for task in tasks:
            cmd = task['command']
            self.assertEqual(cmd[cmd.index('--mode') + 1], 'budget')
            self.assertEqual(cmd[cmd.index('--steps') + 1], '2048')
            self.assertEqual(cmd[cmd.index('--entities') + 1], '16')
            self.assertEqual(cmd[cmd.index('--lr') + 1], '0.001')
        with self.assertRaises(ValueError):
            study_commands(Path('/code'), Path('/output'), Path('/data'), [42, 42], [128])

    def test_budget_summary_uses_paired_differences(self):
        with tempfile.TemporaryDirectory() as tmp:
            for seed, before, after in [(1, 0.01, 0.05), (42, 0.1, 0.11), (123, 0.02, 0.08)]:
                path = Path(tmp) / f'seed_{seed}'
                path.mkdir()
                metrics = {'endpoint_return_mae': 0.05, 'zero_return_mae': 0.04,
                           'normalized_pinball': 0.1, 'quantile_crossing_fraction': 0.05}
                stages = [{'step': step, 'validation': {**metrics, 'group_rank_ic_mean': value}}
                          for step, value in [(128, before), (512, after)]]
                (path / 'budget_temporal.json').write_text(json.dumps({'milestones': stages}))
            result = summarize_study(Path(tmp), [1, 42, 123], [128, 512])
            self.assertTrue(result['test_not_evaluated'])
            self.assertFalse(result['adoption_ready'])
            row = result['rows'][1]
            self.assertEqual(row['n_seeds'], 3)
            self.assertAlmostEqual(row['paired_delta_median']['group_rank_ic_mean'], 0.04)
            self.assertEqual(row['ic_improved_count'], 3)
            self.assertEqual(row['beats_zero_mae_count'], 0)

    def test_objective_commands_pin_inputs_and_separate_artifacts(self):
        tasks = study_commands(Path('/code'), Path('/output'), Path('/data'), [42, 1, 123, 888, 2024], [128, 512, 2048],
                               variants=['prototype'], objectives=OBJECTIVES, expected_manifest='fixed-hash')
        self.assertEqual(len(tasks), 20)
        self.assertEqual(len({t['output'] for t in tasks}), 20)
        for task in tasks:
            cmd = task['command']
            self.assertEqual(cmd[cmd.index('--objective') + 1], task['objective'])
            self.assertEqual(cmd[cmd.index('--expected-manifest-sha256') + 1], 'fixed-hash')
            self.assertEqual(cmd[cmd.index('--variants') + 1], 'prototype')

    def test_objective_summary_separates_baseline_and_adjacent_controls(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name, ic in zip(OBJECTIVES, [0.01, 0.02, 0.1, 0.11], strict=True):
                path = Path(tmp) / study_name(42, name)
                path.mkdir()
                metrics = {'group_rank_ic_mean': ic, 'endpoint_return_mae': 0.05, 'zero_return_mae': 0.04,
                           'endpoint_return_pinball': 0.1, 'endpoint_quantile_crossing_fraction': 0.05}
                data = {'seed': 42, 'objective': name, 'variant': 'prototype',
                        'milestones': [{'step': 2048, 'validation': metrics}]}
                (path / 'budget_prototype.json').write_text(json.dumps(data))
            result = summarize_objectives(Path(tmp), [42, 1], [2048], OBJECTIVES, ['prototype'])
            row = next(r for r in result['rows'] if r['objective'] == 'return_rank')
            self.assertEqual(row['n_seeds'], 1)
            self.assertEqual(row['expected_seeds'], 2)
            self.assertAlmostEqual(row['paired_delta_vs_price_path']['group_rank_ic_mean'], 0.1)
            self.assertAlmostEqual(row['paired_delta_vs_previous']['group_rank_ic_mean'], 0.01)
            self.assertTrue(result['test_not_evaluated'])
            self.assertFalse(result['adoption_ready'])

    def test_endpoint_error_uses_zero_return_as_a_separate_baseline(self):
        group = summarize_return_group([0.1, 0.2, 0.3], [0.0, 0.1, np.nan])
        self.assertAlmostEqual(group["endpoint_return_mae"], 0.1)
        self.assertAlmostEqual(group["zero_return_mae"], 0.05)
        self.assertEqual(group["n_labeled"], 2)


@unittest.skipUnless(HAS_TORCH, "budget trainer tested in isolated torch environment")
class FalconBudgetTrainingTests(unittest.TestCase):
    def test_budget_mode_never_reads_test_and_saves_every_fixed_stage(self):
        class GuardedSplits(dict):
            def __getitem__(self, key):
                if key == "test":
                    raise AssertionError("budget study must not inspect test")
                return super().__getitem__(key)

        torch.set_num_threads(2)
        dates = pd.bdate_range("2024-01-01", periods=60)
        panel = MarketPanel(np.random.default_rng(7).normal(4, 0.1, size=(60, 5, 2)).astype(np.float32),
                            np.ones((60, 5), dtype=bool), dates, tuple(str(i) for i in range(5)), ("log_close", "other"))
        splits = GuardedSplits(train=np.arange(10, 25), validation=np.arange(30, 35))
        args = SimpleNamespace(mode="budget", milestones=[2, 4], seed=42, device="cpu", precision="float32",
                               width=16, heads=4, temporal_layers=1, spatial_layers=1, prototypes=3, patch_size=4,
                               lr=0.001, steps=4, eval_every=2, eval_days=2, entities=3, context=4,
                               horizon=5, batch_dates=1, orthogonality_weight=0.01)
        for objective in OBJECTIVES:
            args.objective = objective
            with self.subTest(objective=objective), tempfile.TemporaryDirectory() as tmp:
                result = train_variant("prototype", panel, splits, args, Path(tmp), {"manifest_sha256": "test"})
                self.assertTrue(result["test_not_evaluated"])
                self.assertEqual(result["objective"], objective)
                self.assertEqual([row["step"] for row in result["milestones"]], [2, 4])
                self.assertTrue(all(len(row["validation"]["daily"]) == 5 for row in result["milestones"]))
                self.assertTrue((Path(tmp) / "prototype_step2.pt").exists())
                self.assertTrue((Path(tmp) / "prototype_step4.pt").exists())
                self.assertFalse((Path(tmp) / "prototype.pt").exists())
                first = torch.load(Path(tmp) / "prototype_step2.pt", weights_only=True)
                final = torch.load(Path(tmp) / "prototype_step4.pt", weights_only=True)
                self.assertEqual(first["trained_steps"], 2)
                self.assertEqual(final["trained_steps"], 4)
                self.assertTrue(any(not torch.equal(first["model_state"][k], final["model_state"][k]) for k in first["model_state"]))


if __name__ == "__main__":
    unittest.main()
