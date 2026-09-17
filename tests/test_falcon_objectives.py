import importlib.util
import os
import unittest

from scripts.falcon_objectives import OBJECTIVES, endpoint_returns, pairwise_rank_loss, training_objective

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch

    from scripts.falcon_model import FalconConfig, FalconForecaster, quantile_loss


class ObjectiveNamesTests(unittest.TestCase):
    def test_four_separate_controls(self):
        self.assertEqual(OBJECTIVES, ("price_path", "price_endpoint", "return_endpoint", "return_rank"))


@unittest.skipUnless(HAS_TORCH, "objective tensor tests run in isolated torch environment")
class FalconObjectiveTests(unittest.TestCase):
    def fixture(self):
        context = torch.full((2, 4, 2, 4), 4.)
        prediction = (4 + torch.linspace(-0.04, 0.05, 2 * 4 * 2 * 5 * 3).reshape(2, 4, 2, 5, 3)).requires_grad_()
        location = torch.full((2, 4, 2, 1), 4.)
        scale = torch.full_like(location, 0.1)
        targets = 4 + torch.linspace(-0.02, 0.07, 2 * 4 * 2 * 5).reshape(2, 4, 2, 5)
        mask = torch.zeros_like(targets, dtype=torch.bool)
        mask[:, :, 0] = True
        output = {"prediction": prediction, "normalized_prediction": torch.asinh((prediction - location[..., None]) / scale[..., None]),
                  "location": location, "scale": scale, "active": torch.ones((2, 4, 2), dtype=torch.bool)}
        return output, context, targets, mask, torch.tensor([0.1, 0.5, 0.9])

    def test_path_dispatch_is_exact_legacy_loss(self):
        out, ctx, target, mask, q = self.fixture()
        actual = training_objective(out, target, q, mask, ctx, "price_path")
        torch.testing.assert_close(actual, quantile_loss(out, target, q, mask), rtol=0, atol=0)

    def test_endpoint_objectives_ignore_earlier_and_other_channel_targets(self):
        out, ctx, target, mask, q = self.fixture()
        changed = target.clone()
        changed[..., :-1] = 99
        changed[:, :, 1] = -99
        for name in OBJECTIVES[1:]:
            with self.subTest(name=name):
                torch.testing.assert_close(training_objective(out, target, q, mask, ctx, name),
                                           training_objective(out, changed, q, mask, ctx, name))

    def test_missing_endpoints_are_masked_before_exp_and_have_zero_direct_gradient(self):
        out, ctx, target, mask, q = self.fixture()
        target[0, 1, 0, -1] = torch.nan
        loss = training_objective(out, target, q, mask, ctx, "return_rank")
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(out["prediction"].grad).all())
        torch.testing.assert_close(out["prediction"].grad[0, 1, 0, -1], torch.zeros(3))

    def test_all_missing_endpoint_targets_are_rejected(self):
        out, ctx, target, mask, q = self.fixture()
        target[:, :, 0, -1] = torch.nan
        for name in OBJECTIVES[1:]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                training_objective(out, target, q, mask, ctx, name)

    def test_return_units_are_invariant_to_price_level_rescaling(self):
        out, ctx, target, mask, q = self.fixture()
        changed = dict(out, prediction=out["prediction"] + 2)
        a = training_objective(out, target, q, mask, ctx, "return_rank")
        b = training_objective(changed, target + 2, q, mask, ctx + 2, "return_rank")
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
        predicted, actual, valid = endpoint_returns(out, target, ctx, mask)
        torch.testing.assert_close(actual, torch.expm1(target[:, :, 0, -1] - ctx[:, :, 0, -1]))
        self.assertEqual(predicted.shape, (2, 4, 3))
        self.assertTrue(valid.all())

    def test_pairs_never_cross_dates_or_include_ties(self):
        scores = torch.tensor([[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]], requires_grad=True)
        target = torch.tensor([[0., 0., 0.], [1., 1., 1.]])
        loss = pairwise_rank_loss(scores, target, torch.ones_like(target, dtype=torch.bool))
        self.assertEqual(float(loss.detach()), 0)
        loss.backward()
        torch.testing.assert_close(scores.grad, torch.zeros_like(scores))

    def test_correct_order_has_lower_pair_loss_and_invalid_labels_do_not_participate(self):
        target = torch.tensor([[0., 0.01, 0.02]])
        valid = torch.ones_like(target, dtype=torch.bool)
        good = pairwise_rank_loss(target, target, valid)
        bad = pairwise_rank_loss(-target, target, valid)
        self.assertLess(float(good), float(bad))
        scores = torch.tensor([[0., 0.01, float("nan")]], requires_grad=True)
        masked = torch.tensor([[True, True, False]])
        loss = pairwise_rank_loss(scores, torch.tensor([[0., 0.01, float("nan")]]), masked)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(scores.grad[0, 2]), 0)

    def test_cuda_amp_forward_backward_for_all_objectives(self):
        if os.environ.get("FALCON_TEST_CUDA") != "1" or not torch.cuda.is_available():
            self.skipTest("CUDA test requires explicit isolated GPU opt-in")
        torch.manual_seed(42)
        model = FalconForecaster(FalconConfig(width=16, heads=4, temporal_layers=1, spatial_layers=1,
                                              prototypes=3, patch_size=4)).cuda()
        context = torch.randn(2, 4, 2, 8, device="cuda") * 0.02 + 3
        targets = torch.randn(2, 4, 2, 5, device="cuda") * 0.02 + 3
        mask = torch.zeros_like(targets, dtype=torch.bool)
        mask[:, :, 0] = True
        for name in OBJECTIVES:
            model.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(context, 5)
                loss = training_objective(output, targets, model.quantiles, mask, context, name)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_endpoint_return_coordinate_is_a_genuine_loss_change(self):
        out, ctx, target, mask, q = self.fixture()
        price = training_objective(out, target, q, mask, ctx, "price_endpoint")
        returns = training_objective(out, target, q, mask, ctx, "return_endpoint")
        self.assertGreater(abs(float((price - returns).detach())), 1e-3)


if __name__ == "__main__":
    unittest.main()
