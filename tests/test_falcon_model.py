import importlib.util
import os
import unittest

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch

    from scripts.falcon_model import FalconConfig, FalconForecaster, quantile_loss


@unittest.skipUnless(HAS_TORCH, "PyTorch is available in the isolated GPU environment")
class FalconModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(123)
        self.x = torch.randn(2, 3, 4, 12)
        self.mask = torch.ones_like(self.x, dtype=torch.bool)

    def model(self, variant="prototype"):
        return FalconForecaster(FalconConfig(variant=variant, width=16, heads=4,
                                temporal_layers=1, spatial_layers=1, prototypes=3,
                                patch_size=4, quantiles=(0.1, 0.5, 0.9))).eval()

    def test_shapes_gradients_and_variable_horizon(self):
        for variant in ["temporal", "dense", "prototype"]:
            with self.subTest(variant=variant):
                model = self.model(variant)
                out = model(self.x, 5, self.mask)
                self.assertEqual(out["prediction"].shape, (2, 3, 4, 5, 3))
                self.assertTrue(torch.isfinite(out["prediction"]).all())
                target = torch.randn(2, 3, 4, 5)
                target[0, 0, 0, 0] = float("nan")
                loss = quantile_loss(out, target, model.quantiles) + 0.01 * model.orthogonality_loss()
                loss.backward()
                grads = [p.grad for p in model.parameters() if p.grad is not None]
                self.assertTrue(grads)
                self.assertTrue(all(torch.isfinite(g).all() for g in grads))
                self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0)
                self.assertEqual(model(self.x, 2)["prediction"].shape, (2, 3, 4, 2, 3))

    def test_observation_mask_ignores_arbitrary_missing_values(self):
        self.mask[..., ::3] = False
        changed = self.x.clone()
        changed[~self.mask] = 1e8
        for variant in ["temporal", "dense", "prototype"]:
            model = self.model(variant)
            a = model(self.x, 5, self.mask)["prediction"]
            b = model(changed, 5, self.mask)["prediction"]
            torch.testing.assert_close(a, b)

    def test_different_dates_in_batch_never_share_context(self):
        for variant in ["temporal", "dense", "prototype"]:
            model = self.model(variant)
            before = model(self.x, 5)["prediction"][0]
            changed = self.x.clone()
            changed[1] = torch.randn_like(changed[1]) * 20 + 100
            after = model(changed, 5)["prediction"][0]
            torch.testing.assert_close(before, after)

    def test_variate_and_entity_permutation_equivariance(self):
        v = torch.tensor([2, 0, 3, 1])
        e = torch.tensor([2, 0, 1])
        for variant in ["temporal", "dense", "prototype"]:
            model = self.model(variant)
            ref = model(self.x, 5)["prediction"]
            got = model(self.x[:, e][:, :, v], 5)["prediction"]
            torch.testing.assert_close(got, ref[:, e][:, :, v], atol=2e-5, rtol=2e-5)

    def test_masked_padding_entities_and_variates_do_not_change_predictions(self):
        for variant in ["temporal", "dense", "prototype"]:
            model = self.model(variant)
            ref = model(self.x, 5)["prediction"]
            expanded = torch.full((2, 4, 5, 12), float("nan"))
            expanded[:, :3, :4] = self.x
            out = model(expanded, 5)
            torch.testing.assert_close(out["prediction"][:, :3, :4], ref, atol=2e-5, rtol=2e-5)
            self.assertTrue(torch.isfinite(out["prediction"]).all())
            self.assertTrue((out["prediction"][:, 3] == 0).all())

    def test_constant_and_all_missing_context_are_finite(self):
        x = torch.full_like(self.x, 3.0)
        x[0] = float("nan")
        out = self.model()(x, 5)
        self.assertTrue(torch.isfinite(out["prediction"]).all())
        torch.testing.assert_close(out["location"][1], torch.full_like(out["location"][1], 3))
        self.assertTrue((out["prediction"][0] == 0).all())
        with self.assertRaises(ValueError):
            quantile_loss(out, torch.full((2, 3, 4, 5), float("nan")), self.model().quantiles)

    def test_positive_affine_equivariance_and_context_only_statistics(self):
        model = self.model()
        a = model(self.x, 5)
        b = model(3 * self.x + 7, 5)
        torch.testing.assert_close(b["prediction"], 3 * a["prediction"] + 7, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(a["location"], self.x.mean(dim=-1, keepdim=True))

    def test_negative_prototype_receives_gradient_and_changes_forecast(self):
        model = self.model()
        first = model(self.x, 5)
        target = torch.randn(2, 3, 4, 5)
        quantile_loss(first, target, model.quantiles).backward()
        self.assertGreater(float(model.mixer.negative_keys.grad.abs().sum()), 0)
        with torch.no_grad():
            model.mixer.negative_keys.add_(torch.randn_like(model.mixer.negative_keys) * 3)
        second = model(self.x, 5)
        self.assertFalse(torch.allclose(first["prediction"], second["prediction"]))

    def test_orthogonality_penalty_is_zero_for_orthogonal_spaces(self):
        model = self.model()
        with torch.no_grad():
            model.mixer.positive_keys.copy_(torch.eye(16)[:3])
            model.mixer.negative_keys.copy_(torch.eye(16)[3:6])
        self.assertEqual(float(model.orthogonality_loss().detach()), 0)
        with torch.no_grad():
            model.mixer.negative_keys.copy_(model.mixer.positive_keys)
        self.assertGreater(float(model.orthogonality_loss().detach()), 0)

    def test_small_optimizer_loop_reduces_training_loss(self):
        model = self.model("prototype").train()
        target = self.x[..., -1:].expand(-1, -1, -1, 3).clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        losses = []
        for _ in range(25):
            optimizer.zero_grad()
            loss = quantile_loss(model(self.x, 3), target, model.quantiles)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            losses.append(float(loss.detach()))
        self.assertLess(losses[-1], losses[0])

    def test_invalid_shapes_and_configs_are_rejected(self):
        with self.assertRaises(ValueError):
            FalconConfig(width=17, heads=4)
        with self.assertRaises(ValueError):
            FalconConfig(quantiles=(0.9, 0.1))
        with self.assertRaises(ValueError):
            self.model()(self.x[0], 5)
        with self.assertRaises(ValueError):
            self.model()(self.x, 0)
        with self.assertRaises(ValueError):
            self.model()(self.x, 5, self.mask[..., :-1])

    @unittest.skipUnless(os.environ.get("FALCON_TEST_CUDA") == "1", "CUDA check is opt-in")
    def test_cuda_forward_backward(self):
        self.assertTrue(torch.cuda.is_available())
        for variant in ["temporal", "dense", "prototype"]:
            model = self.model(variant).to("cuda")
            x = self.x.to("cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(x, 5)
                loss = quantile_loss(out, torch.randn(2, 3, 4, 5, device="cuda"), model.quantiles)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(out["prediction"]).all())


if __name__ == "__main__":
    unittest.main()
