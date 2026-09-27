"""Small standalone contracts for the Opt4 edge-first-linear candidate."""
import unittest
from unittest.mock import patch
import torch
from orb_models.md_stages.opt4_edge_preproject import (
    EdgeLinearReference, projected_preactivation, linear_vjp, validate_inputs,
)


class EdgePreprojectionContracts(unittest.TestCase):
    def test_parameter_chain_reference_is_independent_of_custom_vjp(self):
        from orb_models.md_stages.opt4_preproject_validation import parameter_chain_reference
        n, e, s = torch.zeros(3, 2), torch.zeros(5, 2), torch.zeros(5, dtype=torch.long)
        args = n, e, s, s, torch.zeros(4, 6), torch.zeros(4)
        probe = torch.ones(5, 4)
        with patch("orb_models.md_stages.opt4_edge_preproject.linear_vjp", side_effect=AssertionError("reused VJP")):
            weight = parameter_chain_reference(args, probe, 2, split_forward=True)
            bias = parameter_chain_reference(args, probe, 3, split_forward=False)
        torch.testing.assert_close(weight, torch.zeros(4, 6), rtol=0, atol=0)
        torch.testing.assert_close(bias, torch.full((4,), 2.5), rtol=0, atol=0)

    def test_setup_audit_checks_native_and_candidate(self):
        from orb_models.md_stages.opt4_preproject_validation import validate_preproject_output
        n, e, s = torch.zeros(3, 2), torch.zeros(5, 2), torch.zeros(5, dtype=torch.long)
        args = n, e, s, s, torch.zeros(4, 6), torch.zeros(4)
        ref = EdgeLinearReference()(*args)
        self.assertEqual(validate_preproject_output(ref, ref, args)["status"], "passed")
        with self.assertRaises(AssertionError):
            validate_preproject_output(ref + .01, ref + .01, args)

    def test_linear_chain_rule_and_duplicate_neighbors(self):
        torch.manual_seed(26)
        nodes = torch.randn(3, 2, dtype=torch.float64, requires_grad=True)
        edges = torch.randn(5, 2, dtype=torch.float64, requires_grad=True)
        weight = torch.randn(4, 6, dtype=torch.float64, requires_grad=True)
        bias = torch.randn(4, dtype=torch.float64, requires_grad=True)
        s, r = torch.tensor([0, 0, 1, 2, 2]), torch.tensor([1, 1, 0, 2, 2])
        args = nodes, edges, s, r, weight, bias
        validate_inputs(*args)
        z = projected_preactivation(*args)
        torch.testing.assert_close(torch.nn.functional.silu(z), EdgeLinearReference()(*args), rtol=1e-12, atol=1e-12)
        probe = torch.randn_like(z)
        native = torch.autograd.grad(z, (nodes, edges, weight, bias), probe)
        got = linear_vjp(nodes, edges, s, r, weight, probe, (True, True, False, False, True, True))
        for a, b in zip((got[0], got[1], got[4], got[5]), native):
            torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)

    def test_reject_wrong_width_and_dtype(self):
        n, e, s = torch.zeros(3, 2), torch.zeros(5, 2), torch.zeros(5, dtype=torch.long)
        with self.assertRaisesRegex(ValueError, "3 \\* channels"):
            validate_inputs(n, e, s, s, torch.zeros(4, 5), torch.zeros(4))
        with self.assertRaisesRegex(ValueError, "dtype"):
            validate_inputs(n, e.double(), s, s, torch.zeros(4, 6), torch.zeros(4))

    def test_input_vjp_scatter_occurs_after_full_width_gemm(self):
        nodes, edges = torch.zeros(3, 2), torch.zeros(5, 2)
        s, r = torch.tensor([0, 0, 1, 2, 2]), torch.tensor([1, 1, 0, 2, 2])
        weight, g = torch.randn(4, 6), torch.randn(5, 4)
        expected = g @ weight
        seen = []
        native = torch.Tensor.index_put_
        def scatter(out, indices, values, *, accumulate):
            seen.append((tuple(out.shape), values.detach().clone(), accumulate))
            return native(out, indices, values, accumulate=accumulate)
        with patch.object(torch.Tensor, "index_put_", scatter), \
             patch.object(torch.Tensor, "index_add_", side_effect=AssertionError("old hidden-width reduction")):
            result = linear_vjp(nodes, edges, s, r, weight, g, (True, True, False, False, False, False))
        self.assertEqual([entry[0] for entry in seen], [(3, 2), (3, 2)])
        self.assertTrue(all(entry[2] for entry in seen))
        torch.testing.assert_close(seen[0][1], expected[:, 2:4], rtol=0, atol=0)
        torch.testing.assert_close(seen[1][1], expected[:, 4:], rtol=0, atol=0)
        torch.testing.assert_close(result[1], expected[:, :2], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
