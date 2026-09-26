"""Small standalone contracts for the Opt4 edge-first-linear candidate."""
import unittest
import torch
from orb_models.md_stages.opt4_edge_preproject import (
    EdgeLinearReference, projected_preactivation, linear_vjp, validate_inputs,
)


class EdgePreprojectionContracts(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
