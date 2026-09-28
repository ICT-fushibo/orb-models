"""Standalone packed native VJP contracts; CUDA integration lives in outer tests."""
import unittest
from unittest.mock import patch
import torch
from torch.nn import functional as F
from orb_models.md_stages.opt4_edge_packed_vjp import (
    FrozenWeightPack, _PackedLinear, packed_input_vjp,
)


class PackedVJPContracts(unittest.TestCase):
    def test_linear_gradcheck_and_unchanged_forward(self):
        x = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
        w = torch.randn(5, 4, dtype=torch.float64, requires_grad=True)
        b = torch.randn(5, dtype=torch.float64, requires_grad=True)
        pack = FrozenWeightPack()
        fn = lambda x, w, b: _PackedLinear.apply(x, w, b, pack(w)[0])
        torch.testing.assert_close(fn(x, w, b), F.linear(x, w, b), rtol=0, atol=0)
        self.assertTrue(torch.autograd.gradcheck(fn, (x, w, b)))

    def test_packed_weights_are_nonpersistent_and_setup_only(self):
        pack, w = FrozenWeightPack(2), torch.randn(4, 6)
        first = pack(w)
        with patch.object(pack, '_pack', side_effect=AssertionError('runtime packing')):
            self.assertIs(pack(w)[0], first[0])
        self.assertFalse(pack.state_dict())
        self.assertTrue(torch.equal(first[0], w.T))
        self.assertTrue(torch.equal(first[1], w[:, :2].T))

    def test_edge_only_and_native_scatter_chain(self):
        n, e = torch.zeros(3, 2, dtype=torch.float64), torch.randn(5, 2, dtype=torch.float64)
        w, dz = torch.randn(4, 6, dtype=torch.float64), torch.randn(5, 4, dtype=torch.float64)
        s, r = torch.tensor([0, 0, 1, 2, 2]), torch.tensor([1, 1, 0, 2, 2])
        full, edge = FrozenWeightPack(2)(w)
        features = dz @ w
        expected = torch.zeros_like(n).index_put_((s,), features[:, 2:4], accumulate=True)
        expected = expected + torch.zeros_like(n).index_put_((r,), features[:, 4:], accumulate=True)
        for need_nodes in (False, True):
            got = packed_input_vjp(n, e, s, r, w, dz, (need_nodes, True, False, False, False, False), full, edge)
            torch.testing.assert_close(got[1], features[:, :2], rtol=1e-12, atol=1e-12)
            if need_nodes: torch.testing.assert_close(got[0], expected, rtol=1e-12, atol=1e-12)
            else: self.assertIsNone(got[0])


if __name__ == '__main__': unittest.main()
