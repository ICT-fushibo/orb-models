"""Dependency-light algebra/cache contracts; full CUDA tests are outer tests."""
import unittest
from unittest.mock import patch
import torch
from orb_models.md_stages.opt4_first_edge_vjp import EdgeWeightPack, first_edge_input_vjp
from orb_models.md_stages.opt4_edge_preproject import linear_vjp


class FirstEdgeContracts(unittest.TestCase):
    def test_layouts_match_native_all_masks_and_strides(self):
        for count in (0, 1, 17):
            n = torch.randn(4, 6, dtype=torch.float64)[:, ::2]
            e = torch.randn(count, 6, dtype=torch.float64)[:, ::2]
            w = torch.randn(5, 18, dtype=torch.float64)[:, ::2]
            dz = torch.randn(count, 10, dtype=torch.float64)[:, ::2]
            s, r = torch.arange(count) % 4, (torch.arange(count) * 2) % 4
            packed = EdgeWeightPack(3)(w)
            for nn, ne, nw, nb in ((False, True, False, False), (True, True, True, True),
                                   (True, False, False, False), (False, False, True, True)):
                needs = (nn, ne, False, False, nw, nb)
                want = linear_vjp(n, e, s, r, w, dz, needs)
                for layout in (None, packed):
                    got = first_edge_input_vjp(n, e, s, r, w, dz, needs, layout)
                    for a, b in zip(got, want):
                        if b is None: self.assertIsNone(a)
                        else: torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)

    def test_only_edge_pack_and_checkpoint_unmodified(self):
        w = torch.randn(1024, 768)
        cache = EdgeWeightPack(256)
        result = cache(w)
        self.assertEqual(result.numel() * result.element_size(), 1024 * 1024)
        self.assertEqual(list(dict(cache.named_buffers())), ['edge'])
        self.assertEqual(cache.state_dict(), {})
        with patch.object(cache, '_pack', side_effect=AssertionError('repack')):
            self.assertIs(cache(w), result)
        with torch.no_grad(): w.add_(1)
        torch.testing.assert_close(cache(w), w[:, :256].T, rtol=0, atol=0)
        self.assertEqual(cache.builds, 2)

    def test_parameter_perturbation_and_capture_cache(self):
        w = torch.randn(3, 6, dtype=torch.float64, requires_grad=True)
        cache = EdgeWeightPack(2)
        old = cache(w)
        w.data.add_(1)
        self.assertFalse(torch.equal(cache(w), old))
        target = 'orb_models.md_stages.opt4_first_edge_vjp.capturing'
        with patch(target, return_value=True):
            with self.assertRaisesRegex(RuntimeError, 'audits'): cache(w)
            with self.assertRaisesRegex(RuntimeError, 'not prepared'): cache(w.detach())
        frozen = w.detach()
        cached = cache(frozen)
        with patch(target, return_value=True): self.assertIs(cache(frozen), cached)


if __name__ == '__main__': unittest.main()
