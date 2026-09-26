"""CPU contracts for the current ORB RMSNorm Opt4 route (no retired passes)."""
import unittest
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from md_benchmark.opt4_registry import FusionSetupError
from orb_models import md_stages
from orb_models.md_stages import opt4_rmsnorm_native as native


class NativeRMSNormContractTests(unittest.TestCase):
    def test_opt4_rejects_other_route(self):
        # This is a route contract, not a model/neighbour-backend import test.
        # Load an isolated module with only its delegated runner stubbed.
        stub = SimpleNamespace(run_md=Mock(side_effect=AssertionError("wrong route delegated")))
        spec = importlib.util.spec_from_file_location(
            "orb_models.md_stages._route_contract", Path(native.__file__).with_name("opt4.py")
        )
        opt4 = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"orb_models.md_stages.opt3": stub}), \
             patch.object(md_stages, "opt3", stub, create=True):
            spec.loader.exec_module(opt4)
        with self.assertRaisesRegex(ValueError, "ORBv3 Opt4 route"):
            opt4.run_md(SimpleNamespace(model="dpa4", stage="opt4"))
        stub.run_md.assert_not_called()

    def test_native_forward_keeps_saved_statistic_and_weight_only_mask(self):
        value, weight = torch.randn(3, 4), torch.randn(4)
        output, rstd, upstream, dweight = object(), object(), object(), object()
        fwd = Mock(return_value=(output, rstd))
        bwd = Mock(return_value=(None, dweight))
        aten = SimpleNamespace(
            _fused_rms_norm=SimpleNamespace(default=fwd),
            _fused_rms_norm_backward=SimpleNamespace(default=bwd),
        )
        with patch.object(native.torch.ops, "aten", aten):
            native.require_native_rmsnorm_ops()
            self.assertEqual(native.native_rmsnorm_forward(value, weight, 1e-5), (output, rstd))
            self.assertIs(native.native_rmsnorm_weight_vjp(upstream, value, rstd, weight), dweight)
        fwd.assert_called_once_with(value, [4], weight, 1e-5)
        bwd.assert_called_once_with(upstream, value, [4], rstd, weight, [False, True])

    def test_missing_native_ops_is_typed_setup_error_not_fallback(self):
        with patch.object(native.torch.ops, "aten", SimpleNamespace()):
            with self.assertRaisesRegex(FusionSetupError, "_fused_rms_norm"):
                native.require_native_rmsnorm_ops()


if __name__ == "__main__":
    unittest.main()
