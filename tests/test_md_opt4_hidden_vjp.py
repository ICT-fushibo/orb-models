"""Dependency-light algebra checks; full CUDA/library tests live in outer tests."""
import unittest
from unittest.mock import patch
import torch

from md_benchmark.opt4_hidden_vjp import TunedInputVJP, signature, HiddenVJPRuntimeError
from orb_models.md_stages.opt4_hidden_linear_vjp import _HiddenLinear


class HiddenVJPContracts(unittest.TestCase):
    def test_explicit_vjp_gradcheck_uses_current_parameter(self):
        x = torch.randn(3,4,dtype=torch.float64,requires_grad=True)
        w = torch.randn(5,4,dtype=torch.float64,requires_grad=True)
        b = torch.randn(5,dtype=torch.float64,requires_grad=True)
        grad = torch.randn(3,5,dtype=torch.float64)
        tuner = TunedInputVJP(None,{})
        tuner.ready = True
        tuner.grad_signature = signature(grad)
        tuner.selection = {'backend':'aten-linear-packed','layout':'packed'}
        tuner.output = torch.empty_like(x)
        with patch('torch.cuda.is_current_stream_capturing',return_value=False):
            self.assertTrue(torch.autograd.gradcheck(
                lambda x,w,b: _HiddenLinear.apply(x,w,b,tuner),(x,w,b)))
            with self.assertRaises(HiddenVJPRuntimeError):
                tuner.execute(grad.T,w)
        tuner.reset()
        self.assertFalse(tuner.ready)
        self.assertIsNone(tuner.output)


if __name__ == '__main__': unittest.main()
