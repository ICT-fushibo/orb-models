"""CPU contract checks for the ORBv3 Opt4 route."""

import pytest
import torch

from orb_models.md_stages import opt4
from orb_models.md_stages.opt4_fusion import _EdgeGatherPackReference


def test_opt4_rejects_other_route() -> None:
    with pytest.raises(ValueError, match="ORBv3 Opt4 route"):
        opt4.run_md(type("Request", (), {"model": "dpa4", "stage": "opt4"})())


def test_edge_gather_pack_reference_and_node_vjp_validator() -> None:
    boundary = _EdgeGatherPackReference()
    edges = torch.randn(4, 8, requires_grad=True)
    nodes = torch.randn(3, 8, requires_grad=True)
    senders = torch.tensor([0, 0, 1, 2])
    receivers = torch.tensor([1, 2, 2, 0])
    got = boundary(edges, nodes, senders, receivers)
    torch.testing.assert_close(got[:, :8], edges)
    torch.testing.assert_close(got[:, 8:16], nodes.index_select(0, senders))
    torch.testing.assert_close(got[:, 16:], nodes.index_select(0, receivers))

    probe = torch.cos(torch.arange(got.numel(), dtype=got.dtype)).view_as(got)
    expected = torch.autograd.grad((got * probe).sum(), nodes)[0]
    handled = boundary.validate_vjp(
        expected,
        expected,
        (edges, nodes, senders, receivers),
        1,
        [probe],
    )
    assert handled is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_edge_gather_pack_triton_forward_and_complete_vjp(dtype) -> None:
    from orb_models.md_stages.opt4_gather_pack import FastEqEdgeGatherPack

    row_ptr = torch.tensor([0, 3, 5, 9], device="cuda", dtype=torch.int64)
    candidate = FastEqEdgeGatherPack(row_ptr, max_row=4).cuda()
    reference = _EdgeGatherPackReference().cuda()
    senders = torch.tensor(
        [0, 0, 3, 1, 1, 2, 2, 2, 4], device="cuda", dtype=torch.int64
    )
    receivers = torch.tensor(
        [1, 2, 3, 2, 0, 0, 1, 2, 4], device="cuda", dtype=torch.int64
    )
    edges_ref = torch.randn(8, 9, device="cuda", dtype=dtype).T.requires_grad_(True)
    nodes_ref = torch.randn(8, 35, device="cuda", dtype=dtype).T.requires_grad_(True)
    edges_got = edges_ref.detach().clone().requires_grad_(True)
    nodes_got = nodes_ref.detach().clone().requires_grad_(True)
    expected = reference(edges_ref, nodes_ref, senders, receivers)
    actual = candidate(edges_got, nodes_got, senders, receivers)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    probe = torch.cos(
        torch.arange(actual.numel(), device="cuda", dtype=dtype)
    ).view_as(actual)
    expected_grads = torch.autograd.grad(
        (expected * probe).sum(), (edges_ref, nodes_ref)
    )
    actual_grads = torch.autograd.grad(
        (actual * probe).sum(), (edges_got, nodes_got)
    )
    torch.testing.assert_close(actual_grads[0], expected_grads[0], rtol=0, atol=0)
    reference.validate_vjp(
        actual_grads[1],
        expected_grads[1],
        (edges_ref, nodes_ref, senders, receivers),
        1,
        [probe],
    )
