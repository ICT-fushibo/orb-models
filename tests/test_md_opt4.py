"""CPU contract checks for the ORBv3 Opt4 route."""

import pytest
import torch

from orb_models.md_stages import opt4
from orb_models.md_stages.opt4_fusion import _DualAggregatePack


def test_opt4_rejects_other_route() -> None:
    with pytest.raises(ValueError, match="ORBv3 Opt4 route"):
        opt4.run_md(type("Request", (), {"model": "dpa4", "stage": "opt4"})())


def test_attention_vjp_validator_handles_fp32_dot_reordering() -> None:
    rows = torch.tensor([0, 0, 1, 1])
    boundary = _DualAggregatePack(rows, rows=2)
    updated_edges = torch.full((4, 256), 0.1)
    receive = torch.ones(4, 1)
    send = torch.ones(4, 1)
    nodes = torch.zeros(2, 256)
    senders = torch.tensor([1, 0, 1, 0])
    probe = torch.cos(torch.arange(2 * 3 * 256, dtype=torch.float32)).reshape(
        2, 3 * 256
    )
    terms = updated_edges * probe[:, 256:512].index_select(0, rows)
    truth = terms.double().sum(dim=-1, keepdim=True).float()

    handled = boundary.validate_vjp(
        truth + 2.0e-6,
        truth,
        (updated_edges, receive, send, nodes, senders),
        1,
        [probe],
    )
    assert handled is True
