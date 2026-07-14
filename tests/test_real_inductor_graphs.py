import re

import pytest

from mimir_frontend.inductor_readable import translate_inductor_readable


@pytest.mark.parametrize(
    "case_name,expected_frontier",
    [
        ("faster_rcnn_1", "full with dtype torch.int64"),
        ("gcn_1", "aten.scatter_add"),
        ("moe_1", "torch.max with dim"),
    ],
)
def test_real_inductor_graph_frontier(case_name, expected_frontier):
    with pytest.raises(NotImplementedError, match=re.escape(expected_frontier)):
        translate_inductor_readable(case_name)


def test_real_inductor_mlp_forward_translates_after_addmm_support():
    assert translate_inductor_readable("mlp_1") is not None


def test_real_inductor_lstm_forward_translates_after_split_support():
    assert translate_inductor_readable("lstm_1") is not None
