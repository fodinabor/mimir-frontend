import torch

from mimir_frontend.utils import model_to_mimir


class AddReluModel(torch.nn.Module):
    def forward(self, x, y):
        return torch.relu(x + y)


def test_model_to_mimir_outputs_high_level_tensor_ir():
    ir = model_to_mimir(
        AddReluModel(),
        input_shapes=[(None,), (None,)],
        compile_phase="high_level",
    )

    assert "%tensor.binary" in ir
    assert "%tensor.unary" in ir


def test_model_to_mimir_can_use_default_compile_phase():
    ir = model_to_mimir(
        AddReluModel(),
        input_shapes=[(None,), (None,)],
        compile_phase="default",
    )

    assert isinstance(ir, str)
    assert len(ir) > 0


def test_model_to_mimir_rejects_unknown_compile_phase():
    try:
        model_to_mimir(AddReluModel(), input_shapes=[(None,), (None,)], compile_phase="unknown")
    except ValueError as exc:
        assert "compile_phase" in str(exc)
    else:
        raise AssertionError("expected ValueError")
