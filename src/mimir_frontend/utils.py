from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path
import os
import tempfile

import mim
import torch
from torch import fx

from .translator import FXGraphTranslator


Shape = Sequence[int | None]


def _make_driver(compile_phase: str) -> mim.Driver:
    driver = mim.Driver()
    plugins = ["math", "tensor"]
    if compile_phase == "default":
        plugins.extend(["compile", "opt"])
    driver.load_plugins(plugins)
    return driver


def _make_f32_tensor_type(world: mim.World, elem_type: mim.Def, shape: Shape) -> mim.Def:
    if not shape:
        return elem_type
    if len(shape) == 1 and shape[0] is None:
        return world.arr(world.top_nat(), elem_type)

    dims = [world.top_nat() if dim is None else world.lit_nat(dim) for dim in shape]
    return world.arr(world.tuple(dims), elem_type)


def _translate_with_inputs(
    world: mim.World,
    graph: fx.Graph,
    input_types: Sequence[mim.Def],
) -> tuple[mim.Def, list[mim.Def]]:
    input_lams = [world.mut_con(input_type).set(f"arg{i}") for i, input_type in enumerate(input_types)]
    inputs = [lam.var() for lam in input_lams]
    result = FXGraphTranslator(world).translate(graph, inputs)
    return result, inputs


@contextmanager
def _temporary_cwd():
    old_cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp_dir:
        os.chdir(tmp_dir)
        try:
            yield Path(tmp_dir)
        finally:
            os.chdir(old_cwd)


def _write_def_to_string(defn: mim.Def, name: str, max_depth: int) -> str:
    with _temporary_cwd() as tmp_dir:
        path = tmp_dir / f"{name}.mim"
        defn.write(max_depth, str(path))
        return path.read_text()


def model_to_mimir(
    model: torch.nn.Module,
    input_shapes: Sequence[Shape],
    *,
    compile_phase: str = "high_level",
    name: str = "model",
    max_depth: int = 100,
) -> str:
    """Translate a torch FX model to textual MimIR.

    `compile_phase="high_level"` preserves tensor-level IR such as
    `%tensor.binary` and `%tensor.unary`.

    `compile_phase="default"` loads MimIR's compile/opt plugins with the other
    plugins in one batch, avoiding the incremental plugin-loader crash. The
    utility still dumps the translated expression directly; running
    `world.optimize()` on this free expression is unsafe until the importer can
    emit a closed extern entry function.
    """

    if compile_phase not in {"high_level", "default"}:
        raise ValueError("compile_phase must be 'high_level' or 'default'")

    driver = _make_driver(compile_phase)
    world = driver.world()
    ops = FXGraphTranslator(world).ops
    input_types = [_make_f32_tensor_type(world, ops.F32, shape) for shape in input_shapes]
    graph = fx.symbolic_trace(model).graph

    result, _ = _translate_with_inputs(world, graph, input_types)
    result.set(name)

    return _write_def_to_string(result, name, max_depth)
