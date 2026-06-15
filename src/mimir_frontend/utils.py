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


Shape = Sequence[int | str | None]


def _make_driver(compile_phase: str) -> mim.Driver:
    driver = mim.Driver()
    plugins = ["math", "tensor"]
    if compile_phase == "default":
        plugins.extend(["compile", "opt"])
    driver.load_plugins(plugins)
    return driver


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
    name: str = "mimir_module",
    max_depth: int = 100,
) -> str:
    """Translate a torch FX model to a closed MimIR module function.

    `input_shapes` can contain `None` or `str` for dynamic/symbolic dimensions.
    Module parameters are automatically added as function arguments.
    """

    if compile_phase not in {"high_level", "default"}:
        raise ValueError("compile_phase must be 'high_level' or 'default'")

    driver = _make_driver(compile_phase)
    world = driver.world()
    
    # Pre-trace to identify parameters used in the graph
    traced = fx.symbolic_trace(model)
    graph = traced.graph
    
    # Identify parameters used in get_attr nodes
    param_nodes = [node for node in graph.nodes if node.op == "get_attr"]
    param_names = [node.target for node in param_nodes]
    
    translator = FXGraphTranslator(world, module=traced)
    ops = translator.ops
    
    # 1. Identify symbolic dimensions
    sym_names = []
    input_sym_names = [] # Store the symbol name for each dimension of each input
    for shape in input_shapes:
        this_input_syms = []
        for d in shape:
            if isinstance(d, str):
                if d not in sym_names:
                    sym_names.append(d)
                this_input_syms.append(d)
            elif d is None:
                gen_name = f"n{len(sym_names)}"
                sym_names.append(gen_name)
                this_input_syms.append(gen_name)
            else:
                this_input_syms.append(None)
        input_sym_names.append(this_input_syms)

    # 2. Construct Domain Type: [sym_dims..., tensor_inputs..., params...]
    nat_t = world.type_nat()
    dom_types = [nat_t] * len(sym_names)
    
    # Input Tensors
    tensor_input_types = []
    for shape in input_shapes:
        dims = [world.top_nat() if isinstance(d, (str, type(None))) else world.lit_nat(d) for d in shape]
        tensor_input_types.append(world.arr(world.tuple(dims), ops.F32))
        
    # Parameters
    param_types = []
    for target in param_names:
        attr = traced
        for part in target.split("."):
            attr = getattr(attr, part)
        p_shape = [world.lit_nat(d) for d in attr.shape]
        param_types.append(world.arr(world.tuple(p_shape), ops.F32))

    full_dom_types = dom_types + tensor_input_types + param_types
    
    # Store mapping in translator for use in _shape_dims
    translator.input_sym_names = input_sym_names
    
    # 3. Create the real Module Function
    result_lam = translator.translate_as_function(graph, full_dom_types, name=name, sym_names=sym_names)




    if compile_phase == "default":
        world.optimize()

    return _write_def_to_string(result_lam, name, max_depth)
