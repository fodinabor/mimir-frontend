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


class ShapeEnv:
    def __init__(self, world: mim.World):
        self.world = world
        self._symbol_defs: dict[str, mim.Def] = {}
        self.symbol_names: list[str] = []

    def _register_symbol(self, name: str) -> str:
        if name not in self._symbol_defs:
            self.symbol_names.append(name)
            self._symbol_defs[name] = self.world.mut_con(self.world.type_nat()).var()
        return name

    def normalize_dim(self, dim) -> int | str:
        if isinstance(dim, int):
            return dim
        if dim is None:
            return self._register_symbol(f"n{len(self.symbol_names)}")

        text = str(dim)
        if text.isdecimal():
            return int(text)
        return self._register_symbol(text)

    def normalize_shape(self, shape: Shape) -> tuple[int | str, ...]:
        return tuple(self.normalize_dim(dim) for dim in shape)

    def symbol_def(self, name: str) -> mim.Def:
        return self._symbol_defs[self._register_symbol(name)]


def _shape_from_meta_value(value) -> tuple[int | str, ...]:
    if not hasattr(value, "shape"):
        raise TypeError(f"meta['val'] does not have shape: {type(value)}")
    dims = []
    for dim in value.shape:
        dims.append(dim if isinstance(dim, int) else str(dim))
    return tuple(dims)


def shape_to_mimir_dims(
    world: mim.World,
    shape: Shape,
    *,
    shape_env: ShapeEnv | None = None,
    symbolic: bool = False,
) -> list[mim.Def]:
    dims = []
    for dim in shape:
        if isinstance(dim, int):
            dims.append(world.lit_nat(dim))
        elif isinstance(dim, str):
            if dim.isdecimal():
                dims.append(world.lit_nat(int(dim)))
            elif symbolic:
                if shape_env is None:
                    raise ValueError("shape_env is required when symbolic=True")
                dims.append(shape_env.symbol_def(dim))
            else:
                dims.append(world.top_nat())
        else:
            dims.append(world.top_nat())
    return dims


def tensor_type_from_shape(
    world: mim.World,
    elem_type: mim.Def,
    shape: Shape,
    *,
    shape_env: ShapeEnv | None = None,
    symbolic: bool = False,
) -> mim.Def:
    dims = shape_to_mimir_dims(world, shape, shape_env=shape_env, symbolic=symbolic)
    if not dims:
        return elem_type
    if len(dims) == 1:
        return world.arr(dims[0], elem_type)
    return world.arr(world.tuple(dims), elem_type)


def _infer_input_shapes_from_placeholders(graph: fx.Graph) -> list[Shape]:
    shapes = []
    for node in graph.nodes:
        if node.op != "placeholder":
            continue
        if "val" not in node.meta:
            raise ValueError(f"placeholder {node.name} is missing meta['val']; provide input_shapes explicitly")
        shapes.append(_shape_from_meta_value(node.meta["val"]))
    return shapes


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
    model: torch.nn.Module | fx.GraphModule,
    input_shapes: Sequence[Shape] | None,
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
    
    if isinstance(model, fx.GraphModule):
        traced = model
    else:
        traced = fx.symbolic_trace(model)
    graph = traced.graph

    if input_shapes is None:
        input_shapes = _infer_input_shapes_from_placeholders(graph)
    
    # Identify parameters used in get_attr nodes
    param_nodes = [node for node in graph.nodes if node.op == "get_attr"]
    param_names = [node.target for node in param_nodes]
    
    translator = FXGraphTranslator(world, module=traced)
    ops = translator.ops
    
    shape_env = ShapeEnv(world)
    normalized_input_shapes = [shape_env.normalize_shape(shape) for shape in input_shapes]
    sym_names = list(shape_env.symbol_names)
    input_sym_names = [
        [dim if isinstance(dim, str) else None for dim in shape]
        for shape in normalized_input_shapes
    ]

    # 2. Construct Domain Type: [sym_dims..., tensor_inputs..., params...]
    # In MimIR, the function domain is represented as a Sigma (tuple) type.
    # For example, if a model takes a tensor with a dynamic batch size 'n': (n, 20)
    # The generated MimIR function signature will be:
    # `lam extern mimir_module (n: Nat, arg0: «n, 20; F32», weight: «256, 20; F32»)`
    nat_t = world.type_nat()
    dom_types = [nat_t] * len(sym_names)
    
    # Input Tensors
    tensor_input_types = []
    for shape in normalized_input_shapes:
        tensor_input_types.append(tensor_type_from_shape(world, ops.F32, shape))
        
    # Parameters
    # FX weights/biases (extracted from get_attr) are passed as trailing arguments
    param_types = []
    for target in param_names:
        attr = traced
        for part in target.split("."):
            attr = getattr(attr, part)
        param_types.append(tensor_type_from_shape(world, ops.F32, attr.shape))

    full_dom_types = dom_types + tensor_input_types + param_types
    
    # Store mapping in translator for use in _shape_dims
    translator.input_sym_names = input_sym_names
    
    # 3. Create the real Module Function
    # Translate the entire FX graph into a closed `lam extern`
    result_lam = translator.translate_as_function(graph, full_dom_types, name=name, sym_names=sym_names)





    if compile_phase == "default":
        world.optimize()

    return _write_def_to_string(result_lam, name, max_depth)
