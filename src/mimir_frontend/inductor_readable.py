from __future__ import annotations

from dataclasses import dataclass
import inspect
import re
from pathlib import Path

import mim
import torch
from torch import fx

from .translator import FXGraphTranslator


DEFAULT_INDUCTOR_LOG_ROOT = Path("/Users/zc/courses/compiler/pytorch-play/logs/attn_debug/inductor")


@dataclass
class PartialTranslation:
    result: mim.Def | None
    frontier_node: fx.Node | None
    error: Exception | None


def make_world() -> mim.World:
    driver = mim.Driver()
    driver.load_plugins(["math", "tensor", "affine"])
    return driver.world()


def load_inductor_graph_module(case_or_path: str | Path, root: Path = DEFAULT_INDUCTOR_LOG_ROOT) -> fx.GraphModule:
    path = Path(case_or_path)
    if not path.exists():
        path = root / str(case_or_path) / "fx_graph_readable.py"
    namespace = {"torch": torch, "device": torch.device}
    exec(path.read_text(), namespace)
    return fx.symbolic_trace(namespace["GraphModule"]())


def parse_annotation(annotation: str):
    annotation = str(annotation).strip()
    if (
        len(annotation) >= 2
        and annotation[0] == annotation[-1]
        and annotation[0] in {"'", '"'}
    ):
        annotation = annotation[1:-1]

    if annotation.startswith("Sym("):
        return "sym", annotation[4:-1]

    match = re.fullmatch(r"([a-z0-9]+)\[(.*)\]", annotation)
    if not match:
        raise ValueError(f"unsupported annotation: {annotation}")

    dtype = match.group(1)
    shape_text = match.group(2).strip()
    dims = [] if not shape_text else [part.strip() for part in shape_text.split(",")]
    return "tensor", dtype, dims


def make_mimir_inputs_from_annotations(world: mim.World, graph_module: fx.GraphModule) -> list[mim.Def]:
    ops = FXGraphTranslator(world).ops
    inputs = []

    for parameter in inspect.signature(graph_module.forward).parameters.values():
        parsed = parse_annotation(parameter.annotation)
        if parsed[0] == "sym":
            inputs.append(world.mut_con(world.type_nat()).var())
            continue

        _, dtype, dims = parsed
        elem_type = world.type_bool() if dtype.startswith("b") else ops.F32
        mim_dims = [world.lit_nat(int(dim)) if dim.isdecimal() else world.top_nat() for dim in dims]

        if not mim_dims:
            tensor_type = elem_type
        elif len(mim_dims) == 1:
            tensor_type = world.arr(mim_dims[0], elem_type)
        else:
            tensor_type = world.arr(world.tuple(mim_dims), elem_type)
        inputs.append(world.mut_con(tensor_type).var())

    for node in graph_module.graph.nodes:
        if node.op != "get_attr":
            continue
        attr = graph_module
        for part in node.target.split("."):
            attr = getattr(attr, part)
        if not isinstance(attr, torch.Tensor):
            raise NotImplementedError(f"get_attr {node.target} with value type {type(attr)} is not supported")
        if attr.dtype == torch.bool:
            elem_type = world.type_bool()
        else:
            elem_type = ops.F32
        mim_dims = [world.lit_nat(dim) for dim in attr.shape]
        if not mim_dims:
            tensor_type = elem_type
        elif len(mim_dims) == 1:
            tensor_type = world.arr(mim_dims[0], elem_type)
        else:
            tensor_type = world.arr(world.tuple(mim_dims), elem_type)
        inputs.append(world.mut_con(tensor_type).var())

    return inputs


def translate_inductor_readable(case_or_path: str | Path, root: Path = DEFAULT_INDUCTOR_LOG_ROOT) -> mim.Def:
    graph_module = load_inductor_graph_module(case_or_path, root=root)
    world = make_world()
    translator = FXGraphTranslator(world, module=graph_module)
    inputs = make_mimir_inputs_from_annotations(world, graph_module)
    return translator.translate(graph_module.graph, inputs)


def translate_inductor_readable_prefix(
    case_or_path: str | Path,
    root: Path = DEFAULT_INDUCTOR_LOG_ROOT,
) -> PartialTranslation:
    graph_module = load_inductor_graph_module(case_or_path, root=root)
    world = make_world()
    translator = FXGraphTranslator(world, module=graph_module)
    inputs = make_mimir_inputs_from_annotations(world, graph_module)
    graph = graph_module.graph

    translator.env = {}
    placeholders = [node for node in graph.nodes if node.op == "placeholder"]
    param_nodes = [node for node in graph.nodes if node.op == "get_attr"]
    translator.ops.input_to_syms = {}

    for node, arg in zip(placeholders, inputs[: len(placeholders)]):
        translator.env[node] = arg
    for node, arg in zip(param_nodes, inputs[len(placeholders) :]):
        translator.env[node] = arg

    last_result = None
    for node in graph.nodes:
        if node.op in ("placeholder", "get_attr"):
            continue
        if node.op in ("call_function", "call_method"):
            try:
                last_result = translator.convert_node(node)
                translator.env[node] = last_result
            except Exception as exc:
                return PartialTranslation(last_result, node, exc)
        elif node.op == "output":
            return PartialTranslation(last_result, None, None)
        else:
            return PartialTranslation(last_result, node, NotImplementedError(f"Op {node.op} not implemented"))

    return PartialTranslation(last_result, None, None)
