import torch
from torch import fx
import mim
from .operators import OperatorLibrary
from mim._plugins.compile import compile as mim_compile
import operator
from collections.abc import Callable

class FXGraphTranslator:
    def __init__(self, world: mim.World):
        self.world = world
        self.ops = OperatorLibrary(world)
        self.env: dict[fx.Node, mim.Def] = {}
        self.convert_map: dict[str | Callable, Callable[[fx.Node], mim.Def]] = self.create_convert_map()

    def create_convert_map(self) -> dict:
        m = {}
        
        # Elementwise Binary
        binary_ops = [
            (torch.add, operator.add, "aten.add.default", self.ops.add),
            (torch.sub, operator.sub, "aten.sub.default", self.ops.sub),
            (torch.mul, operator.mul, "aten.mul.default", self.ops.mul),
            (torch.div, operator.truediv, "aten.div.default", self.ops.div),
            (None, None, "aten.div.Tensor", self.ops.div),
            (torch.maximum, None, "aten.maximum.default", self.ops.maximum),
            (torch.minimum, None, "aten.minimum.default", self.ops.minimum),
            (torch.eq, operator.eq, "aten.eq.default", self.ops.eq),
            (None, None, "aten.eq.Tensor", self.ops.eq),
            (torch.ne, operator.ne, "aten.ne.default", self.ops.ne),
            (torch.lt, operator.lt, "aten.lt.default", self.ops.lt),
            (torch.le, operator.le, "aten.le.default", self.ops.le),
            (torch.gt, operator.gt, "aten.gt.default", self.ops.gt),
            (torch.ge, operator.ge, "aten.ge.default", self.ops.ge),
            (torch.clamp_max, None, "aten.clamp_max.default", self.ops.clamp_max),
            (torch.clamp_min, None, "aten.clamp_min.default", self.ops.clamp_min),
        ]
        
        for t, op, name, func in binary_ops:
            wrapper = self._wrap_binary(func)
            if t: m[t] = wrapper
            if op: m[op] = wrapper
            if name: m[name] = wrapper

        # Elementwise Unary
        unary_ops = [
            (torch.relu, "aten.relu.default", self.ops.relu),
            (torch.exp, "aten.exp.default", self.ops.exp),
            (torch.tanh, "aten.tanh.default", self.ops.tanh),
            (torch.sqrt, "aten.sqrt.default", self.ops.sqrt),
            (torch.abs, "aten.abs.default", self.ops.abs),
            (torch.neg, "aten.neg.default", self.ops.neg),
            (torch.sigmoid, "aten.sigmoid.default", self.ops.sigmoid),
            (torch.reciprocal, "aten.reciprocal.default", self.ops.reciprocal),
            (torch.rsqrt, "aten.rsqrt.default", self.ops.rsqrt),
        ]
        
        for t, name, func in unary_ops:
            wrapper = self._wrap_unary(func)
            if t: m[t] = wrapper
            if name: m[name] = wrapper

        # Injective
        m[torch.cat] = self._wrap_unsupported("aten.cat")
        m["aten.cat.default"] = self._wrap_unsupported("aten.cat")
        m[torch.permute] = self._wrap_unsupported("aten.permute")
        m["aten.permute.default"] = self._wrap_unsupported("aten.permute")

        # Reductions
        m[torch.sum] = self._wrap_reduction(self.ops.sum)
        m["aten.sum.default"] = self._wrap_reduction(self.ops.sum)
        m["aten.sum.dim_IntList"] = self._wrap_reduction(self.ops.sum)
        m[torch.amax] = self._wrap_reduction(self.ops.amax)
        m["aten.amax.default"] = self._wrap_reduction(self.ops.amax)
        m[torch.mean] = self._wrap_reduction(self.ops.mean)
        m["aten.mean.default"] = self._wrap_reduction(self.ops.mean)
        m["aten.mean.dim"] = self._wrap_reduction(self.ops.mean)
        m[torch.var_mean] = self._wrap_reduction(self.ops.var_mean)
        m["aten.var_mean.default"] = self._wrap_reduction(self.ops.var_mean)
        m["aten.var_mean.correction"] = self._wrap_reduction(self.ops.var_mean)

        # Linear Algebra
        m[torch.mm] = self._wrap_binary(self.ops.mm)
        m["aten.mm.default"] = self._wrap_binary(self.ops.mm)

        # Convolution
        m[torch.convolution] = self._wrap_unsupported("aten.convolution")
        m["aten.convolution.default"] = self._wrap_unsupported("aten.convolution")

        # Selection
        m[torch.where] = self._wrap_where()
        m["aten.where.self"] = self._wrap_where()

        return m

    def _wrap_binary(self, op_func):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            return op_func(args[0], args[1])
        return convert

    def _wrap_unary(self, op_func):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            return op_func(args[0])
        return convert

    def _wrap_reduction(self, op_func):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            input_def = args[0]
            dim = args[1] if len(args) > 1 else node.kwargs.get("dim", None)
            keepdim = args[2] if len(args) > 2 else node.kwargs.get("keepdim", False)
            return op_func(input_def, dim=dim, keepdim=keepdim)
        return convert

    def _wrap_where(self):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            return self.ops.where(args[0], args[1], args[2])
        return convert

    def _wrap_unsupported(self, name: str):
        def convert(node: fx.Node):
            raise NotImplementedError(f"{name} is not implemented")
        return convert

    def retrieve_args(self, node: fx.Node) -> list:
        return self._retrieve_args(node.args)

    def _retrieve_args(self, args):
        if isinstance(args, fx.Node):
            return self.env[args]
        elif isinstance(args, (list, tuple)):
            return [self._retrieve_args(a) for a in args]
        elif isinstance(args, dict):
            return {k: self._retrieve_args(v) for k, v in args.items()}
        else:
            return args

    def translate(self, graph: fx.Graph, inputs: list[mim.Def]) -> mim.Def:
        self.env = {}
        placeholders = [node for node in graph.nodes if node.op == "placeholder"]
        if len(placeholders) != len(inputs):
            raise ValueError(f"Expected {len(placeholders)} inputs, got {len(inputs)}")
        for node, arg in zip(placeholders, inputs):
            self.env[node] = arg

        for node in graph.nodes:
            if node.op == "placeholder":
                continue
            elif node.op in ("call_function", "call_method"):
                self.env[node] = self.convert_node(node)
            elif node.op == "output":
                res = node.args[0]
                if isinstance(res, fx.Node):
                    return self.env[res]
                elif isinstance(res, (list, tuple)):
                    return self.world.tuple([self.env[n] if isinstance(n, fx.Node) else n for n in res])
                else:
                    return res
            else:
                raise NotImplementedError(f"Op {node.op} not implemented")

    def convert_node(self, node: fx.Node) -> mim.Def:
        target = node.target
        
        if target in self.convert_map:
            return self.convert_map[target](node)
        
        if hasattr(target, "name"):
            name = target.name()
            name = name.replace("::", ".")
            if name in self.convert_map:
                return self.convert_map[name](node)

        if node.op == "call_method" and target in self.convert_map:
             return self.convert_map[target](node)
        
        raise NotImplementedError(f"Target {target} (type {type(target)}) not supported")

def get_high_level_phase(world: mim.World) -> mim.Def:
    from mim._plugins.tensor import tensor as mim_tensor
    
    internal_cleanup = world.annex(mim_compile.internal_cleanup_phase.value)
    lower_tensor = world.annex(mim_tensor.lower_tensor.value)
    fuse_tensor = world.annex(mim_tensor.fuse_tensor.value)
    
    phases = [internal_cleanup, lower_tensor, fuse_tensor, internal_cleanup]
    return world.call(mim_compile.phases, world.lit_bool(False), phases)
