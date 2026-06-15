import torch
from torch import fx
import mim
from .operators import OperatorLibrary
from mim._plugins.compile import compile as mim_compile
import operator

class FXGraphTranslator:
    def __init__(self, world: mim.World):
        self.world = world
        self.ops = OperatorLibrary(world)
        self.env = {}

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
            elif node.op == "call_function":
                self.env[node] = self.convert_call_function(node)
            elif node.op == "call_method":
                self.env[node] = self.convert_call_method(node)
            elif node.op == "output":
                res = node.args[0]
                if isinstance(res, fx.Node):
                    return self.env[res]
                elif isinstance(res, (list, tuple)):
                    return self.world.tuple([self.env[n] if isinstance(n, fx.Node) else n for n in res])
                else:
                    return res
            else:
                # Other ops like get_attr might be needed for weights
                raise NotImplementedError(f"Op {node.op} not implemented")

    def convert_call_function(self, node: fx.Node) -> mim.Def:
        target = node.target
        args = []
        for arg in node.args:
            if isinstance(arg, fx.Node):
                args.append(self.env[arg])
            elif isinstance(arg, (list, tuple)):
                args.append([self.env[a] if isinstance(a, fx.Node) else a for a in arg])
            else:
                args.append(arg)

        kwargs = {}
        for k, v in node.kwargs.items():
            if isinstance(v, fx.Node):
                kwargs[k] = self.env[v]
            else:
                kwargs[k] = v

        if target == torch.add or target == operator.add:
            return self.ops.add(*args)
        elif target == torch.mul or target == operator.mul:
            return self.ops.mul(*args)
        elif target == torch.sub or target == operator.sub:
            return self.ops.sub(*args)
        elif target == torch.div or target == operator.truediv:
            return self.ops.div(*args)
        elif target == torch.relu:
            return self.ops.relu(args[0])
        elif target == torch.sum:
            return self.ops.reduce_sum(args[0], **kwargs)
        
        raise NotImplementedError(f"Function {target} not supported")

    def convert_call_method(self, node: fx.Node) -> mim.Def:
        target = node.target
        args = []
        # first arg is self
        for arg in node.args:
            if isinstance(arg, fx.Node):
                args.append(self.env[arg])
            else:
                args.append(arg)
        
        if target == "relu":
            return self.ops.relu(args[0])
        elif target == "add":
            return self.ops.add(args[0], args[1])
        
        raise NotImplementedError(f"Method {target} not supported")

def get_high_level_phase(world: mim.World) -> mim.Def:
    from mim._plugins.tensor import tensor as mim_tensor
    
    internal_cleanup = world.annex(mim_compile.internal_cleanup_phase.value)
    lower_tensor = world.annex(mim_tensor.lower_tensor.value)
    fuse_tensor = world.annex(mim_tensor.fuse_tensor.value)
    
    phases = [internal_cleanup, lower_tensor, fuse_tensor, internal_cleanup]
    return world.call(mim_compile.phases, world.lit_bool(False), phases)
