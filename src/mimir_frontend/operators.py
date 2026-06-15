import mim
from mim._plugins.math import math, _math_arith, _math_extrema
from mim._plugins.tensor import tensor
from mim._plugins.core import core
from mim._plugins.compile import compile as mim_compile

class OperatorLibrary:
    def __init__(self, world: mim.World):
        self.world = world
        self.f32 = world.annex(math.f32.value)
        self.F32 = world.annex(math.F32.value)
        
        mode0 = world.lit_nat_0()
        
        self.f32_add_op = world.call(_math_arith.add, mode0)
        self.f32_mul_op = world.call(_math_arith.mul, mode0)
        self.f32_sub_op = world.call(_math_arith.sub, mode0)
        self.f32_div_op = world.call(_math_arith.div, mode0)
        self.f32_max_op = world.call(_math_extrema.fmax, mode0)

    def _rank_and_shape(self, tensor_def: mim.Def):
        dims = []
        tensor_type = tensor_def.type()
        while isinstance(tensor_type, mim.Seq):
            dims.append(tensor_type.arity())
            tensor_type = tensor_type.body()
        return self.world.lit_nat(len(dims)), self.world.tuple(dims)

    def _implicit_apps(self, callee, args):
        return self.world.app(callee, self.world.tuple(args))

    def binary(self, op, lhs, rhs):
        rank, shape = self._rank_and_shape(lhs)
        callee = self.world.annex(tensor.binary.value)
        callee = self._implicit_apps(callee, [self.F32, self.F32, self.F32])
        callee = self.world.app(callee, op)
        callee = self._implicit_apps(callee, [rank, shape])
        return self.world.app(callee, [lhs, rhs])

    def unary(self, op, input):
        rank, shape = self._rank_and_shape(input)
        callee = self.world.annex(tensor.unary.value)
        callee = self._implicit_apps(callee, [self.F32, self.F32])
        callee = self.world.app(callee, op)
        callee = self._implicit_apps(callee, [rank, shape])
        return self.world.app(callee, input)

    def add(self, lhs, rhs):
        return self.binary(self.f32_add_op, lhs, rhs)

    def mul(self, lhs, rhs):
        return self.binary(self.f32_mul_op, lhs, rhs)

    def sub(self, lhs, rhs):
        return self.binary(self.f32_sub_op, lhs, rhs)

    def div(self, lhs, rhs):
        return self.binary(self.f32_div_op, lhs, rhs)

    def relu(self, x):
        lam = self.world.mut_lam(self.F32, self.F32)
        v = lam.var()
        zero = self.world.lit(self.F32, 0)
        res = self.world.call(_math_extrema.fmax, self.world.lit_nat_0())
        lam.app(True, res, [v, zero])
        return self.unary(lam, x)

    def broadcast(self, input, s_out):
        pass

    def reduce_sum(self, input, dim=None, keepdim=False):
        return input

    def pooling(self, input, window_shape, stride=None, padding=0):
        return input
