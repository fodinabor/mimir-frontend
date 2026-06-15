import mim
import struct
from mim._plugins.affine import affine
from mim._plugins.math import math
from mim._plugins.tensor import tensor

class OperatorLibrary:
    def __init__(self, world: mim.World):
        self.world = world
        self.f32_config = world.annex(math.f32.value)
        self.F32 = world.annex(math.F32.value)
        self.Bool = world.type_bool()
        self.mode0 = world.lit_nat_0()
        
        def bind_math_axm(axm_enum):
            axm = world.annex(axm_enum.value)
            # %math.arith.add {pe} mode
            axm = world.app(axm, self.f32_config)
            return world.app(axm, self.mode0)

        # Arithmetic
        self.f32_add_axm = bind_math_axm(math.arith.add)
        self.f32_sub_axm = bind_math_axm(math.arith.sub)
        self.f32_mul_axm = bind_math_axm(math.arith.mul)
        self.f32_div_axm = bind_math_axm(math.arith.div)
        
        # Extrema
        self.f32_max_axm = bind_math_axm(math.extrema.fmax)
        self.f32_min_axm = bind_math_axm(math.extrema.fmin)
        
        # Comparisons
        self.f32_eq_axm = bind_math_axm(math.cmp.e)
        self.f32_ne_axm = bind_math_axm(math.cmp.ne)
        self.f32_lt_axm = bind_math_axm(math.cmp.l)
        self.f32_le_axm = bind_math_axm(math.cmp.le)
        self.f32_gt_axm = bind_math_axm(math.cmp.g)
        self.f32_ge_axm = bind_math_axm(math.cmp.ge)
        
        # Unary
        self.f32_exp_axm = bind_math_axm(math.exp.exp)
        self.f32_log_axm = bind_math_axm(math.exp.log)
        self.f32_tanh_axm = bind_math_axm(math.tri.tanh)
        self.f32_sqrt_axm = bind_math_axm(math.rt.sq)
        self.f32_abs_axm = bind_math_axm(math.abs)
        self.f32_neg_axm = bind_math_axm(math.minus)
        
        # Complex
        self.f32_sigmoid_axm = bind_math_axm(math.slf)
        self.f32_rsqrt_axm = bind_math_axm(math.rrt)
        self.affine_index = world.annex(affine.index.value)

    def _rank_and_shape(self, tensor_def):
        dims = self._shape_dims(tensor_def)
        return self.world.lit_nat(len(dims)), self.world.tuple(dims)

    def _shape_dims(self, tensor_def):
        dims = []
        tensor_type = tensor_def.type()
        while isinstance(tensor_type, mim.Seq):
            dims.append(tensor_type.arity())
            tensor_type = tensor_type.body()
        return dims

    def _apply_grouped(self, callee, args):
        return self.world.app(callee, self.world.tuple(args))

    def _normalize_reduce_dims(self, dim, rank):
        if dim is None:
            dims = list(range(rank))
        elif isinstance(dim, int):
            dims = [dim]
        elif isinstance(dim, (list, tuple)):
            dims = list(dim)
        else:
            raise NotImplementedError(f"reduce dim {dim!r} is not supported")

        normalized = []
        for d in dims:
            if not isinstance(d, int):
                raise NotImplementedError(f"reduce dim {d!r} is not supported")
            if d < 0:
                d += rank
            if d < 0 or d >= rank:
                raise ValueError(f"reduce dim {d} out of range for rank {rank}")
            if d not in normalized:
                normalized.append(d)
        return normalized

    def _affine_projection_lam(self, total_rank, output_rank, projections):
        vec_type = self.world.arr(self.world.lit_nat(total_rank), self.affine_index)
        out_type = self.world.arr(self.world.lit_nat(output_rank), self.affine_index)
        lam = self.world.mut_lam(vec_type, out_type)
        iters = lam.var()
        lam.set_body(True, self.world.tuple([iters.proj(total_rank, index) for index in projections]))
        return lam

    def _f32_reduce_lambda(self, op):
        args_type = self.world.arr(self.world.lit_nat(2), self.F32)
        lam = self.world.mut_con([args_type, self.world.cn([self.F32])])
        args = lam.var(0)
        reduced = self.world.app(op, [args.proj(2, 0), args.proj(2, 1)])
        lam.app(True, lam.ret_var(), [reduced])
        return lam

    def binary(self, op, lhs, rhs, out_type=None):
        if out_type is None:
            out_type = self.F32
        rank, shape = self._rank_and_shape(lhs)
        callee = self.world.annex(tensor.binary.value)
        callee = self._apply_grouped(callee, [self.F32, self.F32, out_type])
        callee = self.world.app(callee, op)
        callee = self._apply_grouped(callee, [rank, shape])
        return self.world.app(callee, [lhs, rhs])

    def compare(self, op, lhs, rhs):
        return self.binary(op, lhs, rhs, out_type=self.Bool)

    def unary(self, op, input):
        rank, shape = self._rank_and_shape(input)
        return self._unary_with_types(self.F32, self.F32, op, input, rank, shape)

    def _unary_with_types(self, input_type, output_type, op, input, rank, shape):
        callee = self.world.annex(tensor.unary.value)
        callee = self._apply_grouped(callee, [input_type, output_type])
        callee = self.world.app(callee, op)
        callee = self._apply_grouped(callee, [rank, shape])
        return self.world.app(callee, input)

    def _f32_lit(self, value):
        return self.world.lit(self.F32, value)

    def _f32_float_lit(self, value):
        bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
        return self.world.lit(self.F32, bits)

    def _f32_unary_lambda(self, callee, args_fn):
        lam = self.world.mut_lam(self.F32, self.F32)
        v = lam.var()
        lam.app(True, callee, args_fn(v))
        return lam

    def _f32_pair_to_mean_lambda(self, pair_type):
        lam = self.world.mut_lam(pair_type, self.F32)
        pair = lam.var()
        lam.app(True, self.f32_div_axm, [pair.proj(2, 0), pair.proj(2, 1)])
        return lam

    # Arithmetic
    def add(self, lhs, rhs): return self.binary(self.f32_add_axm, lhs, rhs)
    def sub(self, lhs, rhs): return self.binary(self.f32_sub_axm, lhs, rhs)
    def mul(self, lhs, rhs): return self.binary(self.f32_mul_axm, lhs, rhs)
    def div(self, lhs, rhs): return self.binary(self.f32_div_axm, lhs, rhs)
    
    # Comparison
    def eq(self, lhs, rhs): return self.compare(self.f32_eq_axm, lhs, rhs)
    def ne(self, lhs, rhs): return self.compare(self.f32_ne_axm, lhs, rhs)
    def lt(self, lhs, rhs): return self.compare(self.f32_lt_axm, lhs, rhs)
    def le(self, lhs, rhs): return self.compare(self.f32_le_axm, lhs, rhs)
    def gt(self, lhs, rhs): return self.compare(self.f32_gt_axm, lhs, rhs)
    def ge(self, lhs, rhs): return self.compare(self.f32_ge_axm, lhs, rhs)

    # Extrema
    def maximum(self, lhs, rhs): return self.binary(self.f32_max_axm, lhs, rhs)
    def minimum(self, lhs, rhs): return self.binary(self.f32_min_axm, lhs, rhs)
    def clamp_max(self, x, max_val): return self.minimum(x, max_val)
    def clamp_min(self, x, min_val): return self.maximum(x, min_val)

    # Unary
    def exp(self, x): return self.unary(self.f32_exp_axm, x)
    def log(self, x): return self.unary(self.f32_log_axm, x)
    def tanh(self, x): return self.unary(self.f32_tanh_axm, x)
    def sqrt(self, x): return self.unary(self.f32_sqrt_axm, x)
    def abs(self, x): return self.unary(self.f32_abs_axm, x)
    def neg(self, x): return self.unary(self.f32_neg_axm, x)
    def sigmoid(self, x): return self.unary(self.f32_sigmoid_axm, x)
    def rsqrt(self, x): return self.unary(self.f32_rsqrt_axm, x)
    
    def relu(self, x):
        lam = self._f32_unary_lambda(
            self.f32_max_axm,
            lambda v: [v, self._f32_lit(0)],
        )
        return self.unary(lam, x)

    def reciprocal(self, x):
        lam = self._f32_unary_lambda(
            self.f32_div_axm,
            lambda v: [self._f32_float_lit(1.0), v],
        )
        return self.unary(lam, x)

    # Logical
    def where(self, cond, x, y):
        return self.world.call(tensor.select, [cond, x, y])

    def _reduce_aff(self, input, output_type, reducer, init, dim=None, keepdim=False, return_shape=False):
        input_dims = self._shape_dims(input)
        input_rank = len(input_dims)
        reduce_dims = self._normalize_reduce_dims(dim, input_rank)
        kept_dims = [axis for axis in range(input_rank) if axis not in reduce_dims]

        if keepdim:
            output_dims = [
                self.world.lit_nat(1) if axis in reduce_dims else input_dims[axis]
                for axis in range(input_rank)
            ]
            input_projections = [
                input_rank + reduce_dims.index(axis) if axis in reduce_dims else axis
                for axis in range(input_rank)
            ]
        else:
            output_dims = [input_dims[axis] for axis in kept_dims]
            kept_positions = {axis: pos for pos, axis in enumerate(kept_dims)}
            input_projections = [
                len(kept_dims) + reduce_dims.index(axis)
                if axis in reduce_dims
                else kept_positions[axis]
                for axis in range(input_rank)
            ]

        output_rank = len(output_dims)
        reduce_rank = len(reduce_dims)
        loop_dims = output_dims + [input_dims[axis] for axis in reduce_dims]
        total_rank = output_rank + reduce_rank

        callee = self.world.annex(tensor.map_reduce_aff.value)
        callee = self.world.app(callee, self.world.lit_nat(1))
        callee = self._apply_grouped(callee, [output_type, self.world.lit_nat(output_rank), self.world.lit_nat(reduce_rank)])
        callee = self._apply_grouped(callee, [self.world.tuple(output_dims), self.world.tuple(loop_dims)])
        callee = self._apply_grouped(
            callee,
            [
                self.world.tuple([self.F32]),
                self.world.tuple([self.world.lit_nat(input_rank)]),
                self.world.tuple([self.world.tuple(input_dims)]),
            ],
        )
        callee = self._apply_grouped(callee, [reducer, init])
        callee = self.world.app(
            callee,
            self._affine_projection_lam(total_rank, output_rank, list(range(output_rank))),
        )
        callee = self.world.app(
            callee,
            self.world.tuple([
                self._affine_projection_lam(total_rank, input_rank, input_projections),
            ]),
        )
        result = self.world.app(callee, [input])
        if return_shape:
            return result, output_dims
        return result

    # Reductions
    def sum(self, input, dim=None, keepdim=False):
        return self._reduce_aff(
            input,
            self.F32,
            self._f32_reduce_lambda(self.f32_add_axm),
            self._f32_float_lit(0.0),
            dim=dim,
            keepdim=keepdim,
        )

    def amax(self, input, dim=None, keepdim=False):
        return self._reduce_aff(
            input,
            self.F32,
            self._f32_reduce_lambda(self.f32_max_axm),
            self._f32_lit(0xFF800000),
            dim=dim,
            keepdim=keepdim,
        )

    def _f32_mean_reduce_lambda(self, pair_type):
        args_type = self.world.sigma([pair_type, self.F32])
        lam = self.world.mut_con([args_type, self.world.cn([pair_type])])
        args = lam.var(0)
        acc = args.proj(2, 0)
        value = args.proj(2, 1)
        sum_next = self.world.app(self.f32_add_axm, [acc.proj(2, 0), value])
        count_next = self.world.app(self.f32_add_axm, [acc.proj(2, 1), self._f32_float_lit(1.0)])
        lam.app(True, lam.ret_var(), [self.world.tuple([sum_next, count_next])])
        return lam

    def mean(self, input, dim=None, keepdim=False):
        pair_type = self.world.arr(self.world.lit_nat(2), self.F32)
        reduced, output_dims = self._reduce_aff(
            input,
            pair_type,
            self._f32_mean_reduce_lambda(pair_type),
            self.world.tuple([self._f32_float_lit(0.0), self._f32_float_lit(0.0)]),
            dim=dim,
            keepdim=keepdim,
            return_shape=True,
        )
        rank = self.world.lit_nat(len(output_dims))
        shape = self.world.tuple(output_dims)
        return self._unary_with_types(pair_type, self.F32, self._f32_pair_to_mean_lambda(pair_type), reduced, rank, shape)

    def var_mean(self, input, dim=None, keepdim=False, correction=0):
        raise NotImplementedError("aten.var_mean is not implemented")

    # Linear Algebra
    def mm(self, lhs, rhs):
        raise NotImplementedError("aten.mm is not implemented")

    def convolution(self, x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        raise NotImplementedError("aten.convolution is not implemented")

    # Injective
    def cat(self, tensors, dim=0):
        raise NotImplementedError("aten.cat is not implemented")

    def transpose(self, x, permutation):
        raise NotImplementedError("aten.permute is not implemented")
