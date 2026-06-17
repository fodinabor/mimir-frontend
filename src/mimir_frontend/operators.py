import mim
import struct
from mim._plugins.affine import affine
from mim._plugins.math import math
from mim._plugins.tensor import tensor
from mim._plugins.core import core

class OperatorLibrary:
    def __init__(self, world: mim.World):
        self.world = world
        self.f32_config = world.annex(math.f32.value)
        self.F32 = world.annex(math.F32.value)
        self.Bool = world.type_bool()
        self.mode0 = world.lit_nat_0()
        self.sym_map = {} # Mapping from symbolic name to MimIR Nat variable

        
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

        # Bitwise/Logical
        s_bool = world.lit_nat(2)
        m_scalar = world.lit_nat_0()
        
        def bind_bit_axm(axm_enum):
            axm = world.annex(axm_enum.value)
            axm = world.app(axm, s_bool)
            return world.app(axm, m_scalar)
            
        self.bool_and_axm = bind_bit_axm(core.bit2.and_)
        self.bool_not_axm = bind_bit_axm(core.bit1.neg)

    def _rank_and_shape(self, tensor_def):
        dims = self.shape_of(tensor_def)
        return self.world.lit_nat(len(dims)), self.world.tuple(dims)

    def shape_of(self, value):
        if isinstance(value, mim.Def):
            return self._shape_dims(value)
        if hasattr(value, "meta") and isinstance(value.meta, dict) and "val" in value.meta:
            return self.shape_of(value.meta["val"])
        if hasattr(value, "shape"):
            return list(value.shape)
        raise TypeError(f"shape_of does not support {type(value)}")

    def _shape_dims(self, tensor_def):
        dims = []
        tensor_type = tensor_def.type()
        while isinstance(tensor_type, mim.Seq):
            dims.append(tensor_type.arity())
            tensor_type = tensor_type.body()
        return dims

    def _dim_literal_value(self, dim):
        if isinstance(dim, mim.Lit) and hasattr(dim, "get_nat"):
            return dim.get_nat()
        return None

    def _is_dim_one(self, dim):
        return self._dim_literal_value(dim) == 1

    def _same_dim(self, lhs, rhs):
        lhs_value = self._dim_literal_value(lhs)
        rhs_value = self._dim_literal_value(rhs)
        if lhs_value is not None and rhs_value is not None:
            return lhs_value == rhs_value
        return lhs == rhs

    def _same_shape_dims(self, lhs_dims, rhs_dims):
        return len(lhs_dims) == len(rhs_dims) and all(
            self._same_dim(lhs_dim, rhs_dim)
            for lhs_dim, rhs_dim in zip(lhs_dims, rhs_dims)
        )

    def _broadcast_dim(self, lhs_dim, rhs_dim):
        if self._same_dim(lhs_dim, rhs_dim):
            return lhs_dim
        if self._is_dim_one(lhs_dim):
            return rhs_dim
        if self._is_dim_one(rhs_dim):
            return lhs_dim

        lhs_value = self._dim_literal_value(lhs_dim)
        rhs_value = self._dim_literal_value(rhs_dim)
        if lhs_value is not None and rhs_value is not None:
            raise NotImplementedError(
                f"broadcast incompatible dimensions: {lhs_value} vs {rhs_value}"
            )

        # Symbolic dimensions may be equal at runtime. Keep the left-hand dim
        # instead of rejecting dynamic-shape models prematurely.
        return lhs_dim

    def _broadcast_shape_dims(self, lhs_dims, rhs_dims):
        out_reversed = []
        for offset in range(1, max(len(lhs_dims), len(rhs_dims)) + 1):
            lhs_dim = lhs_dims[-offset] if offset <= len(lhs_dims) else self.world.lit_nat(1)
            rhs_dim = rhs_dims[-offset] if offset <= len(rhs_dims) else self.world.lit_nat(1)
            out_reversed.append(self._broadcast_dim(lhs_dim, rhs_dim))
        return list(reversed(out_reversed))

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

    def _tensor_element_type(self, tensor_def):
        tensor_type = tensor_def.type()
        while isinstance(tensor_type, mim.Seq):
            tensor_type = tensor_type.body()
        return tensor_type

    def binary(self, op, lhs, rhs, out_type=None):
        if isinstance(rhs, (int, float)):
            if out_type is None:
                out_type = self._tensor_element_type(lhs)
            lam = self._f32_unary_lambda(op, lambda v: [v, self._f32_float_lit(float(rhs))], ret_type=out_type)
            return self.unary(lam, lhs, out_type=out_type)
        if isinstance(lhs, (int, float)):
            if out_type is None:
                out_type = self._tensor_element_type(rhs)
            lam = self._f32_unary_lambda(op, lambda v: [self._f32_float_lit(float(lhs)), v], ret_type=out_type)
            return self.unary(lam, rhs, out_type=out_type)
            
        in_type = self._tensor_element_type(lhs)
        if out_type is None:
            out_type = in_type
            
        s_lhs_dims = self.shape_of(lhs)
        s_rhs_dims = self.shape_of(rhs)
        output_dims = self._broadcast_shape_dims(s_lhs_dims, s_rhs_dims)
        
        if not self._same_shape_dims(s_lhs_dims, output_dims):
            lhs = self.expand(lhs, output_dims)
        if not self._same_shape_dims(s_rhs_dims, output_dims):
            rhs = self.expand(rhs, output_dims)

        rank = self.world.lit_nat(len(output_dims))
        shape = self.world.tuple(output_dims)
        callee = self.world.annex(tensor.binary.value)
        callee = self._apply_grouped(callee, [in_type, in_type, out_type])
        callee = self.world.app(callee, op)
        callee = self._apply_grouped(callee, [rank, shape])
        return self.world.app(callee, [lhs, rhs])

    def compare(self, op, lhs, rhs):
        return self.binary(op, lhs, rhs, out_type=self.Bool)

    def unary(self, op, input, out_type=None):
        in_type = self._tensor_element_type(input)
        if out_type is None:
            out_type = in_type
        rank, shape = self._rank_and_shape(input)
        return self._unary_with_types(in_type, out_type, op, input, rank, shape)

    def _unary_with_types(self, input_type, output_type, op, input, rank, shape):
        callee = self.world.annex(tensor.unary.value)
        callee = self._apply_grouped(callee, [input_type, output_type])
        callee = self.world.app(callee, op)
        callee = self._apply_grouped(callee, [rank, shape])
        return self.world.app(callee, input)

    def _f32_float_lit(self, value):
        bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
        return self.world.lit(self.F32, bits)

    def _f32_unary_lambda(self, callee, args_fn, ret_type=None):
        if ret_type is None:
            ret_type = self.F32
        lam = self.world.mut_lam(self.F32, ret_type)
        v = lam.var()
        lam.app(True, callee, args_fn(v))
        return lam

    def _f32_pair_to_mean_lambda(self, pair_type):
        lam = self.world.mut_lam(pair_type, self.F32)
        pair = lam.var()
        lam.set_body(True, self.world.app(self.f32_div_axm, [pair.proj(2, 0), pair.proj(2, 1)]))
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
    
    def clamp_max(self, x, max_val):
        if isinstance(max_val, (int, float)):
            lam = self._f32_unary_lambda(
                self.f32_min_axm,
                lambda v: [v, self._f32_float_lit(float(max_val))]
            )
            return self.unary(lam, x)
        return self.minimum(x, max_val)

    def clamp_min(self, x, min_val):
        if isinstance(min_val, (int, float)):
            lam = self._f32_unary_lambda(
                self.f32_max_axm,
                lambda v: [v, self._f32_float_lit(float(min_val))]
            )
            return self.unary(lam, x)
        return self.maximum(x, min_val)

    def clamp(self, x, min_val=None, max_val=None):
        res = x
        if min_val is not None:
            res = self.clamp_min(res, min_val)
        if max_val is not None:
            res = self.clamp_max(res, max_val)
        return res

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
            lambda v: [self._f32_float_lit(0.0), v],
        )
        return self.unary(lam, x)

    def reciprocal(self, x):
        lam = self._f32_unary_lambda(
            self.f32_div_axm,
            lambda v: [self._f32_float_lit(1.0), v],
        )
        return self.unary(lam, x)

    def bitwise_and(self, lhs, rhs):
        return self.binary(self.bool_and_axm, lhs, rhs, out_type=self.Bool)

    def logical_not(self, x):
        return self.unary(self.bool_not_axm, x, out_type=self.Bool)

    def fma(self, a, b, c):
        return self.add(self.mul(a, b), c)

    def convert_element_type(self, x, dtype):
        import torch
        in_type = self._tensor_element_type(x)
        out_type = None
        if dtype in (torch.float32, torch.float):
            out_type = self.F32
        elif dtype == torch.bool:
            out_type = self.Bool
        else:
            raise NotImplementedError(f"Conversion to {dtype} is not implemented")
            
        if in_type == out_type:
            return x
            
        if in_type == self.Bool and out_type == self.F32:
            lam = self.world.mut_lam(self.Bool, self.F32)
            v = lam.var()
            callee = self.world.annex(core.select.value)
            callee = self.world.app(callee, self.F32)
            res = self._apply_grouped(callee, [v, self._f32_float_lit(1.0), self._f32_float_lit(0.0)])
            lam.set_body(True, res)
            return self.unary(lam, x, out_type=self.F32)
            
        if in_type == self.F32 and out_type == self.Bool:
            lam = self._f32_unary_lambda(
                self.f32_ne_axm,
                lambda v: [v, self._f32_float_lit(0.0)],
                ret_type=self.Bool
            )
            return self.unary(lam, x, out_type=self.Bool)
            
        raise NotImplementedError(f"Conversion from {in_type} to {out_type} is not implemented")

    # Logical
    def where(self, cond, x, y):
        cond_dims = self.shape_of(cond)
        x_dims = self.shape_of(x)
        y_dims = self.shape_of(y)
        output_dims = self._broadcast_shape_dims(self._broadcast_shape_dims(cond_dims, x_dims), y_dims)

        if not self._same_shape_dims(cond_dims, output_dims):
            cond = self.expand(cond, output_dims)
        if not self._same_shape_dims(x_dims, output_dims):
            x = self.expand(x, output_dims)
        if not self._same_shape_dims(y_dims, output_dims):
            y = self.expand(y, output_dims)

        tensor_type = self._tensor_element_type(x)
        rank = self.world.lit_nat(len(output_dims))
        shape = self.world.tuple(output_dims)
        callee = self.world.annex(tensor.select.value)
        callee = self.world.app(callee, tensor_type)
        callee = self._apply_grouped(callee, [rank, shape])
        return self._apply_grouped(callee, [cond, x, y])

    def _extract_shape(self, shape_arg):
        out_shape_list = []
        for d in shape_arg:
            if isinstance(d, int):
                out_shape_list.append(self.world.lit_nat(d))
            elif isinstance(d, mim.Def):
                out_shape_list.append(d)
            else:
                raise ValueError(f"Unsupported shape dimension type: {type(d)}")
        return self.world.tuple(out_shape_list), len(shape_arg)

    def expand(self, input, shape):
        in_rank, in_shape = self._rank_and_shape(input)
        in_dims = self.shape_of(input)
        in_rank_val = len(in_dims)

        shape = list(shape)
        out_rank_val = len(shape)
        rank_offset = out_rank_val - in_rank_val
        resolved_shape = []
        for i, dim in enumerate(shape):
            if dim != -1:
                resolved_shape.append(dim)
                continue
            input_index = i - rank_offset
            if input_index < 0 and i < in_rank_val:
                input_index = i
            if input_index < 0 or input_index >= in_rank_val:
                raise ValueError(f"cannot infer expand dimension {i} from input rank {in_rank_val}")
            resolved_shape.append(in_dims[input_index])

        shape = resolved_shape
        out_shape_tuple, out_rank_val = self._extract_shape(shape)
        out_rank = self.world.lit_nat(out_rank_val)
        
        if self._same_shape_dims(in_dims, list(shape)):
            return input

        elem_type = self._tensor_element_type(input)

        if in_rank_val == 0:
            callee = self.world.annex(tensor.map.value)
            callee = self.world.app(callee, self.world.tuple([elem_type, self.world.lit_nat(0), self.world.tuple([])]))
            lam = self.world.mut_lam(self.world.sigma([]), elem_type)
            lam.set_body(True, input)
            callee = self.world.app(callee, lam)
            callee = self.world.app(callee, self.world.tuple([out_rank, out_shape_tuple]))
            return self.world.app(callee, self.world.tuple([]))

        if in_rank_val == out_rank_val:
            callee = self.world.annex(tensor.broadcast.value)
            callee = self._apply_grouped(callee, [elem_type, out_rank])
            return self.world.app(callee, [in_shape, out_shape_tuple, input])
        else:
            callee = self.world.annex(tensor.broadcast_in_dim.value)
            callee = self._apply_grouped(callee, [elem_type, in_rank, out_rank])
            
            idx_t = self.world.type_idx(out_rank)
            offset = out_rank_val - in_rank_val
            index_mapping = [self.world.lit(idx_t, offset + i) for i in range(in_rank_val)]
            index_tuple = self.world.tuple(index_mapping)
            
            return self.world.app(callee, [in_shape, out_shape_tuple, input, index_tuple])

    def full(self, shape, fill_value, dtype=None):
        import torch
        if dtype is None:
            dtype = torch.float32
            
        if dtype in (torch.float32, torch.float, None):
            elem_type = self.F32
            scalar_def = self._f32_float_lit(float(fill_value))
        elif dtype == torch.bool:
            elem_type = self.Bool
            scalar_def = self.world.lit_tt() if fill_value else self.world.lit_ff()
        else:
            raise NotImplementedError(f"full with dtype {dtype} is not implemented")
            
        out_shape, out_rank_val = self._extract_shape(shape)
        out_rank = self.world.lit_nat(out_rank_val)
        
        callee = self.world.annex(tensor.map.value)
        ni = self.world.lit_nat(0)
        Is = self.world.tuple([])
        callee = self.world.app(callee, self.world.tuple([elem_type, ni, Is]))
        
        lam = self.world.mut_lam(self.world.sigma([]), elem_type)
        lam.set_body(True, scalar_def)
        
        callee = self.world.app(callee, lam)
        callee = self.world.app(callee, self.world.tuple([out_rank, out_shape]))
        
        input_is = self.world.tuple([])
        return self.world.app(callee, input_is)

    def _reduce_aff(self, input, output_type, reducer, init, dim=None, keepdim=False, return_shape=False):
        input_dims = self.shape_of(input)
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
        
        in_elem_type = self._tensor_element_type(input)
        callee = self._apply_grouped(
            callee,
            [
                self.world.tuple([in_elem_type]),
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
            self.world.tuple(
                [self._affine_projection_lam(total_rank, input_rank, input_projections)]
            ),
        )
        result = self.world.app(callee, self.world.tuple([input]))
        if return_shape:
            return result, output_dims
        return result

    def sum(self, input, dim=None, keepdim=False):
        return self._reduce_aff(input, self.F32, self._f32_reduce_lambda(self.f32_add_axm), self._f32_float_lit(0.0), dim=dim, keepdim=keepdim)

    def amax(self, input, dim=None, keepdim=False):
        return self._reduce_aff(input, self.F32, self._f32_reduce_lambda(self.f32_max_axm), self._f32_float_lit(-float("inf")), dim=dim, keepdim=keepdim)

    def _f32_pair_reduce_lambda(self, pair_type):
        args_type = self.world.sigma([pair_type, self.F32])
        lam = self.world.mut_con([args_type, self.world.cn([pair_type])])
        args = lam.var(0)
        pair = args.proj(2, 0)
        value = args.proj(2, 1)
        sum_next = self.world.app(self.f32_add_axm, [pair.proj(2, 0), value])
        count_next = self.world.app(self.f32_add_axm, [pair.proj(2, 1), self._f32_float_lit(1.0)])
        lam.app(True, lam.ret_var(), [self.world.tuple([sum_next, count_next])])
        return lam

    def mean(self, input, dim=None, keepdim=False):
        pair_type = self.world.arr(self.world.lit_nat(2), self.F32)
        reduced, output_dims = self._reduce_aff(
            input,
            pair_type,
            self._f32_pair_reduce_lambda(pair_type),
            self.world.tuple([self._f32_float_lit(0.0), self._f32_float_lit(0.0)]),
            dim=dim,
            keepdim=keepdim,
            return_shape=True,
        )
        rank = self.world.lit_nat(len(output_dims))
        shape = self.world.tuple(output_dims)
        return self._unary_with_types(pair_type, self.F32, self._f32_pair_to_mean_lambda(pair_type), reduced, rank, shape)

    def _f32_var_mean_reduce_lambda(self, acc_type):
        """
        Creates the reduction lambda for var_mean which maintains (sum, sum_sq, count).
        Effectively:
          def reducer(acc, value):
              sum_acc, sum_sq_acc, count_acc = acc
              return (sum_acc + value, sum_sq_acc + value*value, count_acc + 1)
        """
        args_type = self.world.sigma([acc_type, self.F32])
        lam = self.world.mut_con([args_type, self.world.cn([acc_type])])
        args = lam.var(0)
        acc = args.proj(2, 0)
        value = args.proj(2, 1)
        
        sum_acc = acc.proj(3, 0)
        sum_sq_acc = acc.proj(3, 1)
        count_acc = acc.proj(3, 2)
        
        sum_next = self.world.app(self.f32_add_axm, [sum_acc, value])
        
        val_sq = self.world.app(self.f32_mul_axm, [value, value])
        sum_sq_next = self.world.app(self.f32_add_axm, [sum_sq_acc, val_sq])
        
        count_next = self.world.app(self.f32_add_axm, [count_acc, self._f32_float_lit(1.0)])
        
        lam.app(True, lam.ret_var(), [self.world.tuple([sum_next, sum_sq_next, count_next])])
        return lam

    def _f32_acc_to_var_mean(self, acc_type, extract_var=True):
        """
        Finalizer map step for var_mean.
        Extracts `mean` or `var` from the accumulator tuple:
          mean = sum / count
          var = (sum_sq / count) - (mean * mean)
        """
        lam = self.world.mut_lam(acc_type, self.F32)
        acc = lam.var()
        s = acc.proj(3, 0)
        s_sq = acc.proj(3, 1)
        c = acc.proj(3, 2)
        
        mean = self.world.app(self.f32_div_axm, [s, c])
        
        if extract_var:
            mean_sq = self.world.app(self.f32_mul_axm, [mean, mean])
            e_x_sq = self.world.app(self.f32_div_axm, [s_sq, c])
            var = self.world.app(self.f32_sub_axm, [e_x_sq, mean_sq])
            lam.set_body(True, var)
        else:
            lam.set_body(True, mean)
            
        return lam

    def var_mean(self, input, dim=None, keepdim=False, correction=0):
        """
        Translates torch.var_mean into a map-reduce operation that yields a tuple (var, mean).
        
        MimIR Pipeline:
        1. Uses `%tensor.map_reduce_aff` to accumulate `(sum, sum_sq, count)` into a 3-element array.
        2. Projects this accumulator tensor into `var` and `mean` using two separate `%tensor.unary` operations.
        3. Returns a MimIR tuple containing the two resulting tensors.
        """
        if correction != 0:
            raise NotImplementedError("var_mean with correction != 0 is not implemented")
            
        acc_type = self.world.arr(self.world.lit_nat(3), self.F32)
        reduced, output_dims = self._reduce_aff(
            input,
            acc_type,
            self._f32_var_mean_reduce_lambda(acc_type),
            self.world.tuple([self._f32_float_lit(0.0), self._f32_float_lit(0.0), self._f32_float_lit(0.0)]),
            dim=dim,
            keepdim=keepdim,
            return_shape=True,
        )
        
        rank = self.world.lit_nat(len(output_dims))
        shape = self.world.tuple(output_dims)
        
        var_tensor = self._unary_with_types(acc_type, self.F32, self._f32_acc_to_var_mean(acc_type, extract_var=True), reduced, rank, shape)
        mean_tensor = self._unary_with_types(acc_type, self.F32, self._f32_acc_to_var_mean(acc_type, extract_var=False), reduced, rank, shape)
        
        return self.world.tuple([var_tensor, mean_tensor])

    # Linear Algebra
    # Linear Algebra
    def mm(self, lhs, rhs):
        # Ring: [T: *, _0: T, add: [T, T] -> T, mul: [T, T] -> T]
        ring = self.world.tuple([self.F32, self._f32_float_lit(0.0), self.f32_add_axm, self.f32_mul_axm])
        callee = self.world.annex(tensor.product_2d.value)
        callee = self.world.app(callee, ring)
        return self.world.implicit_app(callee, [lhs, rhs])

    def convolution(self, x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        raise NotImplementedError("aten.convolution is not implemented")

    # Injective
    def reshape(self, x, shape):
        in_rank, in_shape = self._rank_and_shape(x)
        out_shape_tuple, out_rank_val = self._extract_shape(shape)
        out_rank = self.world.lit_nat(out_rank_val)
        elem_t = self._tensor_element_type(x)

        callee = self.world.annex(tensor.reshape.value)
        callee = self._apply_grouped(callee, [elem_t, in_rank, out_rank])
        callee = self.world.app(callee, in_shape)
        callee = self.world.app(callee, out_shape_tuple)
        return self.world.app(callee, x)

    def view(self, x, shape):
        return self.reshape(x, shape)

    def slice(self, x, dim, start, end, step=1):
        rank, in_shape = self._rank_and_shape(x)
        in_dims = self.shape_of(x)
        rank_val = len(in_dims)
        elem_t = self._tensor_element_type(x)

        if dim < 0: dim += rank_val
        
        starts = []
        steps = []
        out_dims = []
        
        for i in range(rank_val):
            if i == dim:
                s_in = in_dims[i]
                is_static = isinstance(start, int) and (end is None or isinstance(end, int)) and isinstance(step, int)
                
                actual_start = self.world.lit_nat(start) if isinstance(start, int) else start
                actual_step = self.world.lit_nat(step) if isinstance(step, int) else (self.world.lit_nat(1) if step is None else step)
                
                if end is None or (isinstance(end, int) and end > 1000000000):
                    actual_end = s_in
                else:
                    actual_end = self.world.lit_nat(end) if isinstance(end, int) else end
                
                starts.append(actual_start)
                steps.append(actual_step)
                
                if is_static:
                    v_step = step if step is not None else 1
                    if end is None:
                        out_dims.append(self.world.top_nat())
                    else:
                        out_dims.append(self.world.lit_nat((end - start + v_step - 1) // v_step))
                else:
                    out_dims.append(self.world.top_nat())
            else:
                starts.append(self.world.lit_nat(0))
                steps.append(self.world.lit_nat(1))
                out_dims.append(in_dims[i])
        
        callee = self.world.annex(tensor.slice.value)
        callee = self._apply_grouped(callee, [elem_t, rank])
        callee = self.world.app(callee, in_shape)
        
        callee = self.world.app(callee, self.world.tuple([
            self.world.tuple(starts),
            self.world.tuple(steps),
            self.world.tuple(out_dims)
        ]))
        return self.world.app(callee, x)

    def cat(self, tensors, dim=0):
        num_inputs = tensors.num_projs()
        first_tensor = tensors.proj(num_inputs, 0)
        rank, _ = self._rank_and_shape(first_tensor)
        rank_val = len(self.shape_of(first_tensor))
        elem_t = self._tensor_element_type(first_tensor)
        
        if dim < 0: dim += rank_val
        
        callee = self.world.annex(tensor.concat.value)
        callee = self._apply_grouped(callee, [elem_t, self.world.lit_nat(num_inputs), rank])
        
        idx_t = self.world.type_idx(rank)
        ax = self.world.lit(idx_t, dim)
        callee = self.world.app(callee, ax)
        
        input_shapes = []
        for i in range(num_inputs):
            _, s = self._rank_and_shape(tensors.proj(num_inputs, i))
            input_shapes.append(s)
        
        callee = self.world.app(callee, self.world.tuple(input_shapes))
        return self.world.app(callee, tensors)

    def transpose(self, x, permutation):
        rank, shape = self._rank_and_shape(x)
        elem_t = self._tensor_element_type(x)
        idx_t = self.world.type_idx(rank)
        perm_mim = self.world.tuple([self.world.lit(idx_t, p) for p in permutation])
        
        callee = self.world.annex(tensor.transpose.value)
        callee = self._apply_grouped(callee, [elem_t, rank, shape])
        return self.world.app(callee, [x, perm_mim])

    def _is_one(self, d):
        if isinstance(d, int): return d == 1
        return d == self.world.lit_nat(1)

    def squeeze(self, x, dim=None):
        in_dims = self.shape_of(x)
        if dim is None:
            out_dims = [d for d in in_dims if not self._is_one(d)]
        else:
            if dim < 0: dim += len(in_dims)
            out_dims = []
            for i, d in enumerate(in_dims):
                if i == dim:
                    if not self._is_one(d):
                         out_dims.append(d)
                else:
                    out_dims.append(d)
        return self.reshape(x, out_dims)

    def unsqueeze(self, x, dim):
        in_dims = self.shape_of(x)
        if dim < 0: dim += len(in_dims) + 1
        out_dims = list(in_dims)
        out_dims.insert(dim, 1)
        return self.reshape(x, out_dims)

    def split(self, x, split_size_or_sections, dim=0):
        in_dims = self.shape_of(x)
        rank_val = len(in_dims)
        if dim < 0: dim += rank_val
        
        extent = in_dims[dim]
        extent_value = self._dim_literal_value(extent)
        slices = []
        if isinstance(split_size_or_sections, int):
            split_size = split_size_or_sections
            if extent_value is not None:
                curr = 0
                while curr < extent_value:
                    end = min(curr + split_size, extent_value)
                    slices.append(self.slice(x, dim, curr, end))
                    curr = end
            else:
                raise NotImplementedError("Dynamic split by size not supported")
        else:
            curr = 0
            for size in split_size_or_sections:
                end = curr + size
                slices.append(self.slice(x, dim, curr, end))
                curr = end
        
        return self.world.tuple(slices)
        
    def select(self, x, dim, index):
        sliced = self.slice(x, dim, index, index + 1, 1)
        return self.squeeze(sliced, dim)

    def clone(self, x): return x
    def copy(self, x): return x
