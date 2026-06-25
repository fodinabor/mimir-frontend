import mim
import struct
from mim._plugins.affine import affine
from mim._plugins.math import math, _math_arith, _math_extrema, _math_cmp, _math_exp, _math_tri, _math_rt
from mim._plugins.tensor import tensor
from mim._plugins.core import core, _core_bit1, _core_bit2

from .shape_rules import ShapeRules
from . import expr

class OperatorLibrary:
    def __init__(self, world: mim.World):
        self.world = world
        self.rules = ShapeRules(world)
        self.f32_config = world.annex(math.f32.value)
        self.F32 = world.annex(math.F32.value)
        self.Bool = world.type_bool()
        self.mode0 = world.lit_nat_0()
        self.sym_map = {} # Mapping from symbolic name to MimIR Nat variable
        self._shape_cache: dict[mim.Def, list[mim.Def]] = {}

        
        def bind_math_axm(axm_enum):
            axm = world.annex(axm_enum.value)
            # %_math_arith.add {pe} mode
            axm = world.app(axm, self.f32_config)
            return world.app(axm, self.mode0)

        # Arithmetic
        self.f32_add_axm = bind_math_axm(_math_arith.add)
        self.f32_sub_axm = bind_math_axm(_math_arith.sub)
        self.f32_mul_axm = bind_math_axm(_math_arith.mul)
        self.f32_div_axm = bind_math_axm(_math_arith.div)
        
        # Extrema
        self.f32_max_axm = bind_math_axm(_math_extrema.fmax)
        self.f32_min_axm = bind_math_axm(_math_extrema.fmin)
        
        # Comparisons
        self.f32_eq_axm = bind_math_axm(_math_cmp.e)
        self.f32_ne_axm = bind_math_axm(_math_cmp.ne)
        self.f32_lt_axm = bind_math_axm(_math_cmp.l)
        self.f32_le_axm = bind_math_axm(_math_cmp.le)
        self.f32_gt_axm = bind_math_axm(_math_cmp.g)
        self.f32_ge_axm = bind_math_axm(_math_cmp.ge)
        
        # Unary
        self.f32_exp_axm = bind_math_axm(_math_exp.exp)
        self.f32_log_axm = bind_math_axm(_math_exp.log)
        self.f32_tanh_axm = bind_math_axm(_math_tri.tanh)
        self.f32_sqrt_axm = bind_math_axm(_math_rt.sq)
        self.f32_pow_axm = bind_math_axm(math.pow)
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
            
        self.bool_and_axm = bind_bit_axm(_core_bit2.and_)
        self.bool_not_axm = bind_bit_axm(_core_bit1.neg)

    def _rank_and_shape(self, tensor_def):
        dims = self.shape_of(tensor_def)
        return self.world.lit_nat(len(dims)), self.world.tuple(dims)

    def shape_of(self, value):
        """
        Unified way to retrieve shape dims from a MimIR Def or PyTorch object.
        Priority:
        1. Explicitly mapped symbolic dims (for inputs)
        2. Local shape cache (for derived tensors, preserves 1-dims)
        3. MimIR Type system (Arr arity)
        4. PyTorch metadata (FakeTensor)
        """
        if isinstance(value, mim.Def):
            # 1. Check symbolic map for inputs
            if hasattr(self, "input_to_syms") and value in self.input_to_syms:
                sym_names = self.input_to_syms[value]
                dims = self._shape_dims(value)
                final_dims = []
                for i, name in enumerate(sym_names):
                    if name is not None and name in self.sym_map:
                        final_dims.append(self.sym_map[name])
                    else:
                        final_dims.append(dims[i])
                return final_dims

            # 2. Check cache FIRST to preserve singleton (1) dimensions
            # because MimIR type system normalizes them away.
            cached = self._shape_cache.get(value)
            if cached is not None:
                return list(cached)

            # 3. Try type system
            dims = self._shape_dims(value)
            return dims
        
        # 4. Fallback to metadata
        if hasattr(value, "meta") and isinstance(value.meta, dict) and "val" in value.meta:
            return self.shape_of(value.meta["val"])
        if hasattr(value, "shape"):
             return [self.world.lit_nat(d) if isinstance(d, int) else d for d in value.shape]
             
        raise TypeError(f"shape_of does not support {type(value)}")

    def _remember_shape(self, value, dims):
        if isinstance(value, mim.Def):
            normalized = []
            for dim in dims:
                if isinstance(dim, int):
                    normalized.append(self.world.lit_nat(dim))
                else:
                    normalized.append(dim)
            self._shape_cache[value] = normalized
        return value

    def _shape_dims(self, tensor_def):
        dims = []
        tensor_type = tensor_def.type()
        while isinstance(tensor_type, mim.Seq):
            dims.append(tensor_type.arity())
            tensor_type = tensor_type.body()
        return dims

    def _apply_grouped(self, callee, args):
        return self.world.app(callee, self.world.tuple(args))

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
        """
        Translates to MimIR elementwise binary operation:
            %tensor.binary @(T_in, T_in, T_out) op @(rank, shape) (lhs, rhs)
        Example IR:
            %tensor.binary (%math.F (23, 8)) (%_math_arith.add (23, 8) 0) (2, (10, 20)) (lhs, rhs)
        """
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
            
        # Broadcasting logic
        s_lhs_dims = self.shape_of(lhs)
        s_rhs_dims = self.shape_of(rhs)
        
        if not self.rules.same_shape(s_lhs_dims, s_rhs_dims):
            output_dims = self.rules.broadcast_shape(s_lhs_dims, s_rhs_dims)
            if not self.rules.same_shape(s_lhs_dims, output_dims):
                lhs = self.expand(lhs, output_dims)
            if not self.rules.same_shape(s_rhs_dims, output_dims):
                rhs = self.expand(rhs, output_dims)

        rank, shape = self._rank_and_shape(lhs)
        callee = self.world.annex(tensor.binary.value)
        callee = self._apply_grouped(callee, [in_type, in_type, out_type])
        callee = self.world.app(callee, op)
        callee = self._apply_grouped(callee, [rank, shape])
        res = self.world.app(callee, [lhs, rhs])
        
        output_dims = self.rules.broadcast_shape(s_lhs_dims, s_rhs_dims)
        return self._remember_shape(res, output_dims)

    def compare(self, op, lhs, rhs):
        return self.binary(op, lhs, rhs, out_type=self.Bool)

    def unary(self, op, input, out_type=None):
        """
        Translates to MimIR elementwise unary operation:
            %tensor.unary @(T_in, T_out) op @(rank, shape) input
        """
        in_type = self._tensor_element_type(input)
        if out_type is None:
            out_type = in_type
        rank, shape = self._rank_and_shape(input)
        res = self._unary_with_types(in_type, out_type, op, input, rank, shape)
        return self._remember_shape(res, self.shape_of(input))

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
        """
        Constructs a MimIR lambda (anonymous function) to map over elements.
        Example: `lam v: %_math_extrema.fmax (v, 0.0)`
        """
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
    def pow(self, lhs, rhs): return self.binary(self.f32_pow_axm, lhs, rhs)
    
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
        """
        Translates torch.where(cond, x, y) into MimIR's ternary selection:
            %tensor.select @T @(rank, shape) (cond, x, y)
        """
        cond_dims = self.shape_of(cond)
        x_dims = self.shape_of(x)
        y_dims = self.shape_of(y)
        output_dims = self.rules.broadcast_shape(self.rules.broadcast_shape(cond_dims, x_dims), y_dims)

        if not self.rules.same_shape(cond_dims, output_dims):
            cond = self.expand(cond, output_dims)
        if not self.rules.same_shape(x_dims, output_dims):
            x = self.expand(x, output_dims)
        if not self.rules.same_shape(y_dims, output_dims):
            y = self.expand(y, output_dims)

        tensor_type = self._tensor_element_type(x)
        rank = self.world.lit_nat(len(output_dims))
        shape = self.world.tuple(output_dims)
        callee = self.world.annex(tensor.select.value)
        callee = self.world.app(callee, tensor_type)
        callee = self._apply_grouped(callee, [rank, shape])
        result = self._apply_grouped(callee, [cond, x, y])
        return self._remember_shape(result, output_dims)

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
        """
        Translates torch.expand to %tensor.broadcast_in_dim or %tensor.broadcast.
        """
        in_dims = self.shape_of(input)
        in_rank_val = len(in_dims)

        shape = list(shape)
        out_rank_val = len(shape)
        rank_offset = out_rank_val - in_rank_val
        resolved_shape = []
        for i, dim in enumerate(shape):
            if not (isinstance(dim, int) and dim == -1):
                resolved_shape.append(dim)
                continue
            input_index = i - rank_offset
            if input_index < 0 and i < in_rank_val:
                input_index = i
            if input_index < 0 or input_index >= in_rank_val:
                raise ValueError(f"cannot infer expand dimension {i} from input rank {in_rank_val}")
            resolved_shape.append(in_dims[input_index])

        shape = resolved_shape
        out_shape_tuple, _ = self._extract_shape(shape)
        out_rank = self.world.lit_nat(out_rank_val)
        
        if in_dims == shape:
            return input

        elem_type = self._tensor_element_type(input)

        if in_rank_val == 0:
            callee = self.world.annex(tensor.map.value)
            callee = self.world.app(callee, self.world.tuple([elem_type, self.world.lit_nat(0), self.world.tuple([])]))
            lam = self.world.mut_lam(self.world.sigma([]), elem_type)
            lam.set_body(True, input)
            callee = self.world.app(callee, lam)
            callee = self.world.app(callee, self.world.tuple([out_rank, out_shape_tuple]))
            result = self.world.app(callee, self.world.tuple([]))
            return self._remember_shape(result, shape)

        _, in_shape_tuple = self._rank_and_shape(input)
        if in_rank_val == out_rank_val:
            callee = self.world.annex(tensor.broadcast.value)
            callee = self._apply_grouped(callee, [elem_type, out_rank])
            result = self.world.app(callee, [in_shape_tuple, out_shape_tuple, input])
        else:
            callee = self.world.annex(tensor.broadcast_in_dim.value)
            callee = self._apply_grouped(callee, [elem_type, self.world.lit_nat(in_rank_val), out_rank])
            
            idx_t = self.world.type_idx(out_rank)
            offset = out_rank_val - in_rank_val
            index_mapping = [self.world.lit(idx_t, offset + i) for i in range(in_rank_val)]
            index_tuple = self.world.tuple(index_mapping)
            
            result = self.world.app(callee, [in_shape_tuple, out_shape_tuple, input, index_tuple])
        return self._remember_shape(result, shape)

    def full(self, shape, fill_value, dtype=None):
        """
        Translates torch.full to a 0-input map (%tensor.map with ni=0).
        """
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
            
        out_shape_tuple, out_rank_val = self._extract_shape(shape)
        out_rank = self.world.lit_nat(out_rank_val)
        
        callee = self.world.annex(tensor.map.value)
        ni = self.world.lit_nat(0)
        Is = self.world.tuple([])
        callee = self.world.app(callee, self.world.tuple([elem_type, ni, Is]))
        
        lam = self.world.mut_lam(self.world.sigma([]), elem_type)
        lam.set_body(True, scalar_def)
        
        callee = self.world.app(callee, lam)
        callee = self.world.app(callee, self.world.tuple([out_rank, out_shape_tuple]))
        
        input_is = self.world.tuple([])
        result = self.world.app(callee, input_is)
        return self._remember_shape(result, shape)

    def _reduce_aff(self, input, output_type, reducer, init, dim=None, keepdim=False, return_shape=False):
        """
        Translates reduction operations into `%tensor.map_reduce_aff`.
        Uses ShapeRules.reduce_shape_spec as the canonical source for reduce shape invariants.
        """
        input_dims = self.shape_of(input)
        input_rank = len(input_dims)
        spec = self.rules.reduce_shape_spec(input_dims, dim=dim, keepdim=keepdim)

        output_rank = len(spec.output_dims)
        reduce_rank = len(spec.reduce_dims)
        total_rank = output_rank + reduce_rank

        callee = self.world.annex(tensor.map_reduce_aff.value)
        callee = self.world.app(callee, self.world.lit_nat(1)) # nis = 1 input tensor
        callee = self._apply_grouped(callee, [output_type, self.world.lit_nat(output_rank), self.world.lit_nat(reduce_rank)])
        callee = self._apply_grouped(callee, [self.world.tuple(spec.output_dims), self.world.tuple(spec.loop_dims)])
        
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
                [self._affine_projection_lam(total_rank, input_rank, spec.input_projections)]
            ),
        )
        result = self.world.app(callee, self.world.tuple([input]))
        
        # 2. Record the truth of the resulting shape
        self._remember_shape(result, spec.output_dims)
        
        if return_shape:
            return result, spec.output_dims
        return result

    def sum(self, input, dim=None, keepdim=False):
        """
        Translates to a summation via `%tensor.map_reduce_aff`.
        """
        return self._reduce_aff(input, self.F32, self._f32_reduce_lambda(self.f32_add_axm), self._f32_float_lit(0.0), dim=dim, keepdim=keepdim)

    def amax(self, input, dim=None, keepdim=False):
        """
        Translates to maximum reduction via `%tensor.map_reduce_aff`.
        """
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
        """
        Translates to mean reduction.
        """
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
        res = self._unary_with_types(
            pair_type,
            self.F32,
            self._f32_pair_to_mean_lambda(pair_type),
            reduced,
            rank,
            shape,
        )
        return self._remember_shape(res, output_dims)

    def _f32_var_mean_reduce_lambda(self, acc_type):
        """
        Creates the reduction lambda for var_mean which maintains (sum, sum_sq, count).
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
        self._remember_shape(var_tensor, output_dims)
        mean_tensor = self._unary_with_types(acc_type, self.F32, self._f32_acc_to_var_mean(acc_type, extract_var=False), reduced, rank, shape)
        self._remember_shape(mean_tensor, output_dims)
        
        return self.world.tuple([var_tensor, mean_tensor])

    # Linear Algebra
    def mm(self, lhs, rhs):
        # Ring: [T: *, _0: T, add: [T, T] -> T, mul: [T, T] -> T]
        ring = self.world.tuple([self.F32, self._f32_float_lit(0.0), self.f32_add_axm, self.f32_mul_axm])
        # 0x5463d44130001300 is product_2d
        callee = self.world.annex(0x5463d44130001300)
        callee = self.world.app(callee, ring)
        return self.world.implicit_app(callee, [lhs, rhs])

    def convolution(self, x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        raise NotImplementedError("aten.convolution is not implemented")

    # Injective
    def reshape(self, x, shape):
        """
        Translates to `%tensor.reshape`.
        """
        in_rank, in_shape_tuple = self._rank_and_shape(x)
        out_shape_tuple, out_rank_val = self._extract_shape(shape)
        out_rank = self.world.lit_nat(out_rank_val)
        elem_t = self._tensor_element_type(x)

        callee = self.world.annex(tensor.reshape.value)
        callee = self._apply_grouped(callee, [elem_t, in_rank, out_rank])
        callee = self.world.app(callee, in_shape_tuple)
        callee = self.world.app(callee, out_shape_tuple)
        result = self.world.app(callee, x)
        return self._remember_shape(result, list(shape))

    def view(self, x, shape):
        return self.reshape(x, shape)

    def slice(self, x, dim, start, end, step=1):
        """
        Translates to `%tensor.slice`.
        """
        rank, in_shape_tuple = self._rank_and_shape(x)
        in_dims = self.shape_of(x)
        rank_val = len(in_dims)
        elem_t = self._tensor_element_type(x)

        if dim < 0: dim += rank_val
        
        # 1. Canonical shape transformation
        out_dims = self.rules.slice_shape(in_dims, dim, start, end, step)
        
        # 2. Prep start/step tuples for MimIR
        starts = [self.world.lit_nat(0)] * rank_val
        steps = [self.world.lit_nat(1)] * rank_val
        
        starts[dim] = self.world.lit_nat(start) if isinstance(start, int) else start
        steps[dim] = self.world.lit_nat(step) if isinstance(step, int) else (self.world.lit_nat(1) if step is None else step)

        callee = self.world.annex(tensor.slice.value)
        callee = self._apply_grouped(callee, [elem_t, rank])
        callee = self.world.app(callee, in_shape_tuple)
        
        callee = self.world.app(callee, self.world.tuple([
            self.world.tuple(starts),
            self.world.tuple(steps),
            self.world.tuple(out_dims)
        ]))
        result = self.world.app(callee, x)
        return self._remember_shape(result, out_dims)

    def cat(self, tensors, dim=0):
        """
        Translates to `%tensor.concat`.
        """
        num_inputs = tensors.num_projs()
        first_tensor = tensors.proj(num_inputs, 0)
        rank, _ = self._rank_and_shape(first_tensor)
        rank_val = len(self._shape_dims(first_tensor))
        elem_t = self._tensor_element_type(first_tensor)
        
        if dim < 0: dim += rank_val
        
        # 1. Canonical shape transformation
        input_shapes = []
        input_dims_list = []
        for i in range(num_inputs):
            t = tensors.proj(num_inputs, i)
            input_dims = self.shape_of(t)
            input_dims_list.append(input_dims)
            input_shapes.append(self.world.tuple(input_dims))
            
        out_dims = self.rules.concat_shape(input_dims_list, dim)
        
        callee = self.world.annex(tensor.concat.value)
        callee = self._apply_grouped(callee, [elem_t, self.world.lit_nat(num_inputs), rank])
        
        idx_t = self.world.type_idx(rank)
        ax = self.world.lit(idx_t, dim)
        callee = self.world.app(callee, ax)
        
        callee = self.world.app(callee, self.world.tuple(input_shapes))
        result = self.world.app(callee, tensors)
        return self._remember_shape(result, out_dims)

    def transpose(self, x, permutation):
        """
        Translates to `%tensor.transpose`.
        """
        rank_val, in_shape_tuple = self._rank_and_shape(x)
        in_dims = self.shape_of(x)
        elem_t = self._tensor_element_type(x)
        
        # Canonical shape transformation
        out_dims = self.rules.transpose_shape(in_dims, permutation)
        
        idx_t = self.world.type_idx(rank_val)
        perm_mim = self.world.tuple([self.world.lit(idx_t, p) for p in permutation])
        
        callee = self.world.annex(tensor.transpose.value)
        callee = self._apply_grouped(callee, [elem_t, rank_val, in_shape_tuple])
        result = self.world.app(callee, [x, perm_mim])
        return self._remember_shape(result, out_dims)

    def _is_one(self, d):
        if isinstance(d, int): return d == 1
        return d == self.world.lit_nat(1)

    def squeeze(self, x, dim=None):
        """
        Translates to `reshape` with the canonical `squeeze_shape`.
        """
        in_dims = self.shape_of(x)
        out_dims = self.rules.squeeze_shape(in_dims, dim)
        return self.reshape(x, out_dims)

    def unsqueeze(self, x, dim):
        """
        Translates to `reshape` with the canonical `unsqueeze_shape`.
        """
        in_dims = self.shape_of(x)
        out_dims = self.rules.unsqueeze_shape(in_dims, dim)
        return self.reshape(x, out_dims)

    def split(self, x, split_size_or_sections, dim=0):
        """
        Translates to multiple `slice` operations.
        """
        in_dims = self.shape_of(x)
        rank_val = len(in_dims)
        if dim < 0: dim += rank_val
        
        extent = in_dims[dim]
        extent_val = self.rules._dim_literal_value(extent)
        
        output_shapes = self.rules.split_shapes(in_dims, split_size_or_sections, dim)
        slices = []
        curr = 0
        if isinstance(split_size_or_sections, int):
            split_size = split_size_or_sections
            if extent_val is None:
                raise NotImplementedError("Dynamic split by size not supported")
            while curr < extent_val:
                end = min(curr + split_size, extent_val)
                slices.append(self.slice(x, dim, curr, end))
                curr = end
        else:
            for size, out_shape in zip(split_size_or_sections, output_shapes):
                end = curr + size
                part = self.slice(x, dim, curr, end)
                self._remember_shape(part, out_shape)
                slices.append(part)
                curr = end
        
        return self.world.tuple(slices)
        
    def select(self, x, dim, index):
        """
        Translates to `slice` followed by `squeeze`.
        """
        # slice(index, index + 1) then squeeze(dim)
        sliced = self.slice(x, dim, index, index + 1, 1)
        result = self.squeeze(sliced, dim)
        return self._remember_shape(result, self.rules.select_shape(self.shape_of(x), dim))

    def clone(self, x): return x
    def copy(self, x): return x
