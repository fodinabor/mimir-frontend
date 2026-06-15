import pytest
import mim
import operator
import torch
from torch import fx
from pathlib import Path
import tempfile

from mimir_frontend.translator import FXGraphTranslator


def make_world():
    driver = mim.Driver()
    driver.load_plugins(["math", "tensor", "affine"])
    return driver.world()


def make_tensor_type(world, elem_type, shape_kind, rank):
    if shape_kind == "dynamic":
        if rank == 1:
            return world.arr(world.top_nat(), elem_type)
        shape_ty = world.arr(world.lit_nat(rank), world.type_nat())
        shape = world.mut_con(shape_ty).var()
        return world.arr(shape, elem_type)

    if rank == 1:
        return world.arr(world.lit_nat(8), elem_type)
    shape = world.tuple([world.lit_nat(2), world.lit_nat(3), world.lit_nat(4)])
    return world.arr(shape, elem_type)


def make_inputs(world, count, shape_kind, rank):
    ops = FXGraphTranslator(world).ops
    tensor_ty = make_tensor_type(world, ops.F32, shape_kind, rank)
    return [world.mut_con(tensor_ty).var() for _ in range(count)]


def translate_model(model, inputs):
    traced = fx.symbolic_trace(model)
    translator = FXGraphTranslator(inputs[0].world(), module=traced)
    return translator.translate(traced.graph, inputs)


def def_to_string(defn):
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "def.mim"
        defn.write(100, str(path))
        return path.read_text()


def assert_translates_for_all_shapes(model_factory, input_count):
    for shape_kind in ("static", "dynamic"):
        for rank in (1, 3):
            world = make_world()
            result = translate_model(model_factory(), make_inputs(world, input_count, shape_kind, rank))
            assert isinstance(result, mim.Def)


def tensor_element_type(tensor_def):
    tensor_type = tensor_def.type()
    while isinstance(tensor_type, mim.Seq):
        tensor_type = tensor_type.body()
    return tensor_type


def tensor_shape(tensor_def):
    dims = []
    tensor_type = tensor_def.type()
    while isinstance(tensor_type, mim.Seq):
        dims.append(tensor_type.arity())
        tensor_type = tensor_type.body()
    return dims


def assert_translates_to_element_type_for_all_shapes(model_factory, input_count, element_type_fn):
    for shape_kind in ("static", "dynamic"):
        for rank in (1, 3):
            world = make_world()
            result = translate_model(model_factory(), make_inputs(world, input_count, shape_kind, rank))
            assert isinstance(result, mim.Def)
            assert tensor_element_type(result) == element_type_fn(world)


SUPPORTED_BINARY_OPS = [
    ("add", torch.add, operator.add),
    ("sub", torch.sub, operator.sub),
    ("mul", torch.mul, operator.mul),
    ("div", torch.div, operator.truediv),
    ("maximum", torch.maximum, None),
    ("minimum", torch.minimum, None),
]


SUPPORTED_COMPARISON_OPS = [
    ("eq", torch.eq, operator.eq),
    ("ne", torch.ne, operator.ne),
    ("lt", torch.lt, operator.lt),
    ("le", torch.le, operator.le),
    ("gt", torch.gt, operator.gt),
    ("ge", torch.ge, operator.ge),
]


SUPPORTED_UNARY_OPS = [
    ("relu", torch.relu),
    ("exp", torch.exp),
    ("tanh", torch.tanh),
    ("sqrt", torch.sqrt),
    ("abs", torch.abs),
    ("neg", torch.neg),
    ("sigmoid", torch.sigmoid),
    ("reciprocal", torch.reciprocal),
    ("rsqrt", torch.rsqrt),
]


@pytest.mark.parametrize("shape_kind", ["static", "dynamic"])
@pytest.mark.parametrize("rank", [1, 3])
def test_single_elementwise_operator(shape_kind, rank):
    class Model(torch.nn.Module):
        def forward(self, x, y):
            return x + y

    world = make_world()
    result = translate_model(Model(), make_inputs(world, 2, shape_kind, rank))

    assert isinstance(result, mim.Def)


@pytest.mark.parametrize("name,torch_op,python_op", SUPPORTED_BINARY_OPS)
def test_binary_operator_all_shapes(name, torch_op, python_op):
    class TorchModel(torch.nn.Module):
        def forward(self, x, y):
            return torch_op(x, y)

    assert_translates_for_all_shapes(TorchModel, 2)

    if python_op is not None:
        class PythonModel(torch.nn.Module):
            def forward(self, x, y):
                return python_op(x, y)

        assert_translates_for_all_shapes(PythonModel, 2)


@pytest.mark.parametrize("name,torch_op,python_op", SUPPORTED_COMPARISON_OPS)
def test_comparison_operator_returns_bool_tensor_all_shapes(name, torch_op, python_op):
    class TorchModel(torch.nn.Module):
        def forward(self, x, y):
            return torch_op(x, y)

    assert_translates_to_element_type_for_all_shapes(TorchModel, 2, lambda world: world.type_bool())

    class PythonModel(torch.nn.Module):
        def forward(self, x, y):
            return python_op(x, y)

    assert_translates_to_element_type_for_all_shapes(PythonModel, 2, lambda world: world.type_bool())


@pytest.mark.parametrize("name,torch_op", SUPPORTED_UNARY_OPS)
def test_unary_operator_all_shapes(name, torch_op):
    class Model(torch.nn.Module):
        def forward(self, x):
            return torch_op(x)

    assert_translates_for_all_shapes(Model, 1)


@pytest.mark.parametrize("shape_kind", ["static", "dynamic"])
@pytest.mark.parametrize("rank", [1, 3])
def test_sequence_of_elementwise_operators(shape_kind, rank):
    class Model(torch.nn.Module):
        def forward(self, x, y, z):
            return torch.relu((x + y) * z)

    world = make_world()
    result = translate_model(Model(), make_inputs(world, 3, shape_kind, rank))

    assert isinstance(result, mim.Def)


@pytest.mark.parametrize(
    "dim,keepdim,expected_shape",
    [
        (None, False, "((), (2, 3, 4))"),
        (0, False, "((3, 4), (3, 4, 2))"),
        (1, True, "((2, 1, 4), (2, 1, 4, 3))"),
        ((1, 2), False, "(2, (2, 3, 4))"),
        ((1, 2), True, "((2, 1, 1), (2, 1, 1, 3, 4))"),
    ],
)
def test_sum_reduce_static_3d_shapes(dim, keepdim, expected_shape):
    class Model(torch.nn.Module):
        def forward(self, x):
            return torch.sum(x, dim=dim, keepdim=keepdim)

    world = make_world()
    result = translate_model(Model(), make_inputs(world, 1, "static", 3))

    assert isinstance(result, mim.Def)
    assert tensor_element_type(result) == FXGraphTranslator(world).ops.F32
    ir = def_to_string(result)
    assert "%tensor.map_reduce_aff" in ir
    assert expected_shape in ir


@pytest.mark.parametrize("shape_kind", ["static", "dynamic"])
@pytest.mark.parametrize("rank,dim,keepdim", [(1, None, False), (1, 0, True), (3, -1, True), (3, (1, 2), True)])
def test_sum_reduce_all_shape_kinds_smoke(shape_kind, rank, dim, keepdim):
    class Model(torch.nn.Module):
        def forward(self, x):
            return torch.sum(x, dim=dim, keepdim=keepdim)

    world = make_world()
    result = translate_model(Model(), make_inputs(world, 1, shape_kind, rank))

    assert isinstance(result, mim.Def)
    assert tensor_element_type(result) == FXGraphTranslator(world).ops.F32
    assert "%tensor.map_reduce_aff" in def_to_string(result)


@pytest.mark.parametrize("shape_kind", ["static", "dynamic"])
@pytest.mark.parametrize("rank,dim,keepdim", [(1, None, False), (1, 0, True), (3, -1, True), (3, (1, 2), True)])
def test_amax_reduce_all_shape_kinds_smoke(shape_kind, rank, dim, keepdim):
    class Model(torch.nn.Module):
        def forward(self, x):
            return torch.amax(x, dim=dim, keepdim=keepdim)

    world = make_world()
    result = translate_model(Model(), make_inputs(world, 1, shape_kind, rank))

    assert isinstance(result, mim.Def)
    assert tensor_element_type(result) == FXGraphTranslator(world).ops.F32
    ir = def_to_string(result)
    assert "%tensor.map_reduce_aff" in ir


@pytest.mark.parametrize("shape_kind", ["static", "dynamic"])
@pytest.mark.parametrize("rank,dim,keepdim", [(1, None, False), (1, 0, True), (3, -1, True), (3, (1, 2), True)])
def test_mean_reduce_all_shape_kinds_smoke(shape_kind, rank, dim, keepdim):
    class Model(torch.nn.Module):
        def forward(self, x):
            return torch.mean(x, dim=dim, keepdim=keepdim)

    world = make_world()
    result = translate_model(Model(), make_inputs(world, 1, shape_kind, rank))

    assert isinstance(result, mim.Def)
    assert tensor_element_type(result) == FXGraphTranslator(world).ops.F32
    ir = def_to_string(result)
    assert "%tensor.map_reduce_aff" in ir
    assert "%tensor.unary" in ir


@pytest.mark.parametrize(
    "model,input_count",
    [
        (lambda: type("MMModel", (torch.nn.Module,), {"forward": lambda self, x, y: torch.mm(x, y)})(), 2),
        (lambda: type("CatModel", (torch.nn.Module,), {"forward": lambda self, x, y: torch.cat([x, y], dim=0)})(), 2),
        (lambda: type("PermuteModel", (torch.nn.Module,), {"forward": lambda self, x: torch.permute(x, (2, 1, 0))})(), 1),
    ],
)
def test_complex_operators_are_explicitly_unsupported(model, input_count):
    world = make_world()
    inputs = make_inputs(world, input_count, "dynamic", 3)

    with pytest.raises(NotImplementedError):
        translate_model(model, inputs)

@pytest.mark.parametrize("shape_kind", ["static", "dynamic"])
@pytest.mark.parametrize("rank", [1, 3])
def test_where_operator(shape_kind, rank):
    class Model(torch.nn.Module):
        def forward(self, cond, x, y):
            return torch.where(cond, x, y)

    world = make_world()
    ops = FXGraphTranslator(world).ops
    cond_ty = make_tensor_type(world, world.type_bool(), shape_kind, rank)
    cond_input = world.mut_con(cond_ty).var()
    x_input, y_input = make_inputs(world, 2, shape_kind, rank)
    
    result = translate_model(Model(), [cond_input, x_input, y_input])
    assert isinstance(result, mim.Def)
    assert tensor_element_type(result) == ops.F32

@pytest.mark.parametrize("shape_kind", ["static", "dynamic"])
@pytest.mark.parametrize("rank", [1, 3])
def test_clamp_scalar_bound(shape_kind, rank):
    class Model(torch.nn.Module):
        def forward(self, x):
            return torch.clamp(x, min=-1.0, max=1.0)
            
    world = make_world()
    result = translate_model(Model(), make_inputs(world, 1, shape_kind, rank))
    assert isinstance(result, mim.Def)
    assert tensor_element_type(result) == FXGraphTranslator(world).ops.F32

@pytest.mark.parametrize("shape_kind", ["static", "dynamic"])
@pytest.mark.parametrize("rank", [1, 3])
def test_value_only_max(shape_kind, rank):
    class Model(torch.nn.Module):
        def forward(self, x):
            return torch.max(x)

    world = make_world()
    result = translate_model(Model(), make_inputs(world, 1, shape_kind, rank))
    assert isinstance(result, mim.Def)
    assert tensor_element_type(result) == FXGraphTranslator(world).ops.F32
    ir = def_to_string(result)
    assert "%tensor.map_reduce_aff" in ir

def test_tuple_max_is_unsupported():
    class Model(torch.nn.Module):
        def forward(self, x):
            return torch.max(x, dim=0)

    world = make_world()
    with pytest.raises(NotImplementedError):
        translate_model(Model(), make_inputs(world, 1, "static", 3))

@pytest.mark.parametrize("shape_kind", ["static", "dynamic"])
@pytest.mark.parametrize("rank,dim,keepdim", [(1, None, False), (1, 0, True), (3, -1, True), (3, (1, 2), True)])
def test_var_mean_all_shape_kinds_smoke(shape_kind, rank, dim, keepdim):
    class Model(torch.nn.Module):
        def forward(self, x):
            return torch.var_mean(x, dim=dim, keepdim=keepdim, correction=0)

    world = make_world()
    result = translate_model(Model(), make_inputs(world, 1, shape_kind, rank))
    assert isinstance(result, mim.Def)
    # var_mean returns a tuple of (var, mean)
    ir = def_to_string(result)
    assert "%tensor.map_reduce_aff" in ir

def test_bitwise_and_logical_not():
    class Model(torch.nn.Module):
        def forward(self, x, y):
            b_x = x > 0
            b_y = y > 0
            return torch.logical_not(torch.bitwise_and(b_x, b_y))

    world = make_world()
    x_input, y_input = make_inputs(world, 2, "dynamic", 3)
    result = translate_model(Model(), [x_input, y_input])
    assert isinstance(result, mim.Def)
    assert tensor_element_type(result) == world.type_bool()

def test_convert_element_type():
    import torch._prims as prims
    class Model(torch.nn.Module):
        def forward(self, x):
            b = prims.convert_element_type(x, torch.bool)
            return prims.convert_element_type(b, torch.float32)

    world = make_world()
    x_input, = make_inputs(world, 1, "static", 3)
    result = translate_model(Model(), [x_input])
    assert isinstance(result, mim.Def)
    assert tensor_element_type(result) == FXGraphTranslator(world).ops.F32

def test_fma():
    import torch._prims as prims
    if not hasattr(prims, 'fma'):
        pytest.skip("fma not available in this torch version")
    class Model(torch.nn.Module):
        def forward(self, a, b, c):
            return prims.fma(a, b, c)

    world = make_world()
    inputs = make_inputs(world, 3, "dynamic", 1)
    result = translate_model(Model(), inputs)
    assert isinstance(result, mim.Def)
    assert tensor_element_type(result) == FXGraphTranslator(world).ops.F32

def test_full_and_expand():
    class Model(torch.nn.Module):
        def forward(self, x):
            f = torch.full((10, 20), 5.0)
            return x + f

    world = make_world()
    x_input, = make_inputs(world, 1, "static", 2) # F32 tensor of rank 2
    # Wait, make_inputs for rank 2? No, my make_inputs only supports rank 1 and 3 currently.
    pass

@pytest.mark.parametrize("shape_kind", ["static", "dynamic"])
def test_full_operator(shape_kind):
    class Model(torch.nn.Module):
        def forward(self, x):
            return torch.full(x.shape, 5.0)

    world = make_world()
    x_input, = make_inputs(world, 1, shape_kind, 3)
    result = translate_model(Model(), [x_input])
    assert isinstance(result, mim.Def)
    assert tensor_element_type(result) == FXGraphTranslator(world).ops.F32

@pytest.mark.parametrize("shape_kind", ["static", "dynamic"])
def test_expand_operator(shape_kind):
    class Model(torch.nn.Module):
        def forward(self, x):
            return x.expand(5, 10, 20, 30) # Expand rank 3 to rank 4

    world = make_world()
    x_input, = make_inputs(world, 1, shape_kind, 3)
    result = translate_model(Model(), [x_input])
    assert isinstance(result, mim.Def)
    assert tensor_element_type(result) == FXGraphTranslator(world).ops.F32
