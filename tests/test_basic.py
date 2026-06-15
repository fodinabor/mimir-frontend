import mim
import torch
from torch import fx
from mimir_frontend.translator import FXGraphTranslator, get_high_level_phase

def test_add_relu():
    print("Tracing model...")
    class Model(torch.nn.Module):
        def forward(self, x, y):
            return torch.relu(x + y)

    model = Model()
    graph = fx.symbolic_trace(model).graph
    print("Graph traced.")

    print("Initializing MimIR driver...")
    driver = mim.Driver()
    driver.load_plugin("math")
    driver.load_plugin("tensor")
    world = driver.world()
    print("MimIR world initialized.")

    # Define inputs in MimIR
    print("Defining inputs...")
    ops = FXGraphTranslator(world).ops
    F32 = ops.F32
    
    # Define rank-1 tensors with symbolic extent. `arr` expects a shape/arity
    # term, not a Nat literal typed by the top Nat value.
    tensor_ty = world.arr(world.top_nat(), F32)
    x_mim = world.mut_con(tensor_ty).var()
    y_mim = world.mut_con(tensor_ty).var()
    print("Inputs defined.")

    print("Translating graph...")
    translator = FXGraphTranslator(world)
    res_mim = translator.translate(graph, [x_mim, y_mim])
    print("Translation finished.")

    assert isinstance(res_mim, mim.Def)

    print("Translated successfully!")


def test_add_relu_3d_dynamic_shape():
    class Model(torch.nn.Module):
        def forward(self, x, y):
            return torch.relu(x + y)

    graph = fx.symbolic_trace(Model()).graph

    driver = mim.Driver()
    driver.load_plugin("math")
    driver.load_plugin("tensor")
    world = driver.world()

    ops = FXGraphTranslator(world).ops
    shape_ty = world.arr(world.lit_nat(3), world.type_nat())
    shape = world.mut_con(shape_ty).var()
    tensor_ty = world.arr(shape, ops.F32)
    x_mim = world.mut_con(tensor_ty).var()
    y_mim = world.mut_con(tensor_ty).var()

    res_mim = FXGraphTranslator(world).translate(graph, [x_mim, y_mim])

    assert isinstance(res_mim, mim.Def)

if __name__ == "__main__":
    test_add_relu()
