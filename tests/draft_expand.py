import mim
from mim._plugins.tensor import tensor
from mim._plugins.math import math

def make_world():
    driver = mim.Driver()
    driver.load_plugins(["math", "tensor", "core", "affine", "tuple"])
    return driver.world()

world = make_world()
F32 = world.annex(math.F32.value)

r_in = world.lit_nat(1)
r_out = world.lit_nat(2)
s_in = world.tuple([world.lit_nat(3)])
s_out = world.tuple([world.lit_nat(5), world.lit_nat(3)])

input_tensor = world.mut_con(world.arr(s_in, F32)).var()

callee = world.annex(tensor.broadcast_in_dim.value)
callee = world.app(callee, world.tuple([F32, r_in, r_out]))

idx_t = world.type_idx(r_out)
try:
    idx_val = world.lit(idx_t, 1)
    index_tuple = world.tuple([idx_val])
    
    res = world.app(callee, [s_in, s_out, input_tensor, index_tuple])
    print("broadcast_in_dim Success! Type:", res.type())
except Exception as e:
    print(f"Error map: {e}")
