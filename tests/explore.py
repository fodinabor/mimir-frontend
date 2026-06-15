import mim
import torch
from torch import fx
from mimir_frontend.translator import FXGraphTranslator

def make_world():
    driver = mim.Driver()
    driver.load_plugins(["math", "tensor", "affine"])
    return driver.world()

world = make_world()
ops = FXGraphTranslator(world).ops
# test tuple extraction
tup = world.tuple([world.lit_nat(1), world.lit_nat(2)])
ext = world.extract(tup, world.lit_nat(0))
print("extract:", ext)
