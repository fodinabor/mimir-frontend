import mim

def make_world():
    driver = mim.Driver()
    return driver.world()

world = make_world()
v1 = world.lit_nat(1)
v2 = world.lit_nat(2)

mim.Def.__add__ = lambda self, other: "added!"

try:
    print("v1 + v2 =", v1 + v2)
except Exception as e:
    print("Add failed:", e)

