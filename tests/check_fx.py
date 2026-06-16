import torch
from torch import fx
import operator

class Model(torch.nn.Module):
    def forward(self, x):
        return x[2:5, :, 10:20]

model = Model()
traced = fx.symbolic_trace(model)
print("Graph for slicing:")
print(traced.graph)

class Model2(torch.nn.Module):
    def forward(self, x):
        return x[5]

model2 = Model2()
traced2 = fx.symbolic_trace(model2)
print("\nGraph for indexing:")
print(traced2.graph)
