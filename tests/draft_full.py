import torch
from torch import fx
class Model(torch.nn.Module):
    def forward(self, x):
        return torch.full(x.shape, 5.0)

model = Model()
traced = fx.symbolic_trace(model)
print(traced.graph)
