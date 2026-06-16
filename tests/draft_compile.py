import torch
from mimir_frontend.utils import model_to_mimir

class Model(torch.nn.Module):
    def forward(self, x, y): 
        return torch.relu(x + y)

try:
    print("Testing high_level...")
    ir = model_to_mimir(Model(), [("n", 20), ("n", 20)], compile_phase="high_level")
    print(ir)
    print("Testing default...")
    ir_opt = model_to_mimir(Model(), [("n", 20), ("n", 20)], compile_phase="default")
    print(ir_opt)
    print("Success!")
except Exception as e:
    print(e)
