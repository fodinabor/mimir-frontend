from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
from typing import Sequence

import torch

from .utils import Shape, model_to_mimir


@dataclass(frozen=True)
class ExportSpec:
    model: torch.nn.Module
    input_shapes: Sequence[Shape]
    name: str = "mimir_module"
    compile_phase: str = "high_level"


def export(
    model: torch.nn.Module,
    input_shapes: Sequence[Shape],
    *,
    name: str | None = None,
    compile_phase: str = "high_level",
) -> ExportSpec:
    return ExportSpec(
        model=model,
        input_shapes=input_shapes,
        name=name or model.__class__.__name__.lower(),
        compile_phase=compile_phase,
    )


def load_python_module(path: str | Path) -> ModuleType:
    path = Path(path)
    module_name = f"_mimir_export_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import Python module from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def export_spec_from_module(module: ModuleType) -> ExportSpec:
    if hasattr(module, "export_to_mim"):
        value = module.export_to_mim
        if isinstance(value, ExportSpec):
            return value

        if isinstance(value, torch.nn.Module):
            if not hasattr(module, "input_shapes"):
                raise AttributeError(
                    f"{module.__name__} uses a raw torch.nn.Module export_to_mim; define input_shapes as well"
                )
            name = getattr(module, "export_name", value.__class__.__name__.lower())
            compile_phase = getattr(module, "compile_phase", "high_level")
            return ExportSpec(value, module.input_shapes, name=name, compile_phase=compile_phase)

        raise TypeError("export_to_mim must be an ExportSpec or torch.nn.Module")

    if hasattr(module, "Model") and hasattr(module, "get_inputs") and hasattr(module, "get_init_inputs"):
        init_inputs = list(module.get_init_inputs())
        model = module.Model(*init_inputs)
        inputs = module.get_inputs()
        if isinstance(inputs, tuple):
            inputs = list(inputs)
        if not isinstance(inputs, list):
            inputs = list(inputs)
        input_shapes = [tuple(t.shape) for t in inputs]
        name = getattr(module, "export_name", module.__name__.split(".")[-1].lower())
        compile_phase = getattr(module, "compile_phase", "high_level")
        return ExportSpec(model, input_shapes, name=name, compile_phase=compile_phase)

    raise AttributeError(f"{module.__name__} must define export_to_mim or Model/get_inputs/get_init_inputs")


def export_module_to_mimir(
    module_path: str | Path,
    *,
    compile_phase: str | None = None,
    name: str | None = None,
    max_depth: int = 100,
) -> str:
    module = load_python_module(module_path)
    spec = export_spec_from_module(module)
    return model_to_mimir(
        spec.model,
        spec.input_shapes,
        compile_phase=compile_phase or spec.compile_phase,
        name=name or spec.name,
        max_depth=max_depth,
    )
