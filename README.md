# Mimir Frontend

A PyTorch FX graph importer for MimIR.

## Setup

This project vendors its tested MimIR revision as a submodule. Clone with submodules:

```bash
git clone --recursive <repo-url>
cd mimir-frontend
```

Then build the local MimIR Python binding and sync the Python environment:

```bash
./scripts/bootstrap_mimir.sh
```

The script initializes recursive submodules, creates a Python 3.14 `uv` venv,
builds MimIR's `mim_py` target, and runs `uv sync`.

The `mim` dependency is resolved from the local submodule build output:

```toml
mim = { path = "MimIR/build/mim_py_stage/main" }
```

After editing MimIR locally, rebuild and resync with the same command.

## Running Tests

Use `uv run pytest` to run the tests:

```bash
uv run pytest
```

## Architecture

- `src/mimir_frontend/translator.py`: The core `FXGraphTranslator` class.
- `src/mimir_frontend/operators.py`: MimIR operator definitions using the Python bindings.
- `tests/test_basic.py`: Basic verification tests.

## Operator Support

Currently supports:
- `torch.add`, `torch.mul`, `torch.sub`, `torch.div`
- `torch.relu`
- Basic shape propagation

TODO:
- Comprehensive `reduce_sum` implementation.
- `pooling` (pad + reduce + elementwise).
- Dynamic shape handling refinements.
