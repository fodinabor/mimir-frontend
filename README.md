# Mimir Frontend

A PyTorch FX graph importer for MimIR.

## Setup

This project vendors its tested MimIR revision as a submodule. Clone with submodules:

```bash
git clone --recursive git@github.com:hsqStephenZhang/mimir-frontend.git
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

Use `uv run --no-sync pytest -q` after bootstrap:

```bash
uv run --no-sync pytest -q
```

## Fresh Clone Smoke Test

The expected mentor workflow is:

```bash
git clone --recursive git@github.com:hsqStephenZhang/mimir-frontend.git
cd mimir-frontend
./scripts/bootstrap_mimir.sh
uv run --no-sync python -c 'import mim; print("mim import ok")'
uv run --no-sync pytest -q
```

This was verified from a clean clone with:

```text
166 passed, 6 skipped
```

If pytest reports a cache warning about `.pytest_cache` under a sandboxed runner,
it is an environment permission warning and does not affect the test result.

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
