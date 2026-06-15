# Mimir Frontend

A PyTorch FX graph importer for MimIR.

## Setup

This project uses `uv`. To set up the environment:

```bash
uv sync
```

Note: Ensure that `MimIR` and `pytorch` are available at the relative paths specified in `pyproject.toml`.

## Running Tests

Use `uv run pytest` to run the tests:

```bash
uv run pytest tests/test_basic.py
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
