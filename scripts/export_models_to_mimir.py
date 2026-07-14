from __future__ import annotations

import argparse
from pathlib import Path
import sys

from mimir_frontend.model_export import export_spec_from_module, load_python_module
from mimir_frontend.utils import model_to_mimir


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = REPO_ROOT / "models" / "py"
DEFAULT_OUT_DIR = REPO_ROOT / "models" / "mim"


def discover_model_files(paths: list[Path]) -> list[Path]:
    if not paths:
        paths = [DEFAULT_MODEL_DIR]

    files = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(p for p in path.glob("*.py") if not p.name.startswith("_")))
        else:
            files.append(path)
    return files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Python model files defining export_to_mim into MimIR files."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Model .py files or directories. Defaults to models/py.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--compile-phase",
        choices=["high_level", "default"],
        help="Override the compile phase declared by each model.",
    )
    parser.add_argument("--max-depth", type=int, default=100)
    parser.add_argument("--keep-going", action="store_true", help="Continue exporting other models after a failure.")
    return parser.parse_args()


def export_one(path: Path, out_dir: Path, compile_phase: str | None, max_depth: int) -> Path:
    module = load_python_module(path)
    spec = export_spec_from_module(module)
    ir = model_to_mimir(
        spec.model,
        spec.input_shapes,
        compile_phase=compile_phase or spec.compile_phase,
        name=spec.name,
        max_depth=max_depth,
    )

    out_path = out_dir / f"{spec.name}.mim"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(ir)
    return out_path


def main() -> int:
    args = parse_args()
    files = discover_model_files(args.paths)
    if not files:
        print("no model files found", file=sys.stderr)
        return 1

    failures = []
    for path in files:
        try:
            out_path = export_one(path, args.out_dir, args.compile_phase, args.max_depth)
            print(f"{path} -> {out_path}")
        except Exception as exc:
            failures.append((path, exc))
            print(f"failed to export {path}: {exc}", file=sys.stderr)
            if not args.keep_going:
                return 1

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
