from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile

import mim

from mimir_frontend.inductor_readable import (
    DEFAULT_INDUCTOR_LOG_ROOT,
    translate_inductor_readable,
    translate_inductor_readable_prefix,
)


def def_to_string(defn: mim.Def, max_depth: int) -> str:
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "dump.mim"
        defn.write(max_depth, str(path))
        return path.read_text()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump MimIR for an Inductor fx_graph_readable.py case.")
    parser.add_argument("case", help="Case name such as mlp_1, gcn_1, lstm_0, or a path to fx_graph_readable.py.")
    parser.add_argument("--root", type=Path, default=DEFAULT_INDUCTOR_LOG_ROOT)
    parser.add_argument("--partial", action="store_true", help="Print the last successfully translated node on failure.")
    parser.add_argument("--max-depth", type=int, default=100)
    parser.add_argument("--out", type=Path, help="Write IR to this file instead of stdout.")
    return parser.parse_args()


def emit(ir: str, out: Path | None):
    if out is None:
        print(ir)
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(ir)


def main() -> int:
    args = parse_args()

    try:
        result = translate_inductor_readable(args.case, root=args.root)
        emit(def_to_string(result, args.max_depth), args.out)
        return 0
    except Exception as exc:
        if not args.partial:
            print(f"translation failed at current frontier: {exc}", file=sys.stderr)
            print("rerun with --partial to dump the last successfully translated MimIR node", file=sys.stderr)
            return 1

        partial = translate_inductor_readable_prefix(args.case, root=args.root)
        if partial.frontier_node is not None:
            print(
                f"partial dump stopped before node {partial.frontier_node.name}: {partial.error}",
                file=sys.stderr,
            )
        if partial.result is None:
            print("no node was translated before the frontier", file=sys.stderr)
            return 1

        emit(def_to_string(partial.result, args.max_depth), args.out)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
