from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import sys
import traceback

from mimir_frontend.model_export import export_spec_from_module, load_python_module
from mimir_frontend.utils import model_to_mimir


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KERNELBENCH_DIR = REPO_ROOT / "KernelBench" / "KernelBench" / "level3"
DEFAULT_OUT = REPO_ROOT / "docs" / "kernelbench_level3_report.md"


@dataclass
class ModelResult:
    name: str
    status: str
    blocker: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run KernelBench level3 models through MimIR export and report blockers.")
    parser.add_argument("--kernelbench-dir", type=Path, default=DEFAULT_KERNELBENCH_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-depth", type=int, default=80)
    parser.add_argument("--compile-phase", choices=["high_level", "default"], default="high_level")
    return parser.parse_args()


def summarize_exception(exc: BaseException) -> str:
    text = str(exc).strip().replace("\n", " ")
    if not text:
        text = type(exc).__name__
    text = re.sub(r"\s+", " ", text)
    return text


def collect_results(kernelbench_dir: Path, compile_phase: str, max_depth: int) -> list[ModelResult]:
    results: list[ModelResult] = []
    files = sorted(p for p in kernelbench_dir.glob("*.py") if p.is_file() and not p.name.startswith("_"))
    for path in files:
        try:
            module = load_python_module(path)
            spec = export_spec_from_module(module)
            model_to_mimir(
                spec.model,
                spec.input_shapes,
                compile_phase=compile_phase or spec.compile_phase,
                name=spec.name,
                max_depth=max_depth,
            )
            results.append(ModelResult(path.stem, "SUCCESS", ""))
        except Exception as exc:
            results.append(ModelResult(path.stem, "FAILED", summarize_exception(exc)))
    return results


def render_report(results: list[ModelResult], kernelbench_dir: Path) -> str:
    total = len(results)
    success = sum(1 for r in results if r.status == "SUCCESS")
    failed = total - success
    try:
        display_kernelbench_dir = kernelbench_dir.resolve().relative_to(REPO_ROOT.resolve())
    except ValueError:
        display_kernelbench_dir = kernelbench_dir
    lines = []
    lines.append("# KernelBench Level 3 MimIR 导出报告")
    lines.append("")
    lines.append(f"- KernelBench 目录: `{display_kernelbench_dir}`")
    lines.append("- 导出入口: `scripts/export_models_to_mimir.py`")
    lines.append("- 模型约定: `Model` + `get_inputs()` + `get_init_inputs()`，或 `export_to_mim = export(...)`")
    lines.append("")
    lines.append("## 总体统计")
    lines.append("")
    lines.append(f"- 模型总数: {total}")
    lines.append(f"- 成功: {success}")
    lines.append(f"- 失败: {failed}")
    lines.append("")
    lines.append("## 逐模型结果")
    lines.append("")
    lines.append("| 模型 | 状态 | 阻塞点 |")
    lines.append("| :--- | :--- | :--- |")
    for r in results:
        blocker = r.blocker.replace("|", "\\|")
        lines.append(f"| `{r.name}` | {r.status} | {blocker or '-'} |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    results = collect_results(args.kernelbench_dir, args.compile_phase, args.max_depth)
    report = render_report(results, args.kernelbench_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
