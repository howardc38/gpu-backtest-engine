"""Keep example, benchmark, and deployment dependencies out of the engine core."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_core_has_no_example_tool_cli_or_workflow_dependencies():
    for path in (ROOT / "src/gpu_backtest/core").glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                assert not module.startswith(("gpu_backtest_examples", "gpu_backtest_tools")), path
                assert not module.startswith(("gpu_backtest.cli", "gpu_backtest.workflows")), path
                assert node.level < 2, path
            elif isinstance(node, ast.Import):
                assert not any(
                    a.name.startswith(("gpu_backtest_examples", "gpu_backtest_tools"))
                    for a in node.names
                ), path


def test_runtime_bundle_lists_only_its_explicit_packages():
    from gpu_backtest_tools.runpod.bundle import PACKAGE_FILES

    roots = {
        "gpu_backtest": ROOT / "src/gpu_backtest",
        "gpu_backtest_examples": ROOT / "examples/gpu_backtest_examples",
        "gpu_backtest_tools": ROOT / "tools/gpu_backtest_tools",
    }
    assert set(PACKAGE_FILES) == set(roots)
    for package, root in roots.items():
        actual = {p.relative_to(root).as_posix() for p in root.rglob("*.py")}
        listed = {p for p in PACKAGE_FILES[package] if p.endswith(".py")}
        assert listed == actual
        assert all(
            not p.startswith(("tests/", "docs/", "benchmarks/results/"))
            for p in PACKAGE_FILES[package]
        )
