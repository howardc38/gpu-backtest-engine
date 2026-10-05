"""Check tracked release files against an explicit, reviewed content allowlist."""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {
    ".github/workflows/ci.yml",
    ".gitignore",
    "CONTRIBUTING.md",
    "LICENSE",
    "MANIFEST.in",
    "README.md",
    "benchmarks/results/rtx4090_rsi_20261003.json",
    "benchmarks/results/rtx4090_rsi_matched_billion_20261003.json",
    "benchmarks/results/runpod_validation_20261003.md",
    "docs/benchmarks.md",
    "docs/runpod.md",
    "docs/strategy.md",
    "examples/gpu_backtest_examples/__init__.py",
    "examples/gpu_backtest_examples/data.py",
    "examples/gpu_backtest_examples/rsi/README.md",
    "examples/gpu_backtest_examples/rsi/__init__.py",
    "examples/gpu_backtest_examples/rsi/config.json",
    "examples/gpu_backtest_examples/rsi/generate_data.py",
    "examples/gpu_backtest_examples/rsi/strategy.py",
    "examples/gpu_backtest_examples/rsi/synthetic.csv",
    "pyproject.toml",
    "scripts/check_release.py",
    "src/gpu_backtest/__init__.py",
    "src/gpu_backtest/__main__.py",
    "src/gpu_backtest/cli/__init__.py",
    "src/gpu_backtest/cli/backtest.py",
    "src/gpu_backtest/cli/options.py",
    "src/gpu_backtest/cli/tools.py",
    "src/gpu_backtest/core/__init__.py",
    "src/gpu_backtest/core/data.py",
    "src/gpu_backtest/core/engine.py",
    "src/gpu_backtest/core/grid.py",
    "src/gpu_backtest/core/indicators.py",
    "src/gpu_backtest/core/kernels.py",
    "src/gpu_backtest/core/output.py",
    "src/gpu_backtest/core/strategy.py",
    "tests/README.md",
    "tests/cpu/test_architecture.py",
    "tests/cpu/test_benchmark.py",
    "tests/cpu/test_cli.py",
    "tests/cpu/test_engine_contract.py",
    "tests/cpu/test_examples.py",
    "tests/cpu/test_gpu_check.py",
    "tests/cpu/test_runpod.py",
    "tests/cpu/test_strategy.py",
    "tests/cpu/test_tables.py",
    "tests/gpu/fixtures.py",
    "tests/gpu/simulator_cases.py",
    "tests/gpu/test_hardware.py",
    "tests/gpu/test_simulator.py",
    "tools/gpu_backtest_tools/__init__.py",
    "tools/gpu_backtest_tools/benchmarks/__init__.py",
    "tools/gpu_backtest_tools/benchmarks/cpu.py",
    "tools/gpu_backtest_tools/benchmarks/runner.py",
    "tools/gpu_backtest_tools/checks/__init__.py",
    "tools/gpu_backtest_tools/checks/gpu.py",
    "tools/gpu_backtest_tools/checks/reference.py",
    "tools/gpu_backtest_tools/runpod/__init__.py",
    "tools/gpu_backtest_tools/runpod/bundle.py",
    "tools/gpu_backtest_tools/runpod/client.py",
    "tools/gpu_backtest_tools/runpod/launcher.py",
    "tools/gpu_backtest_tools/runpod/requirements.txt",
    "tools/gpu_backtest_tools/runpod/transport.py",
}
PATTERNS = [
    re.compile("AKIA[A-Z0-9]{16}"),
    re.compile("gh[pousr]_[A-Za-z0-9]{30,}"),
    re.compile("github_pat_[A-Za-z0-9_]{30,}"),
    re.compile("-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile("(?:api_key|password|secret)\\s*=\\s*['\\\"][^'\\\"]{12,}['\\\"]", re.IGNORECASE),
]


def main():
    result = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True)
    if result.returncode:
        raise SystemExit("Initialize and stage the new repository before checking release files")
    paths = set(result.stdout.decode().split("\x00")) - {""}
    extra = paths - ALLOWED
    missing = ALLOWED - paths
    if extra or missing:
        raise SystemExit(
            f"Release allowlist mismatch: unexpected={sorted(extra)}, missing={sorted(missing)}"
        )
    problems = []
    for relative in sorted(paths):
        path = ROOT / relative
        if path.is_symlink():
            problems.append(f"{relative}: symlinks require explicit review")
            continue
        text = path.read_text(encoding="utf-8")
        if any((pattern.search(text) for pattern in PATTERNS)):
            problems.append(f"{relative}: possible credential detected")
    if problems:
        raise SystemExit("\n".join(problems))
    print(
        f"Release content check passed: {len(paths)} reviewed files, no credential-pattern matches"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
