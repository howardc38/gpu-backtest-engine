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
    "pyproject.toml",
    "docs/strategy-contract.md",
    "docs/testing.md",
    "docs/runpod.md",
    "docs/runpod-validation.md",
    "docs/benchmarks.md",
    "benchmarks/results/rtx4090_rsi_20261003.json",
    "benchmarks/results/rtx4090_rsi_matched_billion_20261003.json",
    "examples/generate_data.py",
    "examples/rsi.json",
    "examples/synthetic.csv",
    "scripts/check_release.py",
    "src/gpu_backtest/__init__.py",
    "src/gpu_backtest/__main__.py",
    "src/gpu_backtest/benchmark.py",
    "src/gpu_backtest/benchmark_cpu.py",
    "src/gpu_backtest/charts.py",
    "src/gpu_backtest/cli.py",
    "src/gpu_backtest/common.py",
    "src/gpu_backtest/engine.py",
    "src/gpu_backtest/gpu_check.py",
    "src/gpu_backtest/runpod.py",
    "src/gpu_backtest/runpod_requirements.txt",
    "src/gpu_backtest/pipeline.py",
    "src/gpu_backtest/splits.py",
    "src/gpu_backtest/statistics.py",
    "src/gpu_backtest/strategy.py",
    "src/gpu_backtest/synthetic.py",
    "src/gpu_backtest/tables.py",
    "src/gpu_backtest/strategies/__init__.py",
    "src/gpu_backtest/strategies/rsi_meanrev.py",
    "tests/helpers.py",
    "tests/simulator_cases.py",
    "tests/test_cli.py",
    "tests/test_benchmark.py",
    "tests/test_common_contract.py",
    "tests/test_engine_contract.py",
    "tests/test_gpu.py",
    "tests/test_gpu_check.py",
    "tests/test_runpod.py",
    "tests/test_simulator.py",
    "tests/test_split_contract.py",
    "tests/test_statistics.py",
    "tests/test_strategy.py",
    "tests/test_tables.py",
}
PATTERNS = [
    re.compile(r"AKIA[A-Z0-9]{16}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{30,}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"(?:api_key|password|secret)\s*=\s*['\"][^'\"]{12,}['\"]", re.IGNORECASE),
]


def main():
    result = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True)
    if result.returncode:
        raise SystemExit("Initialize and stage the new repository before checking release files")
    paths = set(result.stdout.decode().split("\0")) - {""}
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
        if any(pattern.search(text) for pattern in PATTERNS):
            problems.append(f"{relative}: possible credential detected")
    if problems:
        raise SystemExit("\n".join(problems))
    print(
        f"Release content check passed: {len(paths)} reviewed files, no credential-pattern matches"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
