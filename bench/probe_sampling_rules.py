"""Targeted mutation checks in disposable copies, never the working source."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MUTATIONS = [
    (
        "accept-every-proposal",
        "src/speculative_decoding/sampling.py",
        "acceptance = min(1.0, p[token] / q[token])",
        "acceptance = 1.0",
        "test_sampling.py",
    ),
    (
        "reverse-residual",
        "src/speculative_decoding/sampling.py",
        "max(pi - qi, 0.0)",
        "max(qi - pi, 0.0)",
        "test_sampling.py",
    ),
    (
        "ignore-stop-token",
        "src/speculative_decoding/sampling.py",
        "if token == stop_token:",
        "if False:",
        "test_joint_and_contracts.py",
    ),
]


def main():
    rows = []
    for name, relative, old, new, test in MUTATIONS:
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / "project"
            shutil.copytree(
                ROOT,
                copied,
                ignore=shutil.ignore_patterns(
                    "artifacts", "data", "checkpoints", "__pycache__", "*.egg-info", ".git", ".venv"
                ),
            )
            path = copied / relative
            source = path.read_text()
            if old not in source:
                raise RuntimeError(f"Mutation target missing: {name}")
            path.write_text(source.replace(old, new))
            environment = dict(os.environ, PYTHONPATH=str(copied / "src"), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
            result = subprocess.run(
                [sys.executable, "-m", "unittest", "discover", "-s", "checks", "-p", test, "-q"],
                cwd=copied,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=90,
            )
            killed = result.returncode != 0 and ("FAIL:" in result.stdout or "ERROR:" in result.stdout)
            rows.append(
                {
                    "mutation": name,
                    "target": relative,
                    "test_file": test,
                    "killed": killed,
                    "returncode": result.returncode,
                    "evidence": result.stdout[-1600:],
                }
            )
    report = {"method": "temporary copied source mutations; original source unchanged", "mutations": rows}
    print(json.dumps(report, indent=2))
    if not all(row["killed"] for row in rows):
        raise SystemExit("A mutation survived or its optional tests were unavailable; inspect evidence.")


if __name__ == "__main__":
    main()
