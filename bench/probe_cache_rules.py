"""Targeted cached/block regressions; requires the optional [hf] dependencies."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
MUTATIONS = [
    (
        "shift-target-conditional-row",
        "src/speculative_decoding/block.py",
        "p = target_rows[index]",
        "p = target_rows[index + 1]",
        "test_block.py",
    ),
    (
        "disable-kv-cache-rollback",
        "src/speculative_decoding/hf.py",
        "self.cache.crop(length)",
        "None",
        "test_optional_cached_hf.py",
    ),
    (
        "omit-bonus-logit-row",
        "src/speculative_decoding/hf.py",
        "logits_to_keep=len(proposals) + 1",
        "logits_to_keep=max(1, len(proposals))",
        "test_optional_cached_hf.py",
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
                ignore=shutil.ignore_patterns("artifacts", "__pycache__", "*.egg-info", ".git", ".venv", "dist"),
            )
            target = copied / relative
            source = target.read_text()
            if old not in source:
                raise RuntimeError(f"Mutation target missing: {name}")
            target.write_text(source.replace(old, new))
            result = subprocess.run(
                [sys.executable, "-m", "unittest", "discover", "-s", "checks", "-p", test, "-q"],
                cwd=copied,
                env=dict(os.environ, PYTHONPATH=str(copied / "src"), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=90,
            )
            rows.append(
                {
                    "mutation": name,
                    "test_file": test,
                    "killed": result.returncode != 0 and ("FAIL:" in result.stdout or "ERROR:" in result.stdout),
                    "returncode": result.returncode,
                    "evidence": result.stdout[-1800:],
                }
            )
    print(
        json.dumps(
            {"method": "targeted mutations in disposable copies; original source unchanged", "mutations": rows},
            indent=2,
        )
    )
    if not all(row["killed"] for row in rows):
        raise SystemExit("A mutation survived or its optional tests were not executed.")


if __name__ == "__main__":
    main()
