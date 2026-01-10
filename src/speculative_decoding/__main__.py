from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

from .sampling import MarkovModel, speculative_sample
from .benchmark import joint_benchmark


def evaluate(samples: int = 3000, seed: int = 17) -> dict:
    if type(samples) is not int or not 100 <= samples <= 100000:
        raise ValueError("samples must be in [100, 100000]")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    cases = [
        ((0.5, 0.5), (0.5, 0.5)),
        ((0.9, 0.1), (0.1, 0.9)),
        ((1.0, 0.0), (0.0, 1.0)),
        ((0.0, 1.0), (1.0, 0.0)),
        ((0.7, 0.3), (1.0, 0.0)),
        ((0.2, 0.8), (0.0, 1.0)),
        ((0.1, 0.2, 0.7), (0.7, 0.2, 0.1)),
        ((0.0, 0.3, 0.7), (0.8, 0.2, 0.0)),
        ((0.3333333333, 0.3333333333, 0.3333333334), (0.1, 0.8, 0.1)),
        ((0.99, 0.005, 0.005), (0.01, 0.49, 0.5)),
        ((0.1, 0.1, 0.1, 0.7), (0.25, 0.25, 0.25, 0.25)),
        ((0.25, 0.25, 0.25, 0.25), (0.0, 0.0, 0.0, 1.0)),
    ]
    rows = []
    for index, (p, q) in enumerate(cases):
        target, draft = MarkovModel(p), MarkovModel(q)
        counts = [0] * len(p)
        accepted = proposed = 0
        for trial in range(samples):
            result = speculative_sample(target, draft, max_new_tokens=1, seed=seed + index * samples + trial)
            counts[result.tokens[0]] += 1
            accepted += result.accepted_tokens
            proposed += result.proposed_tokens
        empirical = [count / samples for count in counts]
        rows.append(
            {
                "id": f"distribution-{index:02d}",
                "target": p,
                "draft": q,
                "samples": samples,
                "counts": counts,
                "empirical": empirical,
                "max_absolute_error": max(abs(a - b) for a, b in zip(p, empirical)),
                "accepted_draft_tokens": accepted,
                "proposed_draft_tokens": proposed,
            }
        )
    # A history-dependent model tests conditionals, not just its first marginal.
    target = MarkovModel((0.6, 0.4), ((0.8, 0.2), (0.1, 0.9)))
    draft = MarkovModel((0.2, 0.8), ((0.3, 0.7), (0.9, 0.1)))
    joint_counts = [0] * 4
    for trial in range(samples):
        a, b = speculative_sample(target, draft, max_new_tokens=2, draft_tokens=2, seed=seed + 1000000 + trial).tokens
        joint_counts[a * 2 + b] += 1
    expected_joint = [0.48, 0.12, 0.04, 0.36]
    return {
        "schema_version": 1,
        "experiment": "synthetic_probability_correctness",
        "model_backend": "exact_markov_tables",
        "learned_model_evaluated": False,
        "seed": seed,
        "python": platform.python_version(),
        "case_count": len(rows) + 1,
        "samples_per_case": samples,
        "cases": rows,
        "history_case": {
            "counts": joint_counts,
            "expected_joint": expected_joint,
            "samples": samples,
            "max_absolute_error": max(abs(c / samples - p) for c, p in zip(joint_counts, expected_joint)),
        },
        "limitations": [
            "Finite Monte Carlo checks are not a proof of implementation correctness.",
            "No HF throughput or speedup was measured; verification calls are serial.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Correctness-first speculative sampling laboratory")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("demo", "evaluate", "joint", "hf", "hf-benchmark"):
        command = sub.add_parser(name)
        command.add_argument("--output", type=Path)
        command.add_argument("--seed", type=int, default=17)
        if name in {"evaluate", "joint"}:
            command.add_argument("--samples", type=int, default=3000)
        if name == "joint":
            command.add_argument("--horizon", type=int, default=4)
            command.add_argument("--draft-tokens", type=int, default=3)
            command.add_argument("--stop-token", type=int, choices=(0, 1))
        if name in {"hf", "hf-benchmark"}:
            command.add_argument("--target", required=True, help="local checkpoint directory")
            command.add_argument("--draft", required=True, help="local checkpoint directory; identical tokenizer")
            command.add_argument("--prompt", required=True)
            command.add_argument("--max-new-tokens", type=int, default=32)
            command.add_argument("--device", default="cpu")
            command.add_argument("--draft-tokens", type=int, default=4)
        if name == "hf":
            command.add_argument("--engine", choices=("serial", "cached-block"), default="serial")
        if name == "hf-benchmark":
            command.add_argument("--warmups", type=int, default=1)
            command.add_argument("--repeats", type=int, default=5)
            command.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.command == "evaluate":
        result = evaluate(args.samples, args.seed)
    elif args.command == "joint":
        result = joint_benchmark(args.samples, args.seed, args.horizon, args.draft_tokens, args.stop_token)
    elif args.command == "hf-benchmark":
        from .hf_benchmark import benchmark_local

        result = benchmark_local(
            args.target,
            args.draft,
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            draft_tokens=args.draft_tokens,
            warmups=args.warmups,
            repeats=args.repeats,
            seed=args.seed,
            device=args.device,
            threads=args.threads,
        )
    elif args.command == "hf":
        from .hf import CachedHFModel, LocalHFModel
        from .hf_benchmark import compatible_prefix
        from .block import speculative_sample_block

        backend = CachedHFModel if args.engine == "cached-block" else LocalHFModel
        target, draft = backend(args.target, device=args.device), backend(args.draft, device=args.device)
        prefix = compatible_prefix(target, draft, args.prompt)
        algorithm = speculative_sample_block if args.engine == "cached-block" else speculative_sample
        sampled = algorithm(
            target,
            draft,
            prefix,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
            draft_tokens=args.draft_tokens,
            max_context=min(target.max_context, draft.max_context),
            stop_token=target.tokenizer.eos_token_id,
        )
        result = {
            "backend": "local_hf_" + args.engine,
            "target_checkpoint": str(Path(args.target).resolve()),
            "draft_checkpoint": str(Path(args.draft).resolve()),
            "text": target.tokenizer.decode(sampled.tokens),
            "sampling": sampled.to_dict(),
            "speedup_claim": None,
            "target_compute": target.stats.to_dict(),
            "draft_compute": draft.stats.to_dict(),
        }
    else:
        result = {
            "backend": "synthetic_markov",
            "learned_model_evaluated": False,
            "sampling": speculative_sample(
                MarkovModel((0.7, 0.2, 0.1)), MarkovModel((0.2, 0.3, 0.5)), max_new_tokens=24, seed=args.seed
            ).to_dict(),
        }
    encoded = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
