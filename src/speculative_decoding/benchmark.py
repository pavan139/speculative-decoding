"""Complete finite target laws versus sampled multi-token trajectories."""

from __future__ import annotations

import math
from collections import Counter

from .sampling import MarkovModel, exact_joint_distribution, speculative_sample


def joint_benchmark(
    samples: int = 3000, seed: int = 17, horizon: int = 4, draft_tokens: int = 3, stop_token: int | None = None
) -> dict:
    if type(samples) is not int or not 100 <= samples <= 100_000:
        raise ValueError("samples must be an integer in [100, 100000]")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    target = MarkovModel((0.6, 0.4), ((0.8, 0.2), (0.1, 0.9)))
    draft = MarkovModel((0.2, 0.8), ((0.3, 0.7), (0.9, 0.1)))
    expected = exact_joint_distribution(target, horizon=horizon, stop_token=stop_token)
    counts = Counter(
        tuple(
            speculative_sample(
                target,
                draft,
                max_new_tokens=horizon,
                draft_tokens=draft_tokens,
                seed=seed + trial,
                stop_token=stop_token,
            ).tokens
        )
        for trial in range(samples)
    )
    support = sorted(set(expected) | set(counts))
    rows = [
        {
            "tokens": list(tokens),
            "target_probability": expected.get(tokens, 0.0),
            "count": counts[tokens],
            "empirical_probability": counts[tokens] / samples,
        }
        for tokens in support
    ]
    errors = [abs(row["target_probability"] - row["empirical_probability"]) for row in rows]
    # Union-bound Hoeffding band for all fixed target-support events; descriptive,
    # not a deterministic pass/fail certificate or a proof about arbitrary models.
    band = math.sqrt(math.log(2 * max(1, len(expected)) / 0.01) / (2 * samples))
    return {
        "experiment": "exact_enumerated_target_joint_vs_monte_carlo_speculative",
        "learned_model_evaluated": False,
        "samples": samples,
        "seed": seed,
        "horizon": horizon,
        "draft_tokens": draft_tokens,
        "stop_token": stop_token,
        "target_mass": math.fsum(expected.values()),
        "total_variation_distance": sum(errors) / 2,
        "max_absolute_error": max(errors),
        "simultaneous_99_percent_hoeffding_band": band,
        "illegal_output_count": sum(count for tokens, count in counts.items() if tokens not in expected),
        "cases": rows,
        "limitations": [
            "Target paths are exhaustively enumerated in float64; sampled agreement remains finite-sample evidence.",
            "This measures distribution correctness, not inference throughput.",
        ],
    }
