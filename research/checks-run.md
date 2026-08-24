# Sampling checks run

The current source digest and executed checks are recorded in [reports/verification.json](../reports/verification.json). Optional tests are counted separately when their dependencies are unavailable.

Finite target-law comparisons and actual random GPT-2/Llama regressions check probability rows, stop rules, context bounds, and cache rollback/reset. They are distinct from the CPU fixture timing in `results/cpu-tiny-benchmark.json`.

Run `python bench/check_sampling.py --require-lint --wheel` to refresh verification. Existing mutation JSON preserves the failure evidence and historical paths from its recorded disposable-copy runs; it is a selected regression check, not an exhaustive mutation score.
