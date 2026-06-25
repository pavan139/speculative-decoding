# Technical review

This document records the earlier probability-contract review. The newer cached/batched inference design and its current test/benchmark evidence are in [cached-block-results.md](cached-block-results.md) and [probability-and-cache.md](probability-and-cache.md).

## Reviewed behavior and fixes

The acceptance rule, positive-residual rejection correction, accepted-history conditioning, bonus token, and stop handling were audited against the target sampling law. No mathematical defect was reproduced in those paths. Additional tests exercise disjoint support, subnormal probabilities, zero-probability events, different draft windows, and stopping on accepted, residual, and bonus tokens.

Reproduced contract defects and fixes:

- Probability vectors accepted booleans and numeric strings, and extremely large finite entries could escape as `OverflowError`. The validator now requires real numeric values and gives a stable `ValueError` for overflowing sums.
- A nominally frozen Markov fixture retained caller-owned mutable lists. Parameters are now copied into normalized immutable tuples.
- Invalid backend vocabulary metadata could pass on a zero-output request, and non-integer seeds were accepted implicitly. Both backends and seed types are now validated before generation.
- The optional HF backend now validates direct prefix IDs, tokenizer/model vocabulary compatibility, explicit context types, persistent BOS handling, and checkpoint context limits before inference.

## Added technical capability

`exact_joint_distribution` independently enumerates a bounded target output tree by multiplying target conditional probabilities. It contains no speculative acceptance/rejection code or RNG. The new `joint` command compares complete target laws with sampled trajectories, including variable-length stop leaves, reporting target mass, per-path probabilities/counts, total variation distance, illegal outputs, and a descriptive simultaneous Hoeffding band. Enumeration has an eight-token horizon limit and a configurable finite leaf budget.

Reference outputs are checked in as [joint-reference.json](../results/joint-reference.json) and [stopped-joint-reference.json](../results/stopped-joint-reference.json). They describe exact synthetic probability tables, not pretrained-model quality or performance.

## Executed verification

Using Python 3.12, CPU PyTorch 2.6, and Transformers 4.57.1:

```bash
PYTHONPATH=src python -m unittest discover -s checks -v
python -m speculative_decoding demo --output artifacts/review-demo.json
python -m speculative_decoding evaluate --samples 3000 --output artifacts/review-evaluation.json
python -m speculative_decoding joint --samples 6000 --horizon 4 --draft-tokens 3 --output results/joint-reference.json
python -m speculative_decoding joint --samples 6000 --horizon 4 --draft-tokens 3 --stop-token 1 --output results/stopped-joint-reference.json
python bench/probe_sampling_rules.py
```

The complete suite passes **30 tests**, including an actual tiny random local GPT-2 checkpoint saved and loaded during the test, full-distribution inference, identical-model speculative sampling, context failures, and a deliberately mismatched tokenizer. No weights are downloaded. Optional HF tests skip explicitly when dependencies are absent; a skip is not model verification.

Three targeted source mutations were tested in temporary copies: accepting every proposal, reversing the residual, and ignoring the stop token. All three were killed by the selected tests. [mutation-results.json](mutation-results.json) retains outcomes and failure evidence. This is a focused regression spot check, not an exhaustive mutation score or proof of correctness.

Ruff formatting and the configured E9/F checks pass. The default evaluation and both new joint CLI commands were executed.

## Remaining limits

Float64 enumeration and finite Monte Carlo agreement do not prove the sampler for arbitrary models. HF verification still recomputes forwards serially; there is no batched verifier, KV-cache optimization, measured serving speedup, or large-checkpoint benchmark. GPU behavior, large vocabularies/checkpoints, and production throughput remain unverified. Checkpoint resource requirements remain the caller's responsibility.
