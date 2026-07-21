# Cached block verification results

## Verification paths

The previous HF backend verified one target conditional per full-prefix forward. The new backend performs real temporal block verification: a single target forward supplies every draft conditional and the bonus distribution, reusing DynamicCache across blocks. A pending-token ledger permits exact branch rollback without an extra alignment forward. The serial implementation remains available as an independent reference.

Implemented capabilities include:

- `BlockProbabilityModel`, an independently testable k+1-row verification contract, and an exact-law block sampler.
- Cached GPT-2 and standard-RoPE Llama inference, including grouped-query attention, explicit positions/masks, branch crop, and failure reset.
- Final-row logit projection using `logits_to_keep=k+1`, avoiding unnecessary prefix vocabulary logits.
- Real physical target/draft forward, prefill, decode, input-token, reuse, crop, and reset counters.
- A three-mode CLI benchmark with excluded warmups and loading, cold caches per request, matched seeds, shared weights, and synchronized CUDA timing support.

See [probability-and-cache.md](probability-and-cache.md) for the index derivation, correctness boundary, complexity, and unsupported cache variants.

## Actual CPU experiment

Executed through the real `hf-benchmark` CLI using `bench/time_random_checkpoints.py`:

```bash
python bench/time_random_checkpoints.py --warmups 2 --repeats 5 --output results/cpu-tiny-benchmark.json
```

Both checkpoints were generated locally from fixed seeds and deleted afterward. They are random fixtures, not pretrained models. Environment: Python 3.12, CPU float32, one torch thread, torch 2.6.0+cpu, Transformers 4.57.1. Target: 31,616 parameters, two GPT-2 layers, width 32. Draft: 6,384 parameters, one layer, width 16. Vocabulary 64; prompt three tokens; output cap 24; draft window four.

| Mode | Median generation time | Target forwards on the four 24-token trials | Target input tokens on those trials |
| --- | ---: | --- | --- |
| Full-prefix serial | 292.9 ms | 24, 24, 24, 24 | 348, 348, 348, 348 |
| Cached serial | 204.3 ms | 24, 24, 24, 24 | 26, 26, 26, 26 |
| Cached block | 109.8 ms | 6, 6, 5, 7 | 29, 29, 27, 34 |

The fifth trial stopped at EOS after five generated tokens; it used five serial target forwards and one cached-block target forward. Every measured cached mode produced the same tokens as the serial reference for its matched seed. Every cached-block request used exactly one physical target forward per verification block, including its initial combined prefill/block forward.

This one fixture run observed a 2.67 time ratio for full-prefix versus cached-block medians. It is **not** a trained-model or GPU speedup claim. Modes ran in a fixed order; small CPU timings remain sensitive to host scheduling. All per-trial timings, output lengths, counters, random seeds, architecture settings, and environment fields are preserved in [the raw report](../results/cpu-tiny-benchmark.json).

## Correctness and failure evidence

The final suite passes **40 tests**, including actual randomly initialized local GPT-2 and grouped-query Llama checkpoints. Tests verify:

- Every batched target conditional against eager full-prefix recomputation across extensions, repeated prefixes, shorter requests, LCP=0/1, and arbitrary branch changes.
- Exact toy serial/block trajectory equivalence and complete stopped joint laws against an independent enumerated target oracle.
- Rejection at the first, middle, and last proposed positions; EOS as accepted, residual, and bonus output; zero-output requests and hard budgets.
- Seeded HF serial/block tokens and event traces, with actual rejections required to exercise rollback.
- One physical target forward per block, initial prefill accounting, exact context boundary, invalid IDs/windows, and cold-request benchmark setup.
- A fault that invokes the real forward, mutates DynamicCache, then raises; cache and ledger reset before reuse.

An independent read-only review additionally checked 5,400 random toy trajectories exactly and GPT-2/Llama full-distribution parity to maximum absolute error 1.49e-8. Floating-point tolerance, rather than bitwise parity across arbitrary kernels/hardware, is the actual-model distribution contract.

Three targeted new regressions are checked by `python bench/probe_cache_rules.py`: shifting conditional row indices, disabling KV crop, and omitting the bonus logit row. Each was killed by the selected tests. [cache-mutation-results.json](cache-mutation-results.json) retains failure evidence. These spot checks supplement the earlier sampler mutation tests; they are not an exhaustive mutation score.

Ruff checks/formatting, existing demo/evaluate/joint CLIs, and the CPU benchmark were executed. Optional model tests explicitly skip without the HF dependencies. No weights were downloaded, no paid inference was used, and no training was run.

## Remaining limits

Supported cached scope is batch size one, no padding, eager attention, standard GPT-2/Llama, unquantized weights, standard RoPE, and DynamicCache on the stated dependency range. Large/pretrained checkpoints, GPU performance, multi-request batching, quantization/offloading, sliding/static/paged caches, nonstandard RoPE, and fused GPU sampling remain unverified or unsupported. Python probability validation and CPU distribution transfer can dominate larger-vocabulary workloads. Fewer target launches can coexist with more token work when proposals are rejected; the recorded counters expose that tradeoff.
