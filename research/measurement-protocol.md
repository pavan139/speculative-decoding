# Sampling and timing protocols

Probability checks come first. Use `evaluate` for categorical marginals and `joint` for every supported path of a small history-dependent model. Include stop-token leaves in the target mass, state the sample denominator, and retain illegal-output counts. A confidence band describes the finite Monte Carlo check; it does not prove the algorithm for arbitrary models.

The serial and block samplers should also match trajectories under the controlled shared-RNG construction. For real models, compare conditional distributions within numerical tolerance before using identical tokens as a regression expectation. Floating-point kernel changes can affect sampled trajectories even when probability differences are small.

For timings, `hf-benchmark` measures full-prefix serial, cached serial, and cached block requests with the same loaded weights. Exclude model loading and designated warmups. Start each measured request with empty caches, include prefill, and synchronize CUDA around timed work when applicable. Keep trial order and early-stop lengths in the report.

```bash
python bench/time_random_checkpoints.py --warmups 2 --repeats 5 --output artifacts/cpu-fixture.json
```

That helper creates small fixed-seed GPT-2 fixtures locally and removes them after the report. Its stored measurement is in `results/cpu-tiny-benchmark.json`. It does not use pretrained weights. Target forwards, draft forwards, input-token work, crops, resets, and accepted/proposed counts are different measurements; report them alongside elapsed time.

A new hardware, cache variant, tokenizer, quantization scheme, or sampling setting defines a new experiment. Preserve the old raw report instead of rewriting it as though it had been run under the new conditions. [cached-block-results.md](cached-block-results.md) records the existing conditions and observed limits.
