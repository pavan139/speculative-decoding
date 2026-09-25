# Sharing the sampler and measurements

A source release should include the finite-law checks, cached-model regressions, measurement helper, and raw reports. Run `python bench/check_sampling.py --require-lint --wheel` and inspect `reports/verification.json` before pushing the repository to its own remote.

The files in `results/` describe synthetic probability tables or locally generated random checkpoints. Keep that distinction in release notes. A pretrained-checkpoint measurement needs its exact revision/digest, tokenizer identity, architecture/cache settings, dtype, device, dependency versions, threads, prompt IDs, output lengths, and synchronization method.

Large checkpoints and private prompt contents do not belong in Git. The public local-model example accepts existing checkpoint directories and disables downloads/remote code. Preserve per-trial counters and failures, including requests that stopped early. A fixed-order warm CPU run is not interchangeable with a cold serving or GPU benchmark.

The GitHub Actions file travels with this repository. It checks the independent package rather than relying on another checkout. Link the raw measurement and source digest when tagging a result so readers can identify exactly what was timed.
