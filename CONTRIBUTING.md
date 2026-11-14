# Changing a sampler or cache path

A sampling change must preserve the target law. Begin with a small distribution where the marginal or full stopped joint can be enumerated. Include disjoint support, a first/middle/last rejection, bonus EOS, and a strict output budget where relevant. Comparing two implementations that share the same error is insufficient.

For cache changes, compare every conditional row with eager full-prefix recomputation across prefix extensions and forks. Check physical KV length against the token ledger after crop and after a forward that mutates the cache then fails. Maintain the supported GPT-2/Llama boundary rather than silently accepting a different cache/RoPE contract.

Run `python -m unittest discover -s checks -v`. With `.[hf]` installed, set `RUN_ML_SMOKE=1` for the random local-model tests. Then run `python bench/check_sampling.py --require-lint --wheel`. The targeted tools in `bench/` operate on temporary source copies and should not rewrite the checkout.

A timing change needs identical weights, prompt IDs, generation limits, seeds, warmup policy, and cache reset behavior. Report physical forwards and input tokens along with time. Do not replace the recorded random-checkpoint result with an extrapolated large-model speedup.
