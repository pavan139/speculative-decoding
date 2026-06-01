# Batched verification and reversible KV-cache design

## Contract and supported scope

`BlockProbabilityModel.block_distributions(B, T)` returns exactly `k+1` distributions, where the committed prefix is `B` of length `m`, proposed draft tokens are `T=(t0,...,t{k-1})`, and row `i` is:

```text
p_i = p(next token | B + T[:i]),  i = 0,...,k
```

Rows 0 through k-1 verify draft tokens; row k is the bonus distribution after every draft token is accepted. A zero-length proposal block returns the ordinary next-token distribution and is used by the cached draft backend.

The actual HF backend supports one unpadded request, eager attention, GPT-2 learned absolute positions or Llama standard RoPE (including grouped-query attention), and `transformers.cache_utils.DynamicCache`. It is tested with Transformers 4.57.1 and CPU PyTorch 2.6. Optional dependency metadata requires Transformers >=4.57.1,<5. Static, sliding-window, offloaded/quantized caches, scaled/dynamic RoPE, encoder-decoder models, and other model families are rejected or unsupported. The serial reference remains independent of this restricted cache path.

## Why one target forward supplies all rows

Let the physical cache contain tokens `C`. Let `l=LCP(C,B)` be their shared prefix length. Retain:

```text
c = min(l, m - 1)
```

Crop the cache to c positions, leaving at least the final committed token pending. One forward consumes:

```text
input = B[c:] + T
position_ids = c,...,m+k-1
attention_mask length = m+k
```

A causal LM's logit at an input token predicts the next token. Therefore row `i` is at output index:

```text
offset + i = (m - c - 1) + i
```

There are `m-c+k` input tokens, so the required rows are exactly the final `k+1` rows. Passing `logits_to_keep=k+1` avoids allocating vocabulary logits for unrelated long-prefix positions. It does not avoid the hidden-state or attention computation for those positions.

The first request performs one combined prefill/block forward. Later blocks consume the pending token plus new proposals in one cached forward. There is **no separate alignment or bonus-token forward**. Physical forward counters test this invariant.

## Commit and rejection rollback

After verification the physical cache describes `B+T`, which may include unaccepted tokens. If j proposals were accepted and correction token r was sampled, the next committed prefix is:

```text
B' = B + T[:j] + [r]
```

`commit(B')` crops the speculative suffix to `min(LCP(C,B'),len(B')-1)`. The correction remains pending. If all k proposals plus bonus b are accepted, the cache already contains `B+T`; only b is pending. A target or draft cache never treats rejected tokens as committed.

Cropping alone cannot restore the logit associated with an earlier prefix. This implementation avoids retaining stale logits entirely: the final required prefix token is always rerun with the next block. Repeated prefixes, shorter prefixes, first-token forks, and LCP=1 follow the same rule. The draft backend uses the same ledger with zero-length proposal blocks; after a rejection its old draft suffix is cropped too.

EOS can occur as an accepted proposal, rejection correction, or bonus. Sampling stops immediately in all cases and commits only emitted tokens. Draft construction stops at draft EOS, although verification may compute a distribution after that token; that unused row is not sampled. Output and context limits are checked before model execution; no context truncation is allowed.

## Acceptance law

Draw a draft token x from its stored q conditional. Accept with `min(1,p(x)/q(x))`. On rejection sample from normalized positive residual `max(p-q,0)`, discard later draft tokens, and start a new block. Only if all proposals are accepted may row k supply a bonus token. The block sampler consumes RNG draws in the same order as the serial sampler. Toy probability tables therefore match trajectories exactly for the same seed.

With actual HF kernels, batched and recomputed logits can differ by floating point roundoff. Tests compare full distributions with explicit tolerances and also check seeded trajectories; bitwise equality across devices/kernels is not a general contract.

## Cache failure and counters

DynamicCache mutates in place during a forward. Holding its old Python object is not a snapshot. If a forward or cache/logit validation fails, the backend clears cache, token ledger, and committed-prefix metadata before reraising the error. A regression fault calls the real forward and then raises, proving recovery after in-place mutation.

`SamplingResult.target_calls` counts target API calls: individual conditional requests in the serial sampler, blocks in the block sampler. Compare `target_compute.forward_calls` across actual backends instead. Counters record physical forward attempts, prefill/cached/full-recompute categories, supplied input tokens, reused KV positions, crops, and resets. Completed benchmarks contain successful calls only. Cache resets and checkpoint loading are excluded from timed generation; each request's prefill is included.

## Complexity and tradeoffs

Without KV reuse, N output tokens require repeated work on growing prefixes. Cached serial inference processes new tokens while retaining O(layers × KV heads × sequence length × head dimension) cache state, but still launches target forwards per conditional. Block verification reduces target launches to roughly one per accepted block plus correction/bonus cycle, while processing up to k proposed tokens and one pending token per cached forward.

Rejected proposals spend real attention/MLP work and temporarily extend KV storage. Larger k can therefore increase input-token work even while reducing launch count. Cropping discards branch visibility; storage views may remain allocated until subsequent cache updates/references are released. The CPU reference converts full probability rows to Python for transparent exact-law sampling, adding O(kV) transfer/storage work; it is not a fused GPU sampler. Multi-request batching, paged caches, and optimized GPU acceptance kernels remain outside this implementation.

## Benchmark interpretation

`hf-benchmark` shares the same loaded weights across full-prefix, cached-serial, and cached-block modes. Each mode runs excluded hardware warmups, then measured cold-cache requests with matched seeds. CUDA timing synchronizes before and after generation. Reports retain every trial's output length and physical compute counters, because EOS and rejection can change work. A small CPU timing ratio is evidence for that fixture and host only, not a promised GPU or production-model speedup.
