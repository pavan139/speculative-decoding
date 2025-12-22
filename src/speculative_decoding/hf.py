"""Optional local-only HF backend. The core sampler does not import ML packages."""

from __future__ import annotations

from pathlib import Path
from dataclasses import asdict, dataclass


@dataclass
class ForwardStats:
    forward_calls: int = 0
    prefill_forwards: int = 0
    decode_forwards: int = 0
    full_recompute_forwards: int = 0
    input_tokens: int = 0
    verified_blocks: int = 0
    distribution_queries: int = 0
    cache_crops: int = 0
    cache_resets: int = 0
    reused_prefix_tokens: int = 0

    def to_dict(self):
        return asdict(self)


class LocalHFModel:
    def __init__(
        self,
        checkpoint: str,
        *,
        device: str = "cpu",
        max_context: int = 2048,
        attention_implementation: str | None = None,
    ):
        if type(max_context) is not int or not 1 <= max_context <= 131072:
            raise ValueError("max_context must be an integer in [1, 131072]")
        if not Path(checkpoint).is_dir():
            raise ValueError("checkpoint must be an existing local directory; downloads are disabled")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise RuntimeError("install the optional [hf] dependencies first") from error
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
        options = {"attn_implementation": attention_implementation} if attention_implementation else {}
        self.model = (
            AutoModelForCausalLM.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False, **options)
            .to(device)
            .eval()
        )
        self.device = device
        configured_context = getattr(self.model.config, "max_position_embeddings", None)
        if type(configured_context) is not int or configured_context <= 0:
            configured_context = getattr(self.model.config, "n_positions", None)
        self.max_context = (
            min(max_context, configured_context)
            if type(configured_context) is int and configured_context > 0
            else max_context
        )
        self.vocab_size = int(self.model.config.vocab_size)
        token_ids = self.tokenizer.get_vocab().values()
        if not token_ids or any(type(token) is not int or not 0 <= token < self.vocab_size for token in token_ids):
            raise ValueError("tokenizer IDs do not fit the model output vocabulary")
        self.stats = ForwardStats()

    def reset_stats(self):
        self.stats = ForwardStats()

    def distribution(self, prefix):
        if not prefix:
            raise ValueError("HF backend requires a persistent nonempty prefix; insert BOS before sampling")
        if any(type(token) is not int or not 0 <= token < self.vocab_size for token in prefix):
            raise ValueError("HF prefix token outside model vocabulary")
        if len(prefix) > self.max_context:
            raise ValueError("HF context budget exceeded")
        tokens = self.torch.tensor([list(prefix)], dtype=self.torch.long, device=self.device)
        self.stats.forward_calls += 1
        self.stats.full_recompute_forwards += 1
        self.stats.distribution_queries += 1
        self.stats.input_tokens += len(prefix)
        with self.torch.inference_mode():
            logits = self.model(input_ids=tokens, use_cache=False).logits[0, -1].float()
            return self.torch.softmax(logits, dim=-1).cpu().tolist()


class CachedHFModel(LocalHFModel):
    """Batch-one GPT-2/Llama dynamic-KV backend with exact branch rollback.

    A committed prefix's final token is left pending. One forward consumes that
    token (and any uncached suffix) plus the whole draft block, yielding every
    target conditional and the bonus distribution. No alignment forward occurs.
    """

    def __init__(self, checkpoint: str, *, device: str = "cpu", max_context: int = 2048):
        super().__init__(checkpoint, device=device, max_context=max_context, attention_implementation="eager")
        if (
            self.model.config.model_type not in {"gpt2", "llama"}
            or getattr(self.model.config, "sliding_window", None)
            or getattr(self.model.config, "rope_scaling", None)
            or getattr(self.model, "is_quantized", False)
        ):
            raise ValueError(
                "cached backend supports standard unquantized GPT-2/Llama without sliding windows or RoPE scaling"
            )
        self.cache = None
        self.cache_tokens: list[int] = []
        self.committed_tokens: list[int] = []

    def reset_cache(self):
        self.cache = None
        self.cache_tokens = []
        self.committed_tokens = []
        self.stats.cache_resets += 1

    def _validate_tokens(self, prefix, *, nonempty=True):
        if nonempty and not prefix:
            raise ValueError("cached HF backend requires a persistent nonempty prefix")
        if any(type(token) is not int or not 0 <= token < self.vocab_size for token in prefix):
            raise ValueError("cached HF token outside vocabulary")
        if len(prefix) > self.max_context:
            raise ValueError("cached HF context budget exceeded")

    def _crop(self, length):
        if length == len(self.cache_tokens):
            return
        if length == 0:
            self.cache = None
            self.stats.cache_resets += 1
        else:
            from transformers.cache_utils import DynamicCache

            if isinstance(self.cache, DynamicCache):
                self.cache.crop(length)
            else:
                raise TypeError("cached decoding supports transformers DynamicCache only")
        self.cache_tokens = self.cache_tokens[:length]
        self.stats.cache_crops += 1

    def _shared(self, prefix):
        count = 0
        for cached, requested in zip(self.cache_tokens, prefix):
            if cached != requested:
                break
            count += 1
        return count

    def commit(self, prefix):
        self._validate_tokens(prefix)
        self._crop(min(self._shared(prefix), len(prefix) - 1))
        self.committed_tokens = list(prefix)

    def block_distributions(self, prefix, proposals):
        self._validate_tokens(prefix)
        self._validate_tokens(proposals, nonempty=False)
        if len(proposals) > 64 or len(prefix) + len(proposals) > self.max_context:
            raise ValueError("block exceeds draft-window or context budget")
        shared = min(self._shared(prefix), len(prefix) - 1)
        self._crop(shared)
        pending = list(prefix[shared:]) + list(proposals)
        positions = self.torch.arange(shared, shared + len(pending), device=self.device)
        inputs = self.torch.tensor([pending], dtype=self.torch.long, device=self.device)
        mask = self.torch.ones((1, shared + len(pending)), dtype=self.torch.long, device=self.device)
        self.stats.forward_calls += 1
        self.stats.prefill_forwards += shared == 0
        self.stats.decode_forwards += shared > 0
        self.stats.input_tokens += len(pending)
        self.stats.reused_prefix_tokens += shared
        self.stats.verified_blocks += bool(proposals)
        try:
            with self.torch.inference_mode():
                # The final k+1 rows are exactly p0..pk, even for a long prefill.
                # Avoid materializing unused prefix-position vocabulary logits.
                output = self.model(
                    input_ids=inputs,
                    past_key_values=self.cache,
                    attention_mask=mask,
                    position_ids=positions.unsqueeze(0),
                    cache_position=positions,
                    use_cache=True,
                    return_dict=True,
                    logits_to_keep=len(proposals) + 1,
                )
                from transformers.cache_utils import DynamicCache

                if not isinstance(output.past_key_values, DynamicCache):
                    raise TypeError("model did not return the supported transformers DynamicCache")
                if output.past_key_values.get_seq_length() != len(prefix) + len(proposals):
                    raise ValueError("physical KV-cache length disagrees with token ledger")
                if output.logits.shape[1] != len(proposals) + 1:
                    raise ValueError("model did not return the requested block logit rows")
                probabilities = self.torch.softmax(output.logits[0].float(), dim=-1).cpu().tolist()
            self.cache = output.past_key_values
            self.cache_tokens = list(prefix) + list(proposals)
        except Exception:
            # DynamicCache mutates in-place inside forward; its previous object
            # is not a rollback snapshot after a partially completed failure.
            self.reset_cache()
            raise
        return probabilities

    def distribution(self, prefix):
        self.stats.distribution_queries += 1
        return self.block_distributions(prefix, [])[0]
