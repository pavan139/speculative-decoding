"""End-to-end local inference timings with controlled warmups and cache resets."""

from __future__ import annotations

import platform
import statistics
import time

from .block import speculative_sample_block
from .hf import CachedHFModel, LocalHFModel
from .sampling import speculative_sample


class FullPrefixView:
    """Reuse model weights while explicitly bypassing its KV-cache backend."""

    def __init__(self, backend: CachedHFModel):
        self.backend = backend
        self.vocab_size = backend.vocab_size

    def distribution(self, prefix):
        return LocalHFModel.distribution(self.backend, prefix)


def compatible_prefix(target, draft, prompt: str) -> list[int]:
    if target.tokenizer.get_vocab() != draft.tokenizer.get_vocab():
        raise ValueError("checkpoint tokenizers must have identical token-to-ID mappings")
    if target.tokenizer.special_tokens_map != draft.tokenizer.special_tokens_map:
        raise ValueError("checkpoint tokenizers must have identical special-token semantics")
    prefix = target.tokenizer.encode(prompt, add_special_tokens=True)
    if not prefix:
        if target.tokenizer.bos_token_id is None:
            raise ValueError("empty prompt requires a tokenizer BOS token")
        prefix = [target.tokenizer.bos_token_id]
    return prefix


def benchmark_local(
    target_path: str,
    draft_path: str,
    prompt: str,
    *,
    max_new_tokens: int = 32,
    draft_tokens: int = 4,
    warmups: int = 1,
    repeats: int = 5,
    seed: int = 17,
    device: str = "cpu",
    threads: int = 1,
) -> dict:
    if (
        any(type(value) is not int for value in (warmups, repeats, threads, seed))
        or not 0 <= warmups <= 20
        or not 1 <= repeats <= 100
        or not 1 <= threads <= 64
    ):
        raise ValueError("integer seed; warmups 0..20, repeats 1..100 and CPU threads 1..64 required")
    target = CachedHFModel(target_path, device=device)
    draft = CachedHFModel(draft_path, device=device)
    prefix = compatible_prefix(target, draft, prompt)
    torch = target.torch
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(threads)
    sync = lambda: torch.cuda.synchronize(device) if torch.device(device).type == "cuda" else None
    rows = {}
    modes = {
        "serial_full_prefix": (FullPrefixView(target), FullPrefixView(draft), speculative_sample),
        "cached_serial": (target, draft, speculative_sample),
        "cached_block": (target, draft, speculative_sample_block),
    }
    try:
        for name, (target_backend, draft_backend, algorithm) in modes.items():
            trials = []
            for trial in range(-warmups, repeats):
                target.reset_cache()
                draft.reset_cache()
                target.reset_stats()
                draft.reset_stats()
                sync()
                start = time.perf_counter()
                result = algorithm(
                    target_backend,
                    draft_backend,
                    prefix,
                    max_new_tokens=max_new_tokens,
                    draft_tokens=draft_tokens,
                    seed=seed + trial,
                    max_context=min(target.max_context, draft.max_context),
                    stop_token=target.tokenizer.eos_token_id,
                )
                sync()
                elapsed = time.perf_counter() - start
                if trial >= 0:
                    trials.append(
                        {
                            "seed": seed + trial,
                            "elapsed_seconds": elapsed,
                            "generated_tokens": len(result.tokens),
                            "tokens_per_second": len(result.tokens) / elapsed,
                            "sampling": result.to_dict(),
                            "target_compute": target.stats.to_dict(),
                            "draft_compute": draft.stats.to_dict(),
                        }
                    )
            rows[name] = {
                "median_seconds": statistics.median(row["elapsed_seconds"] for row in trials),
                "median_tokens_per_second": statistics.median(row["tokens_per_second"] for row in trials),
                "trials": trials,
            }
        for name, result in rows.items():
            result["same_tokens_as_serial"] = all(
                row["sampling"]["tokens"] == baseline["sampling"]["tokens"]
                for row, baseline in zip(result["trials"], rows["serial_full_prefix"]["trials"])
            )
            result["observed_median_time_ratio_vs_serial"] = (
                rows["serial_full_prefix"]["median_seconds"] / result["median_seconds"]
            )
    finally:
        torch.set_num_threads(previous_threads)
    import transformers

    return {
        "experiment": "local_end_to_end_serial_vs_cached_serial_vs_cached_block",
        "device": device,
        "cpu_threads": threads,
        "machine": platform.machine(),
        "processor": torch.cuda.get_device_name(device)
        if torch.device(device).type == "cuda"
        else (platform.processor() or "CPU model unavailable"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "target_checkpoint": target_path,
        "draft_checkpoint": draft_path,
        "target_parameters": sum(parameter.numel() for parameter in target.model.parameters()),
        "draft_parameters": sum(parameter.numel() for parameter in draft.model.parameters()),
        "target_dtype": str(next(target.model.parameters()).dtype),
        "draft_dtype": str(next(draft.model.parameters()).dtype),
        "warmups_per_mode": warmups,
        "measured_repeats": repeats,
        "prompt_tokens": len(prefix),
        "max_new_tokens": max_new_tokens,
        "draft_tokens": draft_tokens,
        "modes": rows,
        "limits": [
            "Hardware warmups are excluded; every request starts with an empty cache and includes prefill.",
            "Loading checkpoints and resetting caches are outside measured time; Python sampling and CPU probability transfer are included.",
            "Time ratios apply only to this run, checkpoints, hardware, and prompt; no general GPU speedup is implied.",
        ],
    }
