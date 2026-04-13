"""Real CPU GPT-2/Llama cache and block tests using random local weights."""

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from speculative_decoding.block import speculative_sample_block
from speculative_decoding.hf import CachedHFModel, LocalHFModel
from speculative_decoding.hf_benchmark import FullPrefixView, benchmark_local
from speculative_decoding.sampling import speculative_sample

AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ("torch", "transformers", "tokenizers"))


def write_checkpoint(path: Path, *, kind="gpt2", seed=11):
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    path.mkdir(parents=True, exist_ok=True)
    vocabulary = {"<bos>": 0, "<eos>": 1, "<unk>": 2, **{f"t{index}": index + 3 for index in range(9)}}
    raw = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    raw.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, bos_token="<bos>", eos_token="<eos>", unk_token="<unk>")
    tokenizer.save_pretrained(path)
    torch.manual_seed(seed)
    if kind == "gpt2":
        model = GPT2LMHeadModel(
            GPT2Config(
                vocab_size=12, n_positions=48, n_ctx=48, n_embd=16, n_layer=2, n_head=2, bos_token_id=0, eos_token_id=1
            )
        )
    else:
        model = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=12,
                max_position_embeddings=48,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=1,
                bos_token_id=0,
                eos_token_id=1,
            )
        )
    model.save_pretrained(path)


@unittest.skipUnless(AVAILABLE, "optional torch/transformers/tokenizers are absent")
class CachedHFTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        for kind in ("gpt2", "llama"):
            write_checkpoint(cls.root / kind, kind=kind)
        write_checkpoint(cls.root / "draft", seed=22)

    @classmethod
    def tearDownClass(cls):
        import torch

        cls.temporary.cleanup()
        torch.set_num_threads(cls.previous_threads)

    def test_all_block_rows_equal_full_prefix_across_forks_and_rollbacks(self):
        import torch

        requests = [
            ([0, 3], [4, 5, 6]),
            ([0, 3, 4, 9], [7]),
            ([0, 3], []),
            ([0, 3], [8]),
            ([0, 9], [4]),
            ([8], [7, 6]),
            ([8, 7], [5]),
            ([8, 7], []),
        ]
        for kind in ("gpt2", "llama"):
            cached = CachedHFModel(str(self.root / kind))
            full = LocalHFModel(str(self.root / kind), attention_implementation="eager")
            for prefix, proposals in requests:
                before = cached.stats.forward_calls
                rows = cached.block_distributions(prefix, proposals)
                self.assertEqual(cached.stats.forward_calls, before + 1)
                self.assertEqual(len(rows), len(proposals) + 1)
                for index, actual in enumerate(rows):
                    expected = full.distribution(prefix + proposals[:index])
                    torch.testing.assert_close(torch.tensor(actual), torch.tensor(expected), rtol=1e-5, atol=1e-7)
                committed = prefix + proposals[:1] + [10]
                cached.commit(committed)
                self.assertEqual(cached.committed_tokens, committed)
                self.assertLessEqual(len(cached.cache_tokens), len(committed) - 1)
                self.assertEqual(cached.cache.get_seq_length() if cached.cache else 0, len(cached.cache_tokens))

    def test_seeded_serial_cached_and_block_sequences_match(self):
        for kind in ("gpt2", "llama"):
            target = CachedHFModel(str(self.root / kind))
            draft = CachedHFModel(str(self.root / "draft"))
            rejected = 0
            for seed in range(12):
                kwargs = dict(prefix=[0, 3], max_new_tokens=12, draft_tokens=3, seed=seed, stop_token=1, max_context=48)
                serial = speculative_sample(FullPrefixView(target), FullPrefixView(draft), **kwargs)
                target.reset_cache()
                draft.reset_cache()
                target.reset_stats()
                draft.reset_stats()
                block = speculative_sample_block(target, draft, **kwargs)
                rejected += block.rejected_tokens
                self.assertEqual(block.tokens, serial.tokens)
                self.assertEqual(block.events, serial.events)
                self.assertEqual(target.stats.forward_calls, block.target_calls)
                self.assertEqual(target.stats.verified_blocks, block.target_calls)
                self.assertEqual(target.stats.prefill_forwards, 1)
                self.assertEqual(target.stats.decode_forwards, max(0, block.target_calls - 1))
                self.assertLessEqual(len(target.cache_tokens), len(kwargs["prefix"]) + len(block.tokens) - 1)
            self.assertGreater(rejected, 0, "seeded model experiment must exercise real rejection/rollback")

    def test_exact_context_invalid_ids_and_cache_reset_after_forward_fault(self):
        cached = CachedHFModel(str(self.root / "gpt2"), max_context=8)
        rows = cached.block_distributions([0, 3, 4, 5, 6], [7, 8, 9])
        self.assertEqual(len(rows), 4)
        before = cached.stats.forward_calls
        for prefix, proposals in (([0] * 8, [3]), ([True], [3]), ([0], [12]), ([0], [3] * 65)):
            with self.subTest(prefix=prefix), self.assertRaises(ValueError):
                cached.block_distributions(prefix, proposals)
        self.assertEqual(cached.stats.forward_calls, before)
        cached.commit([0, 3, 4, 10])
        original = cached.model.forward

        def broken(*args, **kwargs):
            original(*args, **kwargs)
            raise RuntimeError("injected after DynamicCache mutation")

        with patch.object(cached.model, "forward", side_effect=broken):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                cached.block_distributions([0, 3, 4, 10], [11])
        self.assertIsNone(cached.cache)
        self.assertEqual(cached.cache_tokens, [])
        self.assertEqual(cached.committed_tokens, [])
        cached.block_distributions([0, 3], [4])

    def test_benchmark_warmups_cold_request_counters_and_shared_weights(self):
        with self.assertRaisesRegex(ValueError, "integer seed"):
            benchmark_local("", "", "", seed=True)
        result = benchmark_local(
            str(self.root / "gpt2"),
            str(self.root / "draft"),
            "t0 t1",
            max_new_tokens=10,
            draft_tokens=3,
            warmups=1,
            repeats=2,
        )
        self.assertEqual(result["measured_repeats"], 2)
        for mode, values in result["modes"].items():
            self.assertTrue(values["same_tokens_as_serial"])
            self.assertEqual(len(values["trials"]), 2)
            for trial in values["trials"]:
                self.assertGreater(trial["elapsed_seconds"], 0)
                if mode == "cached_block":
                    self.assertEqual(trial["target_compute"]["forward_calls"], trial["sampling"]["target_calls"])
                    self.assertEqual(trial["target_compute"]["prefill_forwards"], 1)


if __name__ == "__main__":
    unittest.main()
