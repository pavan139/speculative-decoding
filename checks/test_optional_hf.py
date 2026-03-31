"""Actual tiny random local checkpoints; no downloads or pretrained claims."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

from speculative_decoding.hf import LocalHFModel
from speculative_decoding.sampling import speculative_sample


AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ("torch", "transformers", "tokenizers"))


@unittest.skipUnless(AVAILABLE, "optional local HF test dependencies are absent")
class LocalHFTests(unittest.TestCase):
    def test_tiny_checkpoint_probabilities_context_and_sampler(self):
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                tokenizer = PreTrainedTokenizerFast(
                    tokenizer_object=Tokenizer(
                        WordLevel({"<bos>": 0, "<eos>": 1, "A": 2, "<unk>": 3}, unk_token="<unk>")
                    ),
                    bos_token="<bos>",
                    eos_token="<eos>",
                    unk_token="<unk>",
                )
                tokenizer.save_pretrained(path)
                torch.manual_seed(9)
                model = GPT2LMHeadModel(
                    GPT2Config(
                        vocab_size=4,
                        n_positions=8,
                        n_ctx=8,
                        n_embd=8,
                        n_layer=1,
                        n_head=1,
                        bos_token_id=0,
                        eos_token_id=1,
                    )
                )
                model.save_pretrained(path)
                backend = LocalHFModel(str(path), max_context=100)
                self.assertEqual(backend.max_context, 8)
                probabilities = backend.distribution([0, 2])
                self.assertEqual(len(probabilities), 4)
                self.assertAlmostEqual(sum(probabilities), 1, places=6)
                for bad in ([], [True], [4], [1.5], [0] * 9):
                    with self.subTest(bad=bad), self.assertRaises(ValueError):
                        backend.distribution(bad)
                result = speculative_sample(backend, backend, [0], max_new_tokens=3, draft_tokens=2, max_context=8)
                self.assertEqual(result.rejected_tokens, 0)
                self.assertEqual(len(result.tokens), 3)
                # A tokenizer/model mismatch must fail before indexing model logits.
                tokenizer.add_tokens(["extra"])
                tokenizer.save_pretrained(path)
                with self.assertRaisesRegex(ValueError, "tokenizer IDs"):
                    LocalHFModel(str(path))
        finally:
            torch.set_num_threads(previous)


if __name__ == "__main__":
    unittest.main()
