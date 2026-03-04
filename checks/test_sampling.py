import random
import unittest

from speculative_decoding.sampling import (
    MarkovModel,
    greedy,
    sample_distribution,
    speculative_sample,
    validate_distribution,
)
from speculative_decoding.__main__ import evaluate


class SamplingTests(unittest.TestCase):
    def test_bad_probabilities_rejected(self):
        for values in ((), (0.2, 0.2), (-0.1, 1.1), (float("nan"), 1.0), (float("inf"), 0.0)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_distribution(values)

    def test_transition_shape_validation(self):
        with self.assertRaises(ValueError):
            MarkovModel((0.5, 0.5), ((1.0, 0.0),))
        with self.assertRaises(ValueError):
            MarkovModel((0.5, 0.5), ((1.0,), (1.0,)))

    def test_zero_probability_token_is_never_sampled(self):
        rng = random.Random(4)
        self.assertEqual({sample_distribution((0, 0, 1), rng) for _ in range(100)}, {2})

    def test_disjoint_support_uses_residual(self):
        result = speculative_sample(MarkovModel((1.0, 0.0)), MarkovModel((0.0, 1.0)), max_new_tokens=9)
        self.assertEqual(result.tokens, [0] * 9)
        self.assertEqual(result.rejected_tokens, 9)
        self.assertTrue(all(event["kind"] == "target_residual" for event in result.events))

    def test_identical_models_accept_every_proposal(self):
        model = MarkovModel((0.4, 0.6))
        result = speculative_sample(model, model, max_new_tokens=13, draft_tokens=3)
        self.assertEqual(result.rejected_tokens, 0)
        self.assertGreater(result.accepted_tokens, 0)
        self.assertTrue(any(event["kind"] == "target_bonus" for event in result.events))

    def test_hard_token_budget(self):
        model = MarkovModel((1.0,))
        for count in range(12):
            self.assertEqual(len(speculative_sample(model, model, max_new_tokens=count, draft_tokens=4).tokens), count)

    def test_explicit_context_budget(self):
        model = MarkovModel((1.0,))
        with self.assertRaises(ValueError):
            speculative_sample(model, model, [0] * 5, max_new_tokens=4, max_context=8)

    def test_invalid_budgets(self):
        model = MarkovModel((1.0,))
        for kwargs in ({"draft_tokens": 0}, {"draft_tokens": 65}, {"max_new_tokens": -1}, {"max_new_tokens": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                speculative_sample(model, model, **kwargs)

    def test_vocabulary_mismatch(self):
        with self.assertRaises(ValueError):
            speculative_sample(MarkovModel((1.0,)), MarkovModel((0.5, 0.5)))

    def test_invalid_prefix(self):
        model = MarkovModel((1.0,))
        for prefix in ([1], [True], [-1]):
            with self.subTest(prefix=prefix), self.assertRaises(ValueError):
                speculative_sample(model, model, prefix)

    def test_same_seed_reproducible(self):
        target, draft = MarkovModel((0.2, 0.8)), MarkovModel((0.7, 0.3))
        self.assertEqual(
            speculative_sample(target, draft, seed=12).to_dict(), speculative_sample(target, draft, seed=12).to_dict()
        )

    def test_target_conditional_uses_accepted_history(self):
        target = MarkovModel((1.0, 0.0), ((0.0, 1.0), (1.0, 0.0)))
        draft = MarkovModel((0.5, 0.5))
        self.assertEqual(speculative_sample(target, draft, max_new_tokens=8).tokens, [0, 1] * 4)

    def test_stop_on_accepted_token(self):
        model = MarkovModel((0.0, 1.0))
        result = speculative_sample(model, model, max_new_tokens=8, stop_token=1)
        self.assertEqual(result.tokens, [1])
        self.assertEqual(result.stop_reason, "stop_token")

    def test_stop_on_residual_token(self):
        result = speculative_sample(MarkovModel((1.0, 0.0)), MarkovModel((0.0, 1.0)), max_new_tokens=8, stop_token=0)
        self.assertEqual(result.tokens, [0])
        self.assertEqual(result.stop_reason, "stop_token")

    def test_greedy_is_deterministic_and_separate(self):
        self.assertEqual(greedy(MarkovModel((0.2, 0.8)), max_new_tokens=4), [1] * 4)

    def test_full_backend_vocabulary_validated(self):
        class Bad:
            vocab_size = 2

            def distribution(self, prefix):
                return (1.0,)

        with self.assertRaises(ValueError):
            speculative_sample(Bad(), MarkovModel((0.5, 0.5)))

    def test_empirical_marginals_and_joint(self):
        result = evaluate(samples=2000)
        self.assertEqual(result["case_count"], 13)
        self.assertLess(max(case["max_absolute_error"] for case in result["cases"]), 0.05)
        self.assertLess(result["history_case"]["max_absolute_error"], 0.05)
        self.assertFalse(result["learned_model_evaluated"])


if __name__ == "__main__":
    unittest.main()
