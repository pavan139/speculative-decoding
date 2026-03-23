import math
import random
import unittest

from speculative_decoding.benchmark import joint_benchmark
from speculative_decoding.sampling import (
    MarkovModel,
    exact_joint_distribution,
    sample_distribution,
    speculative_sample,
    validate_distribution,
)


class ExactJointTests(unittest.TestCase):
    def test_independently_computed_history_law(self):
        model = MarkovModel((0.6, 0.4), ((0.8, 0.2), (0.1, 0.9)))
        expected = {(0, 0): 0.48, (0, 1): 0.12, (1, 0): 0.04, (1, 1): 0.36}
        actual = exact_joint_distribution(model, horizon=2)
        for path, probability in expected.items():
            self.assertAlmostEqual(actual[path], probability)
        self.assertAlmostEqual(math.fsum(actual.values()), 1)

    def test_variable_length_stop_law_and_prefix_conditioning(self):
        model = MarkovModel((0.5, 0.5), ((0.75, 0.25), (0.2, 0.8)))
        actual = exact_joint_distribution(model, [1], horizon=3, stop_token=1)
        expected = {(1,): 0.8, (0, 1): 0.05, (0, 0, 1): 0.0375, (0, 0, 0): 0.1125}
        self.assertEqual(set(actual), set(expected))
        for path in expected:
            self.assertAlmostEqual(actual[path], expected[path])

    def test_zero_horizon_and_sparse_support(self):
        model = MarkovModel((0.0, 1.0, 0.0))
        self.assertEqual(exact_joint_distribution(model, horizon=0), {(): 1.0})
        self.assertEqual(exact_joint_distribution(model, horizon=8, max_sequences=1), {(1,) * 8: 1.0})

    def test_enumeration_is_bounded(self):
        model = MarkovModel((0.5, 0.5))
        for options in ({"horizon": 9}, {"horizon": 4, "max_sequences": 8}, {"max_sequences": True}, {"stop_token": 2}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                exact_joint_distribution(model, **options)

    def test_speculative_law_matches_all_paths_across_windows(self):
        for window, stop in ((1, None), (2, None), (4, None), (3, 1)):
            result = joint_benchmark(samples=4000, seed=101, horizon=3, draft_tokens=window, stop_token=stop)
            with self.subTest(window=window, stop=stop):
                self.assertEqual(result["illegal_output_count"], 0)
                self.assertAlmostEqual(result["target_mass"], 1)
                self.assertLess(result["max_absolute_error"], result["simultaneous_99_percent_hoeffding_band"])


class NumericAndRandomTests(unittest.TestCase):
    def test_markov_parameters_are_copied_into_immutable_normalized_tuples(self):
        initial = [0.5, 0.5]
        transitions = [[0.2, 0.8], [0.8, 0.2]]
        model = MarkovModel(initial, transitions)
        initial[0] = 100
        transitions[0][0] = 100
        self.assertEqual(model.initial, (0.5, 0.5))
        self.assertEqual(model.transitions[0], (0.2, 0.8))

    def test_numeric_contract_rejects_strings_boolean_overflow(self):
        for probabilities in ((True, False), ("0.5", "0.5"), (1e308, 1e308)):
            with self.subTest(probabilities=probabilities), self.assertRaises(ValueError):
                validate_distribution(probabilities)

    def test_normalization_within_declared_tolerance(self):
        self.assertAlmostEqual(math.fsum(validate_distribution((0.5, 0.5000001))), 1)

    def test_extreme_nonzero_probabilities_and_zero_draw(self):
        class ZeroRng:
            def random(self):
                return 0.0

        self.assertEqual(sample_distribution((0.0, 5e-324, 1.0), ZeroRng()), 1)
        target = MarkovModel((1.0, 0.0))
        draft = MarkovModel((5e-324, 1.0))
        self.assertEqual(speculative_sample(target, draft, max_new_tokens=10).tokens, [0] * 10)

    def test_global_rng_unchanged_and_strict_seed(self):
        model = MarkovModel((0.5, 0.5))
        before = random.getstate()
        speculative_sample(model, model, seed=19)
        self.assertEqual(before, random.getstate())
        for seed in (True, "19", 1.2):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                speculative_sample(model, model, seed=seed)

    def test_invalid_backend_metadata_even_for_zero_output(self):
        class BadModel:
            vocab_size = True

            def distribution(self, _prefix):
                return (1.0,)

        valid = MarkovModel((1.0,))
        for target, draft in ((BadModel(), valid), (valid, BadModel())):
            with self.assertRaises(ValueError):
                speculative_sample(target, draft, max_new_tokens=0)

    def test_stop_on_bonus_and_exact_context_boundary(self):
        model = MarkovModel((1.0, 0.0), ((0.0, 1.0), (1.0, 0.0)))
        result = speculative_sample(model, model, max_new_tokens=4, draft_tokens=1, stop_token=1, max_context=4)
        self.assertEqual(result.tokens, [0, 1])
        self.assertEqual(result.events[-1]["kind"], "target_bonus")
        self.assertEqual(result.stop_reason, "stop_token")
        self.assertEqual(result.verified_tokens, 1)
        self.assertEqual(result.target_calls, 2)


if __name__ == "__main__":
    unittest.main()
