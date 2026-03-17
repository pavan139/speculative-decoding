import random
import unittest
from collections import Counter

from speculative_decoding.block import ReferenceBlockModel, speculative_sample_block
from speculative_decoding.sampling import MarkovModel, exact_joint_distribution, speculative_sample


class BlockSamplerTests(unittest.TestCase):
    def test_rejection_at_first_middle_and_last_draft_positions(self):
        target = MarkovModel((1.0, 0.0, 0.0), ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)))
        for rejected_at in range(3):
            initial = (0.0, 1.0, 0.0) if rejected_at == 0 else target.initial
            transitions = list(target.transitions)
            if rejected_at == 1:
                transitions[0] = (0.0, 0.0, 1.0)
            elif rejected_at == 2:
                transitions[1] = (1.0, 0.0, 0.0)
            draft = MarkovModel(initial, tuple(transitions))
            result = speculative_sample_block(ReferenceBlockModel(target), draft, max_new_tokens=3, draft_tokens=3)
            self.assertEqual(result.tokens, [0, 1, 2])
            self.assertEqual(result.events[rejected_at]["kind"], "target_residual")
            self.assertTrue(all(event["kind"] == "accepted_draft" for event in result.events[:rejected_at]))

    def test_serial_and_block_paths_are_identical_across_random_tables(self):
        rng = random.Random(31)
        for _ in range(12):

            def row():
                values = [rng.randrange(5) for _ in range(3)]
                if not sum(values):
                    values[0] = 1
                return tuple(value / sum(values) for value in values)

            target = MarkovModel(row(), tuple(row() for _ in range(3)))
            draft = MarkovModel(row(), tuple(row() for _ in range(3)))
            for window in (1, 2, 4):
                for stop in (None, 0, 2):
                    for seed in range(8):
                        kwargs = dict(prefix=[1], seed=seed, max_new_tokens=8, draft_tokens=window, stop_token=stop)
                        serial = speculative_sample(target, draft, **kwargs)
                        block = speculative_sample_block(ReferenceBlockModel(target), draft, **kwargs)
                        self.assertEqual(block.tokens, serial.tokens)
                        self.assertEqual(block.events, serial.events)
                        for key in (
                            "proposed_tokens",
                            "verified_tokens",
                            "accepted_tokens",
                            "rejected_tokens",
                            "stop_reason",
                        ):
                            self.assertEqual(getattr(block, key), getattr(serial, key))

    def test_block_empirical_stopped_joint_matches_independent_oracle(self):
        target = MarkovModel((0.6, 0.4), ((0.8, 0.2), (0.1, 0.9)))
        draft = MarkovModel((0.2, 0.8), ((0.3, 0.7), (0.9, 0.1)))
        for stop in (None, 1):
            expected = exact_joint_distribution(target, horizon=3, stop_token=stop)
            counts = Counter(
                tuple(
                    speculative_sample_block(
                        ReferenceBlockModel(target),
                        draft,
                        max_new_tokens=3,
                        draft_tokens=2,
                        seed=trial,
                        stop_token=stop,
                    ).tokens
                )
                for trial in range(4000)
            )
            self.assertTrue(set(counts) <= set(expected))
            self.assertLess(
                max(abs(counts[path] / 4000 - probability) for path, probability in expected.items()), 0.035
            )

    def test_target_calls_count_blocks_and_commit_final_prefix(self):
        class Tracked(ReferenceBlockModel):
            def commit(self, prefix):
                self.committed = list(prefix)

        model = MarkovModel((1.0,))
        target = Tracked(model)
        result = speculative_sample_block(target, model, [0], max_new_tokens=16, draft_tokens=4)
        self.assertEqual(result.target_calls, 4)
        self.assertEqual(target.committed, [0] * 17)

    def test_invalid_block_shape_vocabulary_and_probabilities(self):
        class BadBlock:
            vocab_size = 2

            def commit(self, prefix):
                pass

            def block_distributions(self, prefix, proposals):
                return self.rows

        draft = MarkovModel((1.0, 0.0))
        for rows in ([], [(1.0, 0.0)], [(1.0,)] * 3, [(0.2, 0.2)] * 3):
            backend = BadBlock()
            backend.rows = rows
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                speculative_sample_block(backend, draft, max_new_tokens=2, draft_tokens=2)

    def test_stop_bonus_disjoint_support_and_zero_output(self):
        model = MarkovModel((1.0, 0.0), ((0.0, 1.0), (1.0, 0.0)))
        result = speculative_sample_block(
            ReferenceBlockModel(model), model, max_new_tokens=4, draft_tokens=1, stop_token=1
        )
        self.assertEqual(result.tokens, [0, 1])
        self.assertEqual(result.events[-1]["kind"], "target_bonus")
        target, draft = MarkovModel((1.0, 0.0)), MarkovModel((0.0, 1.0))
        result = speculative_sample_block(ReferenceBlockModel(target), draft, max_new_tokens=3, stop_token=0)
        self.assertEqual(result.tokens, [0])
        self.assertEqual(result.events[-1]["kind"], "target_residual")
        self.assertEqual(speculative_sample_block(ReferenceBlockModel(target), draft, max_new_tokens=0).target_calls, 0)


if __name__ == "__main__":
    unittest.main()
