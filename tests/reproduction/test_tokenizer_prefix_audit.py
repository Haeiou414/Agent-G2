import unittest

from reproduction.analyze_tokenizer_prefix import (
    action_based_prefix_steps,
    token_based_prefix_steps,
    token_step_offsets,
)


class TokenizerPrefixAuditTests(unittest.TestCase):
    def test_token_offsets_include_step_formatting(self) -> None:
        offsets, total = token_step_offsets(
            ["short", "a much longer action"],
            lambda text: len(text.split()),
        )
        self.assertEqual(offsets, [3, 9])
        self.assertEqual(total, 9)

    def test_action_equation_reserves_one_policy_step(self) -> None:
        self.assertEqual(action_based_prefix_steps(10, 0.21), 3)
        self.assertEqual(action_based_prefix_steps(10, 1.0), 9)

    def test_token_rule_can_disagree_with_action_equation(self) -> None:
        offsets = [3, 9, 12, 15]
        token_steps = token_based_prefix_steps(offsets, total_tokens=15, ratio=0.5)
        action_steps = action_based_prefix_steps(total_steps=4, ratio=0.5)
        self.assertEqual(token_steps, 2)
        self.assertEqual(action_steps, 2)

        token_steps = token_based_prefix_steps([8, 10, 12, 14], total_tokens=14, ratio=0.5)
        self.assertEqual(token_steps, 1)
        self.assertNotEqual(token_steps, action_steps)


if __name__ == "__main__":
    unittest.main()
