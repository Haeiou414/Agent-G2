import unittest

from reproduction.audit_config import REPO_ROOT


class ExactTokenizerReportTests(unittest.TestCase):
    def test_report_is_pinned_and_contains_exact_half_depth_result(self) -> None:
        report = (
            REPO_ROOT / "reproduction" / "reports" / "tokenizer_prefix_audit.md"
        ).read_text(encoding="utf-8")
        self.assertIn("4556a9bfdf84320267c3a9e9e7b85732ba2835ba", report)
        self.assertIn("9c5ae00e602b8860cbd784ba82a8aa14e8feecec692e7076590d014d7b7fdafa", report)
        self.assertIn("| 0.5 | 1,278 | 36.0% | 0 | 1,278 |", report)


if __name__ == "__main__":
    unittest.main()
