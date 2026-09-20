import unittest

from reproduction.analyze_alfworld_data import (
    DEFAULT_DATA,
    analyze,
    balanced_thresholds,
    render_markdown,
)


class AlfworldDataAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = analyze(DEFAULT_DATA)

    def test_released_dataset_identity_and_size(self) -> None:
        self.assertEqual(
            self.result["sha256"],
            "1f9d2bd6b9a5a88c06a99f521c63828c00a5494ed346aa2953fcc5764f749a66",
        )
        self.assertEqual(self.result["trajectory_count"], 3553)
        self.assertEqual(self.result["length_min"], 3)
        self.assertEqual(self.result["length_max"], 15)

    def test_paper_three_cluster_partition(self) -> None:
        self.assertEqual(self.result["thresholds"], [5, 7, 15])
        self.assertEqual(self.result["cluster_sizes"], {1: 888, 2: 1192, 3: 1473})
        self.assertIn("| 1 | 3–5 | 888", render_markdown(self.result))

    def test_text_length_proxy_changes_many_half_depth_prefixes(self) -> None:
        half = next(row for row in self.result["prefix_proxy"] if row["ratio"] == 0.5)
        self.assertEqual(half["mismatch_count"], 973)
        self.assertAlmostEqual(half["mismatch_percent"], 27.38530819026175)

    def test_balanced_thresholds_rejects_too_many_groups(self) -> None:
        with self.assertRaises(ValueError):
            balanced_thresholds([1, 1, 2], groups=3)


if __name__ == "__main__":
    unittest.main()
