import unittest

from reproduction.audit_config import REPO_ROOT, audit, render_markdown


class ConfigAuditTests(unittest.TestCase):
    def test_checked_out_recipe_is_extracted(self) -> None:
        rows = audit(REPO_ROOT)
        by_name = {str(row["name"]): row for row in rows}
        self.assertEqual(by_name["tasks per step"]["status"], "MATCH")
        self.assertEqual(by_name["rollouts per task"]["status"], "MATCH")
        self.assertEqual(by_name["peak learning rate"]["status"], "MISMATCH")
        self.assertEqual(by_name["difficulty clusters K"]["implementation"], 5.0)
        self.assertEqual(sum(row["status"] == "MISMATCH" for row in rows), 7)

    def test_report_contains_provenance_and_summary(self) -> None:
        report = render_markdown(audit(REPO_ROOT))
        self.assertIn("7 mismatches across 12 audited fields", report)
        self.assertIn("examples/gmsv_trainer/run_alfworld.sh", report)
        self.assertIn("Two runtime discrepancies", report)
        self.assertIn("expert action count", report)
        self.assertIn("separate experiment tracks", report)

    def test_paper_launcher_overrides_every_scalar_mismatch(self) -> None:
        launcher = (REPO_ROOT / "reproduction" / "run_alfworld_paper_table5.sh").read_text(
            encoding="utf-8"
        )
        for expected in (
            "actor_rollout_ref.actor.optim.lr=1e-5",
            "data.max_prompt_length=7000",
            "gmsv.num_difficulty_groups=3",
            "gmsv.lambda_value=1.0",
            "gmsv.mu_global_max=1.0",
            "gmsv.max_length=1.0",
            "trainer.total_epochs=200",
        ):
            self.assertIn(expected, launcher)


if __name__ == "__main__":
    unittest.main()
