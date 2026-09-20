import unittest

from reproduction.scheduler import AgentG2Scheduler, SchedulerConfig


class AgentG2SchedulerTests(unittest.TestCase):
    def test_initial_parameters_match_paper(self) -> None:
        scheduler = AgentG2Scheduler((1, 2, 3), seed=3)
        mu, sigma = scheduler.parameters(1)
        self.assertEqual(mu, 1.0)
        self.assertEqual(sigma, 0.1)

    def test_difficult_cluster_receives_deeper_guidance(self) -> None:
        scheduler = AgentG2Scheduler((1, 2), seed=3)
        for _ in range(4):
            scheduler.update({1: [1.0, 1.0, 1.0, 1.0], 2: [0.0, 0.0, 0.0, 0.0]})
        easy_mu, _ = scheduler.parameters(1)
        hard_mu, _ = scheduler.parameters(2)
        self.assertGreater(hard_mu, easy_mu)

    def test_variance_widens_distribution(self) -> None:
        scheduler = AgentG2Scheduler((1, 2), seed=3)
        for _ in range(3):
            scheduler.update({1: [0.5, 0.5, 0.5, 0.5], 2: [0.0, 1.0, 0.0, 1.0]})
        _, stable_sigma = scheduler.parameters(1)
        _, mixed_sigma = scheduler.parameters(2)
        self.assertEqual(stable_sigma, 0.1)
        self.assertGreater(mixed_sigma, stable_sigma)

    def test_global_baseline_moves_toward_target_accuracy(self) -> None:
        scheduler = AgentG2Scheduler((1,), seed=3)
        scheduler.update({1: [1.0, 1.0]})
        self.assertAlmostEqual(scheduler.mu_global, 0.7)
        scheduler.update({1: [0.0, 0.0]})
        self.assertAlmostEqual(scheduler.mu_global, 0.8)

    def test_prefix_length_reserves_one_policy_action(self) -> None:
        self.assertEqual(AgentG2Scheduler.prefix_length(0.0, 10), 0)
        self.assertEqual(AgentG2Scheduler.prefix_length(0.21, 10), 3)
        self.assertEqual(AgentG2Scheduler.prefix_length(1.0, 10), 9)
        self.assertEqual(AgentG2Scheduler.prefix_length(1.0, 1), 0)

    def test_invalid_success_estimate_is_rejected(self) -> None:
        scheduler = AgentG2Scheduler((1,), SchedulerConfig())
        with self.assertRaises(ValueError):
            scheduler.update({1: [1.2]})


if __name__ == "__main__":
    unittest.main()
