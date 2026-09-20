import unittest
from types import SimpleNamespace

try:
    import numpy as np
    from gmsv.alfworld import ExpertTrajectory, GMSVAlfworldPrefixRuntime, GroupStats

    RUNTIME_AVAILABLE = True
except ModuleNotFoundError:
    RUNTIME_AVAILABLE = False


@unittest.skipUnless(RUNTIME_AVAILABLE, "requires the official runtime dependencies")
class RuntimeScheduleUpdateTests(unittest.TestCase):
    def test_deterministic_mean_sampling_uses_center_without_rng(self) -> None:
        runtime = GMSVAlfworldPrefixRuntime.__new__(GMSVAlfworldPrefixRuntime)
        runtime.gmsv_cfg = {
            "prefix_sample_mode": "deterministic_mean",
            "max_length": 1.0,
        }
        runtime.rng = None

        sampled, clipped = runtime._sample_prefix_ratio(mu=0.63, sigma=0.27)

        self.assertEqual(sampled, 0.63)
        self.assertEqual(clipped, 0.63)

    def test_update_aggregates_rollouts_to_prompt_success_rate(self) -> None:
        runtime = GMSVAlfworldPrefixRuntime.__new__(GMSVAlfworldPrefixRuntime)
        runtime._initialized = True
        runtime.gmsv_cfg = {
            "alpha": 0.2,
            "sigma_mode": "dynamic_sigma",
            "sigma_min": 0.1,
            "gamma": 1.0,
            "lambda_value": 1.0,
            "batch_acc_low": 0.5,
            "batch_acc_high": 0.5,
            "global_step_delta": 0.1,
            "mu_global_min": 0.0,
            "mu_global_max": 0.8,
        }
        runtime.group_stats = {1: GroupStats(accuracy_ema=0.0, variance_ema=0.0)}
        runtime.mu_global = 0.8
        runtime.train_batches_completed = 0

        metrics = runtime.update_after_train_batch(
            {
                "uid": np.asarray(["task-a"] * 4 + ["task-b"] * 4, dtype=object),
                "traj_uid": np.asarray([f"rollout-{idx}" for idx in range(8)], dtype=object),
                "gmsv_difficulty_group": np.ones(8, dtype=np.int64),
                "episode_rewards": np.asarray([1, 0, 0, 0, 1, 1, 1, 0], dtype=np.float32),
                "gmsv_expert_matched": np.ones(8, dtype=bool),
                "gmsv_fixed_no_prefix_train": np.zeros(8, dtype=bool),
            }
        )

        self.assertEqual(metrics["train/matched_prompt_count"], 2.0)
        self.assertEqual(metrics["train/total_prompt_count"], 2.0)
        self.assertEqual(metrics["train/expert_match_rate"], 1.0)
        self.assertEqual(metrics["train/expert_mismatch_rate"], 0.0)
        self.assertEqual(metrics["train/batch_prompt_accuracy"], 0.5)
        self.assertEqual(metrics["scoreboard/group_1/batch_acc"], 0.5)
        self.assertEqual(metrics["scoreboard/group_1/batch_var"], 0.0625)
        self.assertAlmostEqual(metrics["scoreboard/group_1/A_k"], 0.1)
        self.assertAlmostEqual(metrics["scoreboard/group_1/V_k"], 0.0125)
        self.assertEqual(metrics["train/mu_global_after_update"], 0.8)

    def test_prefix_ratio_is_applied_to_action_count_not_token_count(self) -> None:
        class FakeTokenizer:
            def encode(self, text, add_special_tokens=False):
                return text.split()

        runtime = GMSVAlfworldPrefixRuntime.__new__(GMSVAlfworldPrefixRuntime)
        runtime.tokenizer = FakeTokenizer()
        runtime.config = SimpleNamespace(
            env=SimpleNamespace(env_name="alfworld", max_steps=20)
        )
        runtime.gmsv_cfg = {
            "sigma_mode": "fixed_sigma",
            "fixed_sigma": 0.1,
            "lambda_value": 1.0,
            "max_length": 1.0,
            "train_no_prefix_ratio": 0.0,
            "allow_full_prefix": False,
        }
        runtime.group_stats = {1: GroupStats(accuracy_ema=0.5, variance_ema=0.0)}
        runtime.mu_global = 0.5
        runtime._sample_prefix_ratio = lambda mu, sigma: (0.21, 0.21)
        trajectory = ExpertTrajectory(
            trial_id="task-a",
            task_type="pick",
            actions=["short"] + ["an action with many tokens"] * 9,
            thinks=[],
            action_count=10,
            difficulty_group=1,
            step_end_offsets=[1, 6, 11, 16, 21, 26, 31, 36, 41, 46],
            full_prefix_token_ids=list(range(46)),
        )

        plan = runtime._sample_plan(trajectory, is_train=True, trial_id="task-a")

        self.assertEqual(plan.prefix_step_count, 3)
        self.assertEqual(plan.prefix_actions, trajectory.actions[:3])
        self.assertFalse(plan.extended_to_step_end)


if __name__ == "__main__":
    unittest.main()
