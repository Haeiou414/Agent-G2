import os
import subprocess
import unittest

from reproduction.audit_config import REPO_ROOT


LAUNCHER = REPO_ROOT / "reproduction" / "run_alfworld_single_gpu.sh"


class SingleGpuLauncherTests(unittest.TestCase):
    def test_launcher_restores_and_validates_alfworld_data_after_restart(self):
        text = LAUNCHER.read_text()
        self.assertIn('source "$ROOT_DIR/reproduction/autodl_env.sh"', text)
        self.assertIn('$ALFWORLD_DATA/json_2.1.1/train', text)

    def dry_run(self, method: str, seed: int = 7) -> str:
        env = dict(os.environ)
        env["DRY_RUN"] = "1"
        completed = subprocess.run(
            ["bash", str(LAUNCHER), method, str(seed)],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        return completed.stdout

    def test_all_methods_share_the_same_budget(self) -> None:
        for method in (
            "gmsv",
            "grpo",
            "target_acc",
            "fixed_sigma",
            "no_aux_sft",
            "deterministic_mean",
        ):
            command = self.dry_run(method)
            self.assertIn("data.train_batch_size=4", command)
            self.assertIn("env.rollout.n=4", command)
            self.assertIn("env.resources_per_worker.num_cpus=0.05", command)
            self.assertIn("actor_rollout_ref.rollout.enforce_eager=true", command)
            self.assertIn("actor_rollout_ref.rollout.free_cache_engine=true", command)
            self.assertIn("actor_rollout_ref.rollout.gpu_memory_utilization=0.50", command)
            self.assertIn("actor_rollout_ref.model.lora_rank=16", command)
            self.assertIn("actor_rollout_ref.model.lora_alpha=16", command)
            self.assertIn("trainer.total_training_steps=80", command)
            self.assertIn("trainer.max_actor_ckpt_to_keep=1", command)
            self.assertIn("trainer.n_gpus_per_node=1", command)
            self.assertIn("trainer.resume_mode=disable", command)
            self.assertIn("trainer.default_local_dir=outputs/", command)
            self.assertIn("trainer.logger=\\[console\\,jsonl\\]", command)
            self.assertIn("seed7", command)

    def test_method_selects_expected_upstream_recipe(self) -> None:
        self.assertIn("examples/gmsv_trainer/run_alfworld.sh", self.dry_run("gmsv"))
        self.assertIn("examples/grpo_trainer/run_alfworld.sh", self.dry_run("grpo"))
        self.assertIn(
            "examples/fix_acc_hint_trainer/run_alfworld.sh",
            self.dry_run("target_acc"),
        )

    def test_gmsv_uses_paper_faithful_schedule_overrides(self) -> None:
        command = self.dry_run("gmsv")
        for expected in (
            "gmsv.num_difficulty_groups=3",
            "gmsv.lambda_value=1.0",
            "gmsv.mu_global_max=1.0",
            "gmsv.max_length=1.0",
            "gmsv.seed=7",
        ):
            self.assertIn(expected, command)

    def test_ablation_definitions_match_paper_table_three(self) -> None:
        fixed_sigma = self.dry_run("fixed_sigma")
        self.assertIn("gmsv.sigma_mode=fixed_sigma", fixed_sigma)
        self.assertIn("gmsv.fixed_sigma=0.1", fixed_sigma)

        no_aux = self.dry_run("no_aux_sft")
        self.assertIn("gmsv.prefix_sft.enable=false", no_aux)

        deterministic = self.dry_run("deterministic_mean")
        self.assertIn("gmsv.prefix_sample_mode=deterministic_mean", deterministic)

    def test_unknown_method_fails_before_training(self) -> None:
        completed = subprocess.run(
            ["bash", str(LAUNCHER), "unknown", "1"],
            capture_output=True,
            text=True,
            env={**os.environ, "DRY_RUN": "1"},
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("unknown METHOD", completed.stderr)

    def test_run_tag_creates_a_distinct_experiment_name(self) -> None:
        env = {**os.environ, "DRY_RUN": "1", "RUN_TAG": "retry-1"}
        completed = subprocess.run(
            ["bash", str(LAUNCHER), "gmsv", "7"],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        self.assertIn("limited_gmsv_qwen2.5_1.5b_seed7_retry-1", completed.stdout)

    def test_base_model_path_can_use_a_pinned_local_snapshot(self) -> None:
        env = {
            **os.environ,
            "DRY_RUN": "1",
            "BASE_MODEL_PATH": "/data/models/qwen2.5-1.5b-instruct",
        }
        completed = subprocess.run(
            ["bash", str(LAUNCHER), "gmsv", "7"],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        self.assertIn(
            "actor_rollout_ref.model.path=/data/models/qwen2.5-1.5b-instruct",
            completed.stdout,
        )

    def test_lora_budget_can_be_overridden_explicitly(self) -> None:
        env = {
            **os.environ,
            "DRY_RUN": "1",
            "LORA_RANK": "32",
            "LORA_ALPHA": "64",
        }
        completed = subprocess.run(
            ["bash", str(LAUNCHER), "gmsv", "7"],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        self.assertIn("actor_rollout_ref.model.lora_rank=32", completed.stdout)
        self.assertIn("actor_rollout_ref.model.lora_alpha=64", completed.stdout)

    def test_unsafe_run_tag_is_rejected(self) -> None:
        completed = subprocess.run(
            ["bash", str(LAUNCHER), "gmsv", "7"],
            capture_output=True,
            text=True,
            env={**os.environ, "DRY_RUN": "1", "RUN_TAG": "../overwrite"},
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("RUN_TAG", completed.stderr)

    def test_launcher_preserves_training_exit_code_and_console_log(self) -> None:
        script = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn('tee "$RUN_DIR/console.log"', script)
        self.assertIn("RUN_EXIT_CODE=${PIPESTATUS[0]}", script)
        self.assertIn("VERL_RESOLVED_CONFIG_PATH", script)


if __name__ == "__main__":
    unittest.main()
