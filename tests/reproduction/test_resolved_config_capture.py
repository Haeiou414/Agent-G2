import unittest

from reproduction.audit_config import REPO_ROOT


class ResolvedConfigCaptureTests(unittest.TestCase):
    def test_driver_saves_resolved_config_before_ray_initialization(self) -> None:
        source = (REPO_ROOT / "verl" / "trainer" / "main_ppo.py").read_text(encoding="utf-8")
        save_position = source.index("OmegaConf.save")
        ray_position = source.index("if not ray.is_initialized()")
        self.assertLess(save_position, ray_position)
        self.assertIn("VERL_RESOLVED_CONFIG_PATH", source)
        self.assertIn("resolve=True", source[save_position : save_position + 120])


if __name__ == "__main__":
    unittest.main()
