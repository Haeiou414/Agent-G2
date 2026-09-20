import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from reproduction.audit_config import REPO_ROOT


TRACKING_PATH = REPO_ROOT / "verl" / "utils" / "tracking.py"
SPEC = importlib.util.spec_from_file_location("tracking_under_test", TRACKING_PATH)
assert SPEC is not None and SPEC.loader is not None
TRACKING = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRACKING)
_JsonlLoggingAdapter = TRACKING._JsonlLoggingAdapter


class FakeScalar:
    def item(self):
        return 0.75


class JsonlTrackingTests(unittest.TestCase):
    def test_writes_only_json_serializable_scalars(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "metrics.jsonl"
            with patch.dict(os.environ, {"VERL_METRICS_JSONL_PATH": str(output)}):
                logger = _JsonlLoggingAdapter()
                logger.log({"val/success_rate": FakeScalar(), "table": [1, 2]}, step=80)
            row = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(row["step"], 80)
            self.assertEqual(row["metrics"], {"val/success_rate": 0.75})


if __name__ == "__main__":
    unittest.main()
