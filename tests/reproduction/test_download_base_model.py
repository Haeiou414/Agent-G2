import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from reproduction import download_base_model


class BaseModelDownloadTests(unittest.TestCase):
    def test_forwards_pinned_revision_and_verifies_weights(self) -> None:
        payload = b"verified model"
        calls = []

        def fake_snapshot_download(**kwargs):
            calls.append(kwargs)
            target = Path(kwargs["local_dir"])
            target.mkdir(parents=True, exist_ok=True)
            (target / download_base_model.WEIGHTS_NAME).write_bytes(payload)
            return str(target)

        fake_module = mock.Mock(snapshot_download=fake_snapshot_download)
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            sys.modules, {"huggingface_hub": fake_module}
        ), mock.patch.object(download_base_model, "WEIGHTS_SIZE", len(payload)), mock.patch.object(
            download_base_model, "WEIGHTS_SHA256", hashlib.sha256(payload).hexdigest()
        ), mock.patch.object(
            sys,
            "argv",
            ["download_base_model", "--local-dir", directory, "--max-workers", "1"],
        ):
            download_base_model.main()

        self.assertEqual(calls[0]["repo_id"], download_base_model.REPO_ID)
        self.assertEqual(calls[0]["revision"], download_base_model.REVISION)
        self.assertEqual(calls[0]["max_workers"], 1)

    def test_rejects_wrong_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            download_base_model, "WEIGHTS_SIZE", 3
        ), mock.patch.object(download_base_model, "WEIGHTS_SHA256", "0" * 64):
            model_dir = Path(directory)
            (model_dir / download_base_model.WEIGHTS_NAME).write_bytes(b"bad")
            with self.assertRaises(SystemExit):
                download_base_model.verify_weights(model_dir)


if __name__ == "__main__":
    unittest.main()
