import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from reproduction import download_public_checkpoint


class PublicCheckpointDownloadTests(unittest.TestCase):
    def test_forwards_pinned_revision_and_worker_limit(self) -> None:
        payload = b"verified checkpoint"
        calls = []

        def fake_snapshot_download(**kwargs):
            calls.append(kwargs)
            target = Path(kwargs["local_dir"])
            target.mkdir(parents=True, exist_ok=True)
            (target / download_public_checkpoint.WEIGHTS_NAME).write_bytes(payload)
            return str(target)

        fake_module = mock.Mock(snapshot_download=fake_snapshot_download)
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            sys.modules, {"huggingface_hub": fake_module}
        ), mock.patch.object(
            download_public_checkpoint, "WEIGHTS_SIZE", len(payload)
        ), mock.patch.object(
            download_public_checkpoint,
            "WEIGHTS_SHA256",
            hashlib.sha256(payload).hexdigest(),
        ), mock.patch.object(
            sys,
            "argv",
            ["download_public_checkpoint", "--local-dir", directory, "--max-workers", "1"],
        ):
            download_public_checkpoint.main()

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["repo_id"], download_public_checkpoint.REPO_ID)
        self.assertEqual(calls[0]["revision"], download_public_checkpoint.REVISION)
        self.assertEqual(calls[0]["local_dir"], Path(directory))
        self.assertEqual(calls[0]["max_workers"], 1)

    def test_rejects_wrong_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            download_public_checkpoint, "WEIGHTS_SIZE", 3
        ), mock.patch.object(download_public_checkpoint, "WEIGHTS_SHA256", "0" * 64):
            model_dir = Path(directory)
            (model_dir / download_public_checkpoint.WEIGHTS_NAME).write_bytes(b"bad")
            with self.assertRaises(SystemExit):
                download_public_checkpoint.verify_weights(model_dir)


if __name__ == "__main__":
    unittest.main()
