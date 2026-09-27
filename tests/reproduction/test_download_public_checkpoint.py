import sys
import unittest
from pathlib import Path
from unittest import mock

from reproduction import download_public_checkpoint


class PublicCheckpointDownloadTests(unittest.TestCase):
    def test_forwards_pinned_revision_and_worker_limit(self) -> None:
        calls = []

        def fake_snapshot_download(**kwargs):
            calls.append(kwargs)
            return kwargs["local_dir"]

        fake_module = mock.Mock(snapshot_download=fake_snapshot_download)
        with mock.patch.dict(sys.modules, {"huggingface_hub": fake_module}), mock.patch.object(
            sys,
            "argv",
            ["download_public_checkpoint", "--local-dir", "/tmp/model", "--max-workers", "1"],
        ):
            download_public_checkpoint.main()

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["repo_id"], download_public_checkpoint.REPO_ID)
        self.assertEqual(calls[0]["revision"], download_public_checkpoint.REVISION)
        self.assertEqual(calls[0]["local_dir"], Path("/tmp/model"))
        self.assertEqual(calls[0]["max_workers"], 1)


if __name__ == "__main__":
    unittest.main()
