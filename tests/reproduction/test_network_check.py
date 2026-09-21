import os
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from reproduction.check_network import probe, targets


class NetworkCheckTests(unittest.TestCase):
    def test_hf_mirror_is_respected(self) -> None:
        with patch.dict(os.environ, {"HF_ENDPOINT": "https://hf-mirror.com/"}):
            self.assertEqual(targets()["Hugging Face"], "https://hf-mirror.com/")

    def test_conda_channel_is_covered(self) -> None:
        self.assertIn("repo.anaconda.com", targets()["Conda defaults"])

    def test_http_auth_error_is_reachable(self) -> None:
        error = HTTPError("https://api.openai.com/v1/models", 401, "Unauthorized", {}, None)
        with patch("reproduction.check_network.urlopen", side_effect=error):
            self.assertIn("route reachable", probe(("OpenAI API", error.url), 1)[2])

    def test_network_failure_is_not_reachable(self) -> None:
        with patch("reproduction.check_network.urlopen", side_effect=URLError("DNS failure")):
            self.assertIn("UNREACHABLE", probe(("GitHub", "https://github.com/"), 1)[2])


if __name__ == "__main__":
    unittest.main()
