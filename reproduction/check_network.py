"""Check outbound routes needed for AutoDL setup and optional remote Codex."""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def targets() -> dict[str, str]:
    hf_endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    return {
        "GitHub": "https://github.com/",
        "Hugging Face": hf_endpoint + "/",
        "Conda defaults": "https://repo.anaconda.com/pkgs/main/noarch/repodata.json",
        "PyPI": "https://pypi.org/simple/pip/",
        "PyTorch CUDA 12.4": "https://download.pytorch.org/whl/cu124/torch/",
        "ChatGPT": "https://chatgpt.com/",
        "OpenAI API": "https://api.openai.com/v1/models",
    }


def probe(item: tuple[str, str], timeout: float) -> tuple[str, str, str]:
    name, url = item
    request = Request(url, method="HEAD", headers={"User-Agent": "agent-g2-network-check/1"})
    try:
        with urlopen(request, timeout=timeout) as response:
            return name, url, f"HTTP {response.status}"
    except HTTPError as exc:
        # 401/403 still proves DNS, TLS, and HTTP routing; authorization is separate.
        return name, url, f"HTTP {exc.code} (route reachable)"
    except (OSError, URLError) as exc:
        return name, url, f"UNREACHABLE: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=8.0)
    args = parser.parse_args()
    routes = targets()
    with ThreadPoolExecutor(max_workers=len(routes)) as pool:
        results = list(pool.map(lambda item: probe(item, args.timeout), routes.items()))
    for name, url, status in results:
        print(f"{name:18} {status:36} {url}")
    print("HTTP 401/403 means the route responds, not that login or downloads will succeed.")


if __name__ == "__main__":
    main()
