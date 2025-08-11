#!/usr/bin/env python3
"""Run experiments on a rented Vast.ai GPU.

This script reads an API key from ``vast_api_key.txt`` and then:
1. Searches for a Pytorch-CUDA template instance with >=96% reliability,
   an NVIDIA GPU priced between 1-2 USD/hour.
2. Rents the instance with SSH enabled.
3. Copies the current repository to the instance, installs dependencies, and executes ``run_all_experiments.sh`` remotely.
4. Retrieves ``logs/`` and ``saved_models/`` back to the local machine.
5. Destroys the instance to avoid further charges.

The script uses the official ``vastai`` Python package for all API
interaction and assumes ``ssh``/``scp`` are available on the host machine.

This file only contains logic.  It does not run automatically; invoke it as:

    python vast_gpu_runner.py
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Dict

import vast
from urllib.parse import urlparse

# The vast.ai helper functions expect a global ``ARGS`` namespace.
vast.ARGS = argparse.Namespace(curl=False)

# Path to the API key text file.  The file should contain only the key.
API_KEY_PATH = Path("vast_api_key.txt")

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def read_api_key(path: Path) -> str:
    """Read the Vast.ai API key from ``path``."""
    if not path.exists():
        raise FileNotFoundError(
            f"API key file '{path}' not found. Create it with your Vast.ai key."
        )
    return path.read_text().strip()

def build_args(api_key: str) -> argparse.Namespace:
    """Create a minimal ``argparse.Namespace`` understood by ``vast`` helpers."""
    return argparse.Namespace(
        api_key=api_key,
        url=vast.server_url_default,
        retry=3,
        verify=True,
        explain=False,
    )

def find_offer(api_key: str) -> Dict[str, Any]:
    """Search the Vast.ai marketplace for a suitable offer.

    The query requests an offer with:
    - template image supporting PyTorch + CUDA
    - reliability >= 0.96
    - price between 1 and 2 USD/hour
    - NVIDIA GPU and SSH capability
    """
    args = build_args(api_key)
    query = {
        "verified": {"eq": True},
        "reliability": {"gte": 0.96},
        "gpu_name": {"like": "NVIDIA%"},
        "dph": {"between": [1.0, 2.0]},
        "order": [["dph", "asc"]],
        "type": "all",
        "limit": 1,
    }
    url = vast.apiurl(args, "/search/asks/")
    headers = vast.apiheaders(args)
    resp = vast.http_put(args, url, headers=headers, json={"q": query})
    resp.raise_for_status()
    offers = resp.json().get("offers", [])
    if not offers:
        raise RuntimeError("No suitable Vast.ai offers found.")
    return offers[0]

def create_instance(api_key: str, offer: Dict[str, Any]) -> Dict[str, Any]:
    """Create (rent) a Vast.ai instance from ``offer``."""
    args = build_args(api_key)
    payload = {
        "server_id": offer["id"],
        "image": "pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime",
        "disk": 20,
        "ssh": True,
    }
    url = vast.apiurl(args, "/instances/")
    headers = vast.apiheaders(args)
    resp = vast.http_post(args, url, headers=headers, json=payload)
    resp.raise_for_status()
    return resp.json()

def wait_instance_ready(api_key: str, instance_id: int) -> Dict[str, Any]:
    """Poll Vast.ai until the instance is ready for SSH."""
    args = build_args(api_key)
    url = vast.apiurl(args, f"/instances/{instance_id}/")
    headers = vast.apiheaders(args)
    while True:
        resp = vast.http_get(args, url, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        if data.get("state") == "running" and data.get("ssh_uri"):
            return data
        time.sleep(10)

def run_remote(ssh_uri: str) -> None:
    """Copy repo to the remote host, run experiments, and fetch results."""
    repo_dir = Path.cwd()
    parsed = urlparse(f"ssh://{ssh_uri}")
    host = parsed.hostname
    port = parsed.port or 22
    user = parsed.username or "root"
    remote = f"{user}@{host}"

    # Copy repository to the remote machine.
    subprocess.run(["scp", "-P", str(port), "-r", str(repo_dir), f"{remote}:~/repo"], check=True)

    # Execute the experiment script remotely. Install dependencies and then run the experiment script with an
    # environment variable that prevents recursive Vast.ai orchestration.
    remote_cmd = (
        "cd repo && "
        "python3 -m pip install -r requirements.txt && "
        "RUNNING_IN_VAST=1 bash run_all_experiments.sh"
    )
    subprocess.run(["ssh", "-p", str(port), remote, remote_cmd], check=True)

    # Retrieve results back to the local machine.
    subprocess.run(["scp", "-P", str(port), "-r", f"{remote}:~/repo/logs", str(repo_dir)], check=True)
    subprocess.run(["scp", "-P", str(port), "-r", f"{remote}:~/repo/saved_models", str(repo_dir)], check=True)

def destroy_instance(api_key: str, instance_id: int) -> None:
    """Terminate the rented Vast.ai instance."""
    args = build_args(api_key)
    url = vast.apiurl(args, f"/instances/{instance_id}/")
    headers = vast.apiheaders(args)
    vast.http_del(args, url, headers=headers)

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    api_key = read_api_key(API_KEY_PATH)
    offer = find_offer(api_key)
    instance = create_instance(api_key, offer)
    instance_id = instance["id"]
    try:
        ready_info = wait_instance_ready(api_key, instance_id)
        ssh_uri = ready_info["ssh_uri"]
        run_remote(ssh_uri)
    finally:
        destroy_instance(api_key, instance_id)

if __name__ == "__main__":
    main()
