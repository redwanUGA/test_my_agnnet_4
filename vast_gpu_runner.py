#!/usr/bin/env python3
"""Run experiments on a rented Vast.ai GPU.

This script reads an API key from ``vast_api_key.txt`` and then:
1. Searches for a Pytorch-CUDA template instance with >=96% reliability,
   an NVIDIA GPU priced between 1-2 USD/hour.
2. Rents the instance with SSH enabled.
3. Copies the current repository to the instance, installs dependencies, and executes ``run_all_experiments.sh`` remotely.
4. Retrieves ``logs/`` and ``saved_models/`` back to the local machine.
5. Destroys the instance to avoid further charges.

API interaction uses the ``vastai-sdk`` Python package, which exposes the
same functionality as the ``vast`` command line. The script assumes
``ssh``/``scp`` are available on the host machine.

This file only contains logic.  It does not run automatically; invoke it as:

    python vast_gpu_runner.py
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any, Dict

from vastai_sdk import VastAI

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

def find_offer(client: VastAI) -> Dict[str, Any]:
    """Search the Vast.ai marketplace for a suitable offer.

    The query requests an offer with:
    - template image supporting PyTorch + CUDA
    - reliability >= 0.96
    - price between 1 and 2 USD/hour
    - NVIDIA GPU and SSH capability
    """
    query = {
        "verified": {"eq": True},
        "reliability": {"gte": 0.96},
        "gpu_name": {"like": "NVIDIA%"},
        "dph": {"between": [1.0, 2.0]},
    }
    offers = client.search_offers(query=query, type="all", limit=1, order="dph-")
    if not offers:
        raise RuntimeError("No suitable Vast.ai offers found.")
    return offers[0]

def create_instance(client: VastAI, offer: Dict[str, Any]) -> Dict[str, Any]:
    """Create (rent) a Vast.ai instance from ``offer``."""
    payload = {
        "image": "pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime",
        "disk": 20,
        "ssh": True,
    }
    return client.create_instance(id=offer["id"], **payload)

def wait_instance_ready(client: VastAI, instance_id: int) -> Dict[str, Any]:
    """Poll Vast.ai until the instance is ready for SSH."""
    while True:
        data = client.show_instance(id=instance_id)
        if data.get("state") == "running" and data.get("ssh_host") and data.get("ssh_port"):
            return data
        time.sleep(10)

def run_remote(host: str, port: int, user: str = "root") -> None:
    """Copy repo to the remote host, run experiments, and fetch results."""
    repo_dir = Path.cwd()
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

def destroy_instance(client: VastAI, instance_id: int) -> None:
    """Terminate the rented Vast.ai instance."""
    client.destroy_instance(id=instance_id)

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    api_key = read_api_key(API_KEY_PATH)
    client = VastAI(api_key=api_key)
    offer = find_offer(client)
    instance = create_instance(client, offer)
    instance_id = instance["id"]
    try:
        ready_info = wait_instance_ready(client, instance_id)
        host = ready_info["ssh_host"]
        port = ready_info["ssh_port"]
        user = ready_info.get("ssh_user", "root")
        run_remote(host, port, user)
    finally:
        destroy_instance(client, instance_id)

if __name__ == "__main__":
    main()
