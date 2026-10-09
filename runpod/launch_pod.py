"""Create a RunPod GPU pod that runs the 4K/120 pipeline on a clip.

Reads RUNPOD_API_KEY from the environment (never hard-code/commit it).

Example:
  RUNPOD_API_KEY=xxxx python runpod/launch_pod.py \
    --repo-url https://github.com/<user>/craps.git \
    --input-url "https://youtu.be/RozZUnkVcgI" \
    --start 365 --duration 4
"""

import argparse
import json
import os
import sys
import urllib.request

API = "https://api.runpod.io/graphql"


def load_api_key():
    key = os.environ.get("RUNPOD_API_KEY")
    if key:
        return key.strip()
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if os.path.exists(env_path):
        with open(env_path) as fh:
            for line in fh:
                line = line.strip()
                if line.startswith("RUNPOD_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None

MUTATION = (
    "mutation($input: PodFindAndDeployOnDemandInput!) {"
    " podFindAndDeployOnDemand(input: $input) { id name imageName machineId desiredStatus }"
    "}"
)


def gql(api_key, query, variables):
    body = json.dumps({"query": query, "variables": variables}).encode()
    req = urllib.request.Request(
        f"{API}?api_key={api_key}", data=body,
        headers={"content-type": "application/json",
                 "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def main():
    ap = argparse.ArgumentParser(description="Launch a RunPod pod to run the 4K/120 pipeline")
    ap.add_argument("--repo-url", required=True, help="git URL the pod clones")
    ap.add_argument("--input-url", required=True)
    ap.add_argument("--start", type=float, required=True, help="clip start (s)")
    ap.add_argument("--duration", type=float, required=True, help="clip duration (s)")
    ap.add_argument("--gpu-type", default="NVIDIA GeForce RTX 4090")
    ap.add_argument("--image", default="runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04")
    ap.add_argument("--name", default="craps-4k120")
    ap.add_argument("--cloud-type", default="ALL")
    ap.add_argument("--disk", type=int, default=60)
    ap.add_argument("--volume", type=int, default=60)
    ap.add_argument("--public-key-file", default=None,
                    help="SSH public key file; enables SSH and skips dockerArgs")
    args = ap.parse_args()

    api_key = load_api_key()
    if not api_key:
        sys.exit("ERROR: set RUNPOD_API_KEY (env var) or put RUNPOD_API_KEY=... in .env")

    env = [
        {"key": "REPO_URL", "value": args.repo_url},
        {"key": "INPUT_URL", "value": args.input_url},
        {"key": "START", "value": str(args.start)},
        {"key": "DURATION", "value": str(args.duration)},
    ]

    if args.public_key_file:
        with open(args.public_key_file) as fh:
            env.append({"key": "PUBLIC_KEY", "value": fh.read().strip()})
        docker_args = ""
    else:
        docker_args = (
            "bash -c 'cd /workspace && "
            '[ -d craps/.git ] || git clone --depth 1 "$REPO_URL" craps; '
            'REPO_URL="$REPO_URL" INPUT_URL="$INPUT_URL" START="$START" DURATION="$DURATION" '
            "bash craps/runpod/pod_bootstrap.sh'"
        )

    payload = {
        "cloudType": args.cloud_type,
        "gpuCount": 1,
        "volumeInGb": args.volume,
        "containerDiskInGb": args.disk,
        "gpuTypeId": args.gpu_type,
        "name": args.name,
        "imageName": args.image,
        "dockerArgs": docker_args,
        "ports": "8888/http,22/tcp",
        "volumeMountPath": "/workspace",
        "env": env,
    }

    result = gql(api_key, MUTATION, {"input": payload})
    print(json.dumps(result, indent=2))
    pod = (result.get("data") or {}).get("podFindAndDeployOnDemand")
    if pod:
        print(f"\npod id: {pod['id']}  status: {pod.get('desiredStatus')}")
        print("Track it at https://www.runpod.io/console/pods")
        print("Output (when done): /workspace/out/out_4k120.mp4 on the pod's volume")


if __name__ == "__main__":
    main()