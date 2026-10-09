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

MUTATION = (
    "mutation($input: PodFindAndDeployOnDemandInput!) {"
    " podFindAndDeployOnDemand(input: $input) { id name imageName machineId desiredStatus }"
    "}"
)


def gql(api_key, query, variables):
    body = json.dumps({"query": query, "variables": variables}).encode()
    req = urllib.request.Request(
        f"{API}?api_key={api_key}", data=body,
        headers={"content-type": "application/json"}, method="POST",
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
    args = ap.parse_args()

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        sys.exit("ERROR: set RUNPOD_API_KEY in the environment")

    bootstrap = (
        "bash -lc 'cd /workspace && "
        '[ -d craps/.git ] || git clone --depth 1 "$REPO_URL" craps && '
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
        "dockerArgs": bootstrap,
        "ports": "8888/http,22/tcp",
        "volumeMountPath": "/workspace",
        "env": [
            {"key": "REPO_URL", "value": args.repo_url},
            {"key": "INPUT_URL", "value": args.input_url},
            {"key": "START", "value": str(args.start)},
            {"key": "DURATION", "value": str(args.duration)},
        ],
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