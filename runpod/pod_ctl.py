"""Query/stop/terminate a RunPod pod. Reads RUNPOD_API_KEY from env or .env."""

import argparse
import json
import os
import sys
import time
import urllib.request

API = "https://api.runpod.io/graphql"


def load_key():
    key = os.environ.get("RUNPOD_API_KEY")
    if key:
        return key.strip()
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if os.path.exists(env_path):
        with open(env_path) as fh:
            for line in fh:
                if line.strip().startswith("RUNPOD_API_KEY="):
                    return line.strip().split("=", 1)[1].strip().strip('"').strip("'")
    return None


def gql(key, query, variables=None):
    body = json.dumps({"query": query, "variables": variables or {}}).encode()
    req = urllib.request.Request(
        f"{API}?api_key={key}", data=body,
        headers={"content-type": "application/json",
                 "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


STATUS_Q = ("query($id:String!){ pod(input:{podId:$id}){ id desiredStatus "
            "runtime { uptimeInSeconds gpus { gpuUtilPercent memoryUtilPercent } "
            "container { cpuPercent memoryPercent } ports { publicPort privatePort type } } } }")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pod_id")
    ap.add_argument("action", choices=["status", "ssh", "stop", "terminate"], default="status", nargs="?")
    args = ap.parse_args()
    key = load_key()
    if not key:
        sys.exit("no RUNPOD_API_KEY")

    if args.action == "status":
        print(json.dumps(gql(key, STATUS_Q, {"id": args.pod_id}), indent=2))
    elif args.action == "ssh":
        ports_q = ("query($id:String!){ pod(input:{podId:$id}){ runtime { "
                   "ports { ip isIpPublic publicPort privatePort type } } } }")
        for _ in range(40):
            r = gql(key, ports_q, {"id": args.pod_id})
            ports = (((r.get("data") or {}).get("pod") or {}).get("runtime") or {}).get("ports")
            for pr in ports or []:
                if pr.get("privatePort") == 22:
                    print(f"{pr.get('ip')} {pr.get('publicPort')}")
                    return
            time.sleep(10)
        print("no ssh port yet")
    elif args.action == "stop":
        print(json.dumps(gql(key, "mutation($id:String!){ podStop(input:{podId:$id}){ id desiredStatus } }",
                             {"id": args.pod_id}), indent=2))
    else:
        print(json.dumps(gql(key, "mutation($id:String!){ podTerminate(input:{podId:$id}){ id } }",
                             {"id": args.pod_id}), indent=2))


if __name__ == "__main__":
    main()