"""The SGLang commit inside a ROCm image, from its build history on Docker Hub.

    python scripts/ci/image_sglang.py [TAG]   # prints "<tag> <sglang commit>"

TAG defaults to the newest ``rocm/sgl-dev:v*-rocm10-mi35x-YYYYMMDD``, the image the
PR test's plan job picks. The image's ``docker/rocm.Dockerfile`` build records the
commit it checked out as ``SGL_BRANCH=<sha>`` in its history; no image is pulled.
"""

import json
import re
import sys
import urllib.request

REPO = "rocm/sgl-dev"
TAG = re.compile(r"^v[0-9][0-9A-Za-z.]*-rocm10-mi35x-[0-9]{8}$")
MANIFESTS = ", ".join(
    [
        "application/vnd.docker.distribution.manifest.v2+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.oci.image.index.v1+json",
    ]
)


def get(url, token=None, accept=None):
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    if accept:
        req.add_header("Accept", accept)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def newest_tag():
    tags = get(
        f"https://hub.docker.com/v2/repositories/{REPO}/tags"
        "?page_size=100&ordering=last_updated&name=-rocm10-mi35x-"
    )["results"]
    names = [t["name"] for t in tags if TAG.match(t["name"])]
    return max(names, key=lambda n: n.rsplit("-", 1)[-1])


def sglang_commit(tag):
    token = get(
        "https://auth.docker.io/token?service=registry.docker.io"
        f"&scope=repository:{REPO}:pull"
    )["token"]
    base = f"https://registry-1.docker.io/v2/{REPO}"
    m = get(f"{base}/manifests/{tag}", token, MANIFESTS)
    if "manifests" in m:  # a multi-platform index: the image's own manifest
        m = get(f"{base}/manifests/{m['manifests'][0]['digest']}", token, MANIFESTS)
    config = get(f"{base}/blobs/{m['config']['digest']}", token)
    for step in config.get("history", []):
        if hit := re.search(r"SGL_BRANCH=([0-9a-f]{40})", step.get("created_by", "")):
            return hit[1]
    raise SystemExit(f"no SGL_BRANCH=<sha> in {REPO}:{tag}'s history")


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else newest_tag()
    print(tag, sglang_commit(tag))
