#!/usr/bin/env bash
# Body of one slurm_launch.sh job: one model's PR test on a whole MI355X node.
#
# Fetches the PR and base commits onto node-local scratch (the shared home is too
# slow to hold checkouts), copies in the CI scripts the runner checked out from
# the base branch (so a PR cannot change what runs outside the container), starts
# the SGLang ROCm container and runs scripts/ci/run_model.py in it. The run's
# out/ (result.json, logs) goes back to $CI_RUN_DIR/out.tgz on shared storage.
#
# Environment (exported by slurm_launch.sh): CI_MODEL CI_TIER CI_HARNESS CI_REPO
# CI_PR CI_PR_SHA CI_BASE_SHA CI_IMAGE CI_NODE_SCRATCH CI_MODEL_ROOT CI_RUN_DIR
# CI_EXIT_FILE

exec 2>&1   # some schedulers' batch --output captures stdout only
set -uo pipefail

: "${CI_MODEL:?}" "${CI_HARNESS:?}" "${CI_REPO:?}" "${CI_PR_SHA:?}" "${CI_BASE_SHA:?}"
: "${CI_IMAGE:?}" "${CI_NODE_SCRATCH:?}" "${CI_MODEL_ROOT:?}" "${CI_RUN_DIR:?}" "${CI_EXIT_FILE:?}"
JOB_ID="${SLURM_JOB_ID:-${SPUR_JOB_ID:-$$}}"
WS="$CI_NODE_SCRATCH/job-$JOB_ID"
CACHE="$CI_NODE_SCRATCH/cache"
NAME="sgl-amd-mono-ci-$JOB_ID"
# The CI's HF token (read access to gated datasets such as GPQA), if the runner
# account has one. Used once, before any PR code runs, to fill the dataset cache.
HF_TOKEN_FILE="${HF_TOKEN_FILE:-$HOME/.sgl_amd_mono_ci/hf_token}"

echo "[mono-ci-job] start on $(hostname) job=$JOB_ID model=$CI_MODEL tier=${CI_TIER:-full} image=$CI_IMAGE $(date -u +%FT%TZ)"

finish() {
    local rc=$?
    tar czf "$CI_RUN_DIR/out.tgz" -C "$WS/out" . 2>/dev/null || true
    docker rm -f "$NAME" >/dev/null 2>&1 || true
    # The container wrote as root; remove its files from inside one (never pulling).
    { docker image inspect "$CI_IMAGE" >/dev/null 2>&1 \
        && docker run --rm -v "$CI_NODE_SCRATCH:/s" "$CI_IMAGE" rm -rf "/s/job-$JOB_ID" >/dev/null 2>&1; } \
        || rm -rf "$WS" 2>/dev/null || true
    echo "$rc" > "$CI_EXIT_FILE"
    echo "[mono-ci-job] exit $rc $(date -u +%FT%TZ)"
}
trap finish EXIT
set -e

checkout() {  # DIR SHA: the commit's tree, fetched by SHA (a fork's PR head via refs/pull)
    git init -q "$1"
    git -C "$1" fetch -q --depth 1 "$CI_REPO" "$2" \
        || git -C "$1" fetch -q "$CI_REPO" "refs/pull/${CI_PR:?}/head"
    git -C "$1" -c advice.detachedHead=false checkout -q "$2"
    echo "[mono-ci-job] $(basename "$1"): $(git -C "$1" log -1 --format='%h %s')"
}

mkdir -p "$WS/out" "$WS/harness/scripts" "$CACHE"
cp -a "$CI_HARNESS" "$WS/harness/scripts/ci"

# The model's checkpoints, mirrored to local disk: shared storage reads too slowly
# to load servers from. A mirror is redone when its source's config or index
# changes; the container sees the mirror at the source's path.
MIRROR="$CI_NODE_SCRATCH/models"
mkdir -p "$MIRROR$CI_MODEL_ROOT"
mirror() {
    local src=$1 dst="$MIRROR$1" stamp
    [[ -d "$src" ]] || { echo "[mono-ci-job] no checkpoint at $src"; return 0; }
    stamp="$(stat -c '%n %s %Y' "$src"/config.json "$src"/*.index.json 2>/dev/null || true)"
    [[ -f "$dst/.mirror" && "$(cat "$dst/.mirror")" == "$stamp" ]] && return 0
    echo "[mono-ci-job] mirroring $src to $dst $(date -u +%FT%TZ)"
    mkdir -p "$dst"
    rm -f "$dst/.mirror"
    # .cache: huggingface-cli's download metadata, not always readable by others
    rsync -a --delete --exclude .mirror --exclude .cache "$src/" "$dst/"
    echo "$stamp" > "$dst/.mirror"
    echo "[mono-ci-job] mirrored $src $(date -u +%FT%TZ)"
}
for path in $(python3 "$WS/harness/scripts/ci/models.py" paths "$CI_MODEL"); do
    mirror "$path"
done
# CI_STAGE_ONLY=1: fill the mirror and stop (staging a node ahead of the CI)
[[ -n "${CI_STAGE_ONLY:-}" ]] && exit 0

checkout "$WS/pr" "$CI_PR_SHA"
checkout "$WS/base" "$CI_BASE_SHA"

# This CI's containers from an earlier job that did not clean up. The node is
# held whole, so nothing else of ours runs here.
docker ps -aq --filter "name=^sgl-amd-mono-ci-" | xargs -r docker rm -f
echo "[mono-ci-job] GPU use before start:"
rocm-smi --showuse --showmemuse 2>/dev/null | grep -E '^GPU\[' || true

# The cluster's anonymous pulls share one egress IP that Docker Hub rate-limits.
if ! docker image inspect "$CI_IMAGE" >/dev/null 2>&1; then
    for wait in 60 120 240 480 0; do
        docker pull "$CI_IMAGE" && break
        [[ "$wait" == 0 ]] && { echo "cannot pull $CI_IMAGE"; exit 1; }
        echo "pull failed; retrying in ${wait}s"; sleep "$wait"
    done
fi

docker run -d --name "$NAME" \
    --network host --ipc host \
    --device /dev/kfd --device /dev/dri --group-add video \
    --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
    --shm-size 64g --ulimit memlock=-1 --ulimit stack=67108864 \
    -v "$WS:/ci" -v "$CACHE:/cache" -v "$MIRROR$CI_MODEL_ROOT:$CI_MODEL_ROOT:ro" \
    -e HF_HOME=/cache/hf -e PIP_CACHE_DIR=/cache/pip \
    -e CI_MODEL_ROOT="$CI_MODEL_ROOT" -e CI_IMAGE="$CI_IMAGE" -e CI_PR_SHA="$CI_PR_SHA" -e CI_BASE_SHA="$CI_BASE_SHA" \
    -e MONO_CI_NUM_EXAMPLES -e MONO_CI_CONCURRENCY \
    -w /ci "$CI_IMAGE" sleep infinity

if [[ -s "$HF_TOKEN_FILE" ]]; then
    echo "[mono-ci-job] filling the gated dataset cache"
    docker exec --env-file <(printf 'HF_TOKEN=%s\n' "$(tr -d '[:space:]' < "$HF_TOKEN_FILE")") "$NAME" \
        bash -c 'pip install -q sgl-eval==0.1.3 && sgl-eval run gpqa --base-url http://127.0.0.1:9/v1 \
                 --model none --num-examples 1 --out-dir /tmp/warm >/dev/null 2>&1; true'
fi

docker exec "$NAME" python3 /ci/harness/scripts/ci/run_model.py \
    --model "$CI_MODEL" --tier "${CI_TIER:-full}" --pr /ci/pr --base /ci/base --out /ci/out --cache /cache
