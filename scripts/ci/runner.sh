#!/usr/bin/env bash
# runner.sh -- the PR test's GitHub Actions runners on a Slurm login node.
#
# Each runner instance (rN) is pinned to one MI355X CI node (SLURM_NODELIST in
# its .env) and takes one job at a time, so two runners keep the two dedicated
# nodes busy and never put two jobs on one node. The runner does no GPU work: it
# checks out the CI scripts and runs sbatch/squeue (slurm_launch.sh). Each instance is a systemd --user service; login-node
# sessions are reaped at logout, so this needs `loginctl enable-linger`.
#
#   bash runner.sh install N [NODE]         download the runner into rN, pinned to NODE
#                                           (default: the Nth of NODES below); writes the
#                                           cluster settings below into its .env
#   bash runner.sh register N [REPO_URL]    register rN (default https://github.com/sgl-project/sgl-amd-mono);
#                                           reads a registration token on stdin
#   bash runner.sh start|stop|restart N
#   bash runner.sh status
#   bash runner.sh unregister N             reads a removal token on stdin
#
# Tokens: `gh api -X POST repos/<owner>/<repo>/actions/runners/registration-token`
# (a repo admin), valid one hour. Never pass them as arguments:
# /proc/<pid>/cmdline is world-readable on a shared host.

set -euo pipefail
# The login node's own disk: a shared home may be slow to write, and a runner
# writes its checkouts and tools under its directory.
ROOT="${MONO_RUNNER_ROOT:-/var/tmp/sgl-amd-mono-runner-$(id -un)}"
UNIT_DIR="$HOME/.config/systemd/user"
UNIT=sgl-amd-mono-runner
LABELS=sgl-amd-mono-mi355x
# The CI nodes, r1 and r2: two of the six nodes of Crusoe's amd-aifw-sglang-ci
# partition. sglang's own CI runners there must keep off them (SLURM_EXCLUDE in
# their .env), or a job here waits behind theirs.
NODES=(crsuse2-m2m-v3-060 crsuse2-m2m-v3-123)
# Cluster settings (see slurm_launch.sh); the environment at install overrides each
# one, and a later install without it writes these defaults again.
: "${SLURM_PARTITION:=amd-aifw-sglang-ci}"
: "${CI_NODE_SCRATCH:=/mnt/m2m_nobackup/sgl-amd-mono-ci-$(id -un)}"
: "${CI_MODEL_ROOT:=/shared_nfs/huggingface_models}"

write_env() {
  local dir=$1 node=$2
  {
    echo "LANG=C.UTF-8"
    echo "SLURM_PARTITION=$SLURM_PARTITION"
    echo "SLURM_ACCOUNT=${SLURM_ACCOUNT:-$SLURM_PARTITION}"
    echo "SLURM_QOS=${SLURM_QOS:-$SLURM_PARTITION-qos}"
    echo "SLURM_NODELIST=$node"
    echo "CI_NODE_SCRATCH=$CI_NODE_SCRATCH"
    echo "CI_MODEL_ROOT=$CI_MODEL_ROOT"
    echo "CI_SHARED_DIR=$HOME/.sgl_amd_mono_ci"
    # The scheduler client config the login shell sets; a systemd service starts without it.
    env | grep -E '^SPUR_(CLUSTER_NAME|CONTROLLER_ADDR|AUTH_PLUGIN)=' || true
  } > "$dir/.env"
  echo "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" > "$dir/.path"
}

write_unit() {
  mkdir -p "$UNIT_DIR"
  cat > "$UNIT_DIR/$UNIT@.service" <<EOF
[Unit]
Description=sgl-amd-mono PR test GitHub Actions runner %i
After=network-online.target

[Service]
WorkingDirectory=$ROOT/%i
# As the runner's own svc.sh: runsvc.sh passes the stop on to the listener, which
# lets a running job finish its step.
ExecStart=$ROOT/%i/bin/runsvc.sh
Restart=always
RestartSec=30
KillMode=process
KillSignal=SIGTERM
TimeoutStopSec=5min

[Install]
WantedBy=default.target
EOF
  systemctl --user daemon-reload
}

inst() { [[ "$1" =~ ^[0-9]+$ ]] || { echo "instance must be a number" >&2; exit 2; }; echo "r$((10#$1))"; }  # 01 is r1

cmd=${1:-status}; shift || true
case "$cmd" in
  install)
    i=$(inst "${1:?instance}"); n=${i#r}; d="$ROOT/$i"
    node=${2:-}
    if [[ -z "$node" ]] && (( n >= 1 && n <= ${#NODES[@]} )); then node=${NODES[n - 1]}; fi
    [[ -n "$node" ]] || { echo "install $n: give the node (only r1..r${#NODES[@]} have one)" >&2; exit 2; }
    if [[ ! -x "$d/bin/Runner.Listener" ]]; then
      v=$(curl -sSI --max-time 30 https://github.com/actions/runner/releases/latest \
          | tr -d '\r' | awk -F/ 'tolower($1) ~ /^location:/ {print $NF}'); v=${v#v}
      [[ -n "$v" ]] || { echo "cannot resolve runner version" >&2; exit 1; }
      mkdir -p "$d"
      curl -sSfL --max-time 600 \
        "https://github.com/actions/runner/releases/download/v$v/actions-runner-linux-x64-$v.tar.gz" | tar xz -C "$d"
    fi
    write_env "$d" "$node"; write_unit
    echo "$i: runner $("$d/bin/Runner.Listener" --version) in $d, node $node" ;;
  register)
    i=$(inst "${1:?instance}"); url=${2:-https://github.com/sgl-project/sgl-amd-mono}; d="$ROOT/$i"
    [[ -x "$d/config.sh" ]] || { echo "run: install ${1} [NODE]" >&2; exit 2; }
    read -rs token; [[ -n "$token" ]] || { echo "no token on stdin" >&2; exit 2; }
    node=$(sed -n 's/^SLURM_NODELIST=//p' "$d/.env")
    (cd "$d" && ./config.sh --unattended --replace --url "$url" --token "$token" \
       --name "mono-ci-${node##*-}-$i" --labels "$LABELS" --work _work)
    systemctl --user enable --now "$UNIT@$i" ;;
  start|stop|restart)
    i=$(inst "${1:?instance}")
    [[ "$cmd" == start ]] && systemctl --user enable "$UNIT@$i"
    [[ "$cmd" == stop ]] && systemctl --user disable "$UNIT@$i"
    systemctl --user "$cmd" "$UNIT@$i" ;;
  unregister)
    i=$(inst "${1:?instance}"); d="$ROOT/$i"
    systemctl --user disable --now "$UNIT@$i" || true
    read -rs token; (cd "$d" && ./config.sh remove --token "$token") ;;
  status)
    echo "host $(hostname -s), linger: $(loginctl show-user "$(id -un)" -p Linger --value)"
    for d in "$ROOT"/r*; do
      [[ -d "$d" ]] || continue; i=$(basename "$d")
      reg=no; [[ -f "$d/.runner" ]] && reg=$(jq -r '"\(.agentName) -> \(.gitHubUrl)"' "$d/.runner" 2>/dev/null)
      printf '%-4s node=%-22s registered=%-60s service=%s\n' "$i" \
        "$(sed -n 's/^SLURM_NODELIST=//p' "$d/.env")" "$reg" "$(systemctl --user is-active "$UNIT@$i" 2>/dev/null)"
    done ;;
  *) sed -n '2,22p' "$0"; exit 1 ;;
esac
