#!/usr/bin/env bash
# Run a repo command on a Colab GPU from the terminal, detached, via the official Colab CLI
# (`uv tool install google-colab-cli`; auth once with `gcloud auth application-default login
# --scopes=openid,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory`).
# No browser tab is involved, so a sleeping laptop or a hidden tab cannot drop the run.
#
#   colab_job.sh start  <session> <branch> '<command run from the repo root>'
#   colab_job.sh tail   <session> [lines]     # last lines of the job log
#   colab_job.sh fetch  <session> [dest.tgz]  # tar the results dir and download it
#   colab_job.sh stop   <session>             # ALWAYS do this when done: idle VMs burn units
#
# The HF token is sent as a FILE (~/.cache/huggingface/token -> the VM's same path) and read on
# the VM, so its value never appears in a command, a log or the CLI's local history.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
colab() { command colab --auth=adc "$@"; }

action="${1:?usage: colab_job.sh start|tail|fetch|stop <session> ...}"
session="${2:?session name}"

case "$action" in
  start)
    branch="${3:?branch}"; cmd="${4:?command}"
    token="$HOME/.cache/huggingface/token"
    [ -s "$token" ] || { echo "FATAL: $token missing (run: uvx --from huggingface_hub hf auth login)" >&2; exit 1; }
    colab new -s "$session" --gpu T4 --high-mem
    tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
    echo 'import os; os.makedirs("/root/.cache/huggingface", exist_ok=True)' > "$tmp/prep.py"
    colab exec -s "$session" -f "$tmp/prep.py"
    colab upload -s "$session" "$token" /root/.cache/huggingface/token
    cat > "$tmp/launch.py" <<'PY'
import os, subprocess
os.environ["HF_TOKEN"] = open("/root/.cache/huggingface/token").read().strip()
branch, cmd = os.environ["JOB_BRANCH"], os.environ["JOB_CMD"]
script = f"curl -fsSL https://raw.githubusercontent.com/udshah31/sense/{branch}/scripts/kaggle_run.sh | bash -s -- {branch} {cmd!r}"
subprocess.Popen(["bash", "-c", script + " > /content/job.log 2>&1; echo EXIT=$? >> /content/job.log"],
                 start_new_session=True, env=os.environ.copy())
print("job started; log at /content/job.log")
PY
    colab exec -s "$session" -f "$tmp/launch.py" --env "JOB_BRANCH=$branch" --env "JOB_CMD=$cmd"
    ;;
  tail)
    n="${3:-40}"
    echo "print(''.join(open('/content/job.log').readlines()[-$n:]))" | colab exec -s "$session"
    ;;
  fetch)
    dest="${3:-colab-results-$session.tgz}"
    echo "import subprocess; subprocess.run(['tar','czf','/content/results.tgz','-C','/kaggle/working','sense-results'], check=True); print('tarred')" | colab exec -s "$session"
    colab download -s "$session" /content/results.tgz "$dest"
    ;;
  stop)
    colab stop -s "$session"
    ;;
  *) echo "unknown action $action" >&2; exit 2;;
esac
