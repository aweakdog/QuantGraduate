#!/usr/bin/env bash
# quant-strategy 应急更新器: 只从固定仓库的固定分支 **快进** 更新代码目录。
# 由 admin_plane.py 的 update 动作调用 (也可 SSH 手工跑)。所有可变量写死在这里,
# 不从请求里来。流程: 锁 -> 校验 origin -> fetch main -> 拒绝非快进 -> git archive
# 到临时目录 -> 拒绝 symlink/密钥文件 -> 用运行环境的 venv python 编译全部 .py
# (3.10 语法!) + 两个纯逻辑测试 -> rsync 进 scripts/ pipeline/ tests/ (不删、不碰
# data/ .venv/ 配置) -> 写 deployed-commit 标记。重启由单独的 restart 动作做。
#
# 注意: 应急面自身的三个文件 (admin_plane.py / admin_update.sh / admin_bootstrap.sh)
# 被排除, 仓库更新改不了应急边界; 改它们必须走 SSH。
set -euo pipefail
umask 077

REPO="git@github.com:aweakdog/QuantGraduate.git"
BRANCH="main"
LIVE="${QA_LIVE_DIR:-$HOME/quant-strategy}"
STATE="${QA_STATE_DIR:-$HOME/.local/state/quant-admin}"
SOURCE="$STATE/source"
MARKER="$STATE/deployed-commit"
LOCK="$STATE/update.lock"
PY="$LIVE/.venv/bin/python"
CODE_DIRS=(scripts pipeline tests)
SELF_EXCLUDES=(--exclude='admin_plane.py' --exclude='admin_update.sh' --exclude='admin_bootstrap.sh' --exclude='__pycache__/')

mkdir -p "$STATE"
chmod 700 "$STATE"
exec 9>"$LOCK"
flock -n 9 || { echo "another update is already running"; exit 20; }
[[ -x "$PY" ]] || { echo "venv python missing at $PY"; exit 26; }
[[ -d "$LIVE/scripts" ]] || { echo "live dir missing: $LIVE"; exit 26; }

export GIT_SSH_COMMAND="ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=15"
export GIT_CONFIG_GLOBAL=/dev/null
export GIT_CONFIG_NOSYSTEM=1
export GIT_TERMINAL_PROMPT=0

if [[ ! -d "$SOURCE/.git" ]]; then
  git clone --quiet --no-checkout "$REPO" "$SOURCE"
fi
origin="$(git -C "$SOURCE" remote get-url origin)"
[[ "$origin" == "$REPO" ]] || { echo "refusing unexpected origin: $origin"; exit 21; }

git -C "$SOURCE" fetch --quiet --prune origin "refs/heads/$BRANCH"
target="$(git -C "$SOURCE" rev-parse FETCH_HEAD)"
[[ "$target" =~ ^[0-9a-f]{40}$ ]] || { echo "invalid target commit"; exit 22; }

current=""
[[ -f "$MARKER" ]] && current="$(tr -d '[:space:]' < "$MARKER")"
if [[ -z "$current" ]]; then
  echo "no deployed-commit marker; bootstrap it over SSH first (admin_bootstrap.sh)"; exit 23
fi
[[ "$current" =~ ^[0-9a-f]{40}$ ]] || { echo "invalid deployed marker"; exit 23; }
if [[ "$current" == "$target" ]]; then
  echo "already current: $target"; exit 0
fi
git -C "$SOURCE" merge-base --is-ancestor "$current" "$target" || {
  echo "refusing non-fast-forward update from $current to $target"; exit 24
}
echo "fast-forward: ${current:0:10} -> ${target:0:10} ($(git -C "$SOURCE" rev-list --count "$current..$target") commits)"
git -C "$SOURCE" log --oneline "$current..$target" | head -20

stage="$(mktemp -d "$STATE/update-stage.XXXXXX")"
trap 'rm -rf "$stage"' EXIT
git -C "$SOURCE" archive "$target" "${CODE_DIRS[@]}" | tar -x -C "$stage"

"$PY" - "$stage" <<'PYEOF'
import os, sys
root = sys.argv[1]
for base, dirs, files in os.walk(root):
    for name in dirs + files:
        path = os.path.join(base, name)
        if os.path.islink(path):
            raise SystemExit("refusing repository containing symlink: " + path)
        if name in (".env", "quant-web.env") or name.endswith((".pem", ".key", ".db", ".sqlite")):
            raise SystemExit("refusing repository containing secret/data file: " + path)
PYEOF

# 预检 1: 用生产 venv 的 python 编译全部 .py —— 服务器是 3.10, 本机是 3.13, 新语法在这里被拦
"$PY" -m compileall -q "$stage" >/dev/null || { echo "python compile check failed"; exit 25; }
find "$stage" -name '__pycache__' -type d -prune -exec rm -rf {} +
# 预检 2: 两个不碰数据的纯逻辑测试 (口令体系 / 生产线配置), 在暂存目录里跑, import 的是新代码
if [[ -f "$stage/tests/test_web_access_codes.py" && -f "$stage/tests/test_live_config_profiles.py" ]]; then
  ( cd "$stage" && timeout 180 "$PY" -m pytest -q -x -p no:cacheprovider \
      tests/test_web_access_codes.py tests/test_live_config_profiles.py 2>&1 | tail -5 ) \
    || { echo "preflight tests failed"; exit 25; }
fi
find "$stage" -name '__pycache__' -type d -prune -exec rm -rf {} +
find "$stage" -name '.pytest_cache' -type d -prune -exec rm -rf {} +

for d in "${CODE_DIRS[@]}"; do
  [[ -d "$stage/$d" ]] || continue
  mkdir -p "$LIVE/$d"
  rsync -a --no-links "${SELF_EXCLUDES[@]}" "$stage/$d/" "$LIVE/$d/"
done
printf '%s\n' "$target" > "$MARKER"
echo "deployed: $target"
echo "restart required"
