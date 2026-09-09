#!/usr/bin/env bash
# 在 eez041 上 (通过 VPN/校园网 SSH) 一次性引导 quant-admin 应急维护面。幂等, 可重复跑:
#   1. 生成 256 位管理 token (已有则不动)  -> ~/.config/quant-admin.env (0600)
#   2. 生成自签证书 (已有则不动)           -> ~/.config/quant-admin/{cert,key}.pem, 打印 SHA-256 指纹
#   3. 写 deployed-commit 标记 = origin/main 当前提交 (只在标记不存在时; 前提是活树代码已与 main 一致)
#   4. 写并启用 systemd --user unit quant-admin.service (端口 8738), 重启使配置生效
# 用法:  bash ~/quant-strategy/scripts/admin_bootstrap.sh            # 引导/修复
#        bash ~/quant-strategy/scripts/admin_bootstrap.sh --rotate    # 换 token (旧 token 立刻作废)
#        bash ~/quant-strategy/scripts/admin_bootstrap.sh --recert    # 换证书 (本机指纹文件要跟着换)
# 撤销: systemctl --user disable --now quant-admin.service && rm ~/.config/quant-admin.env
set -euo pipefail
umask 077

LIVE="${QA_LIVE_DIR:-$HOME/quant-strategy}"
STATE="${QA_STATE_DIR:-$HOME/.local/state/quant-admin}"
CONF_DIR="$HOME/.config/quant-admin"
ENV_FILE="$HOME/.config/quant-admin.env"
UNIT_DIR="$HOME/.config/systemd/user"
UNIT="$UNIT_DIR/quant-admin.service"
PORT="${QA_PORT:-8738}"
CERT="$CONF_DIR/cert.pem"; KEY="$CONF_DIR/key.pem"
ROTATE=0; RECERT=0
for a in "$@"; do
  case "$a" in
    --rotate) ROTATE=1 ;;
    --recert) RECERT=1 ;;
    *) echo "unknown arg: $a"; exit 2 ;;
  esac
done

mkdir -p "$STATE" "$CONF_DIR" "$UNIT_DIR"
chmod 700 "$STATE" "$CONF_DIR"

# 1. token
if [[ ! -f "$ENV_FILE" || "$ROTATE" == 1 ]]; then
  tok="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
  printf 'QA_ADMIN_TOKEN=%s\nQA_LIVE_DIR=%s\nQA_STATE_DIR=%s\n' "$tok" "$LIVE" "$STATE" > "$ENV_FILE"
  chmod 600 "$ENV_FILE"
  echo "token written to $ENV_FILE  (复制到本机: ~/Library/Application Support/QuantAdmin/admin-token, chmod 600)"
  echo "  TOKEN: $tok"
else
  echo "token exists: $ENV_FILE (--rotate 可换)"
fi

# 2. 自签证书 (10 年, IP SAN 写死 143.89.46.41; 客户端只比指纹不看 CN/SAN)
if [[ ! -f "$CERT" || ! -f "$KEY" || "$RECERT" == 1 ]]; then
  openssl req -x509 -newkey rsa:2048 -nodes -days 3650 -keyout "$KEY" -out "$CERT" \
    -subj "/CN=quant-admin" -addext "subjectAltName=IP:143.89.46.41" >/dev/null 2>&1
  chmod 600 "$KEY" "$CERT"
  echo "certificate written: $CERT"
fi
fp="$(openssl x509 -in "$CERT" -outform DER | sha256sum | awk '{print $1}')"
echo "cert sha256: $fp  (复制到本机: ~/Library/Application Support/QuantAdmin/server-cert.sha256)"

# 3. deployed-commit 标记
if [[ ! -f "$STATE/deployed-commit" ]]; then
  if [[ -d "$LIVE/.git" ]]; then
    head="$(git -C "$LIVE" rev-parse HEAD)"
    printf '%s\n' "$head" > "$STATE/deployed-commit"
    echo "deployed-commit marker set to live HEAD $head"
    echo "  !! 前提: 活树的 scripts/pipeline/tests 已与该提交一致 (git -C $LIVE status 里代码目录应干净)"
  else
    echo "WARN: $LIVE is not a git repo; write $STATE/deployed-commit by hand"
  fi
else
  echo "deployed-commit: $(cat "$STATE/deployed-commit")"
fi

# 4. systemd unit
cat > "$UNIT" <<EOF
[Unit]
Description=quant-web VPN-fallback admin plane (fixed actions over HTTPS)
After=network.target

[Service]
Type=simple
WorkingDirectory=$LIVE
EnvironmentFile=$ENV_FILE
Environment=QA_CERT=$CERT
Environment=QA_KEY=$KEY
ExecStart=/usr/bin/python3 $LIVE/scripts/admin_plane.py --host 0.0.0.0 --port $PORT
Restart=always
RestartSec=5
StandardOutput=append:$STATE/admin_plane.log
StandardError=append:$STATE/admin_plane.log

[Install]
WantedBy=default.target
EOF
chmod +x "$LIVE/scripts/admin_update.sh" "$LIVE/scripts/admin_plane.py" 2>/dev/null || true
systemctl --user daemon-reload
systemctl --user enable --now quant-admin.service >/dev/null 2>&1 || true
systemctl --user restart quant-admin.service
sleep 1
systemctl --user is-active quant-admin.service
echo "listening: $(ss -ltn 2>/dev/null | grep -c ":$PORT ") on port $PORT"
echo "audit log: $STATE/audit.log"
