# 无 VPN 应急维护面 (quant-admin, 2026-09-09 上线)

## 为什么有它

eez041 的 sshd 带来源白名单 (`AllowUsers *@143.89.*.*` 等四段), 校 VPN 一失效, 22 端口能连上但认证阶段被拒
—— 不是公钥问题, `ssh-copy-id` 救不了。而 8xxx 段的 HTTP(S) 公网可达 (quant-web 8737 就是这么被账户们用的)。
于是照 `English/VPN_FALLBACK_MAINTENANCE_PLAYBOOK.md` 的做法, 在 **8738** 起了一个独立的 HTTPS 应急面,
只做几个编译在代码里的固定动作。它 **不是 web shell**: 不接受命令、路径、仓库、分支或任何参数。

正常维护仍走 VPN + SSH。这条通道只在 "VPN 挂了 && 8738 还通" 时用。

## 能做什么 (且只能做这些)

```bash
cd "~/Documents/Documents - yuanhang’s MacBook Pro/code/quant-strategy"
python3 scripts/admin_client.py status    # 部署提交 / quant-web 进程与 HTTP 探活 / 日更链状态 / 账本文件 / 磁盘
python3 scripts/admin_client.py health    # quant-web 是否应答、unit 是否 active、更新器/venv/标记是否齐
python3 scripts/admin_client.py logs      # web_server.log 与 daily_rebuild.log 各末 100 行 (口令/token 已脱敏)
python3 scripts/admin_client.py backup    # 把 data/live 的 state_/plan_/confirm_ json 打成 tar.gz 存到服务器状态目录
python3 scripts/admin_client.py update    # 只从 github aweakdog/QuantGraduate 的 main **快进** 更新 scripts/ pipeline/ tests/
python3 scripts/admin_client.py restart   # systemctl --user restart quant-web; 客户端等到 MainPID 变化且 HTTP 应答才算成功
python3 scripts/admin_client.py daily     # systemctl --user start quant-daily (重跑日更链; 正在跑时拒绝)
```

## 应急改代码的标准流程

```bash
# 1. 本地改、跑测试、提交、推到 main (update 只认 origin/main, 且只认快进)
git add -p && git commit -m "..." && git push origin main
# 2. 无 VPN 时
python3 scripts/admin_client.py backup
python3 scripts/admin_client.py update      # 输出里会列出 fast-forward 的提交; "already current" 说明没推上去
python3 scripts/admin_client.py restart     # 改了 web_server/action_page/live_config 才需要; 只改 live_signal 不用
python3 scripts/admin_client.py status
```

`update` 在服务器上做的事 (`scripts/admin_update.sh`, 所有变量写死):
flock 互斥 → 校验 origin 是固定仓库 → fetch main → **拒绝非快进** (公网接口不给回滚) → `git archive` 到临时目录 →
拒绝 symlink / `.pem` `.key` `.env` `.db` → 用生产 venv 的 **python 3.10** `compileall` 全部 .py (本机是 3.13, 新语法在这一步被拦) →
在暂存目录跑 `test_web_access_codes` + `test_live_config_profiles` → `rsync --no-links` 进三个代码目录 (不删文件、不碰 `data/` `.venv/` 配置) →
写 `~/.local/state/quant-admin/deployed-commit` → 把活树自己的 git HEAD 对齐到该提交 (mixed reset, 不碰工作区)。
重启是单独动作, 为的是先看见更新校验结果。

**两条纪律**:
- 通过 SSH 手工 `scp` 到 041 的改动, 必须随后 commit+push; 否则下次应急 `update` 会把它覆盖回 main 的版本。
  (更新器每次都把 041 活树的 `git HEAD` 对齐到已部署提交, 所以 `git -C ~/quant-strategy status` 里代码目录有东西 = 有未推的手工改动。)
- `status` 里 `daily.active_state` 是 `activating` 时 (日更链正在跑, 21:30 起约 1~2h) 不要 `update`/`daily`。

## 安全边界

- 独立 256 位 token, 只在服务器 `~/.config/quant-admin.env` (0600) 与本机
  `~/Library/Application Support/QuantAdmin/admin-token` (0600); 不进 git/URL/命令行/聊天。
- 请求签名 HMAC-SHA256(token, `ts\nnonce\nMETHOD\npath\nsha256(body)`), 头 `X-QA-Timestamp/Nonce/Signature`;
  服务器只收 ±60s 内、nonce 120s 内不重复的请求; 常量时间比较; 单 IP 10 分钟连错 5 次锁 (429, 正确签名也拒)。
- 自签证书, 客户端固定整张证书的 SHA-256 (`~/Library/Application Support/QuantAdmin/server-cert.sha256`);
  不匹配立即中止, 不是 `curl -k`。换证书后必须 SSH 重新核对再改本地指纹。
- 带浏览器 `Origin` 的请求 401; 响应 `no-store`; 认证失败只回泛化 401。
- 独立进程 `quant-admin.service`, 用系统 `/usr/bin/python3` 跑纯 stdlib 的 `scripts/admin_plane.py`:
  quant-web 崩了、venv 坏了、`web_server.py` 语法错, 它都还活着能 update+restart 自救。
- 更新器排除 `admin_plane.py` / `admin_update.sh` / `admin_bootstrap.sh` 自身: 一次仓库更新改不了应急边界, 改它们必须 SSH。
- 审计: `~/.local/state/quant-admin/audit.log` 每请求一行 JSON (ts/ip/action/result/detail), 不记 token、签名、请求头。
- 进程以 `yliog` 跑, 无 sudo; 动作里没有任何删除/回滚/强制项。

## 引导 / 换钥 / 撤销 (都要 SSH)

```bash
ssh eez041.ece.ust.hk
bash ~/quant-strategy/scripts/admin_bootstrap.sh            # 幂等: token/证书/标记/unit; 打印 token 与证书指纹
bash ~/quant-strategy/scripts/admin_bootstrap.sh --rotate   # 换 token (旧的立刻作废), 本机 admin-token 跟着换
bash ~/quant-strategy/scripts/admin_bootstrap.sh --recert   # 换证书, 本机 server-cert.sha256 跟着换
# 撤销
systemctl --user disable --now quant-admin.service && rm ~/.config/quant-admin.env
```

本机凭据落盘 (两文件都 `chmod 600`, 客户端会拒绝权限不对的文件):

```bash
mkdir -p "$HOME/Library/Application Support/QuantAdmin" && chmod 700 "$HOME/Library/Application Support/QuantAdmin"
ssh eez041.ece.ust.hk 'grep ^QA_ADMIN_TOKEN= ~/.config/quant-admin.env | cut -d= -f2-' > "$HOME/Library/Application Support/QuantAdmin/admin-token"
ssh eez041.ece.ust.hk 'openssl x509 -in ~/.config/quant-admin/cert.pem -outform DER | sha256sum | cut -d" " -f1' > "$HOME/Library/Application Support/QuantAdmin/server-cert.sha256"
chmod 600 "$HOME/Library/Application Support/QuantAdmin/"*
```

## 它救不了的故障

8738 不可达 / 证书或 token 文件丢了 / `admin_plane.py` 自己坏了 / 主机、磁盘、账号故障 / GitHub 不可达。
这些只能等 VPN、校园网 SSH 或找管理员。`refusing non-fast-forward update` 也是刻意的: 回滚请走 SSH 人工判断。

## 排错

| 现象 | 含义 |
|---|---|
| `HTTP 401: unauthorized` | 本机 token/指纹文件、或本机时钟偏了 >60s; 别连试, 5 次锁 10 分钟 |
| `TLS certificate fingerprint mismatch` | 证书换了 / 连错主机 / 中间人; 停, SSH 核对 |
| `HTTP 503: admin interface disabled` | 服务器 env 里没 token, 需 SSH 重新 bootstrap |
| `HTTP 409: another admin action is running` | 上一个 update/backup 还没完 |
| `update failed (25)` | 预检失败: 3.10 编译不过或两组测试挂了, 修了再推 |
| `update failed (24)` | 非快进: main 被 force-push 或落后于服务器标记 |

验收 (2026-09-09): 鉴权 10 项回归 `tests/test_admin_plane.py`; 两次真实快进 (`13143af → f1a235d → 本提交`) 走完
fetch→archive→3.10 compileall→26 测试→rsync→标记→HEAD 对齐, 单次 17s; restart 闭环 PID 2364027→2378569 + HTTP 探活。
**校外无 VPN 的可达性还要用手机热点跑一次 `status` 确认** (8738 在校防火墙实测放行的 8000-8929 段内, 但没在校外亲测)。
