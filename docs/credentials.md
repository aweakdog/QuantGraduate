# 账号密码与环境变量备忘

> **本仓库为私有仓库**，凭据明文保存。若将来改为公开仓库，见文末「转公开前必做」。

真实值统一存放在仓库根目录的 **`.env`**，源码里不再硬编码，改密码只改这一处。

---

## 1. 同花顺 thsdk 正式账号

| 项 | 值 |
|---|---|
| 用户名 | `KQ2026` |
| 密码 | `lin88888888` |
| 环境变量 | `THS_USERNAME` / `THS_PASSWORD` |

**用途**：拉 1 分钟 K 线、主力资金流、龙虎榜事件、板块概念。

**限制**：`thsdk` 只能在 **Windows** 上跑。Mac 上这些脚本用不了，日线数据改用 akshare（`scripts/update_kline_akshare.py`，无需账号）。

**用到的脚本**：

```
pipeline/daily_pull.py            pipeline/pull_oos_events.py
pipeline/pull_1min_thsdk.py       pipeline/pull_oos_events_v2.py
pipeline/pull_oos_100_ff.py       pipeline/lib/ths_utils.py
scripts/pull_daily_120.py         scripts/backfill_recent_kline.py
scripts/auto_wf_wechat.py
```

---

## 2. 实盘网页的四个密码

跑在服务器 `eez041.ece.ust.hk:8080` 上的实盘看板。这些**不在 `.env` 里**，
而是在服务器的 `~/.config/quant-web.env`（由 systemd 的 `EnvironmentFile=` 读入，权限 600）。

| 变量 | 值 | 护住什么 | 有效期 |
|---|---|---|---|
| `QUANT_VIEW_PASSWORD` | `611611` | **能不能打开这个站**：首页、`/pro`、推荐榜、所有 `/api/*` | 30 天 |
| `QUANT_OPS_PASSWORD` | `611611` | **首页改账操作**：改名、切记账方式、确认成交、校准现金、存取现金、删持仓、重置 | 12 小时 |
| `QUANT_PRO_OPS_PASSWORD` | `123321123` | **运维面板 `/pro` 的写操作**：生成今日信号、人工对账、访问日志查询 | 12 小时 |
| `QUANT_BT_PASSWORD` | `Llyh19990710` | 隐藏的回测页 `/backtest`（首页不链接到它） | 7 天 |

四道口令**互不通用**：签名密钥掺入了不同的用途字符串（`ops:` / `pro-ops:` / `bt:`），
所以回测页的 cookie 当不了写权限用，运维面板的口令也当不了首页改账用。
能看不等于能改——即使已登录，改账仍要单独输对应口令。

要点：

- **没设口令时相关功能直接禁用而不是放行**。`QUANT_VIEW_PASSWORD` 缺失时
  **整站返回 503**（页面会写清怎么配）；`QUANT_OPS_PASSWORD` 缺失时所有改账接口 503。
  这是故意的：宁可用不了，不可裸奔。
- 输一次密码后发 cookie（HMAC 签名 + 过期时间戳，服务端不存 session，HttpOnly）。
- 因为签名密钥就是密码本身，**改密码会令所有已发 cookie 失效**（这是好事）。
- 防爆破：单 IP **连错 5 次锁 30 分钟**，每次失败延时 0.7 秒。计数在内存里，
  重启服务即解锁。注意锁定期间**正确密码也会被拒**，这是有意的。

### 已知的安全边界

- **站点是 HTTP 不是 HTTPS**，口令在网络上是明文传输的。同一网络里能抓包的人
  可以拿到它。要彻底解决需要证书 + 反向代理，目前没做。
- `QUANT_VIEW_PASSWORD` 是 6 位纯数字（共 100 万种），靠上面的限流撑着。
  在 `/pro` 的「访问日志」面板里若看到**口令失败次数持续上涨**，就该换长口令。
- uvicorn 直接监听 `0.0.0.0:8080`，**前面没有可信代理**，所以
  `X-Forwarded-For` 是访客可伪造的，日志里的来源只认 TCP 对端地址。

### 访问日志

`/pro` 页 →「👁 访问日志」按钮（要 ops 口令）。用来判断网址有没有泄漏。

| 文件 | 内容 |
|---|---|
| `data/live/access_stats.json` | 每天/每 IP 的**精确**计数，留 90 天 |
| `data/live/access_log.jsonl` | 事件流；同 IP 同路径 5 分钟内合并为一条，4MB 轮转 |
| `data/live/ip_geo_cache.json` | IP 归属地缓存 |

- **只有事件流去重，计数永远精确**。去重是因为页面自身每 60 秒轮询一次，
  否则你自己的浏览器一天就能刷出上千行，把陌生访问淹没。
- **登录接口永不去重**——它们就是密码尝试本身，合并了就看不见爆破。
- 归属地走 `ip-api.com`，**只在你打开日志页时才查**并缓存；离线时降级为「未解析」，
  不影响 IP 和时间。不想把 IP 发给第三方就点面板上的「归属地:关」。
- 请求体一律不记（登录密码就在里面）。

改密码：

```bash
ssh eez041.ece.ust.hk
vi ~/.config/quant-web.env
systemctl --user restart quant-web.service
```

---

## 3. 其他环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `QUANT_MODE` | `backtest` | `live` = 走 wencai 实时；`backtest` = 走价格代理 |
| `QUANT_DATA_DIR` | 空 | 数据根目录，留空则用仓库内 `data/` |
| `THS_PYTHON` | 空 | Windows 上 thsdk 专用 Python 解释器路径 |
| `UFD_ROUTER` | 空 | UFD 路由，可选 |

> 凭据共三组：同花顺账号（§1，在 `.env`）+ 网页两个密码（§2，在服务器
> `~/.config/quant-web.env`）。没有企业微信 webhook、neo4j 密码或其他 API key。

---

## 4. 怎么生效

**Python 代码**：`pipeline/config.py` 启动时自动 `load_dotenv()` 读 `.env`，无需手动操作。

```python
from pipeline.config import settings
settings.THS_USERNAME   # KQ2026
settings.THS_PASSWORD   # lin88888888
```

**Shell / 直接跑脚本**：

```bash
set -a; source .env; set +a
python pipeline/pull_1min_thsdk.py
```

**优先级**：已存在的环境变量 > `.env`（`override=False`），所以临时覆盖只需 `THS_PASSWORD=xxx python ...`。

---

## 5. 换密码怎么办

- **同花顺**：改 `.env` 里的 `THS_PASSWORD` 一处即可，源码不用动。
- **网页密码**：改服务器 `~/.config/quant-web.env`，然后重启 `quant-web.service`。

两者改完都顺手更新本文档对应的表格。

---

## 6. 转公开前必做

一旦要把仓库改成 Public，**按顺序**执行：

1. 在同花顺**修改密码**（历史提交里的旧密码等同已泄露，改仓库可见性不能撤销这一点）
   —— 同时改掉 `QUANT_OPS_PASSWORD` 与 `QUANT_BT_PASSWORD`，本文档 §2 也写了明文
2. `.gitignore` 里恢复 `.env` 屏蔽（把 `# .env` 的注释去掉）
3. `git rm --cached .env`
4. 删除本文档，或抹掉其中的密码
5. 用 `git filter-repo` 清洗历史里的 `.env` 和本文档

第 1 步最关键：只要密码进过任何一次公开提交，就必须视为已泄露。
