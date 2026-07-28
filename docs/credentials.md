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

## 2. 其他环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `QUANT_MODE` | `backtest` | `live` = 走 wencai 实时；`backtest` = 走价格代理 |
| `QUANT_DATA_DIR` | 空 | 数据根目录，留空则用仓库内 `data/` |
| `THS_PYTHON` | 空 | Windows 上 thsdk 专用 Python 解释器路径 |
| `UFD_ROUTER` | 空 | UFD 路由，可选 |

> 全项目**只有同花顺一组凭据**，没有企业微信 webhook、neo4j 密码或其他 API key。

---

## 3. 怎么生效

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

## 4. 换密码怎么办

改 `.env` 里的 `THS_PASSWORD` 一处即可，源码不用动。改完顺手更新本文档第 1 节的表格。

---

## 5. 转公开前必做

一旦要把仓库改成 Public，**按顺序**执行：

1. 在同花顺**修改密码**（历史提交里的旧密码等同已泄露，改仓库可见性不能撤销这一点）
2. `.gitignore` 里恢复 `.env` 屏蔽（把 `# .env` 的注释去掉）
3. `git rm --cached .env`
4. 删除本文档，或抹掉其中的密码
5. 用 `git filter-repo` 清洗历史里的 `.env` 和本文档

第 1 步最关键：只要密码进过任何一次公开提交，就必须视为已泄露。
