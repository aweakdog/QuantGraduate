"""
auto_wf_wechat.py — 交易日14:48 完整管线

流程:
  1. iFinD MCP 抓今日实时行情 (价格/量/额/O/H/L) → 作为今日收盘
  2. thsdk news() 同花顺7x24快讯采集
  3. 更新 raw_kline/{code6}.parquet (追加今日14:48数据)
  4. 增量特征工程 (pipeline.feature_engine, QUANT_DATA_DIR 指向本工程)
  5. 合并为 training_data_v23.parquet (覆盖)
  6. WF 回测 (训练 2022-09-01 / 回测 2026-07-01~今日)
  7. 输出 JSON (当日持仓 + 次日推荐)

被 WorkBuddy 定时任务调度, 输出到 stdout, 由 WorkBuddy 解析后通过 wecomcli-msg 推送微信.

用法:
    python auto_wf_wechat.py
输出 (JSON):
    {"date":"...", "holdings":[...], "next_rec":[...], "text":"...", "error":null}
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
import urllib.request
from datetime import date, datetime
from pathlib import Path

# ── 1. 路径 ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent                # quant-strategy 项目根目录
DATA_DIR = PROJECT_DIR / "data"
WEB_DIR = PROJECT_DIR / "Web"

sys.path.insert(0, str(WEB_DIR))               # backend.trainer
os.environ["QUANT_DATA_DIR"] = str(DATA_DIR)   # 特征工程用本工程数据
os.chdir(str(WEB_DIR))

import pandas as pd
import numpy as np

from backend.trainer import train, TrainParams
from backend.paths import stock_name, latest_training_data

_t_start = time.time()  # 全局计时

# ── 2. iFinD MCP 客户端 ──────────────────────────────────────────────────────

def _mcp_client():
    """返回 iFinD MCP 调用函数 (闭包, 复用 opener)."""
    with open(r"C:\Users\admin\.workbuddy\mcp.json") as f:
        cfg = json.load(f)
    mcp = cfg["mcpServers"]["ifind-stock"]
    token = mcp["headers"]["Authorization"]
    url = mcp["url"]
    opener = urllib.request.build_opener()

    _req_id = [0]

    def call(method: str, params: dict = None) -> dict:
        _req_id[0] += 1
        body = json.dumps({
            "jsonrpc": "2.0", "id": _req_id[0],
            "method": method, "params": params or {}
        }).encode()
        req = urllib.request.Request(
            url, data=body,
            headers={"Authorization": token, "Content-Type": "application/json"},
            method="POST")
        with opener.open(req, timeout=30) as resp:
            return json.loads(resp.read())

    # 初始化
    call("initialize", {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "auto-wf", "version": "2.0"}
    })
    try:
        call("notifications/initialized")
    except Exception:
        pass
    return call


_MCP = None  # lazy init


def _mcp() -> callable:
    global _MCP
    if _MCP is None:
        _MCP = _mcp_client()
    return _MCP


def _fmt_stock(item: dict) -> str:
    code = str(item.get("code", ""))
    score = item.get("pred_score", 0.0)
    name = stock_name(code, fallback=code)
    return f"{name}({code} {score:+.4f})"


def _fmt_list(items: list[dict], top_n: int) -> str:
    return "\n".join(
        f"  {i+1}. {_fmt_stock(item)}" for i, item in enumerate(items[:top_n])
    )


def _code6(code: str) -> str:
    """从 '300308.SZ' 或 '300308' 提取6位代码."""
    c = code.strip().upper()
    if "." in c:
        c = c.split(".")[0]
    return c[:6]


def _is_trading_day() -> bool:
    """粗略判断今天是否交易日 (周一~五 + 非节假日). 精确判断靠 thsdk 是否有行情."""
    return datetime.now().weekday() < 5


# ── 3. 实时行情获取 (iFinD MCP) ──────────────────────────────────────────────

def _fetch_realtime_quotes(pool_codes: list[str]) -> dict[str, dict]:
    """
    通过 iFinD MCP stock_highfreq_quotes 获取实时行情 (batch=10).

    Returns: {code6: {close, open, high, low, volume, amount}}
    昨收价从现有 kline 读取.
    """
    mcp = _mcp()
    quotes: dict[str, dict] = {}

    # 分批调用 (MCP 单次上限 10 只)
    batch_size = 10
    kline_dir = DATA_DIR / "raw" / "kline"

    # 预读昨收价 (kline 最后一条的收盘价)
    last_close: dict[str, float] = {}
    for c in pool_codes:
        c6 = _code6(c)
        kf = kline_dir / f"{c6}.parquet"
        if kf.exists():
            try:
                dk = pd.read_parquet(kf)
                if len(dk) > 0:
                    last_close[c6] = float(dk["收盘价"].iloc[-1])
            except Exception:
                pass

    for start in range(0, len(pool_codes), batch_size):
        batch = pool_codes[start:start + batch_size]
        symbols = ",".join(_code6(c) for c in batch)
        try:
            resp = mcp("tools/call", {
                "name": "stock_highfreq_quotes",
                "arguments": {
                    "symbols": symbols,
                    "indicators": "开盘价,最高价,最低价,最新价,成交量,成交额",
                    "data_mode": "real_time",
                    "interval": 1,
                }
            })
        except Exception as e:
            print(f"  [MCP] batch {start} 失败: {e}")
            continue

        content = resp.get("result", {}).get("content", [])
        for c in content:
            text = c.get("text", "")
            try:
                outer = json.loads(text)
            except json.JSONDecodeError:
                continue
            # iFinD MCP data 字段是嵌套 JSON 字符串
            inner_raw = outer.get("data", "")
            if isinstance(inner_raw, str):
                try:
                    inner = json.loads(inner_raw)
                except json.JSONDecodeError:
                    continue
            elif isinstance(inner_raw, dict):
                inner = inner_raw
            else:
                continue
            tables = inner.get("tables", [])
            if not tables:
                continue
            # tables[0] = header, tables[1..] = data rows
            header = tables[0]
            try:
                open_idx = header.index("开盘价")
                high_idx = header.index("最高价")
                low_idx = header.index("最低价")
                latest_idx = header.index("最新价")
                vol_idx = header.index("成交量")
                amt_idx = header.index("成交额")
            except ValueError:
                continue
            for row in tables[1:]:
                code_str = str(row[0])  # 证券代码
                c6 = code_str[:6]
                quotes[c6] = {
                    "close": float(row[latest_idx]),
                    "open": float(row[open_idx]),
                    "high": float(row[high_idx]),
                    "low": float(row[low_idx]),
                    "volume": int(float(row[vol_idx])) if row[vol_idx] else 0,
                    "amount": float(row[amt_idx]) if row[amt_idx] else 0.0,
                    "last_close": last_close.get(c6, 0.0),
                }
        # MCP 限频保护
        time.sleep(0.05)

    print(f"[DATA] 实时行情 (iFinD MCP): 成功 {len(quotes)}/{len(pool_codes)}, "
          f"耗时 {time.time() - _t_start:.1f}s")
    return quotes


# ── 4. 事件采集 (同花顺7x24快讯) ──────────────────────────────────────────

def _fetch_events(pool_codes: list[str]) -> int:
    """
    通过 thsdk news() 获取同花顺7x24快讯, 追加到 events_v2.parquet.
    精确到秒, 盘中快讯, 来源可靠.
    """
    events_path = DATA_DIR / "raw" / "events_ifind" / "events_v2.parquet"
    events_path.parent.mkdir(parents=True, exist_ok=True)
    new_rows: list[dict] = []

    try:
        from thsdk import THS
        KQ = {"username": os.environ.get("THS_USERNAME", ""), "password": os.environ.get("THS_PASSWORD", "")}
        with THS(KQ) as ths:
            r = ths.news()
            if not r.success or not r.data:
                print("  [EVENTS] thsdk news 返回空")
                return 0
            for item in r.data:
                if not isinstance(item, dict):
                    continue
                title = item.get("Title", "") or ""
                news_time = int(item.get("Time", 0))
                props = item.get("Properties", "") or ""
                summary = ""
                for line in props.split("\n"):
                    if line.startswith("summ="):
                        summary = line[5:]
                        break
                if not title or len(title) < 5:
                    continue
                # 只保留今天的 (86400s ≈ 1天)
                if abs(datetime.now().timestamp() - news_time) > 86500:
                    continue
                new_rows.append({
                    "code": "__market__",
                    "date": datetime.fromtimestamp(news_time).strftime("%Y-%m-%d"),  # 仅日期, 不含时间
                    "eventType": "flash_news",
                    "eventDesc": f"{title}: {summary[:200]}" if summary else title[:500],
                    "eventLevel": "P2",
                    "source": "ths_7x24",
                })
    except Exception as e:
        print(f"  [EVENTS] thsdk news 失败: {e}")
        return 0

    # 去重 + 保存
    if new_rows:
        df_new = pd.DataFrame(new_rows)
        if events_path.exists():
            df_old = pd.read_parquet(events_path)
            df_all = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df_all = df_new
        df_all.to_parquet(events_path, index=False)
        print(f"[EVENTS] 同花顺7x24快讯: {len(new_rows)} 条 (共 {len(df_all)} 条)")
    else:
        print(f"[EVENTS] 同花顺7x24快讯: 0 条")
    return len(new_rows)


# ── 5. 更新 raw_kline ────────────────────────────────────────────────────────

def _append_kline(quotes: dict[str, dict], today: date) -> int:
    """
    将今日14:48快照追加到 raw_kline/{code6}.parquet.
    如果当天已有数据(非交易日/重复运行), 跳过.
    Returns: 更新股票数
    """
    kline_dir = DATA_DIR / "raw" / "kline"
    kline_dir.mkdir(parents=True, exist_ok=True)
    ts = pd.Timestamp(today)

    _KLINE_COLS = ["时间", "收盘价", "成交量", "总金额", "开盘价", "最高价", "最低价"]

    def _read_or_recreate(path: Path) -> pd.DataFrame:
        """读取 parquet; 若损坏则删除重建空表."""
        if not path.exists():
            return pd.DataFrame(columns=_KLINE_COLS)
        try:
            df = pd.read_parquet(path)
            # 验证必要列
            if all(c in df.columns for c in _KLINE_COLS):
                return df
            print(f"  [KLINE] {path.stem} 缺少列, 重建")
        except Exception as e:
            print(f"  [KLINE] {path.stem} 损坏 ({e}), 删除重建")
        # 删除损坏文件, 重建空表
        path.unlink(missing_ok=True)
        return pd.DataFrame(columns=_KLINE_COLS)

    updated = 0
    for code6, q in quotes.items():
        f_path = kline_dir / f"{code6}.parquet"
        try:
            df = _read_or_recreate(f_path)
            # 去重: 检查今天数据是否已存在
            if len(df) > 0 and ts in pd.to_datetime(df["时间"]).values:
                continue

            new_row = pd.DataFrame([{
                "时间": ts, "收盘价": q["close"], "成交量": q["volume"],
                "总金额": q["amount"], "开盘价": q["open"],
                "最高价": q["high"], "最低价": q["low"],
            }])
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_parquet(f_path, index=False)
            updated += 1
        except Exception as e:
            print(f"[KLINE] {code6} 致命错误: {e}")
            continue

    print(f"[KLINE] 更新 {updated}/{len(quotes)} 只 (含今日 {today})")
    return updated


# ── 6. 增量特征工程 ──────────────────────────────────────────────────────────

def _run_feature_engine() -> bool:
    """运行增量特征工程, 更新 features/{code6}.parquet + training_data_v15.parquet."""
    try:
        from pipeline.feature_engine import build_all
        print("[FEATURE] 开始增量特征构建...")
        t0 = time.time()
        build_all(incremental=True, max_workers=16)
        print(f"[FEATURE] 完成: {time.time() - t0:.1f}s")
        return True
    except Exception as e:
        print(f"[FEATURE] 失败: {e}")
        traceback.print_exc()
        return False


# ── 7. 合并为 v23 ────────────────────────────────────────────────────────────

def _build_v23() -> bool:
    """
    将 feature_engine 产出的 training_data_v15.parquet 复制为 v23.
    若 v15 不存在, 返回 False.
    """
    v15_path = DATA_DIR / "processed" / "training_data_v15.parquet"
    v23_path = DATA_DIR / "processed" / "training_data_v23.parquet"
    if not v15_path.exists():
        print("[V23] training_data_v15.parquet 不存在, 跳过")
        return False
    try:
        df = pd.read_parquet(v15_path)
        df["code"] = df["code"].astype(str)
        df["date"] = pd.to_datetime(df["date"])  # 保持 datetime64 类型, 与 trainer 兼容
        df.to_parquet(v23_path, index=False)
        print(f"[V23] 已覆盖 v23: {len(df):,} 行, {len(df.columns)} 列, "
              f"max_date={df['date'].max()}")
        # 同步 v22 (只保留 v22 原有列, 确保版本链一致)
        v22_path = DATA_DIR / "processed" / "training_data_v22.parquet"
        if v22_path.exists():
            v22_cols = set(pd.read_parquet(v22_path, columns=[]).columns
                          if v22_path.stat().st_size > 0 else [])
            if v22_cols:
                v22_df = df[list(v22_cols.intersection(df.columns))]
                v22_df.to_parquet(v22_path, index=False)
                print(f"  [V23] v22 已同步: {len(v22_df)} 行")
        return True
    except Exception as e:
        print(f"[V23] 合并失败: {e}")
        return False


# ── 8. WF 回测 ──────────────────────────────────────────────────────────────

def _run_backtest() -> dict:
    """执行 WF 回测, 返回结果 dict."""
    today = date.today()

    # 检查训练数据是否存在且包含今日
    train_data = latest_training_data()
    if train_data and train_data.exists():
        df_check = pd.read_parquet(train_data, columns=["date"])
        max_date = pd.to_datetime(df_check["date"]).max().date()
        print(f"[BACKTEST] 训练数据最新日期: {max_date}")
    else:
        print("[BACKTEST] 训练数据不存在!")
        return {"error": "训练数据不存在"}

    params = TrainParams(
        train_start=date(2022, 9, 1),
        test_start=date(2026, 7, 1),
        test_end=today,
        top_n=4,  # Top4 > Top3 (8区间验证 7/8, Sharpe +5%)
        skip_next_rec=False,
    )
    t0 = time.time()
    result = train(params)
    elapsed = time.time() - t0
    print(f"[BACKTEST] 完成: {elapsed:.0f}s")

    daily = result.daily_returns
    if not daily:
        return {"error": "回测结果为空 (可能无有效交易日)", "date": str(today)}

    last = daily[-1]
    last_date = last.get("date", "")
    holdings = last.get("holdings", [])
    next_rec = last.get("next_rec", [])

    return {
        "date": last_date,
        "holdings": [
            {"code": str(h.get("code", "")), "score": h.get("pred_score", 0.0)}
            for h in holdings[:4]
        ],
        "next_rec": [
            {"code": str(h.get("code", "")), "score": h.get("pred_score", 0.0)}
            for h in next_rec[:4]
        ],
        "error": None,
        "text": (
            f"📊 量化回测日报 ({last_date})\n"
            f"训练 2022-09-01→{today} | 回测 2026-07-01→{today}\n\n"
            f"📋 最后交易日持仓 ({last_date}):\n{_fmt_list(holdings, 4)}\n\n"
            f"▶ 次日推荐:\n{_fmt_list(next_rec, 4)}"
        ),
    }


# ── 9. 主流程 ────────────────────────────────────────────────────────────────

def main() -> None:
    output: dict = {"date": str(date.today()), "holdings": [], "next_rec": [],
                    "error": None, "text": ""}

    try:
        # ── A) 非交易日快速退出 ──
        if not _is_trading_day():
            output["error"] = "非交易日"
            output["text"] = "📊 今日非交易日, 跳过回测"
            print(json.dumps(output, ensure_ascii=False))
            return

        # ── B) 加载股票池 ──
        from backend.paths import watchlist_path, load_universe_codes
        pool_codes = load_universe_codes("关注圈")
        if not pool_codes:
            output["error"] = "股票池为空"
            output["text"] = "❌ 股票池为空, 无法回测"
            print(json.dumps(output, ensure_ascii=False))
            return
        print(f"[POOL] 股票池: {len(pool_codes)} 只")

        # ── C) 获取实时行情 (14:48 视为收盘) ──
        print(f"\n{'='*50}")
        print(f"  14:48 自动割草回测 — {date.today()}")
        print(f"{'='*50}")
        print(f"[PHASE 1/6] 抓取实时行情 (iFinD MCP)...")
        quotes = _fetch_realtime_quotes(pool_codes)
        if not quotes:
            print("[QUOTES] 无实时数据, 降级为现有数据回测")
        else:
            # ── D) 采集事件 ──
            print(f"\n[PHASE 2/6] 采集今日事件...")
            _fetch_events(pool_codes)

            # ── E) 更新 kline ──
            print(f"\n[PHASE 3/6] 更新 raw_kline...")
            _append_kline(quotes, date.today())

            # ── F) 增量特征工程 ──
            print(f"\n[PHASE 4/6] 增量特征工程...")
            ok = _run_feature_engine()
            if ok:
                print(f"\n[PHASE 5/6] 合并训练数据 v23...")
                _build_v23()
            else:
                print("[FEATURE] 特征工程失败, 使用现有数据回测")

        # ── G) WF 回测 ──
        print(f"\n[PHASE 6/6] WF 回测...")
        output = _run_backtest()

    except Exception as e:
        output = {
            "date": str(date.today()),
            "holdings": [],
            "next_rec": [],
            "error": f"{type(e).__name__}: {e}",
            "text": f"❌ 自动回测失败: {type(e).__name__}: {e}",
        }
        traceback.print_exc()

    # ── 输出 ──
    print(f"\n{'='*50}")
    print(f"  总耗时: {time.time() - _t_start:.0f}s")
    print(f"{'='*50}")
    print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()
