"""实盘参数的唯一来源 (single source of truth)

为什么需要这个文件
──────────────────
live_signal.py 会把一组参数存成 state.json 里的 "config" 指纹, 每次启动都拿
当前参数和指纹比对, 不一致就直接报错退出 (防止参数变了却继续用旧持仓记账,
导致持仓/现金错位)。

而调用 live_signal.py 的地方有三处 —— daily_rebuild.py(定时任务)、
web_server.py(网页手动触发/对账)、人工命令行。之前这三处各自硬编码
`--regime-filter off`, 只要有一处漏改, 定时任务就会因指纹不匹配而整天失败,
而且失败发生在收盘后无人值守的时段。

因此: 所有参数集中在此, 三处都从这里取, 物理上无法漂移。

改参数的正确流程
────────────────
1. 改本文件的 BASE_PARAMS 或 PROFILES
2. 因为指纹变了, 必须重置对应条线 (会清空持仓记录, 现金回到 capital):
       python scripts/live_signal.py $(python scripts/live_config.py --profile steady5w --args) \
              --capital 50000 --init
   或用封好的脚本:  python scripts/init_profiles.py --profile steady5w
3. 若券商账户里还有实际持仓, 重置后用 --sync 把真实持仓填回去

注意: 下列参数中只有 FINGERPRINT_KEYS 里的会进指纹; features_from / capital
不进指纹 (它们不影响持仓记账)。
"""
import argparse
import json
from pathlib import Path

# ── 所有 profile 共用的基础参数 ─────────────────────────
BASE_PARAMS = {
    # 池子: 497 只 PIT 自选股。已验证扩到 2303 只会彻底失效
    # (PIT1500_50K_n5: -67.4%, 夏普 -1.01, IR -1.59, 回撤 -73.2%, 两段都输)
    "train-file": "training_data_pit_v24.parquet",
    "pit-universe": "universe_pit.parquet",
    "label": "5d",
    # 每 5 个交易日整体换仓。必须与回测的 --hold-days 一致:
    # wf_v35 默认 5, 而 live_signal.py 的 argparse 默认是 10 —— 历史上线上
    # 跑的是 10, 与所有回测结论口径不符, 所以这里显式钉死 5。
    "hold-days": 5,
    "portfolio-mode": "periodic",
    "exec-mode": "t1close",
    # 滑点只用于估算成交价与结算, 取回测同值(0.2%)偏保守
    "slippage": 0.002,
    # breadth 择时在当前配置下反而有害, 实测 (hold=5, 滑点0.2%):
    #   5万/3只  off +292.4% IR 1.37  vs  breadth  +50.9% IR 0.34
    #   5万/5只  off +220.2% IR 1.27  vs  breadth  +11.8% IR -0.10
    #   2万/3只  off +161.1% IR 0.97  vs  breadth   -4.3% IR -0.28
    # (早期 h10 配置下 breadth 看似大幅有效, 是因为它在补救一个本身就坏的基线)
    "regime-filter": "off",
}

# 复用哪个回测结果里锁定的特征列表 (不现场筛选, 保证与回测完全一致)
# 必须是"用修复后的 drop_market_wide 跑出来的"结果, 否则会带入市场级泄漏特征
FEATURES_FROM = "wf_daily_REGRESS_CHK_ts2022-09-01_te2026-07-27_cap50000.json"

# ── 四条并行线 ─────────────────────────────────────
# 持仓数不能自由选 —— 回测引擎买不起一手(100股)时会直接跳过该股换下一名
# (wf_v35_breadth_alpha.py 的 buy_lot_too_big 分支), 所以 每只预算 = 本金/持仓数
# 直接决定了能买的最高股价。本金越小、持仓越多, alpha 被资金约束吃得越多。
#
# 实测前沿 (497池 / hold=5 / t1close / 滑点0.2% / 不择时):
#   本金  持仓   总收益    夏普    回撤     IR   费用/本金  两段都正
#    2万   2只  +228.8%   1.17  -41.2%   1.11    24.1%    ✓ [0.19, 1.71]
#    2万   3只  +161.1%   1.05  -35.5%   0.97    28.6%    ✓ [0.68, 1.20]
#    2万   4只   +25.9%   0.42  -37.9%   0.19    27.8%    ✗
#    2万   5只    +9.4%   0.26  -36.4%  -0.05    34.4%    ✗
#    2万   8只   -39.6%  -0.74  -48.7%  -1.52    55.0%    ✗
#    5万   2只  +120.4%   0.86  -38.4%   0.76    17.5%    ✗
#    5万   3只  +292.4%   1.39  -28.6%   1.37    28.2%    ✓ [0.62, 1.86]
#    5万   4只  +226.4%   1.25  -33.2%   1.20    29.0%    ✓
#    5万   5只  +220.2%   1.30  -31.1%   1.27    28.6%    ✓ [0.89, 1.56]
#    5万   8只   +13.3%   0.31  -33.4%  -0.01    22.7%    ✗
#
# 结论: 2万本金的持仓上限是 3 只 (4 只起 IR 崩塌、后半段转负)。
# "稳妥 vs 激进" 用持仓数区分: 持仓越少越集中(收益高、回撤大、个股尾部风险高)。
PROFILES = {
    "steady2w": {
        "name": "稳妥 2万",
        "capital": 20000.0,
        "tranche-n": 3,
        "desc": "2万本金下可行的最分散方案。回测 IR 0.97, 回撤 -35.5%, 两段都跑赢。",
    },
    "aggr2w": {
        "name": "激进 2万",
        "capital": 20000.0,
        "tranche-n": 2,
        "desc": "只持 2 只, 每只预算更大所以高价股也能买。回测 IR 1.11 但回撤 -41.2%。",
    },
    "steady5w": {
        "name": "稳妥 5万",
        "capital": 50000.0,
        "tranche-n": 5,
        "desc": "5 只分散, 单股爆雷影响最小。回测 IR 1.27, 回撤 -31.1%。",
    },
    "aggr5w": {
        "name": "激进 5万",
        "capital": 50000.0,
        "tranche-n": 3,
        "desc": "回测指标最优 (IR 1.37, 回撤 -28.6%), 但仅 3 只, 个股集中度高。",
    },
}

# 默认展示哪条线
DEFAULT_PROFILE = "steady5w"

# 这些键会进 state.json 的 config 指纹 (与 live_signal.fingerprint() 对齐)
FINGERPRINT_KEYS = (
    "train_file", "pit_universe", "label", "hold_days", "tranche_n",
    "portfolio_mode", "exec_mode", "slippage", "regime_filter",
    "regime_ma", "regime_breadth", "regime_confirm", "reversal_guard",
)


def state_file(pid):
    """每条线一份状态, 互不影响 (data/live/ 下的文件名)"""
    return f"state_{pid}.json"


# ── 运行时可改的设置 ────────────────────────────────────
# 名字和"自动记账开关"是用户在网页上随时改的, 不能写进代码, 也不能写进
# state_<id>.json —— 状态文件规定只由 live_signal.py 单方写入, 网页再去写
# 就会和收盘后的定时任务撞车、造成账目错位。所以单独一个小文件。
SETTINGS_PATH = Path(__file__).resolve().parents[1] / "data" / "live" / "profile_settings.json"

# 四条线共用的预测缓存 (见 signal_args)。删掉它只会导致下次重训一遍, 无副作用。
PREDS_CACHE_PATH = Path(__file__).resolve().parents[1] / "data" / "live" / "preds_cache.json"

# auto=True  : 按 T+1 真实行情自动记账 (纸面跟踪, 默认)
# auto=False : 实盘模式 —— 不自动记账, 等你填真实成交价才入账。
#              这条线会停在"待确认成交"状态, 不会自动往前推进。
DEFAULT_SETTING = {"name": None, "auto": True}

NAME_MAX = 16


def load_settings():
    """读全部设置; 文件不存在或损坏都退回默认, 绝不因此让出信号失败"""
    try:
        raw = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    out = {}
    for pid, v in raw.items():
        if pid in PROFILES and isinstance(v, dict):
            out[pid] = {"name": v.get("name") or None,
                        "auto": bool(v.get("auto", True))}
    return out


def save_settings(s):
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = SETTINGS_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(SETTINGS_PATH)          # 原子替换, 避免读到写一半的文件


def setting(pid):
    return {**DEFAULT_SETTING, **load_settings().get(pid, {})}


def display_name(pid):
    """用户改过就用用户的, 否则用代码里的默认名"""
    return setting(pid)["name"] or PROFILES[pid]["name"]


def is_auto(pid):
    return setting(pid)["auto"]


def set_name(pid, name):
    """name 传 None 或空串 = 恢复默认名"""
    if pid not in PROFILES:
        raise KeyError(pid)
    name = (name or "").strip()
    if len(name) > NAME_MAX:
        raise ValueError(f"名字最长 {NAME_MAX} 个字")
    s = load_settings()
    s.setdefault(pid, dict(DEFAULT_SETTING))["name"] = name or None
    save_settings(s)
    return display_name(pid)


def set_auto(pid, auto):
    if pid not in PROFILES:
        raise KeyError(pid)
    s = load_settings()
    s.setdefault(pid, dict(DEFAULT_SETTING))["auto"] = bool(auto)
    save_settings(s)
    return bool(auto)


def signal_args(pid, include_features=True):
    """展开成指定 profile 的 live_signal.py 命令行参数"""
    if pid not in PROFILES:
        raise KeyError(f"未知 profile: {pid} (可选: {', '.join(PROFILES)})")
    p = PROFILES[pid]
    params = dict(BASE_PARAMS)
    params["tranche-n"] = p["tranche-n"]
    out = []
    for k, v in params.items():
        out += [f"--{k}", str(v)]
    out += ["--state", state_file(pid)]
    # 手动模式的线不得自动记账。放在这里而不是各调用方自己判断,
    # 是因为 daily_rebuild / web_server / 命令行 三处都走 signal_args,
    # 写在这里就不可能出现"定时任务绕过了开关把账记了"这种事。
    # 此参数不在 FINGERPRINT_KEYS 里, 所以来回切换不会弄坏已有持仓。
    if not is_auto(pid):
        out += ["--require-confirm"]
    # 四条线的模型完全一样(只有建仓环节按 tranche-n 不同), 各训一遍是 4 倍
    # CPU 白烧。共用一个预测缓存: 当天第一条线训练并写入, 其余直接读。
    # 缓存键含信号日与全部训练输入, 任一项变化都会自动重训, 不会读到过期预测。
    # 同样不在 FINGERPRINT_KEYS 里, 加它不影响已有持仓。
    out += ["--preds-cache", str(PREDS_CACHE_PATH)]
    if include_features and FEATURES_FROM:
        out += ["--features-from", FEATURES_FROM]
    return out


def init_args(pid):
    """首次建立(或重置)该条线的参数 —— 会清空持仓记录"""
    return signal_args(pid) + ["--capital", str(PROFILES[pid]["capital"]), "--init"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="打印实盘参数")
    ap.add_argument("--profile", help="只看指定 profile")
    ap.add_argument("--args", action="store_true", help="输出可直接拼到命令行的参数串")
    a = ap.parse_args()
    if a.args:
        print(" ".join(signal_args(a.profile or DEFAULT_PROFILE)))
    else:
        print("共用参数:")
        for k, v in BASE_PARAMS.items():
            print(f"  --{k:16s} {v}")
        print(f"  --features-from   {FEATURES_FROM}\n")
        for pid, p in PROFILES.items():
            mark = " (默认)" if pid == DEFAULT_PROFILE else ""
            nm = display_name(pid)
            alias = "" if nm == p["name"] else f"  [原名 {p['name']}]"
            print(f"{pid:10s} {nm}{mark}{alias}")
            print(f"           本金 {p['capital']:,.0f} / {p['tranche-n']} 只 / "
                  f"每只预算 {p['capital']/p['tranche-n']:,.0f}")
            print(f"           记账 {'自动(按行情)' if is_auto(pid) else '手动(等确认成交)'}")
            print(f"           状态 {state_file(pid)}")
