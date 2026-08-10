"""回归测试: 延迟发布的接口不得把"当日残缺"永久冻结进本地分片

问题背景 (2026-08-09)
────────────────────
把 tushare 资金流/两融接进 daily_rebuild 日更时发现: margin_detail 是**延迟发布**的。
2026-08-07(周五) 当天只拿到 1994 只, 而前几个交易日都是 4422 只 —— 交易所的两融明细
T+1 才出全, 遇上周末还要多等。

而 pull_by_date 原本的增量逻辑是:

    done = set(have[date_kw])          # 本地已有哪些日期
    todo = [d for d in days if d not in done]

"一个日期只要拉过就永不再拉"。于是那份 1994 只的残缺会被**永久冻结**, 且全程不报错:
后续任何一天重跑都会认为 08-07 已完成。对 mtss_balance 这类存量列, 缺的那 2428 只
会被 ffill 成陈旧余额而不是真值 —— 又是一次"静默降级"。

修复是 --refresh-days N: 强制重拉最近 N 个交易日。重拉之所以安全, 靠的是
save_by_year 按 (ts_code, date) 主键 drop_duplicates(keep="last") 覆盖写回,
新数据只多不少。本文件把这两侧行为一起锁住:

  1. save_by_year: 同一天先写残缺再写完整, 结果必须是完整的且无重复行
  2. pull_by_date: refresh_days=N 必须且仅重拉尾部 N 个交易日, 不碰更早的
  3. refresh_days=0 (全量回填的默认值) 必须保持"已有就跳过", 不浪费配额
"""
import sys
from pathlib import Path

import pandas as pd

try:
    import pytest
except ModuleNotFoundError:      # 本仓库运行环境没装 pytest, 不能因此就跑不了回归
    pytest = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pipeline.pull_tushare as pt  # noqa: E402


class _Skip(Exception):
    pass


def _skip(msg):
    if pytest is not None:
        pytest.skip(msg)
    raise _Skip(msg)


# 10 个连续"交易日", 最后一天是残缺的那天
DAYS = [f"2026080{i}" if i < 10 else f"202608{i}" for i in range(1, 11)]
STOCKS = ["000001.SZ", "000002.SZ", "600000.SH", "600519.SH"]


class _FakePro:
    """假 tushare 客户端: 记录被问过哪些日期, 每天返回全部 4 只"""

    def __init__(self):
        self.asked = []

    def margin_detail(self, **kw):
        d = kw["trade_date"]
        self.asked.append(d)
        return pd.DataFrame({"ts_code": STOCKS,
                             "trade_date": [d] * len(STOCKS),
                             "rzrqye": [1e8, 2e8, 3e8, 4e8]})


def _patch(monkeypatch_dir):
    """把落盘目录与交易日历都换成受控的, 避免碰到真实数据"""
    pt.out_dir = lambda name: monkeypatch_dir       # noqa: E731
    pt.trade_days = lambda pro, start, end: list(DAYS)  # noqa: E731


def _seed_local(tmp, days, stocks):
    """预置本地分片: days 里每天写 stocks 这些股票"""
    rows = [{"ts_code": c, "trade_date": d, "rzrqye": 1.0}
            for d in days for c in stocks]
    pt.save_by_year("margin_detail", pd.DataFrame(rows))


def _read(tmp):
    fs = sorted(tmp.glob("*.parquet"))
    if not fs:
        return pd.DataFrame(columns=["ts_code", "trade_date", "rzrqye"])
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)


def _tmpdir(tag):
    import tempfile
    d = Path(tempfile.mkdtemp(prefix=f"ts_refresh_{tag}_"))
    return d


def test_save_by_year_overwrites_partial_day():
    """同一天先写残缺(1只)再写完整(4只): 必须变成 4 行且无重复键。

    这是 --refresh-days 能起作用的前提。若 save_by_year 是追加而非按主键覆盖,
    重拉就会制造重复行, 反而把数据搞坏。
    """
    tmp = _tmpdir("save")
    _patch(tmp)
    day = DAYS[-1]

    pt.save_by_year("margin_detail", pd.DataFrame(
        {"ts_code": STOCKS[:1], "trade_date": [day], "rzrqye": [1.0]}))
    assert len(_read(tmp)) == 1, "预置的残缺数据应当只有 1 行"

    pt.save_by_year("margin_detail", pd.DataFrame(
        {"ts_code": STOCKS, "trade_date": [day] * 4, "rzrqye": [9.0] * 4}))

    got = _read(tmp)
    assert len(got) == 4, f"重拉后该天应补齐到 4 只, 实际 {len(got)}"
    assert len(got.drop_duplicates(subset=["ts_code", "trade_date"])) == 4, \
        "重拉制造了重复行 —— save_by_year 的主键去重失效"
    assert (got["rzrqye"] == 9.0).all(), \
        "keep='last' 未生效: 旧的残缺值覆盖了新拉到的值"


def test_refresh_days_repulls_only_trailing_days():
    """refresh_days=3: 必须且仅重拉最后 3 个交易日, 更早的一天都不碰。

    上界同样重要 —— 多拉就是白烧配额, 全量回填时会放大成几千次多余调用。
    """
    tmp = _tmpdir("trail")
    _patch(tmp)
    _seed_local(tmp, DAYS, STOCKS)          # 本地 10 天全都"有"

    pro = _FakePro()
    pt.pull_by_date(pro, "margin_detail", DAYS[0], DAYS[-1], sleep=0,
                    refresh_days=3)

    assert pro.asked == DAYS[-3:], \
        f"应只重拉尾部 3 天 {DAYS[-3:]}, 实际问了 {pro.asked}"


def test_refresh_days_zero_keeps_incremental():
    """refresh_days=0(全量回填默认值): 本地已有的日期必须一律跳过。

    这条守住"断点续传"这个原有语义 —— 别为了修尾部残缺把全量回填变成全量重拉。
    """
    tmp = _tmpdir("zero")
    _patch(tmp)
    _seed_local(tmp, DAYS, STOCKS)

    pro = _FakePro()
    pt.pull_by_date(pro, "margin_detail", DAYS[0], DAYS[-1], sleep=0,
                    refresh_days=0)

    assert pro.asked == [], f"不该有任何调用, 实际问了 {pro.asked}"


def test_refresh_days_heals_partial_coverage():
    """真实事故场景: 尾日只有 1 只(残缺), 重拉后必须补齐到 4 只, 且历史不丢。"""
    tmp = _tmpdir("heal")
    _patch(tmp)
    _seed_local(tmp, DAYS[:-1], STOCKS)                 # 前 9 天完整
    _seed_local(tmp, DAYS[-1:], STOCKS[:1])             # 末日残缺: 只有 1 只

    before = _read(tmp)
    assert (before["trade_date"] == DAYS[-1]).sum() == 1

    pro = _FakePro()
    pt.pull_by_date(pro, "margin_detail", DAYS[0], DAYS[-1], sleep=0,
                    refresh_days=2)

    after = _read(tmp)
    assert (after["trade_date"] == DAYS[-1]).sum() == 4, \
        "末日残缺未被补齐 —— 这正是会被永久冻结的那种情况"
    for d in DAYS[:-2]:
        assert (after["trade_date"] == d).sum() == 4, f"{d} 的历史数据被弄丢了"
    assert len(after.drop_duplicates(subset=["ts_code", "trade_date"])) == len(after), \
        "补齐过程制造了重复行"


def test_refresh_days_exceeding_range_is_safe():
    """refresh_days 大于区间总天数不得越界/异常, 退化为重拉全部即可。"""
    tmp = _tmpdir("over")
    _patch(tmp)
    _seed_local(tmp, DAYS, STOCKS)

    pro = _FakePro()
    pt.pull_by_date(pro, "margin_detail", DAYS[0], DAYS[-1], sleep=0,
                    refresh_days=999)

    assert pro.asked == DAYS, f"应重拉全部 {len(DAYS)} 天, 实际 {len(pro.asked)} 天"


def test_daily_rebuild_passes_refresh_days():
    """daily_rebuild 必须真的把 --refresh-days 传下去。

    单测覆盖不到"有没有接线"这一层: 参数实现对了但调用方没传, 线上依旧会冻结残缺。
    所以直接断言源码里那条命令带着这个参数。
    """
    src = (Path(__file__).resolve().parents[1] / "scripts" / "daily_rebuild.py"
           ).read_text(encoding="utf-8")
    if "pull_tushare" not in src:
        _skip("daily_rebuild 尚未接入 pull_tushare")
    assert "--refresh-days" in src, \
        "daily_rebuild 调用 pull_tushare 时没传 --refresh-days, 尾部残缺仍会被冻结"
    for iface in ("moneyflow", "margin_detail"):
        assert iface in src, f"daily_rebuild 的日更链路里缺 {iface}"


def _main():
    """无 pytest 环境下直接 python tests/xxx.py 就能跑。"""
    names = [n for n in sorted(globals()) if n.startswith("test_")]
    failed = []
    for n in names:
        try:
            globals()[n]()
            print(f"  PASS  {n}")
        except _Skip as e:
            print(f"  SKIP  {n}: {e}")
        except AssertionError as e:
            failed.append(n)
            print(f"  FAIL  {n}: {str(e)[:200]}")
    print(f"\n{len(names) - len(failed)}/{len(names)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
