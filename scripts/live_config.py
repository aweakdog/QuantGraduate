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
    # 2026-08-16 全线切 tick1 矩阵 (= v24 + 逐笔lag1微观结构列, daily_rebuild §4.5 产出,
    # 末日与 v24 强制对齐, 断供超 3 交易日则当晚硬失败而不是静默陈旧)。
    # 注意 train_file 在指纹里: 切换日已对全部 state 做过备份+迁移。
    "train-file": "training_data_pit_v24_tick1.parquet",
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
    # MP1c 信号强度门槛 (2026-08-22 用户拍板全线开): pred < 0.002 的候选不买,
    # 空槽位持现金不递补 —— 第二层择时"模型没信心的日子宁可不买"。
    # 门控流(实盘同构 GFG1 缓存) 120 重放五判据全过: 窗B 中位 +16.05pp(19/20 正)
    # + 回撤中位 +9.95pp + 夏普 +0.19; 窗A 完美 wash(Δ0.00, 亏损 5/5 不增) ——
    # 0.002≈日常 pred 的 p97, 强势窗几乎不触发。0.003 臂 B 更肥(+24.5)但 A -1.2
    # 且亏损种子 6/5 微增, 按预注册取 0.002。
    # ⚠ 与下方 08-04 "别再叠加 min-pred" 旧警告的关系: 那条是当年买入侧排名
    # 过滤把部署率压到 10% 的教训; 0.002 绝对值门槛只砍弱信号日, 且 MP1c 已在
    # 含 breadth 择时 + GF1 门控的完整生产流上重判通过, 该警告对本行不适用。
    # 纯执行层不进指纹, 删本行即回滚。
    "min-pred": 0.002,
    # ⚠ 下面这段历史结论已于 2026-08-04 被证伪, 保留原文以免重犯:
    #   "breadth 择时在当前配置下反而有害, 实测 (hold=5, 滑点0.2%):
    #      5万/3只  off +292.4% IR 1.37  vs  breadth  +50.9% IR 0.34
    #      5万/5只  off +220.2% IR 1.27  vs  breadth  +11.8% IR -0.10
    #      2万/3只  off +161.1% IR 0.97  vs  breadth   -4.3% IR -0.28"
    # 问题出在依据是【单次回测】: 同 IC 下 20 个种子的总收益能从 -31% 到 +190%,
    # 上面那个 +292.4% 正是右尾抽样。用 20种子x双窗口重测(scripts/eval_grid.py):
    #      3只 窗口A -43.2% -> -23.7% | 窗口B +11.8% -> +73.6%
    #      5只 窗口A -22.2% ->  -9.8% | 窗口B -11.2% -> +45.3%
    # 两个窗口、两种持仓数下收益与回撤【同时】改善, breadth 择时是目前最有效的
    # 单一改动。详见 docs/findings_2026-08-04_regime_dependent_alpha.md
    # 2026-08-04 已按线逐一验证并打开。5 条线 x 2 窗口 = 10 个格子全部改善,
    # 无一反向 (20种子中位, scripts/eval_grid.py 的 live_* 配置):
    #   条线              窗口B 收益   亏损种子     窗口A 收益   回撤中位
    #   5万/5只/主板     33.5 -> 58.2  5/20 -> 0/20  -31.7 -> -26.4  -52.9 -> -38.0
    #   5万/3只/主板     26.9 -> 47.2  4/20 -> 1/20  -43.4 -> -32.2  -62.8 -> -42.0
    #   10万/3只/主板    25.5 -> 53.5  3/20 -> 1/20  -47.0 -> -37.1  -65.2 -> -47.3
    #   2万/2只/主板     34.5 -> 83.4  4/20 -> 0/20  -60.7 -> -48.8  -73.2 -> -59.3
    #   2万/3只/全市场  -16.9 -> 46.2 13/20 -> 1/20  -19.0 ->  -3.5  -50.3 -> -34.3
    # 代价: 约 44% 的交易日空仓, 平均部署率降到约 50%。
    # 注意别再叠加 min-pred / 买入侧排名过滤 —— 主板池子只有 516 只, 再叠两层
    # 过滤会把平均部署率压到 10%, 实测把 +58.2% 打到 +12.9%。
    # (此处旧文写的 "roll-rank" 指当年那个买入侧过滤旗标; 2026-08-18 上线的
    # --roll-rank 是卖出侧宽续持, 部署率只升不降, 不在此警告范围内。)
    "regime-filter": "breadth",
    # 这三个必须显式钉死 —— 与 hold-days 同一类坑: live_signal.py 的 argparse
    # 默认是 regime-breadth=0.35 / regime-confirm=1, 而所有回测(wf_v35)用的是
    # 0.40 / 2。不写在这里, 一旦把 regime-filter 打开, 线上就会用一套没被回测
    # 验证过的阈值跑, 而且不会有任何报错。
    "regime-ma": 20,
    "regime-breadth": 0.40,
    "regime-confirm": 2,
    # 5种子集成 (2026-08-02): 同数据同超参只换 random_state, 预测取平均。
    # 动机: 同 IC 下单种子的前3名选择方差是最大脆弱点 —— 5个种子单跑
    # IR 0.57~1.21 (均值0.92), 集成后 1.23~1.30 且两段更均衡。
    # 显式钉死种子列表: live_signal 默认值将来若改, 这里保证线上不悄悄变。
    "seed-ensemble": "42,7,123,2024,31337",
    # 整手粒度救济(lot-flex)不在这里: 它是每条线自己的参数, 见 PROFILES。
    # 2万线开 0.5 (两份独立数据都验证有效: +0.79/+0.28), 5万线关
    # (两份数据都偶负: -0.04/-0.21, 虽在噪声内但真金线不上未验证正收益的东西)。
    # 详见 docs/findings_2026-08-02_lot_flex.md
}

# 复用哪个回测结果里锁定的特征列表 (不现场筛选, 保证与回测完全一致)
# 必须是"用修复后的 drop_market_wide 跑出来的"结果, 否则会带入市场级泄漏特征
#
# 2026-08-09 换成 F1B (71 特征)。相比上一版 REGRESS_CHK 的 80 特征, 三处变化:
#   1. fund_flow 改由 tushare moneyflow.net_mf_amount 供给(万元→元, 比值 10000.0028
#      / 相关 0.9996)。旧源同花顺只覆盖 243/519 只且停更在 2026-06-30, 属于死列。
#   2. 排除 dde_net* —— 同源已死且无替代品(与 tushare 各口径相关仅 0.48~0.91,
#      不能拿来回填)。留着只是让模型拿旧年代的分裂点去切一个 NaN 列。
#   3. 排除 con_* 板块联动 —— expC/expD 对照(特征数拉平后)证实其有害, IC 差 0.0043。
#
# 双窗口 3 种子实测 (对照 expC, 同为 restored 矩阵 + 同样排除):
#   窗口B 2022-09~2026-07  IC 0.0540 -> 0.0549   IR 0.63 -> 0.83   回撤 -25.6 -> -23.1
#   窗口A 2020-07~2022-08  IC -0.0084 -> -0.0069 (仍为负, 该窗口依旧不成立)
# 注意 IC 增益只有 +0.0009, 与种子间波动同量级; 三个种子一致不差于基线, 但别把
# IR/收益那段涨幅当业绩预期 —— 3 个种子对集中度运气的抵抗力很有限。
# 真正的上线理由是结构性的: 把一个已死的列换成每日更新的列。
#
# 附带收益: 在用特征里的恒常量从 11 个降到 0 个 (daily_rebuild 的零方差体检实测),
# 即模型名义 71 特征、实际有效也是 71 个。
#
# 2026-08-13 换成 FBTR (75 特征)。相比 F7/F1B, 这是"剔前视+剔死源"的终态口径:
#   1. 排除全部静态映射特征 con_*/tev_*/leader_*/has_leader —— 三个映射文件
#      (watchlist_216/watchlist/supply_chain_map) 都是 2026 年的认知快照套全历史,
#      构成前视。20 种子同窗配对实测: 剔除后策略仍有超额(见 FBTR 系列 JSON)。
#   2. 排除 dde_net*/mf_* —— 源已死且无前向替代品(main_force_net 与任何 tushare
#      口径相关最高 0.59, 不能回填), 留着只会让模型切 NaN 列。
#   3. 保留 fund_flow/mtss(tushare moneyflow/margin_detail 日更供给) 与 13 个宏观
#      特征 —— 宏观源 2026-08-13 起由 pipeline.pull_macro 日更(tushare 汇率/美债 +
#      akshare SOX/商品指数/中债), 全部源与旧 iFinD 值逐日对账通过, 断供已修复。
# 动机是结构性的: F7 特征集里仍有前视源(con_*/tev_*)与死列(dde/mf), 前者夸大
# 回测成绩, 后者随时间腐烂。FBTR 的 75 列全部"活的 + 无前视 + 有日更管线"。
#
# 2026-08-15 换成 V24B (80 特征)。FBTR 那份的问题不在"选得不好", 而在它是
# 【在旧矩阵 training_data_pit_2019 (630只) 上选出来、却搬到线上矩阵
# training_data_pit_v24 (519只) 上用】。换池子后有一列发生了性质变化:
#   ev_decay_n_5d 在线上池只覆盖 27.0% 的股票, 且这 27% 里 90.7% 落在
#   watchlist_216.json 内 —— 它不再是一个事件特征, 而退化成"这只票在不在
#   2026 年那份人工名单里"的覆盖掩码, 即用未来的认知给历史打标记。
#   这是 docs §8.4 那一类前视, 只是这次伪装成了"缺失值模式"。
#
# 20 种子同窗配对实测 (两臂都在线上矩阵 + 滑点 0, 见 wf_daily_V24A/V24B_*):
#          总收益    年化    夏普   超额年化    IR    alpha年化  亏损种子
#   V24A   +8.50%   3.05%   0.24   -2.30%   -0.10    1.40%     3/20
#   V24B  +31.40%  10.55%   0.53   +5.00%   +0.22    8.25%     1/20
#   配对: V24B 更好 18/20 种子, 收益差中位 +21.9pp
# 关键不是收益高低, 而是 V24A 的超额为负 —— 线上那套跑不过等权买全池。
#
# 逐年留一年检验 (scripts/yearly_robust.py, 剔掉某年后其余年份累计中位):
#          全部年   剔2023  剔2024  剔2025  剔2026
#   V24A     6.3     5.4    20.4    -9.9 ⚠   4.5
#   V24B    27.8    34.2    21.9   +12.0 ✓  12.8
# V24A 完全靠 2025 单年撑着, V24B 四列全正。
#
# ⚠ 两点保留, 别把上面的数字当业绩预期:
#   1. IR 0.22 在 2.85 年样本上 t 值约 0.37, 统计上不显著。只能说"没坏",
#      不能说"证明有 alpha"。真正的上线理由是结构性的: 去掉一个前视列。
#   2. V24B 回测跑在 100000/tranche-n 3 (= aggr10w 那条线) 上, 而
#      DEFAULT_PROFILE 是 steady5w (50000/tranche-n 5)。特征集是全线共用的,
#      但默认线那个操作点尚未用 V24B 特征单独回测过, 待补。
#
# 一个反直觉的观察, 记下来免得下次被骗: V24A 的 IC t 值 6.375 反而比 V24B 的
# 3.730 更高, 却是负超额。掩码特征能给出更漂亮的横截面 IC, 组合层拿不到。
# => 判断特征集好坏不能只看 IC, 必须看组合层的超额与 IR。
#
# 闸门: scripts/check_deploy_gate.py V24B 四关全过 (20种子特征集完全一致 /
# 无覆盖掩码 / 无零方差 / 覆盖率达标); 同一闸门拦下 V24A。
# 回归测试: tests/test_no_coverage_mask_features.py 切换前 2 failed, 切换后应全绿。
#
# ── 2026-08-16 全线切 V24PUT (V24B → V24PUT) ─────────────
# 配方 = V24B 基线 + 逐笔微观结构列(lag1, d-1可得) - 12个 qfq 价格单位列(atr/macd全族,
# 前复权锚常数泄漏通道)。两个操作点均 20 种子、滑点0、t1close、breadth 择时:
#   100000/n3 (aggr10w): 总收益中位 +81.4% / IR 0.71 / 回撤 -18.2% / 0/20 亏损
#   50000/n5 (steady5w): 总收益中位 +97.3% / IR 0.94 / 回撤 -13.8% / 0/20 亏损
# 配对: PUT>T1 16/20, PUB>B 17/20; 逐笔 alpha 随 staleness 衰减(lag5 近于无),
# 故 daily_rebuild §4.5 把 tick1 断供设为硬失败。闸门 check_deploy_gate V24PUT/
# V24PUT5W 均 4/4 过。逐笔数据 2023 起 => 2020-07~2022-08 窗口无法评估本配方,
# 各 desc 里的窗口A 数字仍是 V24B 旧特征时代的, 作“同家族在熟市的参考下限”。
# 旧值: FEATURES_FROM = "wf_daily_V24B_s42_ts2022-09-01_te2026-07-27_cap100000.json"
#
# ── 2026-09-01 分点分配 (T1A/T1B 增广特征集, 逐线覆盖) ─────────
# 这里仍是全局默认(80 列 V24PUT); 各 profile 可用 "features-from" 覆盖成自己的
# 特征表 —— T1A = +4 列逐笔订单结构(lag1, D-1 可得), T1B = +4 列K线情绪分解。
# 列由 daily_rebuild §4.3 每晚建进矩阵(在用则硬失败), 分配证据与逐线判决见
# docs/factor_family_ledger.md「T1A/T1B」与 experiment_board 2026-08-31 LSW /
# 2026-09-01 T1AL·T1AV(lag1 生产口径复核)。features_from 不进指纹, 回滚 = 删该行。
FEATURES_FROM = "wf_daily_V24PUT_s42_ts2022-09-01_te2026-07-27_cap100000.json"

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
# ── desc 的写法约定 (2026-08-04 重写) ──────────────────────
# 旧 desc 写的是【单次回测】的数字, 例如 "回测近3年 +108% / 回撤 -29%"。
# 已证明那是右尾抽样: 同一配置换随机种子, 总收益能从 -31% 到 +190%, 而 20 个
# 种子的 IC 全部落在 0.0133~0.0170 的极窄区间 —— 收益差 220pp 纯属集中度运气。
# 这些数字会直接显示在网页上, 等于把抽奖结果当成业绩承诺。
#
# 新约定: 一律给【20 种子中位数】+【最差种子】+【两个窗口】。
#   窗口B 2022-09~2026-07 (943天) —— 这段信号多数时间为正
#   窗口A 2020-07~2022-08 (530天) —— 这段信号多数时间为负, 从未用于调参
# 必须把窗口A 一起写出来, 否则又变成只报好消息。
# 数据来源: scripts/eval_grid.py 的 live_* 配置, docs/findings_2026-08-04_*.md
PROFILES = {
    # ── GF1-G1 门控融合 (2026-08-22 全线开启, 用户拍板) ──
    # 弱势态(全市场等权指数<MA60, 3天确认滞回)自动改用 CGO 防守模型(锁定特征+6
    # 列行为因子)选股, 强势态用原 base 模型。择时过滤器不变(叠在外层)。
    # 20 种子四判据全过(窗A +9.5pp/窗B +7.65pp/窗B 0亏损/窗C -1.65pp 均过预注册门槛),
    # 滑窗后首个过 gate 机制, 证据链见 experiment_board 2026-08-22。
    # 不进指纹(不影响持仓记账), 关掉即回滚。
    # base5w_steady/base5w_aggr 也开(2026-08-22 用户改口径): 它俩的定位从"跟踪老
    # 策略"改为"始终跟随最新策略的无干预镜像", 只比较人为操作的影响。
    "steady2w": {
        "name": "稳妥 2万",
        "capital": 20000.0,
        "tranche-n": 3,
        "lot-flex": 0.5,
        # opened = 这条线在网站上建立的日期(开户日)。不是参数, 不进命令行与指纹,
        # 只供页面展示。重置(从头再来)不改它 —— 开户日记的是"上站时间",
        # 不是账本起点。回填依据: git 77e8a92 (实盘上线, 2026-07-31)。
        "opened": "2026-07-31",
        # FDRR8 执行层 (2026-08-19 补测后开启, 最后一条开的真实线)。
        # PUTFM 缓存重放, 20bp 从严 20 种子同面板: +40.8%/夏普0.61 ->
        # +56.1%/0.77, 回撤 -27.6 -> -27.0, 0/20 亏损, 15/20 种子更好
        # (中位 +19.2pp; 尾部 5% -30.9pp 较主板线宽, 全市场 n3 噪声大)。
        "fill-daily": True,
        "roll-rank": 8,
        # "gate-ma60": True,  # GF1 已关 2026-08-27 (GFV3: v3 干净矩阵复核翻案, 见看板)
        # IND1 行业上限 (2026-08-22 用户拍板上线, 仅 3 只线): 同一申万一级最多持 2 只。
        # 240 重放四判据全过(唯一过关臂): 窗B 收益 +0.85pp + 回撤中位 +2.95pp +
        # 窗A 完美 wash + 亏损种子不增 —— 3 只组合的免费回撤保险。cap1 太狠掉 alpha
        # (B -5.9pp)已否; 5 只组合险改善不过门槛不开。纯执行层不进指纹, 关掉即回滚。
        "ind-cap": 2,
        # ⚠ T1A/T1B 分点分配(2026-09-01)本线【不上】: LSW 在 lag0 下测得 T1A +19.4pp,
        # 但 lag1(生产真能拿到的口径, T1AV 复核)塌成 +3.45pp/11-20 = wash ——
        # 这个点位的增量大头恰好是当日信息, 是六个点里唯一一个。别因为别的线都切了
        # 就顺手给这里也切 T1A —— 那是拿 xmy 的真金去买一个已知拿不到的回测数字。
        "desc": "3 只分散 · 全市场 · 同行业最多 2 只(IND1, 2026-08-22) · 大盘弱势自动空仓 · 闲钱日补+到期宽续持 · 弱信号日持币不买(MP1c, 2026-08-22)。"
                "V24PUT(含逐笔)回测(20种子中位, 20bp从严口径, 2026-08-19 补测): "
                "2022-09~2026-07 +56%(0/20 亏损), 夏普 0.77, 回撤中位 -27%; "
                "同面板不开执行层升级的基线 +41%(升级纯增益)。"
                "逐笔数据2023起, 2020-07~2022-08 窗无法评估(旧特征同家族该窗 -4%, "
                "12/20 亏损, 作参考下限)。约46%交易日空仓。"
                "执行层升级后非换仓日也可能有零星买入/续持行。",
    },
    "aggr2w": {
        "name": "激进 2万",
        "capital": 20000.0,
        "tranche-n": 2,
        "lot-flex": 0.5,
        "opened": "2026-07-31",
        # 账户没开创业板/科创板权限 (2026-08-02, 用户预计两年后才开)。
        # 模型级隔离: 训练/候选/基准只看主板, 而非执行层事后跳过
        # (事后跳过实测丢掉 ~70% 利润, 见 docs/progress_2026-08-02.md)。
        "skip-boards": "30,688",
        # FDRR8 执行层 (2026-08-18, 同 steady5w 注释)。20bp从严口径 20种子同面板:
        # +34.7%/夏普0.56 -> +86.0%/0.96, 最差种子 +0.3%->+21.4%, 回撤 -25.6->-23.9。
        "fill-daily": True,
        "roll-rank": 8,
        # "gate-ma60": True,  # GF1 已关 2026-08-27 (GFV3: v3 干净矩阵复核翻案, 见看板)
        "desc": "只持 2 只(最集中) · 仅主板 · 大盘弱势自动空仓 · 闲钱日补+到期宽续持 · 弱信号日持币不买(MP1c, 2026-08-22)。"
                "V24PUT(含逐笔)回测(20种子中位, 非单次跑, 2026-08-18 引擎口径): "
                "2022-09~2026-07 +139%(最差种子 +72%, 20个种子无一亏损), 夏普 1.30, "
                "回撤中位 -21%; 同面板不开执行层升级的基线是 +78%(升级纯增益, 非模型变化)。"
                "逐笔数据2023起, 2020-07~2022-08 窗无法评估(旧特征同家族该窗 -49%, 20/20 全亏, "
                "作参考下限)。约46%交易日空仓。执行层升级后非换仓日也可能有零星买入/续持行。",
    },
    # PX2 的真人账户 (2026-08-18 开户)。策略与 aggr2w 完全一致, 但独立记账:
    # 各自的 state/plan/确认互不影响。参数改动请同步改 aggr2w 与这里两处
    # (有测试钉着两者一致, 见 tests/test_live_config_profiles.py)。
    "aggr2w_px2": {
        "name": "PX2 激进2万",
        "capital": 20000.0,
        "tranche-n": 2,
        "lot-flex": 0.5,
        "opened": "2026-08-18",
        "skip-boards": "30,688",
        "fill-daily": True,
        "roll-rank": 8,
        # "gate-ma60": True,  # GF1 已关 2026-08-27 (GFV3: v3 干净矩阵复核翻案, 见看板)
        "desc": "PX2 的账户 · 策略与「激进 2万」相同(独立记账): 只持 2 只(最集中) · "
                "仅主板 · 大盘弱势自动空仓 · 闲钱日补+到期宽续持 · 弱信号日持币不买(MP1c, 2026-08-22)。"
                "V24PUT(含逐笔)回测(20种子中位, 非单次跑, 2026-08-18 引擎口径): "
                "2022-09~2026-07 +139%(最差种子 +72%, 20个种子无一亏损), 夏普 1.30, "
                "回撤中位 -21%; 同面板不开执行层升级的基线是 +78%。"
                "逐笔数据2023起, 2020-07~2022-08 窗无法评估(旧特征同家族该窗 -49%, 20/20 全亏, "
                "作参考下限)。约46%交易日空仓。",
    },
    "steady5w": {
        "name": "稳妥 5万",
        "capital": 50000.0,
        "tranche-n": 5,
        "opened": "2026-07-31",
        "skip-boards": "30,688",   # 同 aggr2w: 账户无创/科板权限
        # ── FDRR8 执行层 (2026-08-18 上线, 20 种子官方验证) ──
        # 闲钱日补(fill-daily) + 到期仍在前8就续持(roll-rank 8)。同模型同预测,
        # 纯执行层对照(20bp从严口径): +32.8%/夏普0.61 -> +73.5%/夏普1.00,
        # 超额年化 4.65->13.35, 0/20 亏损种子。两旗标均不进指纹, 关掉即回滚。
        # 运维节奏: 非换仓日也可能出 1-2 行补买/续持, 操作日从每5天变潜在每天。
        "fill-daily": True,
        "roll-rank": 8,
        # "gate-ma60": True,  # GF1 已关 2026-08-27 (GFV3: v3 干净矩阵复核翻案, 见看板)
        # T1B 特征集 (2026-09-01 分点分配): K线情绪分解 4 列, 日K收盘即有无 PIT 隷患。
        # 本点位 20 种子配对 vs 同点 EBPB 基线: +16.15pp, 15/20 正(六点里最强),
        # 同点 T1A 只有 +7.1 —— n5 分散持仓吃情绪宽度, 不吃买入深度。证据链见
        # factor_family_ledger「T1B」与看板 LSW(08-31)。不进指纹, 回滚=删此行。
        "features-from": "features_V24PUT_T1B.json",
        "desc": "5 只分散(单股爆雷影响最小) · 仅主板 · K线情绪特征 T1B(2026-09-01) · 大盘弱势自动空仓 · 闲钱日补+到期宽续持 · 弱信号日持币不买(MP1c, 2026-08-22)。"
                "V24PUT(含逐笔)回测(20种子中位, 非单次跑, 2026-08-18 引擎口径): "
                "2022-09~2026-07 +110%(最差种子 +53%, 20个种子无一亏损), 夏普 1.33, "
                "回撤中位 -14%; 同面板不开执行层升级的基线是 +77%(升级纯增益, 非模型变化)。"
                "逐笔数据2023起, 2020-07~2022-08 窗无法评估(旧特征同家族该窗 -26%, 20/20 全亏, "
                "作参考下限)。约46%交易日空仓。执行层升级后非换仓日也可能有零星买入/续持行。",
    },
    "aggr5w": {
        "name": "激进 5万",
        "capital": 50000.0,
        "tranche-n": 3,
        "opened": "2026-07-31",
        "skip-boards": "30,688",   # 同 aggr2w: 账户无创/科板权限
        # FDRR8 执行层 (2026-08-18, 同 steady5w 注释)。20bp从严口径 20种子同面板:
        # +33.5%/夏普0.57 -> +91.6%/1.05, 最差种子 +14.6%->+51.1%, 回撤 -22.4->-18.8。
        "fill-daily": True,
        "roll-rank": 8,
        # "gate-ma60": True,  # GF1 已关 2026-08-27 (GFV3: v3 干净矩阵复核翻案, 见看板)
        "ind-cap": 2,   # IND1 (2026-08-22): 3 只线同行业限 2 只, 详见 steady2w 注释
        # T1A 特征集 (2026-09-01 分点分配): 逐笔订单结构 4 列(lag1, D-1 可得 ——
        # 供应商逐笔包 D+1 早 07:20 才到, 信号链 D 17:30 跑, 故只能用昨日值;
        # T1AL 20 种子实测 lag1 不输 lag0: bare 配对 +28.8 vs +18.6, 直接配对 wash)。
        # 本点位 lag1 全配置复核(T1AV): +16.85pp, 11/20 正 (lag0 时 +8.5)。
        # 证据链见 factor_family_ledger「T1A」。不进指纹, 回滚=删此行。
        "features-from": "features_V24PUT_T1A.json",
        "desc": "仅 3 只集中 · 仅主板 · 同行业最多 2 只(IND1, 2026-08-22) · 订单结构特征 T1A(2026-09-01) · 大盘弱势自动空仓 · 闲钱日补+到期宽续持 · 弱信号日持币不买(MP1c, 2026-08-22)。"
                "V24PUT(含逐笔)回测(20种子中位, 非单次跑, 2026-08-18 引擎口径): "
                "2022-09~2026-07 +146%(最差种子 +98%, 20个种子无一亏损), 夏普 1.42, "
                "回撤中位 -17%; 同面板不开执行层升级的基线是 +79%(升级纯增益, 非模型变化)。"
                "逐笔数据2023起, 2020-07~2022-08 窗无法评估(旧特征同家族该窗 -32%, 20/20 全亏, "
                "作参考下限)。约46%交易日空仓。执行层升级后非换仓日也可能有零星买入/续持行。",
    },
    "aggr10w": {
        "name": "激进 10万",
        "capital": 100000.0,
        "tranche-n": 3,
        "opened": "2026-08-02",   # git 3ed59f8 新增此线
        "skip-boards": "30,688",   # 同 aggr2w: 账户无创/科板权限
        # FDRR8 执行层 (2026-08-18, 同 steady5w 注释)。20bp从严口径 20种子同面板:
        # +37.8%/夏普0.60 -> +106.8%/1.15, 最差种子 +13.1%->+44.9%, 回撤 -22.5->-19.0。
        "fill-daily": True,
        "roll-rank": 8,
        # "gate-ma60": True,  # GF1 已关 2026-08-27 (GFV3: v3 干净矩阵复核翻案, 见看板)
        "ind-cap": 2,   # IND1 (2026-08-22): 3 只线同行业限 2 只, 详见 steady2w 注释
        # T1A 特征集 (2026-09-01 分点分配, 同 aggr5w 注释): 本点位 lag1 全配置
        # 复核(T1AV): +16.4pp, 13/20 正 (lag0 时 +20.35, 六点里最强的 T1A 点)。
        "features-from": "features_V24PUT_T1A.json",
        "desc": "仅 3 只集中, 10万本金 · 仅主板 · 同行业最多 2 只(IND1, 2026-08-22) · 订单结构特征 T1A(2026-09-01) · 大盘弱势自动空仓 · 闲钱日补+到期宽续持 · 弱信号日持币不买(MP1c, 2026-08-22)。"
                "V24PUT(含逐笔)回测(20种子中位, 非单次跑, 2026-08-18 引擎口径): "
                "2022-09~2026-07 +159%(最差种子 +81%, 20个种子无一亏损), 夏普 1.46, "
                "回撤中位 -17%; 同面板不开执行层升级的基线是 +79%(升级纯增益, 非模型变化)。"
                "逐笔数据2023起, 2020-07~2022-08 窗无法评估(旧特征同家族该窗 -37%, 20/20 全亏, "
                "作参考下限)。约46%交易日空仓。执行层升级后非换仓日也可能有零星买入/续持行。",
    },
    "fyf100w": {
        "name": "FYF 100万",
        "capital": 1_000_000.0,
        "tranche-n": 8,
        "opened": "2026-08-18",
        "skip-boards": "30,688",   # 同其他实盘线: 仅主板
        # 选型 2026-08-18(20bp从严, 20种子, PUT5W缓存重放 @100万):
        #   n3+FDRR8 +106%/夏普1.15/回撤-18.7 | n5 +81/1.06/-15.1 | n8 +79/1.15/-13.4
        # 取 n8: 夏普与 n3 打平、回撤浅 5pp, 单槽 12.5万 执行贴回测
        # (n3 单槽 33万 的冲击成本未建模, 纸面多的 27pp 不可全信)。
        "fill-daily": True,
        "roll-rank": 8,
        # "gate-ma60": True,  # GF1 已关 2026-08-27 (GFV3: v3 干净矩阵复核翻案, 见看板)
        # T1B 特征集 (2026-09-01 分点分配): 本点位(fyf100w) 20 种子配对 +8.60pp,
        # 13/20 正, 同点 T1A 更弱 —— 属取最大估计非统计显著, 账本里标注。无 PIT 隷患。
        "features-from": "features_V24PUT_T1B.json",
        "desc": "FYF 实盘 · 100万本金 · 8 只分散 · 仅主板 · K线情绪特征 T1B(2026-09-01) · 大盘弱势自动空仓 · "
                "闲钱日补+到期宽续持 · 弱信号日持币不买(MP1c, 2026-08-22)。V24PUT(含逐笔)回测(20种子中位, 非单次跑, "
                "2026-08-18 引擎口径, 20bp滑点从严): 2022-09~2026-07 +79%"
                "(最差种子 +49%, 20个种子无一亏损), 夏普 1.15, 回撤中位 -13%; "
                "同面板不开执行层升级的基线是 +44%。同参数 0bp 口径参考: 主板 n8 "
                "在 1000万 本金为 +127%(本金越小整手摩擦越明显, 100万 介于两者)。"
                "逐笔数据2023起, 2020-07~2022-08 窗无法评估。约46%交易日空仓。"
                "非换仓日也可能有零星买入/续持行。",
    },
    # ── 不可更改的基准线 ───────────────────────────
    # 上面几条是给真人用的, 会被改名、手工记账、校准现金、删持仓 ——
    # 跑上一段时间后就无法分辨"赚亏是策略本身的还是人为干预的"。
    # 这两条锁死: 只能纯纸面自动记账, 任何写操作都被拒, 作为参照组。
    #
    # 【重要, 2026-08-04 修正】这两条曾被描述为"与稳妥/激进5万参数逐字一致",
    # 那是错的: steady5w / aggr5w 都带 skip-boards="30,688"(账户没开创业板与
    # 科创板权限), 而这两条没有。所以"真实账户 - 基准线"的差额里, 主要成分是
    # 【可投资范围不同】, 而不是当初想测的"人为干预的代价"。
    #
    # 经决定保持全市场不变(2026-08-04), 因此它们的定位改为:
    #   "如果当初开了创业板/科创板权限, 同样的模型会跑成什么样"
    # 想测"人为干预的代价"需要另建两条带 skip-boards 的纸面对照线。
    "base5w_steady": {
        "name": "基准·稳妥5万",
        "capital": 50000.0,
        "tranche-n": 5,
        "locked": True,
        "opened": "2026-07-31",   # git fa49a02 两条基准线同日上站
        # 2026-08-22 用户改口径: 基准线定位从"冻结老策略"改为"始终跟随最新策略",
        # 唯一对照变量 = 有无人为干预(改名/改账/漏确认等)。故补齐 FDRR8 + GF1。
        # ⚠ 时序断层: 2026-08-22 之前的净值是老策略(无 FDRR8/门控)跑出来的,
        # 做长期对比时以此日为界。三开关均不进指纹, 持仓/现金不受影响。
        "fill-daily": True,
        "roll-rank": 8,
        # "gate-ma60": True,  # GF1 已关 2026-08-27 (GFV3: v3 干净矩阵复核翻案, 见看板)
        # 2026-09-01 跟随镜像对象 steady5w 切 T1B(08-22 口径: 始终跟随最新策略)。
        "features-from": "features_V24PUT_T1B.json",
        "desc": "全市场版的 5 只持仓参照组(含创业板/科创板)。永久纸面自动记账且"
                "始终跟随最新策略(2026-08-22 起叠 FDRR8+GF1 门控, 2026-08-27 关 GF1; 此前为老策略时序), "
                "不可改名/不可改记账/不可校准现金/不可删持仓。"
                "用途: 与真实线对比时隔离人为干预的影响(另含创科板可投范围差异)。"
                "旧策略参考回测(20种子中位, 不含新叠层): 2022-09~2026-07 +45%"
                "(1/20 亏损); 2020-07~2022-08 -10%(18/20 亏损)。",
    },
    "base5w_aggr": {
        "name": "基准·激进5万",
        "capital": 50000.0,
        "tranche-n": 3,
        "locked": True,
        "opened": "2026-07-31",
        # 同 base5w_steady: 2026-08-22 起始终跟随最新策略, 只隔离人为干预变量。
        "fill-daily": True,
        "roll-rank": 8,
        # "gate-ma60": True,  # GF1 已关 2026-08-27 (GFV3: v3 干净矩阵复核翻案, 见看板)
        "ind-cap": 2,   # IND1 (2026-08-22): 基准线跟随最新策略, 3 只线同步开
        # 2026-09-01 跟随镜像点位(50k/n3+cap2, 同 aggr5w)切 T1A(08-22 口径同上)。
        "features-from": "features_V24PUT_T1A.json",
        "desc": "全市场版的 3 只持仓参照组(含创业板/科创板)。永久纸面自动记账且"
                "始终跟随最新策略(2026-08-22 起叠 FDRR8+GF1 门控, 2026-08-27 关 GF1; 此前为老策略时序), "
                "不可改名/不可改记账/不可校准现金/不可删持仓。"
                "用途: 与真实线对比时隔离人为干预的影响(另含创科板可投范围差异)。"
                "旧策略参考回测(20种子中位, 不含新叠层): 2022-09~2026-07 +74%"
                "(1/20 亏损); 2020-07~2022-08 -24%(17/20 亏损)。",
    },
    # ── 1000万 假想基准线 (2026-08-18 上站) ───────────────
    # 用途: "如果本金是 1000万, 同一套模型+执行层会跑成什么样" 的前向纸面跟踪。
    # 口径与真实线一致(主板-only + FDRR8), 与 base5w_* 的全市场定位不同。
    # 选型依据 (载荷重放 20 种子 x n3/n5/n8 x 基线/FDRR8, 0bp, /tmp/b10m.log):
    #   n3 FDRR8 +139.6%/夏普1.39/回撤-18.8/最差+61.1
    #   n5 FDRR8 +137.8%/夏普1.50/回撤-14.8/最差+78.6
    #   n8 FDRR8 +126.9%/夏普1.58/回撤-12.0/最差+90.8  ← 选这个
    # n8 中位收益只让 ~11pp, 换来最高夏普/最浅回撤/最厚最差种子 —— 基准线
    # 看风险调整后收益。1000万下无整手粒度约束, n8 不再有 2万时代的资金病
    # (当年 5万/n8 只有 +13.3% 是槽位买不起股, 不是 n8 本身的错)。
    # ⚠ 容量假设: 单槽 125万 打进 630 池中小盘的冲击成本引擎不建模,
    # 回测数字是"无冲击"上界, 只作方向参照, 别当可执行承诺。
    "bench10m": {
        "name": "基准·1000万",
        "capital": 10_000_000.0,
        "tranche-n": 8,
        "locked": True,
        "opened": "2026-08-18",
        "skip-boards": "30,688",   # 与真实线同口径(仅主板)
        "fill-daily": True,        # FDRR8: 与 4 条主板真实线同执行层
        "roll-rank": 8,
        # "gate-ma60": True,  # GF1 已关 2026-08-27, 基准线随真实线同步 (GFV3 翻案)
        "desc": "假想 1000万 本金的策略基准(2026-08-18 起前向纸面跟踪) · 8 只分散 · "
                "仅主板 · 大盘弱势自动空仓 · 闲钱日补+到期宽续持 · 弱信号日持币不买(MP1c, 2026-08-22) · 永久锁定不可人为干预。"
                "V24PUT(含逐笔)回测(20种子中位, 非单次跑, 2026-08-18 引擎口径): "
                "2022-09~2026-07 +127%(最差种子 +91%, 20个种子无一亏损), 夏普 1.58, "
                "回撤中位 -12%; 同面板不开执行层升级的基线是 +78%。"
                "2020-07~2022-08 窗无法评估(逐笔数据2023起; 同家族旧特征在5万级操作点"
                "该窗 -26%~-49% 全亏, 作参考下限)。约46%交易日空仓。"
                "⚠ 大资金冲击成本未建模: 单槽 125万 对中小盘的滑点回测不计, "
                "数字是无冲击上界, 作方向参照而非业绩承诺。",
    },
    "bench10m_fm": {
        "name": "基准·1000万全市场",
        "capital": 10_000_000.0,
        "tranche-n": 8,
        "locked": True,
        "opened": "2026-08-18",
        # 全市场: 不设 skip-boards, 与 bench10m(仅主板)成对照
        "fill-daily": True,
        "roll-rank": 8,
        # "gate-ma60": True,  # GF1 已关 2026-08-27, 同 bench10m
        "desc": "假想 1000万 本金的全市场策略基准(2026-08-18 起前向纸面跟踪) · "
                "8 只分散 · 含创业板/科创板 · 大盘弱势自动空仓 · 闲钱日补+到期宽续持 · 弱信号日持币不买(MP1c, 2026-08-22) · "
                "永久锁定不可人为干预。与基准·1000万(仅主板)唯一差别是可投范围, "
                "用于前向检验\"全市场超额集中在 20cm 板块\"的回测结论。"
                "V24PUT(含逐笔)回测(20种子中位, 非单次跑, 2026-08-18 引擎口径): "
                "2022-09~2026-07 +178%(最差种子 +130%, 20个种子无一亏损), 夏普 1.54, "
                "回撤中位 -15%; 同面板不开执行层升级的基线是 +121%。"
                "2020-07~2022-08 窗无法评估(逐笔数据2023起)。"
                "⚠ 大资金冲击成本未建模: 单槽 125万, 且 20cm 小票流动性更薄, "
                "无冲击上界口径比主板版更宽松, 作方向参照而非业绩承诺。",
    },
}

# 锁死的条线: 不得改名、不得切记账方式、不得现金校准/出入金/删持仓/
# 手工确认成交。这是唯一判定入口, 前端隐控件、后端拒请求都读它。
LOCKED = tuple(pid for pid, p in PROFILES.items() if p.get("locked"))

# 默认展示哪条线
DEFAULT_PROFILE = "steady5w"

# ══════════════════════════════════════════════════════════════
# 网页访问口令表 (2026-08-18 用户指定)
# ══════════════════════════════════════════════════════════════
# 口令即身份: 登录框只有一个输入位, 输哪个口令拿哪个身份。
#   ro    -> 全部线可见, 只读 (任何修改请求都被拦)
#   admin -> 全部线可见, 全部可写 (改账不再另问 ops 口令)
#   acct  -> 只看见自己名下的线(卡片/切换列表按 pids 过滤, 越权 403),
#            改自己的线不需要再输 ops 口令 —— 口令本身就是身份凭证
# 旧的 QUANT_VIEW_PASSWORD / QUANT_VIEW_RO_PASSWORD 环境变量继续有效
# (老 cookie 不失效), 语义不变: 环境变量 full 看全站但改账仍要 ops 口令。
# 改/删口令直接改这张表再重启; token 签名密钥掺口令本身, 改动即作废,
# 且只作废被改的那个口令, 不影响其他人的登录状态。
ACCESS_CODES = {
    "213213": {"role": "ro"},
    "611611": {"role": "admin"},
    "px":  {"role": "acct", "pids": ("steady5w", "aggr2w_px2")},  # Px 本尊 + PX2
    "llx": {"role": "acct", "pids": ("aggr5w",)},
    "phy": {"role": "acct", "pids": ("aggr2w",)},
    "xjb": {"role": "acct", "pids": ("aggr10w",)},
    "fyf": {"role": "acct", "pids": ("fyf100w",)},
    "xmy": {"role": "acct", "pids": ("steady2w",)},  # 线上改名为 plmm_2
}

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
# capital=None: 用 PROFILES 里的代码默认值; 非 None 则是网页上重置时改过的本金。
DEFAULT_SETTING = {"name": None, "auto": True, "capital": None}

NAME_MAX = 16

# 本金的合理区间。下限不是拍的: 一个槽位连最便宜的股票一手(100股)都买
# 不起的话, 策略根本无法运行。上限只是防手抖多敲一个 0。
CAPITAL_MIN = 5000.0
CAPITAL_MAX = 10_000_000.0


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
            try:
                cap = float(v["capital"]) if v.get("capital") is not None else None
            except (TypeError, ValueError):
                cap = None
            if cap is not None and not (CAPITAL_MIN <= cap <= CAPITAL_MAX):
                cap = None          # 设置文件被手改坏了 -> 退回代码默认
            out[pid] = {"name": v.get("name") or None,
                        "auto": bool(v.get("auto", True)),
                        "capital": cap}
    return out


def save_settings(s):
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = SETTINGS_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(SETTINGS_PATH)          # 原子替换, 避免读到写一半的文件


def setting(pid):
    return {**DEFAULT_SETTING, **load_settings().get(pid, {})}


def is_locked(pid):
    """基准线: 不接受任何人为修改"""
    return bool(PROFILES.get(pid, {}).get("locked"))


def main_board_only(pid):
    """这条线是否只买主板 (账户没开创业板/科创板权限)"""
    return bool(PROFILES.get(pid, {}).get("skip-boards"))


def display_name(pid):
    """用户改过就用用户的, 否则用代码里的默认名。

    基准线一律用代码名 —— 即使设置文件被手改过也不认, 否则参照组被改名后
    就认不出来了。
    """
    if is_locked(pid):
        return PROFILES[pid]["name"]
    return setting(pid)["name"] or PROFILES[pid]["name"]


def is_auto(pid):
    """基准线永远自动记账 —— 它存在的意义就是纯纸面跟踪策略本身"""
    if is_locked(pid):
        return True
    return setting(pid)["auto"]


def capital_of(pid):
    """该条线当前的本金 —— 网页上“从头再来”改过就用改过的。

    本金不在 FINGERPRINT_KEYS 里, 所以改它不会与已有持仓冲突; 但它只在
    --init 时生效(作为起始现金), 所以光改它不重置是没用的 —— 因此
    set_capital 只由重置接口调用。

    基准线永远用代码里的值: 它就该一直待在那里不动。
    """
    if pid not in PROFILES:
        raise KeyError(pid)
    if is_locked(pid):
        return float(PROFILES[pid]["capital"])
    return float(setting(pid)["capital"] or PROFILES[pid]["capital"])


def set_capital(pid, cap):
    """改本金。仅供重置流程调用 —— 单独改它不会影响已建立的账。"""
    if pid not in PROFILES:
        raise KeyError(pid)
    if is_locked(pid):
        raise PermissionError(f"{PROFILES[pid]['name']} 是基准线, 本金固定不可改")
    cap = float(cap)
    if not (CAPITAL_MIN <= cap <= CAPITAL_MAX):
        raise ValueError(
            f"本金要在 {CAPITAL_MIN:,.0f} ~ {CAPITAL_MAX:,.0f} 之间")
    s = load_settings()
    s.setdefault(pid, dict(DEFAULT_SETTING))["capital"] = cap
    save_settings(s)
    return cap


def set_name(pid, name):
    """name 传 None 或空串 = 恢复默认名"""
    if pid not in PROFILES:
        raise KeyError(pid)
    if is_locked(pid):
        raise PermissionError(f"{PROFILES[pid]['name']} 是基准线, 不可改名")
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
    if is_locked(pid):
        raise PermissionError(f"{PROFILES[pid]['name']} 是基准线, 记账方式锁定为纸面自动")
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
    # 条线级参数覆盖: profile 里出现的参数名(带横线的 key)直接覆盖基础参数。
    # 目前只有 lot-flex: 2万线频繁触发"买不起一手"救济有效, 5万线未验证有效不开
    for k, v in p.items():
        if "-" in k:
            params[k] = v
    # 分线特征集(2026-09-01 T1A/T1B 分点分配): include_features=False 时
    # 连线自己的 features-from 也不发, 语义与全局开关一致。
    if not include_features:
        params.pop("features-from", None)
    out = []
    for k, v in params.items():
        # 布尔旗标 (如 fill-daily): True 只发旗标不带值, False/None 不发 ——
        # argparse 的 store_true 参数不接受 "--flag True" 这种写法。
        if v is True:
            out += [f"--{k}"]
        elif v is False or v is None:
            continue
        else:
            out += [f"--{k}", str(v)]
    out += ["--state", state_file(pid)]
    # 手动模式的线不得自动记账。放在这里而不是各调用方自己判断,
    # 是因为 daily_rebuild / web_server / 命令行 三处都走 signal_args,
    # 写在这里就不可能出现"定时任务绕过了开关把账记了"这种事。
    # 此参数不在 FINGERPRINT_KEYS 里, 所以来回切换不会弄坏已有持仓。
    if not is_auto(pid):
        out += ["--require-confirm"]
    # 同模型的线共用一个预测缓存: 当天第一条线训练并写入, 其余直接读。
    # 缓存键含信号日与全部训练输入, 任一项变化都会自动重训, 不会读到过期预测。
    # 同样不在 FINGERPRINT_KEYS 里, 加它不影响已有持仓。
    # 主板-only 的线模型不同(训练集/截面都变了), 用独立缓存文件 ——
    # 否则两类线在同一文件里互相覆盖, 每天 6 次全部 cache miss 白训。
    _cache = PREDS_CACHE_PATH
    if params.get("skip-boards"):
        _cache = _cache.with_name(f"{_cache.stem}_mb{_cache.suffix}")
    # 门控线在弱势态用 86 特征, 与未开门控的线(如冻结参照线)缓存键不同 ——
    # 共用一个文件会在弱势日互相覆盖对方的缓存造成反复重训, 故分文件。
    if params.get("gate-ma60"):
        _cache = _cache.with_name(f"{_cache.stem}_gf{_cache.suffix}")
    # 分特征集的线也要分缓存文件: 缓存文件是单条目的(键不同即整体覆盖),
    # T1A/T1B/基线特征各是不同模型, 共用一个文件会在夜链里循环互踩,
    # 每条线都 cache miss 白训一遍。文件名后缀 = 特征表名去掉 features_ 前缀。
    _ff = params.get("features-from")
    if _ff:
        _slug = Path(_ff).stem.lower().removeprefix("features_")
        _cache = _cache.with_name(f"{_cache.stem}_{_slug}{_cache.suffix}")
    out += ["--preds-cache", str(_cache)]
    # 线自己的 features-from 已在上面的参数循环里发出; 只有没覆盖的线才发全局表。
    # (不能两个都发: argparse 后者覆盖前者, 全局表会把分线表顶掉。)
    if include_features and FEATURES_FROM and "features-from" not in params:
        out += ["--features-from", FEATURES_FROM]
    return out


def init_args(pid):
    """首次建立(或重置)该条线的参数 —— 会清空持仓记录"""
    return signal_args(pid) + ["--capital", str(capital_of(pid)), "--init"]


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
            cap = capital_of(pid)
            changed = "" if cap == p["capital"] else f"  [代码默认 {p['capital']:,.0f}]"
            print(f"           本金 {cap:,.0f}{changed} / {p['tranche-n']} 只 / "
                  f"每只预算 {cap/p['tranche-n']:,.0f}")
            print(f"           记账 {'自动(按行情)' if is_auto(pid) else '手动(等确认成交)'}")
            print(f"           状态 {state_file(pid)}")
