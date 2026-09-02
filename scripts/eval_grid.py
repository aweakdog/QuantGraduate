"""多种子 x 双窗口 评估框架 —— 让"这个配置好"变成可检验的陈述

为什么需要这个
──────────────
1) 单次回测的收益是随机数。3 只持仓下, 20 个随机种子的总收益从 -31% 到 +190%,
   而它们的 IC 全部落在 0.0133~0.0170 这个极窄区间。也就是说同样的预测能力,
   收益能差 220 个百分点。任何"某配置 +292%"的结论都可能只是抽中了右尾。
   => 所以本框架的任何指标都取【多种子分布】, 报中位数 / 最差 / 分位数,
      而不是单次值。

2) data/processed/ 下已有 90+ 个同窗口(2022-09~2026-07)的回测结果, 这段数据
   被用于选参数上百次。在同一段数据上反复挑最优, 挑出来的是噪声。
   => 所以配置必须在【两个独立窗口】上同时达标才采纳。

窗口定义与"无未来函数"的前提
────────────────────────────
    窗口A  2020-07-01 ~ 2022-08-31 (530 交易日)  从未用于任何调参
    窗口B  2022-09-01 ~ 2026-07-27 (943 交易日)  历史上被反复使用

    关键: 80 个入选特征必须【按窗口各自现场筛选】。现行线上锁定的特征集
    (wf_daily_REGRESS_CHK_*.json) 是用 2022-09 之前的数据筛的, 而这段数据
    正好覆盖窗口A的整个测试期 —— 直接拿来测窗口A就是未来函数, 结果会虚高。
    因此本框架对每个窗口各跑一次"筛特征"运行(只用该窗口起点之前的数据),
    再把结果锁给该窗口的所有种子。两个窗口因此用不同的 80 个特征, 这是正确的:
    我们检验的是"同一套流程按时间点重复应用", 不是"同一组特征".

三个阶段 (都可断点续跑, 已有产物自动跳过)
──────────────────────────────────────
    features : 每窗口 1 次, 现场筛特征并落盘, 供该窗口后续所有运行复用
    caches   : 每窗口 x 每种子 1 次模型运行, 落盘预测缓存 (贵, 一次性)
    eval     : 每配置 x 每窗口 x 每种子 1 次, --load-preds 跳过训练 (便宜)

因为 regime / min-pred / tranche-n / roll-rank / lot-flex / 成本 全是执行层参数,
换这些不需要重训 —— 建好缓存后, 每个新想法只需跑 eval 阶段。

用法
────
    python scripts/eval_grid.py features                # 先筛两个窗口的特征
    python scripts/eval_grid.py caches --seeds 20       # 建 2x20 预测缓存
    python scripts/eval_grid.py eval --configs base,n5  # 评估配置
    python scripts/eval_grid.py report --configs base,n5

注意: 本框架只读 data/processed 下的实验产物, 不碰线上任何文件。
"""
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
LOGDIR = ROOT / "data" / "processed" / "eval_logs"
# 解释器不能硬编码 .venv/bin/python: 分布式跑任务时 eez040/042 上没有 .venv
# (缺 python3-venv 且装它要 sudo, 改用了 pip --user), 硬编码会让子进程启动时
# 报 FileNotFoundError。用当前解释器 —— 无论是 .venv 还是系统 python3 都对。
PY = sys.executable

# 实验数据源: 2019 起的矩阵 + 修复了退市股缺口的 PIT 池
TRAIN_FILE = "training_data_pit_2019.parquet"
PIT_UNIVERSE = "universe_pit_2019.parquet"

# ── 两个独立窗口 ────────────────────────────────────────────
# 每个窗口的"筛特征"运行 tag 由 feat_tag(win, variant) 生成, 见下方函数 ——
# full 变体沿用 EVALFEAT_{win} 这个原名, 以便复用已经跑好的产物。
WINDOWS = {
    "A": {"test_start": "2020-07-01", "test_end": "2022-08-31",
          "desc": "2021抱团+2022熊市, 未用于调参"},
    "B": {"test_start": "2022-09-01", "test_end": "2026-07-27",
          "desc": "历史上被反复用于选参数"},
}

DEFAULT_SEEDS = [42, 7, 123, 2024, 31337, 1, 2, 3, 5, 11,
                 17, 23, 55, 77, 99, 202, 314, 512, 888, 1234]

# 所有运行共享的模型层参数。改这里等于换模型, 缓存必须重建。
MODEL_ARGS = ["--train-file", TRAIN_FILE, "--pit-universe", PIT_UNIVERSE,
              "--label", "5d", "--objective", "l2"]


def model_args(variant):
    """按变体解析出模型层参数 —— 主要是替换训练矩阵文件。

    为什么不能只把 --train-file 放在变体的 "model" 列表里: eval 阶段的命令
    【不带】v["model"](那时靠 --load-preds 跳过训练), 但 wf_v35 仍会把
    train_file 纳入缓存 meta 的一致性校验, 于是 features/caches 阶段用了新矩阵、
    eval 阶段却传回旧矩阵名, 直接报"预测缓存不匹配"。所以必须在这一层替换。
    """
    tf = VARIANTS[variant].get("train_file")
    if not tf:
        return list(MODEL_ARGS)
    a = list(MODEL_ARGS)
    a[a.index("--train-file") + 1] = tf
    return a

# ── 股票池变体 ──────────────────────────────────────────────
# 线上 5 条线里有 4 条带 --skip-boards 30,688 (账户没开创业板/科创板权限)。
# 这【必须】在模型层隔离: wf_v35 里是 `if SKIP_BOARDS and not args.load_preds`,
# 也就是说带 --load-preds 时只在执行层事后跳过, 不再从训练样本/截面 demean/
# 候选池里剔除 —— 那是另一个策略(项目文档实测丢掉约 70% 利润)。
# 所以主板线要单独建一套缓存, 不能复用全市场的。
#
# 产物命名: full 沿用原名(不动已建好的 40 个缓存), mb 另起 _mb 前缀。
VARIANTS = {
    "full": {"model": [], "exec": [], "feat": "EVALFEAT", "cache": "preds_eval",
             "ev": "EV", "desc": "全市场 (对应线上 steady2w 与两条基准线)"},
    "mb":   {"model": ["--skip-boards", "30,688"],
             "exec": ["--skip-boards", "30,688"],
             "feat": "EVALFEAT_MB", "cache": "preds_eval_mb", "ev": "EVMB",
             "desc": "仅主板 (对应线上 aggr2w/steady5w/aggr5w/aggr10w)"},

    # ── 受控实验 (2026-08-05): 只开 --drop-market-wide, 其余一切不变 ──
    # 动机: 431 个候选特征里有 159 个【截面变异比 = 0】—— 当日对所有股票取
    # 同一个值(利率/汇率/商品/外盘指数/tev_all_* 全市场事件聚合)。而标签是
    # 按日期 demean 的截面收益, 所以这类特征对"今天哪只股票比别人强"预测力
    # 恒为 0, 在树里唯一用处是切分日期区间 = 记忆训练期。
    # 上轮入选的 80 个特征里就有 12 个是这类(usdcnh/cn5y/us2y_ma5/usdind_ma20/
    # cn_commodity_idx_*/sox_chg_5d_ma5/tev_all_bull_5d/tev_all_bear_5d/...),
    # 放宽到变异比<0.05 是 18 个。
    # wf_v35 本来就有 --drop-market-wide 专门防这件事(帮助文本建议 0.01),
    # 但从未启用过, 默认 0。
    #
    # 这可能解释三件之前想不通的事:
    #   两窗口特征筛选只重叠 35/80 —— 没有截面信息的特征靠"哪段日期恰好能
    #     切出来"入选, 换窗口自然全变
    #   窗口A 20/20 全亏 —— 模型在训练段靠记忆日期, 到测试期失效
    #   IC 只有 0.03~0.05 —— 80 个里有 12~18 个贡献的是记忆而非 alpha
    #
    # 变体只改这一个参数 -> 结果可直接与 full/mb 对比, 差异就是它的影响。
    # 272 个候选里筛 80, 筛选仍然有效(不像叠加早期可得性白名单后只剩 90 个,
    # 那样 90 里筛 80 等于不筛, 会把"改了三样东西"混成一个结果无法归因)。
    "full_dmw": {"model": ["--drop-market-wide", "0.01"],
                 "exec": [], "feat": "EVALFEAT_DMW", "cache": "preds_eval_dmw",
                 "ev": "EVDMW",
                 "desc": "全市场 + 剔除截面无变异特征 (受控实验)"},
    "mb_dmw":   {"model": ["--skip-boards", "30,688",
                           "--drop-market-wide", "0.01"],
                 "exec": ["--skip-boards", "30,688"],
                 "feat": "EVALFEAT_MBDMW", "cache": "preds_eval_mbdmw",
                 "ev": "EVMBDMW",
                 "desc": "仅主板 + 剔除截面无变异特征 (受控实验)"},

    # ── 低覆盖率特征该不该用 (2026-08-05) ────────────────────────
    # 发现: con_*(概念聚合) 与 tev_*(事件衰减) 在【任何年份】都只覆盖 10~17%
    # 的样本行, 其余 83~90% 是 NaN 被 ffill().fillna(0) 填成 0。
    #   con_amount:      2015年 10% -> 2026年 17%
    #   tev_decay_bear:  2015年 11% -> 2026年 17%
    # 也就是说这不是"早期缺失", 2026 年也一样缺。而上轮入选的 80 个里有 9 个
    # con_* 和 2 个 tev_*。
    #
    # 危险在于: 模型可能学到的是"这只股票在不在这个数据源里", 而不是概念动量
    # 本身。这和市场级特征是同一类病 —— 特征携带的信息不是它名义上代表的东西:
    #   市场级特征   -> 实际在标记"日期"
    #   低覆盖率特征 -> 实际在标记"这只股票在不在某个数据源里"
    # 后者更隐蔽, 因为"在不在数据源里"往往与市值/流动性相关, 所以看起来有预测力。
    #
    # 在 mb_dmw 之上只加这一个改动, 所以差异可直接归因。
    "mb_dmw_nocon": {"model": ["--skip-boards", "30,688",
                               "--drop-market-wide", "0.01",
                               "--exclude-feats", "con_*,tev_*"],
                     "exec": ["--skip-boards", "30,688"],
                     "feat": "EVALFEAT_MBNOCON", "cache": "preds_eval_mbnocon",
                     "ev": "EVMBNOCON",
                     "desc": "mb_dmw 再剔除低覆盖(10~17%)的 con_*/tev_*"},

    # ── 财务特征该不该用 = 值不值得花钱买历史 ─────────────────────
    # 财务数据来自 iFinD, 2018-05 才有; 2018 前是 0%, 2019 后也只有 64~70%
    # (三成股票始终没有)。上轮入选 80 个里有 8 个财务特征。
    # 买 2015-2018 的历史只能把那段从 0% 提到约 70%, 剩下 30% 买不到。
    # 所以【先测有没有用, 再决定要不要买】—— 已有教训: 12 个市场级特征入选了,
    # 但剔掉后窗口B 的 IC 只掉 0.0024, 入选不等于有用。
    "mb_dmw_nofund": {"model": ["--skip-boards", "30,688",
                                "--drop-market-wide", "0.01",
                                "--exclude-feats",
                                "revenue*,profit*,eps*,bps*,roe*,total_assets*,"
                                "debt_ratio*,operate_cf*,gross_margin*,pe*,pb*"],
                      "exec": ["--skip-boards", "30,688"],
                      "feat": "EVALFEAT_MBNOFUND", "cache": "preds_eval_mbnofund",
                      "ev": "EVMBNOFUND",
                      "desc": "mb_dmw 再剔除全部财务特征 (判断值不值得买历史)"},

    # ── 融券余额特征 (2026-08-07) ─────────────────────────────────
    # 16 张 tushare 表逐字段过 IC + top5 分层双关卡后, 唯一存活的是 rqye(融券余额)。
    # 它是全场【唯一在窗口A 的 top5 超额为正】的因子(+0.316%, 按流通市值归一后),
    # 而窗口A 正是我们唯一的堵点 —— 模型在那里的 top5 超额是 -0.164%。
    #
    # 诚实标注先验很弱: 独立采样(每5日, 消除5日前瞻重叠)的 t 值只有 A 1.3 / B 0.6,
    # 低于事前定的 t>2。所以这一轮的目的是【证死或证实】, 不是期待它翻盘。
    # 对照组已有的 daily_basic 全线否决(IC 两窗同号但 top5 全为零或负), 说明
    # "IC 好看" 完全不足以推断 top5 能赚钱, 这次也一样要看端到端结果。
    #
    # 只加 2 列, 训练矩阵由 add_margin_features.py 在原矩阵上 merge 得到 ——
    # 除这 2 列外与 mb_dmw 用的矩阵逐格相同, 所以差异可直接归因。
    # 不走 feature_engine 全量重建是刻意的: 那会引入无关的重算漂移
    # (见 diag_rebuild_drift.py), 把"加了特征"和"重算了一遍"混在一起。
    "mb_dmw_rq": {"model": ["--skip-boards", "30,688",
                            "--drop-market-wide", "0.01"],
                  "exec": ["--skip-boards", "30,688"],
                  "feat": "EVALFEAT_MBRQ", "cache": "preds_eval_mbrq",
                  "ev": "EVMBRQ",
                  "train_file": "training_data_pit_2019_rq.parquet",
                  "desc": "mb_dmw + 融券余额2列 (rq_bal_mv, rq_bal_pct)"},
}
# 注意 mb 变体的一个已知口径问题: eval 阶段带 --load-preds 时 df 不再被板块过滤,
# 所以产物里的 benchmark 仍是全市场等权, 而策略只买主板 -> excess/IR 不可直接
# 与 full 变体比。本框架的门槛只用 收益/夏普/回撤/亏损种子数, 都不依赖基准,
# 所以不影响结论; 但看 IR 时要记得这点。

# 所有运行共享的执行层默认值 (与线上 BASE_PARAMS 对齐)
#
# 滑点 2026-08-06 从 0.002 改为 0.0005 —— 这不是调参, 是修一个高估了 4 倍的
# 成本假设。实测 (mb_dmw, 3只, 20种子, 窗口B 收益中位):
#     滑点 0.20%  +34.5%      滑点 0.10%  +65.2%
#     滑点 0.05%  +86.5%      滑点 0      +104.2%
# 0.2%/边 意味着往返 0.52%(含佣金印花税), 一年 48 次调仓 = -25%/年 的地板,
# 比选股能力的影响大得多。而 0.2% 是把三个成分都按最坏情况叠加算出来的:
#     价差    A股 tick 0.01元, 10~50元的股票半价差只有 0.01~0.05%
#     冲击    5万元 / 1.28亿日成交额 = 0.04‰, 基本为 0
#     时点偏差 尾盘成交 vs 按收盘价记账, 这部分【对称零均值】, 不是成本
# 0.05% 是对前两项(真成本)的现实估计。保留 base3_slip0/slip10 两个对照配置,
# 它们显式覆盖这个默认值, 用来随时复查这个假设的敏感性。
EXEC_BASE = ["--hold-days", "5", "--portfolio-mode", "periodic",
             "--exec-mode", "t1close", "--slippage", "0.0005"]

# 改了 EXEC_BASE 就必须同步这里 —— 见 collect() 的陈旧结果校验。
# ev_tag 里【不含】执行层参数, 所以改了 EXEC_BASE 却不重跑, 旧参数跑出来的结果
# 文件会因文件名重合而被静默当成新结果。本轮已经踩过两次同类的坑(rsync 不同步
# 导致整段实验静默失败、换仓相位混进持有天数对比), 所以这里做校验: 对不上的
# 产物排除出判定并在报告末尾列出, 不允许"看起来有结果"。
EXEC_EXPECT = {"slippage": 0.0005, "exec_mode": "t1close"}

# collect() 发现的陈旧产物, 在报告末尾统一列出
STALE_SEEN = []

# ── 待评估的配置 ────────────────────────────────────────────
# 只允许放【执行层】参数, 否则 --load-preds 会因缓存不匹配而拒绝运行
# (这正是我们想要的保护: 模型层参数变了就必须重建缓存)。
#
# 不能放这里的东西 (会静默变成另一个语义):
#   --skip-boards  wf_v35 里是 `if SKIP_BOARDS and not args.load_preds` ——
#       带 --load-preds 时只在执行层跳过, 不再从训练/截面/候选池里剔除。
#       这是"事后跳过"变体, 实测丢掉约 70% 利润, 与线上主板线的语义不同。
#       要评估主板线必须另建一套缓存(模型层就隔离掉那些板块)。
#   --objective / --n-features / --neutralize-style / --train-start
#       都是模型层, 改了必须重建缓存 (脚本会因缓存不匹配直接报错, 是好事)。
#
# 关于"试很多配置"的风险: 试得越多, 靠运气过门槛的概率越高。这里的对冲是
# 要求【两个独立窗口同时达标】—— 一个纯噪声的改动要在两段不同 regime 上
# 同时表现好, 概率低得多。但也别无节制地加: 保持在 10 个以内, 且每个都要
# 有事前的机制解释, 而不是"扫一遍看哪个数字最大"。
CONFIGS = {
    # ── 基线 ──
    "base3":  {"desc": "3只持仓, 不择时 (对应线上 aggr5w/aggr10w 的持仓数)",
               "args": ["--tranche-n", "3", "--initial-capital", "50000",
                        "--regime-filter", "off"]},
    "base5":  {"desc": "5只持仓, 不择时 (对应线上 steady5w)",
               "args": ["--tranche-n", "5", "--initial-capital", "50000",
                        "--regime-filter", "off"]},

    # ── 第二轮: 目标是压最差种子的回撤与离散度, 不是抬中位数 ──
    # 机制: 大盘转弱时整体空仓。3只持仓的深回撤基本都发生在系统性下跌中,
    # 个股选得再准也躲不开 beta(实测 beta 1.43)。注意历史上 breadth 择时在
    # 单次 n=3 回测里被判定"有害", 但那个结论本身就落在噪声里, 需要重测。
    "g3_regime": {"desc": "3只 + breadth择时(弱势空仓), 压系统性回撤",
                  "args": ["--tranche-n", "3", "--initial-capital", "50000",
                           "--regime-filter", "breadth", "--regime-ma", "20",
                           "--regime-breadth", "0.40", "--regime-confirm", "2"]},
    # 机制: 波动率目标仓位 —— 高波动期自动降低敞口
    "g3_vol":    {"desc": "3只 + 波动率目标仓位, 高波动期降敞口",
                  "args": ["--tranche-n", "3", "--initial-capital", "50000",
                           "--regime-filter", "off", "--vol-target"]},
    # 机制: 信号不够强就不买, 留现金。避免在模型没把握时硬凑3只
    # 门槛取 0.006: 实测每日前3名预测均值的中位数是 窗口A 0.0111 / 窗口B 0.0062,
    # 所以 0.006 大约过滤掉窗口B 一半的换仓日、窗口A 四分之一。
    # (先试过 0.002, 95~100% 的日子都能过, 等于没加门槛)
    # 注意这是绝对阈值, 两个窗口的预测值分布本身不同 -> 它在两窗口上的松紧不一致,
    # 这本身就是个鲁棒性隐患, 结果要结合这点看。
    "g3_minpred": {"desc": "3只 + 信号强度门槛0.006, 打分不够就留现金",
                   "args": ["--tranche-n", "3", "--initial-capital", "50000",
                            "--regime-filter", "off", "--min-pred", "0.006"]},
    # 机制: 不追已经急涨的股。3只持仓下单只追高被打脸的伤害被放大3倍
    "g3_rev":    {"desc": "3只 + 反转护栏(剔除近5日涨幅前10%)",
                  "args": ["--tranche-n", "3", "--initial-capital", "50000",
                           "--regime-filter", "off", "--reversal-guard", "0.10"]},
    # 机制: 降换手省费用。基线费用占本金 16%, 每年烧 4%, 是确定性损失
    "g3_roll":   {"desc": "3只 + 卖出容忍(仍在前8名就续持), 降换手省费用",
                  "args": ["--tranche-n", "3", "--initial-capital", "50000",
                           "--regime-filter", "off", "--roll-rank", "8"]},
    # 5只版本只带最有希望的两个, 避免配置数膨胀
    "g5_regime": {"desc": "5只 + breadth择时",
                  "args": ["--tranche-n", "5", "--initial-capital", "50000",
                           "--regime-filter", "breadth", "--regime-ma", "20",
                           "--regime-breadth", "0.40", "--regime-confirm", "2"]},
    "g5_roll":   {"desc": "5只 + 卖出容忍(前10名续持)",
                  "args": ["--tranche-n", "5", "--initial-capital", "50000",
                           "--regime-filter", "off", "--roll-rank", "10"]},

    # ── 第三轮: 量化"信号-执行延迟"值多少钱 ──
    # 现状是 T 日收盘出信号 -> T+1 尾盘成交, 中间隔了一整个交易日, 信号衰减一天。
    # 若改成 T 日 14:50 拿盘中快照出信号、当天 14:50-15:00 成交, 延迟就压到 ~10 分钟。
    #
    # 这里用 --exec-mode close (T日收盘价成交) 来近似那个方案。必须清楚它的性质:
    #   * 特征仍是用【收盘价】算的, 而成交也在收盘 -> 严格说是同时性未来函数,
    #     所以这个结果是【上界】, 不是可实现收益。
    #   * 要把它变成可实盘, 前提是 14:50 用当时快照重算特征。14:50 价≈收盘价,
    #     但尾盘 10 分钟的偏移、以及资金流/换手等当日未收口的特征, 都会打折。
    # 用途: 与 base3/base5 对比, 差值就是"消除一天延迟"最多能拿回多少。
    # 若差值很小, 就不必为 14:50 方案改造流水线(也不必买历史分钟数据)。
    "t0close3": {"desc": "3只 + T日尾盘成交(14:50快照方案的上界, 含同时性)",
                 "args": ["--tranche-n", "3", "--initial-capital", "50000",
                          "--regime-filter", "off", "--exec-mode", "close"]},
    "t0close5": {"desc": "5只 + T日尾盘成交(同上, 5只版)",
                 "args": ["--tranche-n", "5", "--initial-capital", "50000",
                          "--regime-filter", "off", "--exec-mode", "close"]},

    # ── 持有天数扫描 (2026-08-06) ────────────────────────────
    # 动机来自 diag_entry_path.py 的实测: 把 top3 选票在信号日之后的收益拆成
    # 逐日单腿, 发现信号衰减【远比 5 天慢】—— 毛收益一直累到 t+18 才见顶,
    # d22 才归零, 衰减周期约三周。而往返成本 0.46% 是每个周期固定付一次,
    # 所以拉长持有 = 摊薄成本, 只要多出来那几天的毛收益还大于 0 就赚。
    #
    # 上界估算 (买入固定 close_t+1, 忽略涨停/停牌/整手约束):
    #   持有天数   4      5(现状)   6      7      10     15
    #   窗口B年化  +11.1%  +16.3%  +18.9% +20.7% +17.2% +18.1%
    #   窗口A年化  -28.0%  -24.7%  -22.7% -20.9% -21.0% -19.7%
    # 关键是【两个窗口同号】: A 从 -24.7 改善到 -20.9, B 从 +16.3 改善到 +20.7。
    # 这是至今唯一一个 A/B 一致的改动方向 (vol-target/reversal-guard/14:50
    # 全都是一窗好一窗坏)。而且 6~16 天是个 17~21% 的平台, 不是尖峰, 不像拟合。
    #
    # 旁证: --roll-rank(卖出容忍) 之前被测出有效, 它本质上就是在变相拉长持有期。
    # 所以这里不叠加 roll-rank, 先单独量纯持有天数的效应, 避免归因混在一起。
    "hold6_3":  {"desc": "3只, 持有6天 (现状5天; 摊薄往返成本)",
                 "args": ["--tranche-n", "3", "--initial-capital", "50000",
                          "--regime-filter", "off", "--hold-days", "6"]},
    "hold7_3":  {"desc": "3只, 持有7天 (上界估算的最优点)",
                 "args": ["--tranche-n", "3", "--initial-capital", "50000",
                          "--regime-filter", "off", "--hold-days", "7"]},
    "hold8_3":  {"desc": "3只, 持有8天 (验证最优点附近是平台还是尖峰)",
                 "args": ["--tranche-n", "3", "--initial-capital", "50000",
                          "--regime-filter", "off", "--hold-days", "8"]},
    "hold10_3": {"desc": "3只, 持有10天 (换仓降到年24次)",
                 "args": ["--tranche-n", "3", "--initial-capital", "50000",
                          "--regime-filter", "off", "--hold-days", "10"]},
    "hold15_3": {"desc": "3只, 持有15天 (接近信号衰减完的三周)",
                 "args": ["--tranche-n", "3", "--initial-capital", "50000",
                          "--regime-filter", "off", "--hold-days", "15"]},
    "hold4_3":  {"desc": "3只, 持有4天 (对齐label的t+1->t+5; 预期变差, 做反向对照)",
                 "args": ["--tranche-n", "3", "--initial-capital", "50000",
                          "--regime-filter", "off", "--hold-days", "4"]},
    "hold7_5":  {"desc": "5只, 持有7天 (最优持有期的5只版)",
                 "args": ["--tranche-n", "5", "--initial-capital", "50000",
                          "--regime-filter", "off", "--hold-days", "7"]},
    # 若持有天数确实有效, 再看它与已验证的两个机制能不能叠加
    "hold7_rg": {"desc": "3只, 持有7天 + breadth择时 (两个已验证机制叠加)",
                 "args": ["--tranche-n", "3", "--initial-capital", "50000",
                          "--regime-filter", "breadth", "--regime-ma", "20",
                          "--regime-breadth", "0.40", "--regime-confirm", "2",
                          "--hold-days", "7"]},

    # 持有天数 x 现实滑点: 两个独立的成本改善叠加, 是当前最有希望的候选。
    # 拉长持有降低【付费次数】, 修正滑点假设降低【每次的单价】, 互不冲突。
    "hold7_rg_slip05": {"desc": "3只 + 持有7天 + breadth择时 + 滑点0.05%/边 (成本双改善)",
                        "args": ["--tranche-n", "3", "--initial-capital", "50000",
                                 "--regime-filter", "breadth", "--regime-ma", "20",
                                 "--regime-breadth", "0.40", "--regime-confirm", "2",
                                 "--hold-days", "7", "--slippage", "0.0005"]},
    "hold7_slip05": {"desc": "3只 + 持有7天 + 滑点0.05%/边 (不带择时, 便于归因)",
                     "args": ["--tranche-n", "3", "--initial-capital", "50000",
                              "--regime-filter", "off",
                              "--hold-days", "7", "--slippage", "0.0005"]},

    # ── 滑点敏感性: 0.2%/边 这个假设到底扛了多少锅 ──
    # 2026-08-14 用 Wind L2 逐笔快照实测标定完毕(1851 个股票日, 617 只 x 3 日),
    # 下面三个成分不再是估算, 是量出来的:
    #   价差   永远不利。窗口内 (卖1-买1)/2/中价 中位 2.7bp, 5%~95% = 0.4~15bp
    #   冲击   永远不利, 但 5万元 / 1.28亿日成交额 = 0.04‰, 基本为 0
    #   时点偏差 取决于在哪个窗口成交, 这是最关键的一项:
    #     14:57-15:00 收盘集合竞价 —— 单一清算价撮合, 该价【就是】收盘价。
    #       实测 集合竞价VWAP/收盘价-1 中位 0.000000 (95%分位 3e-7), 即偏差恰为 0。
    #       容量: 集合竞价成交额占全日中位 1.06%(5%分位 0.55%), 我们 1~3万/笔
    #       只占其零点几个百分点, 完全吃得下。涨跌停(单边报价)才成不了交, 占比中位 0%。
    #     14:50-14:57 连续竞价 —— VWAP/收盘价-1 中位 -5.2bp, 5%~95% = -28bp~+16bp,
    #       这一段才有"零均值但高方差"的时点偏差。
    # 结论: exec_mode=t1close 走收盘集合竞价时, 真实单边成本 = 0bp(不穿价) ~ 2.7bp(穿价),
    #   不是 20bp。与 35 笔实盘确认单的实测(中位 0bp)独立吻合。
    # 所以 slip0 不是"乐观上界"而是集合竞价下的真值; slip05 是留了近 20 倍余量的保守版。
    # 剂量响应(FBTR2, 20种子中位总收益): 0bp +49.5% | 5bp +36.5% | 10bp +24.6% | 20bp +6.4%
    # 复现: scripts/tick_close_exec.py -> data/processed/tick_exec/<date>.parquet
    "base3_slip0":  {"desc": "3只, 零滑点(只留佣金印花税) —— 走收盘集合竞价的实测真值",
                     "args": ["--tranche-n", "3", "--initial-capital", "50000",
                              "--regime-filter", "off", "--slippage", "0"]},
    "base3_slip05": {"desc": "3只, 滑点0.05%/边 (半价差的现实估计)",
                     "args": ["--tranche-n", "3", "--initial-capital", "50000",
                              "--regime-filter", "off", "--slippage", "0.0005"]},
    "base3_slip10": {"desc": "3只, 滑点0.10%/边 (保守但不离谱)",
                     "args": ["--tranche-n", "3", "--initial-capital", "50000",
                              "--regime-filter", "off", "--slippage", "0.001"]},
    "g3_rg_slip05": {"desc": "3只 + breadth择时, 滑点0.05%/边 (最优配置的现实成本版)",
                     "args": ["--tranche-n", "3", "--initial-capital", "50000",
                              "--regime-filter", "breadth", "--regime-ma", "20",
                              "--regime-breadth", "0.40", "--regime-confirm", "2",
                              "--slippage", "0.0005"]},

    # ── 第三轮: 只组合"在两个窗口上都单独有效"的机制 ──
    # 第二轮结果: regime择时 与 min-pred 都在 A(少亏) 和 B(多赚) 两侧同向改善;
    # vol-target 与 reversal-guard 在 B 上把 +11.8% 变成 -23.0% / -13.4%, 已淘汰。
    # 两个有效机制作用层次不同(一个大盘级空仓, 一个信号级信心), 值得叠加。
    "c3_rg_mp":  {"desc": "3只 + breadth择时 + 信号门槛0.006",
                  "args": ["--tranche-n", "3", "--initial-capital", "50000",
                           "--regime-filter", "breadth", "--regime-ma", "20",
                           "--regime-breadth", "0.40", "--regime-confirm", "2",
                           "--min-pred", "0.006"]},
    "c5_rg_mp":  {"desc": "5只 + breadth择时 + 信号门槛0.006",
                  "args": ["--tranche-n", "5", "--initial-capital", "50000",
                           "--regime-filter", "breadth", "--regime-ma", "20",
                           "--regime-breadth", "0.40", "--regime-confirm", "2",
                           "--min-pred", "0.006"]},
    "c3_rg_mp_rl": {"desc": "3只 + breadth择时 + 信号门槛 + 卖出容忍(省费用)",
                    "args": ["--tranche-n", "3", "--initial-capital", "50000",
                             "--regime-filter", "breadth", "--regime-ma", "20",
                             "--regime-breadth", "0.40", "--regime-confirm", "2",
                             "--min-pred", "0.006", "--roll-rank", "8"]},
    "c5_rg_mp_rl": {"desc": "5只 + breadth择时 + 信号门槛 + 卖出容忍",
                    "args": ["--tranche-n", "5", "--initial-capital", "50000",
                             "--regime-filter", "breadth", "--regime-ma", "20",
                             "--regime-breadth", "0.40", "--regime-confirm", "2",
                             "--min-pred", "0.006", "--roll-rank", "10"]},

    # ── 第四轮: 用 IC 自身择时, 而不是用大盘 ──
    # 依据: breadth 择时对"IC 变号"的识别力很差 —— 7 个坏半年里 4 个当时广度
    # 正常(2021H2 空仓仅 25%), 而空仓率最高的两段(2024H1 57%, 2026 56%)恰恰
    # IC 最好。它有效是靠降敞口压回撤, 不是靠识别 alpha 变号。
    # 而 IC 自身有弱持续性(过去/未来相关 0.10~0.15, 126天窗口下同号命中 65~68%),
    # 所以直接用"已闭合的历史 IC"择时在原理上更对口。
    # 注意 --ic-timing 的未来函数已于 2026-08-04 修复(原实现用当天 IC 决定当天
    # 空仓, 而当天 IC 要 5 天后才知道), 修复前的任何结论都不可用。
    # 样本量警告: 6 年只有约 11 个独立的 126 天区间, 这类规则极易过拟合,
    # 必须两窗口同时达标, 且即便达标证据也很薄。
    "t3_ic":     {"desc": "3只 + IC择时(连续3天已闭合IC为负则空仓)",
                  "args": ["--tranche-n", "3", "--initial-capital", "50000",
                           "--regime-filter", "off", "--ic-timing"]},
    "t3_ic_rg":  {"desc": "3只 + IC择时 + breadth择时(两者取或)",
                  "args": ["--tranche-n", "3", "--initial-capital", "50000",
                           "--regime-filter", "breadth", "--regime-ma", "20",
                           "--regime-breadth", "0.40", "--regime-confirm", "2",
                           "--ic-timing"]},
    "t5_ic":     {"desc": "5只 + IC择时",
                  "args": ["--tranche-n", "5", "--initial-capital", "50000",
                           "--regime-filter", "off", "--ic-timing"]},

    # ── 与线上 5 条线逐字对应的配置 ──────────────────────────
    # 为什么不能拿上面通用配置的结论直接上线: 本金这个变量翻转过结论
    # (5万下 3只最好, 100万下 20只最好), 而线上有 2万/5万/10万 三种本金、
    # 2/3/5 三种持仓数、两条线开着 lot-flex。必须按线各测一遍。
    # 每条线都有 _off / _rg 一对, 唯一差别是择时开关, 便于直接读出增量。
    # 变体要选对: steady2w 是全市场(--variant full), 其余四条是主板(--variant mb)。
    "live_steady5w_off": {"desc": "线上 steady5w: 5万/5只/主板, 不择时",
                          "args": ["--tranche-n", "5", "--initial-capital", "50000",
                                   "--regime-filter", "off"]},
    "live_steady5w_rg":  {"desc": "线上 steady5w + breadth择时",
                          "args": ["--tranche-n", "5", "--initial-capital", "50000",
                                   "--regime-filter", "breadth", "--regime-ma", "20",
                                   "--regime-breadth", "0.40", "--regime-confirm", "2"]},
    "live_aggr5w_off":   {"desc": "线上 aggr5w: 5万/3只/主板, 不择时",
                          "args": ["--tranche-n", "3", "--initial-capital", "50000",
                                   "--regime-filter", "off"]},
    "live_aggr5w_rg":    {"desc": "线上 aggr5w + breadth择时",
                          "args": ["--tranche-n", "3", "--initial-capital", "50000",
                                   "--regime-filter", "breadth", "--regime-ma", "20",
                                   "--regime-breadth", "0.40", "--regime-confirm", "2"]},
    "live_aggr10w_off":  {"desc": "线上 aggr10w: 10万/3只/主板, 不择时",
                          "args": ["--tranche-n", "3", "--initial-capital", "100000",
                                   "--regime-filter", "off"]},
    "live_aggr10w_rg":   {"desc": "线上 aggr10w + breadth择时",
                          "args": ["--tranche-n", "3", "--initial-capital", "100000",
                                   "--regime-filter", "breadth", "--regime-ma", "20",
                                   "--regime-breadth", "0.40", "--regime-confirm", "2"]},
    "live_aggr2w_off":   {"desc": "线上 aggr2w: 2万/2只/主板/lot-flex0.5, 不择时",
                          "args": ["--tranche-n", "2", "--initial-capital", "20000",
                                   "--lot-flex", "0.5", "--regime-filter", "off"]},
    "live_aggr2w_rg":    {"desc": "线上 aggr2w + breadth择时",
                          "args": ["--tranche-n", "2", "--initial-capital", "20000",
                                   "--lot-flex", "0.5",
                                   "--regime-filter", "breadth", "--regime-ma", "20",
                                   "--regime-breadth", "0.40", "--regime-confirm", "2"]},
    "live_steady2w_off": {"desc": "线上 steady2w: 2万/3只/全市场/lot-flex0.5, 不择时",
                          "args": ["--tranche-n", "3", "--initial-capital", "20000",
                                   "--lot-flex", "0.5", "--regime-filter", "off"]},
    "live_steady2w_rg":  {"desc": "线上 steady2w + breadth择时",
                          "args": ["--tranche-n", "3", "--initial-capital", "20000",
                                   "--lot-flex", "0.5",
                                   "--regime-filter", "breadth", "--regime-ma", "20",
                                   "--regime-breadth", "0.40", "--regime-confirm", "2"]},

    # ── 空槽位逐日补买 (--fill-daily) ────────────────────────
    # 动机来自实盘: 2026-08-04 aggr5w 目标 3 只只成交 2 只(挂单没买上),
    # 于是 17,161 元现金要空置到 08-10 的下一个换仓日。
    # 现有 is_rebal 逻辑有个不对称: `_n_matured == len(lots)` 在【空仓】时
    # (0==0) 成立所以天天重试建仓, 但【部分成交】(2/3)时不成立就得干等 5 天。
    # fill-daily 让非换仓日也补满空槽位。注意它会增加换手 -> 增加手续费,
    # 而手续费是这套系统最大的确定性损失, 所以必须验证净效果而非想当然。
    # 在当前线上配置(已开 breadth 择时)之上加, 才是真正的增量对比。
    "live_steady5w_rg_fd": {"desc": "线上 steady5w + 择时 + 空槽逐日补买",
                            "args": ["--tranche-n", "5", "--initial-capital", "50000",
                                     "--regime-filter", "breadth", "--regime-ma", "20",
                                     "--regime-breadth", "0.40", "--regime-confirm", "2",
                                     "--fill-daily"]},
    "live_aggr5w_rg_fd":   {"desc": "线上 aggr5w + 择时 + 空槽逐日补买",
                            "args": ["--tranche-n", "3", "--initial-capital", "50000",
                                     "--regime-filter", "breadth", "--regime-ma", "20",
                                     "--regime-breadth", "0.40", "--regime-confirm", "2",
                                     "--fill-daily"]},
    # ── 追高上限 (--max-chase): 把实盘的真实成交率建进回测 ──────
    # 起因: 用户实盘是盘尾挂限价单, 跳空高开的票打不到价位就买不上
    # (2026-08-04 aggr5w: 601138 高开后收盘 +7.2%, 一股没成交)。
    # 而回测默认假设 100% 按执行日收盘价成交, 所以回测天然乐观。
    # 两个方向都要量化: 漏掉这些票本身可能有利(实测那批买进去平均 -0.15%
    # vs 能买到的 +0.80%), 但空出来的槽位留现金又是拖累。净效果只能实测。
    "live_aggr5w_rg_ch1": {"desc": "线上 aggr5w + 择时 + 高开>1%放弃(留现金)",
                           "args": ["--tranche-n", "3", "--initial-capital", "50000",
                                    "--regime-filter", "breadth", "--regime-ma", "20",
                                    "--regime-breadth", "0.40", "--regime-confirm", "2",
                                    "--max-chase", "0.01"]},
    "live_aggr5w_rg_ch3": {"desc": "线上 aggr5w + 择时 + 高开>3%放弃(留现金)",
                           "args": ["--tranche-n", "3", "--initial-capital", "50000",
                                    "--regime-filter", "breadth", "--regime-ma", "20",
                                    "--regime-breadth", "0.40", "--regime-confirm", "2",
                                    "--max-chase", "0.03"]},
    "live_steady5w_rg_ch3": {"desc": "线上 steady5w + 择时 + 高开>3%放弃(留现金)",
                             "args": ["--tranche-n", "5", "--initial-capital", "50000",
                                      "--regime-filter", "breadth", "--regime-ma", "20",
                                      "--regime-breadth", "0.40", "--regime-confirm", "2",
                                      "--max-chase", "0.03"]},

    "live_steady2w_rg_fd": {"desc": "线上 steady2w + 择时 + 空槽逐日补买",
                            "args": ["--tranche-n", "3", "--initial-capital", "20000",
                                     "--lot-flex", "0.5",
                                     "--regime-filter", "breadth", "--regime-ma", "20",
                                     "--regime-breadth", "0.40", "--regime-confirm", "2",
                                     "--fill-daily"]},
}

# ── 换仓相位对照 (2026-08-06) ────────────────────────────────
# 为什么需要这个: wf_v35 的换仓日判定是 `i % HOLD_DAYS == REBAL_OFFSET`,
# 所以 periodic 模式下每个 (HOLD_DAYS, OFFSET) 组合对应【完全不同的一组
# 换仓日期】。之前的持有天数扫描只跑了 offset=0, 等于每个持有天数只抽了
# 一个相位样本 —— 结果 hold8 在窗口A 出现 +3.9%, 而它的两个邻居 hold7/hold10
# 是 -38.2%/-46.9%, 一天之差跳 42pp, 且 A/B 两窗口好坏不相关。这是相位噪声
# 的典型特征, 不是持有天数的效应。
#
# diag_entry_path.py 的上界估算是在【所有信号日】上取平均(等于平均了所有
# 相位), 所以它预测的是平滑平台; 回测只取一个相位, 两者不可比。
#
# 这里做两件事:
#   base3_ph1..ph4  —— 量【现有基线本身】的相位离散度。base3 是 5 个相位里
#                      的 1 个, 如果 5 个相位之间摆动几十个点, 那我们所有
#                      配置的绝对水位都带着这个未量化的运气成分。
#                      (同相位配置之间的配对比较不受影响, 仍然有效)
#   hold8_ph1..ph7  —— 判定 hold8 那个 +3.9% 是真效应还是抽到好相位。
#                      若 8 个相位的中位数回落到邻居水平, 就是噪声。
def _phase_cfgs():
    out = {}
    _rg = ["--regime-filter", "breadth", "--regime-ma", "20",
           "--regime-breadth", "0.40", "--regime-confirm", "2"]
    _c3 = ["--tranche-n", "3", "--initial-capital", "50000"]
    _c5 = ["--tranche-n", "5", "--initial-capital", "50000"]
    for hold, base_args, label in (
        (5, _c3 + ["--regime-filter", "off"], "base3"),
        (8, _c3 + ["--regime-filter", "off", "--hold-days", "8"], "hold8_3"),
        # 上面两组已跑完, 结论: hold8 的优势是相位运气, base3 的 ph0 是最好相位。
        #
        # 下面是 5 只版的当前最佳候选(口径已从 3 只改为 5 只)。3只版的 ph0 数字
        # (B窗夏普0.78/回撤最差-41.8%)是首次通过门槛的, 但 base3 实测五个相位在
        # B 窗摆动 21pp、亏损种子数从 0/20 到 6/20, 所以必须相位平均才算数。
        (7, _c5 + _rg + ["--hold-days", "7", "--slippage", "0.0005"],
         "h7_rg_slip05_n5"),
        (5, _c5 + _rg + ["--slippage", "0.0005"], "g5_rg_slip05"),
    ):
        for off in range(hold):
            # off=0 用本名。base3/hold8_3 已在 CONFIGS 里跑过, 不覆盖也不重跑
            name = label if off == 0 else f"{label}_ph{off}"
            if name in CONFIGS:
                continue
            out[name] = {
                "desc": f"{label} 换仓相位{off}/{hold} (量相位噪声)",
                "args": base_args + ["--rebal-offset", str(off)],
            }
    return out


CONFIGS.update(_phase_cfgs())

# ── 验收门槛 ────────────────────────────────────────────────
# 必须【两个窗口同时】满足。中位数看"典型情况", 最差种子看"运气不好时"——
# 后者才是"不是偶发现象"的真正检验: 一个配置如果只有中位数好看但最差种子
# 亏 30%, 那你实盘抽到那个种子的概率并不低。
#
# 门槛怎么定出来的 (全部实测于窗口B, 943天, 5万本金):
#     配置              夏普中位  夏普最差  回撤中位  回撤最差  收益中位
#     3只 (20种子)        0.45    -0.06    -45.1%   -63.8%    +46.6%
#     5只 (5种子)         0.33     0.02    -49.7%   -53.8%    +22.2%
#     10只 (5种子)        0.16     0.13    -39.8%   -48.6%     +1.0%
#     20只/100万 (天花板)  0.61     0.54    -37.5%   -39.1%    +70.0%
#
# 注意最后一行: 即便完全分散 + 本金放大 20 倍, 夏普中位数也只有 0.61。
# 这就是这套 alpha 的真实水位, 不存在"夏普 1.4"那种东西 —— 历史上记下的
# 1.39 是单次跑抽中右尾的结果。因此门槛必须贴着 0.5 附近定, 定在 0.8/1.0
# 等于要求一个不存在的东西, 只会逼着自己去过拟合。
#
# 重点放在【最差种子】而非中位数: 中位数从 0.45 抬到 0.50 是锦上添花,
# 把最差种子从 -0.06/-63.8% 收到 0.15/-50% 才是"能不能睡着觉"的区别。
#
# ── 2026-08-06 进展: 窗口B 已达标, 窗口A 确认无 alpha ──────────
# g5_rg_slip05 (5只 + breadth择时 + 滑点0.05%), 5 个换仓相位 x 20 种子:
#     窗口B 相位中位: 收益 +79.2%  夏普中位 0.77  夏普最差 0.58
#                     回撤中位 -28.9%  回撤最差 -37.7%  亏损种子 0/20
#     -> 五项门槛【全部通过】, 且 5 个相位每一个都单独通过, 不是挑相位挑出来的
#     窗口A 相位中位: 收益 -16.4%, 亏损种子 3~20/20
#
# 关于上面那句"不存在夏普 1.4": 仍然成立, 但天花板要上修 —— 之前的水位是在
# 滑点 0.2% 下测的, 而那个假设高估成本 4 倍。5只/0.05%滑点下 B 窗夏普中位 0.77,
# 最好相位 0.92。所以门槛 0.50 现在偏松, 但【先别动】: 窗口A 还是 -16%,
# 在两窗口都达标之前收紧门槛只会掩盖真正的问题。
#
# 窗口A 的失败已排除四类原因, 是该时段真实缺 alpha, 不是执行层或数据层缺失:
#     不是成本      —— 零滑点下仍 -20.5%
#     不是持有天数  —— 相位平均后 4~15 天无一致效应
#     不是相位运气  —— 12 个相位全为负
#     不是缺估值特征 —— daily_basic 全部字段在 A 窗 top5 为零或为负
THRESHOLDS = {
    "sharpe_median": 0.50,   # 略高于现状 0.45, 低于天花板 0.61
    "sharpe_worst": 0.15,    # 稳健性核心: 20 个种子全部为正
    "maxdd_median": -42.0,   # 不得比这更深
    "maxdd_worst": -50.0,    # 稳健性核心: 从 -63.8% 收进来
    "max_loss_seeds_pct": 10.0,  # 亏损种子占比上限 (现状 4/20 = 20%)
}


def log(msg):
    print(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


def out_path(tag, win):
    w = WINDOWS[win]
    return PROC / f"wf_daily_{tag}_ts{w['test_start']}_te{w['test_end']}_cap50000.json"


def out_path_cap(tag, win, cap):
    w = WINDOWS[win]
    return PROC / f"wf_daily_{tag}_ts{w['test_start']}_te{w['test_end']}_cap{int(cap)}.json"


def feat_tag(win, variant):
    return f"{VARIANTS[variant]['feat']}_{win}"


def cache_name(win, seed, variant):
    return f"{VARIANTS[variant]['cache']}_{win}_s{seed}.pkl"


def ev_tag(cname, win, seed, variant):
    return f"{VARIANTS[variant]['ev']}_{cname}_{win}_s{seed}"


def features_json(win, variant):
    """该窗口+变体 锁定特征的来源文件名 (features 阶段的产物)"""
    return out_path(feat_tag(win, variant), win).name


def win_args(win):
    w = WINDOWS[win]
    return ["--test-start", w["test_start"], "--test-end", w["test_end"]]


def run_parallel(jobs, jobs_cap, phase):
    """jobs: [(name, cmd_list)] —— 并发上限内跑完, 返回失败列表

    每个进程都要把 800MB 矩阵读进来再建特征(约 4 分钟), 所以并发数受内存约束,
    不是 CPU。默认 12 是在 503GB 机器上实测安全的值。
    """
    LOGDIR.mkdir(parents=True, exist_ok=True)
    pending, running, failed = list(jobs), [], []
    total = len(pending)
    while pending or running:
        while pending and len(running) < jobs_cap:
            name, cmd = pending.pop(0)
            lf = open(LOGDIR / f"{phase}_{name}.log", "w")
            p = subprocess.Popen(cmd, cwd=str(ROOT), stdout=lf, stderr=subprocess.STDOUT)
            running.append((name, p, lf))
            log(f"  启动 {name} ({total - len(pending) - len(running) + 1}/{total} 已排定)")
        time.sleep(5)
        for item in running[:]:
            name, p, lf = item
            if p.poll() is None:
                continue
            running.remove(item)
            lf.close()
            if p.returncode != 0:
                failed.append(name)
                log(f"  ✗ {name} rc={p.returncode} (见 {LOGDIR / f'{phase}_{name}.log'})")
            else:
                log(f"  ✓ {name}")
    return failed


# ── 阶段1: 每窗口现场筛特征 ─────────────────────────────────
def phase_features(args):
    v = VARIANTS[args.variant]
    jobs = []
    for win in args.windows:
        dst = out_path(feat_tag(win, args.variant), win)
        if dst.exists() and not args.force:
            log(f"窗口{win}[{args.variant}]: 特征已存在, 跳过 ({dst.name})")
            continue
        # 不传 --features-from = 现场筛选, 且脚本内部只用首个出信号日之前的
        # 数据做筛选(见 wf_v35 里 select_features(..., FIRST_PRED)), 无未来函数
        cmd = [PY, "-u", "scripts/wf_v35_breadth_alpha.py", *model_args(args.variant), *v["model"],
               *win_args(win), *EXEC_BASE,
               "--n-features", "80", "--tranche-n", "3",
               "--initial-capital", "50000", "--regime-filter", "off",
               "--lgb-seed", "42", "--tag", feat_tag(win, args.variant)]
        jobs.append((f"{args.variant}_win{win}", cmd))
    if not jobs:
        return
    log(f"阶段 features[{args.variant}]: {len(jobs)} 个运行 (每窗口一次现场筛选)")
    failed = run_parallel(jobs, args.jobs, "features")
    if failed:
        raise SystemExit(f"features 阶段失败: {failed}")
    for win in args.windows:
        sel = json.loads(out_path(feat_tag(win, args.variant), win).read_text())
        log(f"窗口{win}[{args.variant}] 锁定 {len(sel['selected_features'])} 个特征 "
            f"-> {features_json(win, args.variant)}")


# ── 阶段2: 建预测缓存 (贵, 一次性) ──────────────────────────
def phase_caches(args):
    v = VARIANTS[args.variant]
    jobs = []
    for win in args.windows:
        fj = out_path(feat_tag(win, args.variant), win)
        if not fj.exists():
            raise SystemExit(f"窗口{win}[{args.variant}] 还没筛特征, 先跑: "
                             f"eval_grid.py features --variant {args.variant}")
        for seed in args.seeds:
            if (PROC / cache_name(win, seed, args.variant)).exists() and not args.force:
                continue
            cmd = [PY, "-u", "scripts/wf_v35_breadth_alpha.py", *model_args(args.variant), *v["model"],
                   *win_args(win), *EXEC_BASE,
                   "--features-from", fj.name,
                   "--tranche-n", "3", "--initial-capital", "50000",
                   "--regime-filter", "off", "--lgb-seed", str(seed),
                   "--save-preds", cache_name(win, seed, args.variant),
                   "--tag", f"EVALCACHE_{args.variant}_{win}_s{seed}"]
            jobs.append((f"{args.variant}_{win}_s{seed}", cmd))
    if not jobs:
        log(f"阶段 caches[{args.variant}]: 全部已存在, 跳过")
        return
    log(f"阶段 caches[{args.variant}]: {len(jobs)} 个模型运行待跑 (这一步最耗时)")
    failed = run_parallel(jobs, args.jobs, "caches")
    if failed:
        raise SystemExit(f"caches 阶段失败: {failed}")


# ── 阶段3: 评估配置 (便宜, 复用缓存) ────────────────────────
def phase_eval(args):
    v = VARIANTS[args.variant]
    jobs = []
    for cname in args.configs:
        cfg = CONFIGS[cname]
        cap = _cap_of(cfg["args"])
        for win in args.windows:
            for seed in args.seeds:
                cache = PROC / cache_name(win, seed, args.variant)
                if not cache.exists():
                    raise SystemExit(f"缺预测缓存 {cache.name}, 先跑 caches 阶段")
                tag = ev_tag(cname, win, seed, args.variant)
                if out_path_cap(tag, win, cap).exists() and not args.force:
                    continue
                cmd = [PY, "-u", "scripts/wf_v35_breadth_alpha.py", *model_args(args.variant),
                       *win_args(win), *EXEC_BASE,
                       "--features-from", features_json(win, args.variant),
                       "--load-preds", cache.name, *cfg["args"], *v["exec"],
                       "--tag", tag]
                jobs.append((f"{args.variant}_{cname}_{win}_s{seed}", cmd))
    if not jobs:
        log(f"阶段 eval[{args.variant}]: 全部已存在, 跳过")
        return
    log(f"阶段 eval[{args.variant}]: {len(jobs)} 个执行层运行待跑")
    failed = run_parallel(jobs, args.jobs, "eval")
    if failed:
        raise SystemExit(f"eval 阶段失败: {failed}")


def _cap_of(arglist):
    if "--initial-capital" in arglist:
        return float(arglist[arglist.index("--initial-capital") + 1])
    return 100000.0


# ── 汇总 ────────────────────────────────────────────────────
def _expected_exec(cname):
    """该配置最终生效的执行层参数 (配置自己的 args 覆盖 EXEC_BASE)"""
    exp = dict(EXEC_EXPECT)
    a = CONFIGS[cname]["args"]
    if "--slippage" in a:
        exp["slippage"] = float(a[a.index("--slippage") + 1])
    if "--exec-mode" in a:
        exp["exec_mode"] = a[a.index("--exec-mode") + 1]
    exp["rebal_offset"] = (int(a[a.index("--rebal-offset") + 1])
                           if "--rebal-offset" in a else 0)
    return exp


def collect(cname, win, seeds, variant):
    """读回结果, 顺带校验产物里记录的执行层参数与当前配置一致。

    ev_tag 里不含执行层参数, 所以改了 EXEC_BASE(比如滑点 0.002 -> 0.0005) 而
    没重跑时, 旧参数的产物会因文件名重合被当成新结果 —— 那会得出一个完全错误
    但看起来毫无异常的结论。这里把参数不符的产物【排除出判定】并大声警告,
    而不是静默混用, 也不硬抛错(否则 0.002 时代的上千个历史产物全都读不出来)。
    """
    cap = _cap_of(CONFIGS[cname]["args"])
    exp = _expected_exec(cname)
    rows, stale = [], []
    for seed in seeds:
        p = out_path_cap(ev_tag(cname, win, seed, variant), win, cap)
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        bad = {}
        for k, v in exp.items():
            got = d.get(k)
            # rebal_offset 是 2026-08-06 才加的字段, 老产物里没有 -> 视为 0
            if got is None and k == "rebal_offset":
                got = 0
            if isinstance(v, float):
                if got is None or abs(float(got) - v) > 1e-12:
                    bad[k] = (got, v)
            elif got != v:
                bad[k] = (got, v)
        if bad:
            stale.append(bad)
            continue
        rows.append(d["summary"])
    if stale:
        det = "; ".join(f"{k}: 产物={a!r} 期望={b!r}"
                        for k, (a, b) in stale[0].items())
        STALE_SEEN.append(f"{cname}/{win}: {len(stale)}个 ({det})")
    return rows


def summarize(rows):
    if not rows:
        return None
    def arr(k):
        return np.array([r[k] for r in rows], dtype=float)
    ret, shp, dd = arr("total_return_pct"), arr("sharpe"), arr("max_dd_pct")
    return {
        "n_seeds": len(rows),
        "ret_median": float(np.median(ret)), "ret_worst": float(ret.min()),
        "ret_best": float(ret.max()),
        "ret_p25": float(np.percentile(ret, 25)), "ret_p75": float(np.percentile(ret, 75)),
        "n_loss": int((ret < 0).sum()),
        "sharpe_median": float(np.median(shp)), "sharpe_worst": float(shp.min()),
        "maxdd_median": float(np.median(dd)), "maxdd_worst": float(dd.min()),
        "fee_median": float(np.median(arr("total_cost_pct"))),
        "ic_median": float(np.median(arr("ic_mean"))),
        "bench": float(np.median(arr("benchmark_total_pct"))),
        "n_below_bench": int((ret < np.median(arr("benchmark_total_pct"))).sum()),
    }


def verdict(sa, sb):
    """两窗口同时达标才算过"""
    if not sa or not sb:
        return "数据不全", []
    fails = []
    for win, s in (("A", sa), ("B", sb)):
        if s["sharpe_median"] < THRESHOLDS["sharpe_median"]:
            fails.append(f"{win}:夏普中位{s['sharpe_median']:.2f}<{THRESHOLDS['sharpe_median']}")
        if s["sharpe_worst"] < THRESHOLDS["sharpe_worst"]:
            fails.append(f"{win}:夏普最差{s['sharpe_worst']:.2f}<{THRESHOLDS['sharpe_worst']}")
        if s["maxdd_median"] < THRESHOLDS["maxdd_median"]:
            fails.append(f"{win}:回撤中位{s['maxdd_median']:.1f}%<{THRESHOLDS['maxdd_median']}%")
        if s["maxdd_worst"] < THRESHOLDS["maxdd_worst"]:
            fails.append(f"{win}:回撤最差{s['maxdd_worst']:.1f}%<{THRESHOLDS['maxdd_worst']}%")
        loss_pct = s["n_loss"] / s["n_seeds"] * 100
        if loss_pct > THRESHOLDS["max_loss_seeds_pct"]:
            fails.append(f"{win}:亏损种子{loss_pct:.0f}%>{THRESHOLDS['max_loss_seeds_pct']:.0f}%")
    return ("通过" if not fails else "不通过"), fails


def phase_report(args):
    print()
    print("=" * 108)
    print("多种子 x 双窗口评估报告")
    print(f"  股票池变体: {args.variant} —— {VARIANTS[args.variant]['desc']}")
    print(f"  窗口A {WINDOWS['A']['test_start']} ~ {WINDOWS['A']['test_end']}  ({WINDOWS['A']['desc']})")
    print(f"  窗口B {WINDOWS['B']['test_start']} ~ {WINDOWS['B']['test_end']}  ({WINDOWS['B']['desc']})")
    print(f"  门槛(两窗口同时): 夏普中位>={THRESHOLDS['sharpe_median']} 最差>={THRESHOLDS['sharpe_worst']}"
          f" | 回撤中位>={THRESHOLDS['maxdd_median']}% 最差>={THRESHOLDS['maxdd_worst']}%"
          f" | 亏损种子<={THRESHOLDS['max_loss_seeds_pct']:.0f}%")
    print("=" * 108)
    out = {}
    for cname in args.configs:
        sa = summarize(collect(cname, "A", args.seeds, args.variant))
        sb = summarize(collect(cname, "B", args.seeds, args.variant))
        v, fails = verdict(sa, sb)
        out[cname] = {"A": sa, "B": sb, "verdict": v, "fails": fails}
        print()
        print(f"### {cname} —— {CONFIGS[cname]['desc']}")
        print("%-4s %6s %9s %9s %9s %8s %8s %9s %9s %7s %7s" % (
            "窗口", "种子", "收益中位", "收益最差", "收益最好", "夏普中位",
            "夏普最差", "回撤中位", "回撤最差", "亏损数", "费用%"))
        for win, s in (("A", sa), ("B", sb)):
            if not s:
                print(f"{win:<4} (无数据)")
                continue
            print("%-4s %6d %9.1f %9.1f %9.1f %8.2f %8.2f %9.1f %9.1f %7s %7.1f" % (
                win, s["n_seeds"], s["ret_median"], s["ret_worst"], s["ret_best"],
                s["sharpe_median"], s["sharpe_worst"], s["maxdd_median"],
                s["maxdd_worst"], f"{s['n_loss']}/{s['n_seeds']}", s["fee_median"]))
        print(f"  结论: {v}" + (f"  |  未达标项: {'; '.join(fails)}" if fails else ""))
    _print_cross_table(out, args)
    _print_phase_table(out)
    suffix = "" if args.variant == "full" else f"_{args.variant}"
    dst = PROC / f"eval_grid_report{suffix}.json"
    dst.write_text(json.dumps({"thresholds": THRESHOLDS, "windows": WINDOWS,
                               "variant": args.variant, "seeds": args.seeds,
                               "results": out},
                              ensure_ascii=False, indent=2))
    if STALE_SEEN:
        print()
        print("!" * 108)
        print("以下产物的执行层参数与当前配置不符, 已【排除出判定】(不是缺数据):")
        for s in STALE_SEEN:
            print(f"  {s}")
        print("要用这些配置就必须按当前参数重跑; 旧产物文件名会重合, 需先删除。")
        print("!" * 108)
    print()
    print(f"报告已写入 {dst}")


def _print_cross_table(out, args):
    """横向对比表, 按【两窗口里更差的那个夏普最差值】排序。

    为什么按最差种子排而不是按中位数或收益: 我们要挑的是"运气不好时也能接受"
    的配置。按收益排会把 3 只持仓那种右尾配置排到前面, 那正是要避免的陷阱。
    """
    if len(out) < 2:
        return
    print()
    print("=" * 108)
    print("横向对比 (按两窗口中较差的「夏普最差种子」排序 —— 挑的是下限, 不是上限)")
    print("=" * 108)
    rows = []
    for cname, r in out.items():
        sa, sb = r["A"], r["B"]
        if not sa or not sb:
            continue
        rows.append((
            cname,
            min(sa["sharpe_worst"], sb["sharpe_worst"]),
            min(sa["sharpe_median"], sb["sharpe_median"]),
            min(sa["maxdd_worst"], sb["maxdd_worst"]),
            min(sa["ret_median"], sb["ret_median"]),
            max(sa["n_loss"] / sa["n_seeds"], sb["n_loss"] / sb["n_seeds"]) * 100,
            max(sa["fee_median"], sb["fee_median"]),
            r["verdict"],
        ))
    rows.sort(key=lambda x: -x[1])
    print("%-12s %10s %10s %10s %10s %9s %8s %8s" % (
        "配置", "夏普最差", "夏普中位", "回撤最差", "收益中位", "亏损种子%", "费用%", "结论"))
    print("%-12s %10s %10s %10s %10s %9s %8s %8s" % (
        "", "(取较差窗口)", "(取较差)", "(取较差)", "(取较差)", "(取较高)", "", ""))
    for r in rows:
        print("%-12s %10.2f %10.2f %10.1f %10.1f %9.0f %8.1f %8s" % r)
    print()
    print("门槛: 夏普最差>=%.2f 夏普中位>=%.2f 回撤最差>=%.1f%% 亏损种子<=%.0f%%" % (
        THRESHOLDS["sharpe_worst"], THRESHOLDS["sharpe_median"],
        THRESHOLDS["maxdd_worst"], THRESHOLDS["max_loss_seeds_pct"]))


PH_RE = re.compile(r"^(.*)_ph(\d+)$")


def _print_phase_table(out):
    """把同一配置的多个换仓相位聚合成【相位中位数】再判定。

    为什么必须有这张表: wf_v35 的换仓日判定是 i %% HOLD_DAYS == REBAL_OFFSET,
    每个 offset 对应完全不同的一组换仓日期。实测 base3 的 5 个相位在窗口B
    收益从 +13.2% 到 +34.5%(摆动 21pp), 亏损种子数从 0/20 到 6/20 —— 而我们
    长期只跑相位 0, 且相位 0 恰好是 5 个里最好的那个。hold8 在窗口A 那个
    +3.9% 也是同样的假象(8 个相位中位数回落到 -32%)。

    所以: 只跑了单相位的配置, 其数字一律视为【上界估计】而非结论。
    """
    groups = {}
    for cname, r in out.items():
        m = PH_RE.match(cname)
        base = m.group(1) if m else cname
        if base in out or not m:      # 有 ph0 本名的才成组, 否则自成一组
            groups.setdefault(base, []).append((cname, r))
    multi = {k: v for k, v in groups.items() if len(v) > 1}
    if not multi:
        single = [k for k in groups if k in out]
        if single:
            print()
            print("[相位提醒] 以下配置只有单个换仓相位的数据, 其数字应视为上界"
                  f"估计而非结论: {', '.join(sorted(single))}")
            print("           相位噪声实测可达 21pp(窗口B收益), 要下结论请补跑 "
                  "--rebal-offset 1..HOLD_DAYS-1")
        return
    print()
    print("=" * 108)
    print("相位中位数 (对同一配置的全部换仓相位取中位 —— 这才是可采信的数字)")
    print("=" * 108)
    print("%-20s %4s %3s %9s %9s %9s %9s %9s %9s" % (
        "配置", "相位", "窗", "收益中位", "夏普中位", "夏普最差",
        "回撤中位", "回撤最差", "亏损%"))
    agg = {}
    for base, items in sorted(multi.items()):
        synth = {}
        for win in ("A", "B"):
            ss = [r[win] for _, r in items if r[win]]
            if not ss:
                continue
            def med(k):
                return float(np.median([s[k] for s in ss]))
            synth[win] = {
                "n_seeds": ss[0]["n_seeds"],
                "ret_median": med("ret_median"), "ret_worst": med("ret_worst"),
                "ret_best": med("ret_best"),
                "sharpe_median": med("sharpe_median"),
                "sharpe_worst": med("sharpe_worst"),
                "maxdd_median": med("maxdd_median"),
                "maxdd_worst": med("maxdd_worst"),
                "fee_median": med("fee_median"),
                "n_loss": float(np.median([s["n_loss"] for s in ss])),
                "ic_median": med("ic_median"), "bench": med("bench"),
                "ret_p25": med("ret_p25"), "ret_p75": med("ret_p75"),
                "n_below_bench": ss[0]["n_below_bench"],
            }
            s = synth[win]
            print("%-20s %4d %3s %9.1f %9.2f %9.2f %9.1f %9.1f %9.0f" % (
                base, len(ss), win, s["ret_median"], s["sharpe_median"],
                s["sharpe_worst"], s["maxdd_median"], s["maxdd_worst"],
                s["n_loss"] / s["n_seeds"] * 100))
        if len(synth) == 2:
            v, fails = verdict(synth["A"], synth["B"])
            agg[base] = (v, fails)
            print(f"{'':<20} 相位中位判定: {v}"
                  + (f"  |  未达标: {'; '.join(fails)}" if fails else ""))
    print()
    print("注: 亏损% 是各相位「亏损种子占比」的中位数。单个相位全 0/20 不算"
          "稳健, 全部相位都 0/20 才算。")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=["features", "caches", "eval", "report", "all"])
    ap.add_argument("--variant", default="full", choices=list(VARIANTS),
                    help="股票池变体: full=全市场 | mb=仅主板(线上4条线用的口径)。"
                         "两者的特征/缓存/产物完全分开, 不会互相覆盖")
    ap.add_argument("--windows", default="A,B", help="逗号分隔, 默认 A,B")
    ap.add_argument("--seeds", default="20",
                    help="种子个数(取内置列表前 N 个), 或逗号分隔的具体种子")
    ap.add_argument("--configs", default=",".join(CONFIGS),
                    help=f"逗号分隔, 可选: {','.join(CONFIGS)}")
    ap.add_argument("--jobs", type=int, default=12, help="并发进程上限(受内存约束)")
    ap.add_argument("--force", action="store_true", help="已有产物也重跑")
    args = ap.parse_args()

    args.windows = [w.strip() for w in args.windows.split(",") if w.strip()]
    for w in args.windows:
        if w not in WINDOWS:
            raise SystemExit(f"未知窗口 {w}, 可选 {list(WINDOWS)}")
    if args.seeds.isdigit():
        n = int(args.seeds)
        if n > len(DEFAULT_SEEDS):
            raise SystemExit(f"内置种子只有 {len(DEFAULT_SEEDS)} 个")
        args.seeds = DEFAULT_SEEDS[:n]
    else:
        args.seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    args.configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    for c in args.configs:
        if c not in CONFIGS:
            raise SystemExit(f"未知配置 {c}, 可选 {list(CONFIGS)}")

    log(f"窗口={args.windows} 种子={len(args.seeds)}个 配置={args.configs} 并发={args.jobs}")
    if args.phase in ("features", "all"):
        phase_features(args)
    if args.phase in ("caches", "all"):
        phase_caches(args)
    if args.phase in ("eval", "all"):
        phase_eval(args)
    if args.phase in ("report", "all"):
        phase_report(args)


if __name__ == "__main__":
    try:  # htop 低调化 (见 scripts/proctitle.py)
        from proctitle import lowkey
        lowkey("mltask/grid")
    except Exception:
        pass
    main()
