# 生产线回测到最新日 (活文档, 每次 te 变了重跑)

> 纪律(2026-09-05, 用户: "alpha 会 decay, 回测每次都要更新到最新日期"): 这张表的 `--test-end` 永远是生产矩阵的最新日. 每条线用 `live_config.signal_args` 的**逐字配置**(唯一去掉的是 `--seed-ensemble`, 换成 20 个单种子看分布), 生产矩阵 `training_data_pit_v24_tick1.parquet`, 测试起点 2023-09-19(矩阵 2022-09-01 起, 留一年训练). 报**中位**, 不报单次.
>
> **性质**: min-pred / roll-rank / fill-daily / ind-cap / T1A·T1B 这些执行层与特征选择全是在 2022-09~2026-07 上挑出来的, 所以 B' 窗对现行配置是**样本内**; 每次刷新时**只有上次 te 之后的那一段是样本外**, 看"最近 3 月 / 6 月"分段时记住这点. 它回答"现行配置到今天为止长什么样", 不回答"能不能持续".

## 跑法

```bash
# 040 (128 核, 12 并行, 80 训练 + 40 重放 ≈ 9.5 小时; 脚本存档 /tmp/pl_040.sh, 摘要 /tmp/pl_summary.py)
# 4 个模型类各训 20 种子(--save-preds), 同类的另一条线 --load-preds 重放执行层:
#   PLA 主板+T1A : aggr5w(训) -> aggr10w(放)      PLB 主板+T1B : steady5w(训) -> fyf100w(放)
#   PLC 主板+基线80: aggr2w(训)                    PLF 全市场+基线80: steady2w(训)
common="--train-file training_data_pit_v24_tick1.parquet --pit-universe universe_pit.parquet --label 5d --objective l2 \
  --hold-days 5 --portfolio-mode periodic --exec-mode t1close --slippage 0.002 \
  --regime-filter breadth --regime-ma 20 --regime-breadth 0.40 --regime-confirm 2 \
  --min-pred 0.002 --fill-daily --roll-rank 8 --test-start 2023-09-19 --test-end <矩阵最新日>"
# 各线 (与 python scripts/live_config.py --profile X --args 逐字对应):
#   aggr5w  : --tranche-n 3 --initial-capital 50000   --skip-boards 30,688 --ind-cap 2 --features-from features_V24PUT_T1A.json
#   aggr10w : --tranche-n 3 --initial-capital 100000  --skip-boards 30,688 --ind-cap 2 --features-from features_V24PUT_T1A.json
#   steady5w: --tranche-n 5 --initial-capital 50000   --skip-boards 30,688 --features-from features_V24PUT_T1B.json
#   fyf100w : --tranche-n 8 --initial-capital 1000000 --skip-boards 30,688 --features-from features_V24PUT_T1B.json
#   aggr2w  : --tranche-n 2 --initial-capital 20000 --lot-flex 0.5 --skip-boards 30,688 --features-from wf_daily_V24PUT_s42_ts2022-09-01_te2026-07-27_cap100000.json
#   steady2w: --tranche-n 3 --initial-capital 20000 --lot-flex 0.5 --ind-cap 2 --features-from wf_daily_V24PUT_s42_ts2022-09-01_te2026-07-27_cap100000.json
# 种子: 42 7 123 2024 31337 1 2 3 5 11 17 23 55 77 99 202 314 512 888 1234
```

产物: `data/processed/wf_daily_PL_<线>_s<种子>_ts2023-09-19_te<te>_cap<本金>.json`, 预测缓存 `preds_PL{A,B,C,F}_s<种子>.json`(040).

## 2026-09-04 (跑于 09-05 12:53 → 22:32, 040)

窗 2023-09-19 → 2026-09-04 (约 3 年). 基准 = 等权买入持有 `fwd_1d_ret` 日均: 训练线(aggr5w/steady5w/aggr2w)带 `--skip-boards` 时 df 级剪掉创/科板, 基准是**主板等权 +22.2%**; 重放线(aggr10w/fyf100w 走 `--load-preds`, wf 在该模式下不做 df 级板块过滤)与全市场线的基准是**全市场等权 +26.4%** —— 只影响基准/IR 这两列, 不影响策略路径(候选池来自主板预测缓存 + 执行层再跳一次板块). "空仓" = 择时判弱势的交易日占比.

| 线 | 配置 | 总收益 中位 (Q1/Q3) | 最差 | 正 | 赢基准 | 年化 | 夏普 | 回撤 | IR | 费用 | 空仓 | 最近 6 月 策略/基准 | 最近 3 月 策略/基准 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **aggr5w** | 5万/3只/主板/T1A/ind-cap 2 | **+110.8%** (+87.6 / +131.3) | +57.6 | 20/20 | 20/20 (+22.2%) | +30.0% | 1.16 | −17.9% | +0.89 | 9.9% | 45% | +30.1 / −7.5 (20/20 赢) | +10.2 / +0.0 (20/20 赢) |
| **aggr10w** | 10万/3只/主板/T1A/ind-cap 2 | **+96.3%** (+83.0 / +117.8) | +64.4 | 20/20 | 20/20 (+26.4%) | +26.8% | 1.06 | −18.1% | +0.71 | 9.7% | 45% | +29.2 / −6.8 (20/20 赢) | +10.6 / −2.2 (20/20 赢) |
| **steady5w** | 5万/5只/主板/T1B | **+103.4%** (+91.3 / +114.4) | +73.3 | 20/20 | 20/20 (+22.2%) | +28.4% | 1.27 | −16.0% | +0.92 | 8.8% | 45% | +27.9 / −7.5 (20/20 赢) | +8.8 / +0.0 (20/20 赢) |
| **fyf100w** | 100万/8只/主板/T1B | **+87.2%** (+80.1 / +104.3) | +71.1 | 20/20 | 20/20 (+26.4%) | +24.7% | 1.18 | −15.1% | +0.72 | 8.4% | 45% | +29.6 / −6.8 (20/20 赢) | +7.3 / −2.2 (20/20 赢) |
| **aggr2w** | 2万/2只/主板/基线80/lot-flex 0.5 | **+100.0%** (+48.5 / +155.8) | +11.2 | 20/20 | 18/20 (+22.2%) | +27.6% | 1.01 | −24.2% | +0.75 | 8.8% | 45% | +37.4 / −7.5 (20/20 赢) | +11.7 / +0.0 (20/20 赢) |
| **steady2w** | 2万/3只/全市场/基线80/lot-flex 0.5/ind-cap 2 | **+55.0%** (+50.2 / +68.4) | +15.4 | 20/20 | 19/20 (+26.4%) | +16.6% | 0.73 | −24.9% | +0.37 | 10.2% | 45% | +13.1 / −6.8 (20/20 赢) | +8.9 / −2.2 (20/20 赢) |

读法:
- 四条真金主板线(aggr5w / aggr10w / steady5w / fyf100w)形状一致: 中位 +87 ~ +111%, 20/20 正, 20/20 赢基准, **最差种子 +57.6% 起**, 回撤中位 −15 ~ −18%. 持仓越多分布越窄(steady5w Q1/Q3 差 23pp, aggr5w 差 44pp; fyf100w 8 只最窄且回撤最小), 中位略低 —— 这是分散化该有的样子.
- 对照 bare 生产等价臂(X19E22L, 同窗同矿, V24PUT 80 列, 不带 min-pred/fill-daily/roll-rank/ind-cap): 中位 +56.8%, 最差 +15.9, 回撤 −19.8. 全套执行层 + T1A 把中位抬了 ~54pp、把最差抬了 ~42pp, 回撤没变差. 这些件各自都过过 20 种子门, 叠起来的量级与各自的增量之和同阶(MP1c +16 / T1A +17~29 / roll-rank·fill-daily 各若干).
- 2 万两条线: aggr2w 中位与 5 万线相当但分布宽一倍(Q1 +48.5 / Q3 +155.8, 最差 +11.2), 是 2 只持仓的集中度; steady2w 全市场池最弱(+55.0, IR 0.37), 与 07-29 "2 万本金买不起赚钱的票"的结论一致.
- **最近 6 月(2026-03-06 → 09-04)**: 全线 +13 ~ +37% vs 基准 −7%, 120/120 个(线×种子)赢基准. **最近 3 月**: +7 ~ +12% vs 基准 0 / −2, 同样 120/120. 其中 07-28 → 09-04 五周是上次 te 之后的真样本外.

## 历史

| te | 跑于 | aggr5w 中位 | steady5w 中位 | aggr10w 中位 | fyf100w 中位 | aggr2w 中位 | steady2w 中位 | 备注 |
|---|---|---|---|---|---|---|---|---|
| 2026-09-04 | 09-05 | +110.8% | +103.4% | +96.3% | +87.2% | +100.0% | +55.0% | 首版; 逐字生产配置(含 T1A/T1B 分点) |
