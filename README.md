# 六维量化ML选股系统

> 多因子选股 + 供应链事件驱动 + XGBoost ML 评分  
> 覆盖 A 股全市场，因子维度 301+，每日自动打分排名

## 目录结构

```
quant-strategy/
├── pipeline/                   # 数据采集与特征工程
│   ├── config.py               # 统一配置（路径/权重/日志）
│   ├── logger.py                # 统一日志（RotatingFile + 控制台）
│   ├── ifind_funda_pipeline.py # iFinD 基本面采集（MCP API）
│   ├── batch_fund_flow_ifind.py # iFinD 资金流并行拉取
│   ├── batch_update_fundamentals.py # akshare/thsdk 基本面更新
│   ├── collect_fund_flow.py    # 资金流采集器
│   ├── data_engine.py          # 六维数据引擎
│   ├── feature_engine.py       # 特征工程（301 因子）
│   ├── supplier_self_event.py  # 供应商自身事件检测
│   ├── chain_map_merger.py     # 供应链数据源融合
│   ├── chain_leader_monitor.py # 链主事件监控
│   ├── chain_leader_scorer.py  # 链主事件评分
│   ├── xgb_scorer.py           # XGBoost 评分模型
│   ├── westock_data.py         # westock 数据接口
│   ├── gen_fund_b64.py         # Base64 编码生成
│   ├── lib/ths_utils.py        # thsdk 工具函数
│   ├── test_batch_tiny.py      # 小批量采集测试
│   ├── test_fund_flow_single.py # 单只资金流测试
│   └── test_ths_conn.py        # thsdk 连接测试
├── strategies/                 # 10 个通用打分策略
│   ├── r1_fund_tech.py         # R1: 资金面+技术面
│   ├── r2_event_fund_tech.py   # R2: 事件+资金+技术
│   ├── r3_generalized.py       # R3: 泛化因子
│   ├── r4_adaptive.py          # R4: 自适应权重
│   ├── r5_pct_ratio.py         # R5: 百分比排名
│   ├── r6_pct50.py             # R6: 中位数切割
│   ├── r7_causal.py            # R7: 因果事件
│   ├── r8_news_events.py       # R8: 新闻事件
│   ├── r9_trend_filter.py      # R9: 趋势过滤
│   ├── r10_causal_events.py    # R10: 因果事件增强
│   └── 6factor_ref/            # 6因子参考实现
├── tests/                      # 90 个 pytest 测试用例
│   ├── conftest.py
│   ├── test_config.py          # 配置 17 项
│   ├── test_ifind_pipeline.py  # iFinD 管线 9 项
│   ├── test_logger.py          # 日志 7 项
│   ├── test_pipeline.py        # 数据管线 10 项
│   ├── test_strategies.py      # 策略语法 49 项
│   └── test_supplier_self_event.py # 事件检测 8 项
├── backtest/                   # 回测框架
│   ├── chain_event_backtest.py # 供应链事件回测
│   ├── master_backtest.py      # 综合回测
│   ├── corning_backtest.py     # 康宁专项回测
│   └── results/                # 回测结果
├── engine/                     # 执行引擎
│   ├── smlogin/                # SuperMind 认证 + WS 执行
│   ├── supermind_executor.py   # WS 执行引擎
│   ├── jupyter_ws_exec.py      # WS 执行（简化版）
│   └── jupyter_rest_test.py    # REST API 测试
├── neo4j/                      # Neo4j 供应链图谱（本地运行）
└── data/                       # 数据文件
    ├── universe/               # 关注圈、供应链映射、概念主题
    └── raw/                    # 原始数据（K线/资金流/基本面/公告）
```

## 系统架构

```
外部数据源 → 采集管线 → 特征工程 → 策略打分 → 综合评分 → 排序选股
                  ↑                        ↑
             iFinD MCP API           XGBoost ML 模型
             akshare/thsdk           16 个策略加权
             westock data
```

## 策略演进

| 阶段 | 策略体系 | 因子维度 | ML 集成 | 状态 |
|:----:|----------|:--------:|:-------:|:----:|
| 基线 | R1~R2 资金事件驱动 | 6 因子 | ❌ | ✅ 稳定 |
| 扩展 | R3~R10 全因子 | 73 特征 | ✅ XGBoost v1 | ✅ 稳定 |
| v2.2 | 三池泛化 + score-weighted | 301 特征 | ✅ XGBoost v2 | ✅ 当前 |
| v3 | 供应链事件评分（5 维） | 301 + 事件 | ✅ XGBoost + 图 | 🔄 开发中 |

### 10 策略权重体系

各策略独立打分（0-100 归一化），最终评分 = 加权合成：

| 策略 | 权重 | 侧重 |
|:----|:----:|------|
| R1 资金技术 | 8% | 资金流 + 技术指标 |
| R2 事件资金技术 | 12% | 事件驱动 + 资金验证 |
| R3 泛化因子 | 12% | 宽基多因子 |
| R4 自适应权重 | 10% | 动态权重调整 |
| R5 百分比排名 | 12% | 相对强度排名 |
| R6 中位数切割 | 10% | 二分位分类 |
| R7 因果事件 | 12% | 因果推断信号 |
| R8 新闻事件 | 8% | 新闻热度指标 |
| R9 趋势过滤 | 8% | 趋势方向过滤 |
| R10 因果事件增强 | 8% | 因果信号增强 |

## 核心依赖

- **Python 3.11** — 主运行环境（Hermes venv）
- **iFinD MCP API** — 基本面数据
- **thsdk 正式账号** — 历史资金流 + wencai NLP
- **akshare** — 资金流备选源
- **pandas / numpy / polars** — 数据处理
- **xgboost / scikit-learn** — ML 评分模型
- **Neo4j 2025.12** — 供应链图谱存储
- **pytest** — 测试框架（90 用例）

## 数据覆盖

| 数据类型 | 起始日期 | 覆盖范围 |
|---------|:--------:|:--------:|
| K 线日线 | 2021-06-29 | 5,533 只 |
| 资金流 | 2020-01-02 | 398 只 |
| 基本面 | 2019-12-31 | 117 只有效 |
| 公告 | 2000-03-28 | 200 只 |
| **宏观数据** | **2005~2026** | **70 项指标** |
| ㅤ利率/货币政策 | 2005+ | 16 项（中美欧日英等央行利率、LPR、SHIBOR、存准率） |
| ㅤ宏观经济指标 | 2005+ | 24 项（GDP、CPI、PPI、PMI、工业增加值、消费零售等） |
| ㅤ货币金融 | 2005+ | 5 项（M0/M1/M2、社融、信贷、外汇储备） |
| ㅤ大宗商品/贵金属 | 2010+ | 8 项（黄金、白银、原油、OPEC、大宗商品价格指数） |
| ㅤ航运指数 | 2010+ | 3 项（BDI、BCI、BPI 波罗的海指数） |
| ㅤ半导体 | 2000+ | 1 项（全球半导体 SOX 指数） |
| ㅤ就业/消费 | 2010+ | 3 项（非农、失业率、零售销售） |
| ㅤ债市/利差 | 2015+ | 1 项（中美利差） |
| ㅤ景气指数 | 2010+ | 1 项（中国景气指数） |
| 训练数据 | 2010-01-04 | 全量 |

## 技能

- `a-stock-ml` — 六维量化 ML 选股系统核心技能

## 开发基础设施

### pre-commit 质量门禁

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

自动运行：ruff 检查/格式化、mypy 类型检查、尾随空白清理、YAML/JSON 校验、大文件检测。

### CI 流水线

GitHub Actions 配置见 `.github/workflows/ci.yml`，每次 push 自动执行：
- ruff lint + format 检查
- mypy 类型检查（pipeline/ engine/）
- pytest 单元测试

### 测试

```bash
# 运行全量测试
pytest tests/ -v

# 测试覆盖率报告
pytest tests/ --cov=pipeline --cov-report=term-missing
```

## 关键约束

- ❌ f-string → ✅ `%s` 或 `.format()` 或拼接（SuperMind WS 执行）
- ❌ Playwright → ✅ WS 直连（首选）
- ✅ iFinD TOKEN 硬编码（模块级，仅采集时使用）
- ✅ 所有路径通过 `config.py` 动态解析
