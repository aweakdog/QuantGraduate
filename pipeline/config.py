"""
配置中心 — 集中管理项目配置、路径、环境变量

使用方式:
  from pipeline.config import settings
  settings.DATA_DIR  # 自动解析的 data 路径
  settings.MODE      # "live" 或 "backtest"

环境变量覆盖:
  QUANT_DATA_DIR=/custom/path  python pipeline/feature_engine.py
  QUANT_MODE=backtest          python pipeline/chain_leader_scorer.py
"""
import os
from pathlib import Path
from typing import Literal

# 仓库根目录下的 .env 自动加载 (账号密码集中存放于此, 见 docs/credentials.md)
# 已存在的环境变量优先, .env 只作兜底, 不覆盖显式设置的值。
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
except ImportError:
    pass


class Settings:
    """项目全局配置"""

    # ─── 项目根目录 ────────────────────────────────────────
    # 自动从本文件位置向上解析
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

    # ─── 数据目录 ──────────────────────────────────────────
    # 优先使用环境变量 QUANT_DATA_DIR，否则用默认路径
    DATA_DIR: Path = Path(
        os.environ.get("QUANT_DATA_DIR", str(PROJECT_ROOT / "data"))
    )

    # ─── 子目录快捷属性 ────────────────────────────────────
    @property
    def KLINE_DIR(self) -> Path:
        return self.DATA_DIR / "raw" / "kline"

    @property
    def FUND_FLOW_DIR(self) -> Path:
        return self.DATA_DIR / "raw" / "fund_flow"

    @property
    def UNIVERSE_DIR(self) -> Path:
        return self.DATA_DIR / "universe"

    @property
    def PROCESSED_DIR(self) -> Path:
        return self.DATA_DIR / "processed"

    @property
    def FEATURES_DIR(self) -> Path:
        return self.PROCESSED_DIR / "features"

    @property
    def MODEL_DIR(self) -> Path:
        return self.PROCESSED_DIR / "model"

    @property
    def MACRO_DIR(self) -> Path:
        return self.DATA_DIR / "raw" / "macro"

    @property
    def TUSHARE_DIR(self) -> Path:
        return self.DATA_DIR / "raw" / "tushare"

    @property
    def CJSC_DIR(self) -> Path:
        """中信建投客户端导出的本地行情/财务/板块数据 (见 pipeline/decode_cjsc.py)

        目录结构沿用导出包原样: {SH,SZ}/{60,300,86400}/{code}.DAT
        分钟线只有约 1 年历史(2025-07 ~ 2026-07), 日线到 2001 年。
        """
        return self.DATA_DIR / "raw" / "cjsc"

    # ─── 供应链图谱路径 ────────────────────────────────────
    @property
    def SUPPLY_CHAIN_PATH(self) -> Path:
        return self.UNIVERSE_DIR / "supply_chain_map.json"

    @property
    def CHAIN_LEADER_UNIVERSE_PATH(self) -> Path:
        return self.UNIVERSE_DIR / "chain_leader_universe.json"

    @property
    def CHAIN_MAP_MERGED_PATH(self) -> Path:
        """统一后的供应链数据（合并 supply_chain_map + chain_leader_universe）"""
        return self.UNIVERSE_DIR / "chain_map_merged.json"

    @property
    def WATCHLIST_PATH(self) -> Path:
        """⚠ 含事后选股偏差, 禁止用于回测/特征构建/选股池

        watchlist.json == watchlist_216.json, 是 2026-07-18 手工按题材挑的 216 只
        概念股(钠离子电池/光纤概念/旅游概念...), 却被用在 2022-2026 的回测里。
        2026-08-14 实测这份名单本身就是 alpha:
          名单∩universe 的 130 只   2022-09~2026-08 等权 +174.0% (年化+30.6%, 夏普1.12)
          全 universe 630 只                            +56.2% (年化+12.5%, 夏普0.70)
          名单外 500 只                                 +33.4% (年化 +7.9%, 夏普0.51)
          -> 年化超额 +16.7%, IR 1.42, 除 2022 外逐年跑赢
        旧资金流源(iFinD)只覆盖这份名单, 于是老矩阵里 fund_flow_* 只有 243 只有值,
        LightGBM 用"是否为 NaN"这个切分隐式把选股范围锁进了这个池子, 白拿 16.7%/年。
        这正是 FBTR +27.1% 的全部来源: 换成 tushare 全覆盖(掩码消失)塌 23.7pp,
        直接剔掉资金流列(掩码同样消失)塌到 -16.2% —— 塌的不是信息, 是掩码。
        复现: scripts/coverage_bias_test.py

        选股池请用 data/universe/universe_pit_*.parquet (按当期市值/成交额 PIT 构建)。
        本属性只保留给"给这些票拉基本面/公告"一类与选股无关的取数用途。
        """
        return self.UNIVERSE_DIR / "watchlist.json"

    # ─── 输出路径 ──────────────────────────────────────────
    @property
    def BACKTEST_DIR(self) -> Path:
        return self.PROJECT_ROOT / "backtest"

    @property
    def TRAINING_DATA_PATH(self) -> Path:
        return self.PROCESSED_DIR / "training_data.parquet"

    @property
    def SCORES_PATH(self) -> Path:
        return self.PROCESSED_DIR / "daily_scores.json"

    # ─── THS Python 解释器路径 ────────────────────────────
    @property
    def THS_PYTHON(self) -> str:
        return os.environ.get(
            "THS_PYTHON",
            r"C:\Users\admin\.workbuddy\binaries\python\envs\ths\Scripts\python.exe",
        )

    @property
    def MODEL_PATH(self) -> Path:
        return self.MODEL_DIR / "xgb_model.pkl"

    # ─── 运行模式 ──────────────────────────────────────────
    # "live" 或 "backtest"，影响 wencai vs 价格代理的选择
    MODE: Literal["live", "backtest"] = os.environ.get("QUANT_MODE", "live")  # type: ignore

    # ─── UFD 路由 (通过环境变量设置, 不硬编码) ─────────────
    UFD_ROUTER: str = os.environ.get("UFD_ROUTER", "")

    # ─── THS 凭据（通过环境变量设置，不在代码中硬编码）───────
    @property
    def THS_USERNAME(self) -> str:
        return os.environ.get("THS_USERNAME", "")

    @property
    def THS_PASSWORD(self) -> str:
        return os.environ.get("THS_PASSWORD", "")

    # ─── Tushare Pro token ────────────────────────────────
    @property
    def TUSHARE_TOKEN(self) -> str:
        return os.environ.get("TUSHARE_TOKEN", "")

    # ─── 评分权重 ──────────────────────────────────────────
    SCORE_WEIGHTS = {
        "event_intensity": 0.35,
        "binding": 0.25,
        "fund": 0.15,
        "history": 0.15,
        "self_event": 0.10,
    }

    # ─── 自身事件修正系数 ──────────────────────────────────
    SELF_EVENT_MULTIPLIERS = {
        "severe_negative": 0.85,
        "moderate_negative": 0.92,
        "minor_negative": 0.95,
        "neutral": 1.00,
        "minor_positive": 1.05,
        "major_positive": 1.15,
    }

    # ─── 启动校验 ──────────────────────────────────────────
    _validated: bool = False

    def validate(self, strict: bool = False) -> list[str]:
        """校验关键路径和配置，返回警告列表

        检查：
          - 数据根目录存在
          - 关注圈文件存在
          - 供应链映射存在
          - K线目录存在

        Args:
            strict: 如果 True，关键缺失会打印警告；False 只收集不输出

        Returns:
            警告信息列表
        """
        warnings: list[str] = []
        checks = [
            ("数据根目录", self.DATA_DIR, True),
            ("关注圈文件", self.WATCHLIST_PATH, True),
            ("供应链映射", self.SUPPLY_CHAIN_PATH, False),
            ("K线目录", self.KLINE_DIR, True),
            ("资金流目录", self.FUND_FLOW_DIR, False),
            ("特征输出目录", self.FEATURES_DIR, False),
            ("模型目录", self.MODEL_DIR, False),
        ]
        for label, path, critical in checks:
            exists = path.exists()
            if not exists:
                msg = f"配置校验: {label} 不存在 → {path}"
                if critical:
                    if strict:
                        print(f"  ⚠️  {msg}")
                    warnings.append(msg)
                else:
                    warnings.append(f"{msg}（可选）")

        if self.MODE not in ("live", "backtest"):
            warnings.append(f"配置校验: MODE={self.MODE!r} 非法，应为 live 或 backtest")

        self._validated = True
        return warnings


# 单例导出
settings = Settings()
