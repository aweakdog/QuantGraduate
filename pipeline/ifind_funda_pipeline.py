"""iFinD 基本面数据采集管线 — 批量获取并存储为 fundamentals parquet"""
import json, requests, pandas as pd, os, sys, time, re
from pathlib import Path
from typing import Optional

DATA_DIR = r'D:\myAI\claude-workspace\quant-strategy\data'
WATCH_PATH = os.path.join(DATA_DIR, 'universe', 'watchlist.json')
FUNDA_DIR = os.path.join(DATA_DIR, 'raw', 'fundamentals')
os.makedirs(FUNDA_DIR, exist_ok=True)

URL = "https://api-mcp.51ifind.com:8643/ds-mcp-servers/hexin-ifind-ds-stock-mcp"
TOKEN = "eyJhbGciOiJSU0EtT0FFUC0yNTYiLCJlbmMiOiJBMjU2R0NNIn0.EK9QgcrA9nSdoeX97Ol3gduuVyqoMJjtIFPAyXTksV4T-hKnzLqkW0Q1j-02SSnHyIzSMgGqD74Rj1lZto2oIAynV5gHiZXjTgX8ARvE1NtnBNLbdHMADuomjMNRHAXpPE83sCL4ehLGL6zb5_n8XVzLwr_RuJ4SZiekMR3sEMGNePywrP2flMO_K6R0suTvFTlSWU5WxOYMKqLxUOciZnZqTnxUs6_Lnj6He4XBEgul2VdJX4w6lcPq5ibDx7CDp-8SzW_FW0CBkREtIWBbyuqHaQyWdnUbg6nPoCo3sD3ipTL3ereUqX33GY8mn8dYfIFKZShADp5kGziTtqWRLQ.gQm_IZm7qxG8OKz-.mHYCbUproLUp1qLvMntUQ5rq6e27ORuzqnhXvhkIVFbA5UsTZBq_1UqJuq4XlN5EuI6j2o91dgWFz2vIHhm7482C1vcpwDTlUC48j_UymGR03dX8iiriSA-qE7ZQJLx50YFrG7aFw5sALibKzwDGVETilkI9upyDUu5s7tMg3cIhj0GUWU-8xso-AZf_frGahYyEzZsK4EHKHBxxVmE5IghBnJcTvjvB-Hs46nrhbeQ2wr2aSP82bq8JtXaHvstS6CC_63YS_jB7KWBF1sqkP25138A7y31xzOlmMEW_GIuDOElFXT2SXJ3qbGQmYBg_EwPMyGdy4rs2xk74WmQNyCzrdRIY86zPOlWqyt2EHJi9GHwxOjgAWdJ8eQ0o_kCsw7lyvYoWwQbyeqcs2rOTrhLeMbI.-AlkZXTtGcQkQqGT_JGRzw"
HEADERS = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}

# 列名映射（中文→标准名）
COL_MAP = {
    "营业总收入": "revenue",
    "营业总收入(单位:元)": "revenue",
    "营业收入(单位:元)": "revenue",
    "营业收入": "revenue",
    "净利润(单位:元)": "profit",
    "净利润": "profit",
    "归属于母公司所有者的净利润(单位:元)": "profit",
    "归属于母公司所有者的净利润": "profit",
    "基本每股收益(单位:元)": "eps",
    "基本每股收益": "eps",
    "每股收益EPS-扣除稀释": "eps",
    "每股收益EPS-基本": "eps",
    "每股收益EPS-稀释": "eps",
    "每股净资产BPS": "bps",
    "每股净资产": "bps",
    "净资产收益率ROE": "roe",
    "净资产收益率": "roe",
    "市盈率(PE,LYR)": "pe",
    "市盈率(PE,TTM)": "pe",
    "市盈率(PE)": "pe",
    "市盈率": "pe",
    "市净率(PB,最新)": "pb",
    "市净率(PB,MRQ)": "pb",
    "市净率(PB)": "pb",
    "市净率": "pb",
    "总市值": "mcap",
    "总资产": "total_assets",
    "资产总计": "total_assets",
    "资产负债率": "debt_ratio",
    "毛利率": "gross_margin",
    "销售毛利率": "gross_margin",
    "经营活动现金流净额": "operate_cf",
    "经营活动现金流": "operate_cf",
    "经营性现金流": "operate_cf",
    "现金流净额": "operate_cf",
    "现金流量净额": "operate_cf",
    "现金流": "operate_cf",
}

def mcp_call(name: str, args: dict, max_retries: int = 3) -> dict:
    """调用 iFinD MCP API，带重试 + 指数退避

    处理 429 限流（10s→20s→40s 退避）和网络错误（5s→10s→20s 退避）。

    Args:
        name: MCP 工具名
        args: 参数字典
        max_retries: 重试次数（默认 3，含首次）

    Returns:
        API 响应字典，或 {}（全部重试失败时）
    """
    payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
               "params": {"name": name, "arguments": args}}
    for attempt in range(max_retries):
        try:
            r = requests.post(URL, json=payload, headers=HEADERS, timeout=120)
            if r.status_code == 429 and attempt < max_retries - 1:
                wait = 10 * (2 ** attempt)  # 10s → 20s → 40s
                print(f"    [429] 速率限制，等待 {wait}s 后重试 ({attempt+1}/{max_retries})")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            if attempt < max_retries - 1:
                wait = 5 * (2 ** attempt)
                print(f"    [重试 {attempt+1}/{max_retries}] 等待 {wait}s: {e}")
                time.sleep(wait)
                continue
            print(f"    [失败] 超出重试次数: {e}")
            return {}
    return {}

def parse_table(md_text: str) -> Optional[pd.DataFrame]:
    """解析 iFinD markdown 表格"""
    lines = md_text.strip().split('\n')
    table_lines = [l for l in lines if l.startswith('|') and l.endswith('|')]
    if len(table_lines) < 3:
        return None
    # 第一行表头，第二行分隔符，之后数据
    header = [h.strip() for h in table_lines[0].split('|')[1:-1]]
    data = []
    for line in table_lines[2:]:
        cells = [c.strip() for c in line.split('|')[1:-1]]
        while len(cells) < len(header):
            cells.append('')
        data.append(cells[:len(header)])
    return pd.DataFrame(data, columns=header)

def extract_financials(code6: str) -> Optional[pd.DataFrame]:
    """获取单只股票基本面数据

    通过 iFinD MCP API 查询 ROE/PE/PB/营收/净利润 等财务指标，
    返回 Markdown 表格解析后的 DataFrame。

    Args:
        code6: 6 位股票代码

    Returns:
        DataFrame（原始列名），或 None（无数据/查询失败）
    """
    # 构建查询 — 尽可能覆盖多年
    query = f"{code6}.SZ 2019-2025年度 财务数据 ROE PE PB 营业收入 净利润 每股收益 每股净资产 总资产 资产负债率 毛利率 现金流"
    if code6.startswith('6'):
        query = f"{code6}.SH 2019-2025年度 财务数据 ROE PE PB 营业收入 净利润 每股收益 每股净资产 总资产 资产负债率 毛利率 现金流"

    try:
        resp = mcp_call("get_stock_financials", {"query": query})
        content = resp.get("result", {}).get("content", [])
        if not content:
            return None

        text = content[0].get("text", "")
        try:
            inner = json.loads(text)
            md_text = inner.get("data", {}).get("answer", text)
        except json.JSONDecodeError:
            md_text = text

        df = parse_table(md_text)
        if df is None or len(df) == 0:
            return None

        return df
    except (requests.RequestException, json.JSONDecodeError, KeyError, IndexError, ValueError) as e:
        print(f"    ERROR: {e}")
        return None

def standardize_financials(df_raw: Optional[pd.DataFrame], code6: str) -> Optional[pd.DataFrame]:
    """标准化列名并提取需要的字段

    将原始财务数据 DataFrame（含中文列名）转换为统一英文列名格式，
    过滤无效年份（<2010 或 >2030），标准化亿/万元数值为 float。

    Args:
        df_raw: 原始财务数据（来自 extract_financials）
        code6: 6 位股票代码（用于添加 code 列）

    Returns:
        标准化后的 DataFrame，列: [date, code, revenue, profit, eps, bps, roe, pe, pb,
        mcap, total_assets, debt_ratio, gross_margin, operate_cf]
        或 None（输入为空/无日期列时）
    """
    if df_raw is None or len(df_raw) == 0:
        return None

    # 找到日期列（年度/报告期）
    date_col = None
    for col in df_raw.columns:
        if '年度' in col or '报告期' in col:
            date_col = col
            break
    if date_col is None:
        # 尝试列 2（通常是年度列）
        if len(df_raw.columns) > 2:
            date_col = df_raw.columns[2]
        else:
            return None

    # 构建结果
    result = {"date": [], "code": []}

    # 需要映射的目标列
    target_cols = ["revenue", "profit", "eps", "bps", "roe", "pe", "pb", "mcap",
                   "total_assets", "debt_ratio", "gross_margin", "operate_cf"]
    for t in target_cols:
        result[t] = []

    # 列名映射：对每个原始列名，看是否能匹配到目标列
    col_map = {}  # target → source_col_name
    for src_col in df_raw.columns:
        src_clean = src_col.replace(' ', '').replace('　', '')
        for pattern, target in COL_MAP.items():
            if pattern in src_clean:
                if target not in col_map:
                    col_map[target] = src_col
                break

    # 逐行处理
    for _, row in df_raw.iterrows():
        # 解析日期
        date_str = str(row.get(date_col, '')).strip()
        if not date_str or date_str == 'nan':
            continue

        # 转换为标准的 YYYY-MM-DD 格式年末日期
        year_match = re.search(r'(\d{4})', date_str)
        if not year_match:
            continue
        year = int(year_match.group(1))
        # 过滤不合理年份（iFinD 偶发乱码）
        if year < 2010 or year > 2030:
            continue
        std_date = f"{year}-12-31"  # 年度默认年末

        result["date"].append(std_date)
        result["code"].append(code6)

        for target in target_cols:
            src = col_map.get(target)
            if src and src in df_raw.columns:
                val = row[src]
                # 清洗数值（去除亿/万等单位）
                if val and str(val).strip() and str(val) != 'nan':
                    cleaned = str(val).replace(',', '').replace('亿', 'e8').replace('万', 'e4')
                    try:
                        result[target].append(float(cleaned))
                    except (ValueError, TypeError):
                        result[target].append(None)
                else:
                    result[target].append(None)
            else:
                result[target].append(None)

    result_df = pd.DataFrame(result)
    if len(result_df) == 0:
        return None

    # 去重（同日期保留首行）
    result_df = result_df.drop_duplicates(subset=["date"], keep="first")

    return result_df

if __name__ == "__main__":
    # 读取关注圈
    with open(WATCH_PATH, encoding='utf-8') as f:
        watch = json.load(f)
    stocks = watch.get("watchlist", [])
    print(f"关注圈: {len(stocks)} 只股票")

    # 检查已有数据
    existing = {f.replace('.parquet', '') for f in os.listdir(FUNDA_DIR)}
    print(f"已有基本面数据: {len(existing)} 只")
    missing = [(s['code'][:6], s['code'], s['name']) for s in stocks if s['code'][:6] not in existing]
    print(f"待采集: {len(missing)} 只")

    if not missing:
        print("全部已有数据，跳过采集")
        exit(0)

    # 批量采集（API 速率限制）
    success = 0
    failed = 0
    batch_size = 5  # 每批5只

    for i in range(0, len(missing), batch_size):
        batch = missing[i:i+batch_size]
        for code6, full_code, name in batch:
            print(f"  [{i+1+success}/{len(missing)}] {code6} {name}...", end=" ")
            sys.stdout.flush()

            df_raw = extract_financials(code6)
            if df_raw is None:
                print("[FAIL] 无数据")
                failed += 1
                continue

            # 标准化
            std = standardize_financials(df_raw, code6)
            if std is None or len(std) == 0:
                print("[FAIL] 标准化失败")
                failed += 1
                continue

            # 保存
            out_path = os.path.join(FUNDA_DIR, f"{code6}.parquet")
            std.to_parquet(out_path, index=False)
            print(f"[OK] {len(std)}年")
            success += 1

            time.sleep(0.5)  # 请求间隔

        # 每批后稍作等待
        time.sleep(1)
        print(f"  进度: 成功{success} + 失败{failed} / 共{len(missing)}")

    print(f"\n采集完成! 成功{success}, 失败{failed}")
    print(f"FUNDA 目录: {FUNDA_DIR} ({len(os.listdir(FUNDA_DIR))} 文件)")

