"""看 c1_base 实际选中的 80 个特征里, 各数据源域各占多少 —— 决定补数优先级"""
import json
import pathlib

P = pathlib.Path('data/processed')
cands = sorted(P.glob('wf_daily_*c1_base*.json')) or sorted(P.glob('wf_daily_*.json'))
if not cands:
    raise SystemExit('找不到回测结果 json')

f = max(cands, key=lambda x: x.stat().st_mtime)
d = json.loads(f.read_text())
feats = d.get('selected_features', [])
print(f"结果文件: {f.name}")
print(f"选中特征: {len(feats)}\n")


def domain(c):
    if c.startswith('mkt_'):
        return '大盘(K线派生)'
    if c.startswith(('ovn_', 'us_', 'sox_', 'spx_', 'ixic_')):
        return '外盘/隔夜'
    if c.startswith(('mf_', 'dde_', 'fund_flow', 'super_large', 'large_net', 'mtss')):
        return '资金流/两融'
    if c.startswith(('ev_', 'tev_')):
        return '事件'
    if c.startswith('ann_'):
        return '公告'
    if c.startswith(('con_', 'concept_')):
        return '概念板块'
    if c.startswith(('leader_', 'has_leader')):
        return '供应链'
    if c in ('cn_pmi', 'us_ism_pmi') or c.startswith(('cn_', 'usd', 'cny')):
        return '宏观'
    if c.startswith(('cmdty_', 'comm_', 'gold', 'oil')):
        return '商品'
    return '个股技术面(K线派生)'


from collections import Counter
cnt = Counter(domain(c) for c in feats)
print(f"{'数据源域':<20}{'特征数':>7}{'占比':>9}   补数难度")
print("-" * 62)
DIFF = {
    '个股技术面(K线派生)': '已解决 (新浪K线)',
    '大盘(K线派生)': '已解决 (新浪K线)',
    '外盘/隔夜': '可解决 (新浪美股)',
    '宏观': '可解决 (bond_zh_us_rate等)',
    '商品': '待查',
    '概念板块': '已解决 (K线派生)',
    '供应链': '静态映射, 无需更新',
    '资金流/两融': '受阻 (东财封禁)',
    '事件': '受阻 (iFinD不可用)',
    '公告': '受阻 (东财封禁)',
}
for k, v in cnt.most_common():
    print(f"{k:<20}{v:>7}{v/len(feats)*100:>8.1f}%   {DIFF.get(k,'?')}")

blocked = sum(v for k, v in cnt.items() if k in ('资金流/两融', '事件', '公告'))
print(f"\n受阻域合计: {blocked}/{len(feats)} = {blocked/len(feats)*100:.1f}%")

print("\n--- Top20 特征 (按选择顺序) ---")
for i, c in enumerate(feats[:20], 1):
    print(f"  {i:2d}. {c:38s} [{domain(c)}]")
