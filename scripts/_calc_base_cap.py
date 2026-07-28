import pandas as pd, numpy as np
from pathlib import Path

df = pd.read_parquet('data/processed/training_data_v24.parquet')
codes = df['code'].unique()
KLINE_DIR = Path('data/raw/kline')
prices = {}
for code in codes:
    code6 = str(code)[:6]
    p = KLINE_DIR / f'{code6}.parquet'
    if not p.exists():
        continue
    kl = pd.read_parquet(p, columns=['close'])
    prices[code] = float(kl['close'].iloc[-1])

sorted_codes = sorted(prices.items(), key=lambda x: x[1])
print('Price distribution:')
for pct in [10, 25, 50, 75, 90]:
    p = np.percentile(list(prices.values()), pct)
    print(f'  P{pct}: price={p:.2f} -> 1 lot={p*100:.0f}')

for base_pct in [60, 70, 80]:
    base_cap = 100000 * base_pct / 100
    cumsum = 0
    n = 0
    for code, price in sorted_codes:
        lot_cost = price * 100
        if cumsum + lot_cost > base_cap:
            break
        cumsum += lot_cost
        n += 1
    max_price = sorted_codes[n - 1][1] if n > 0 else 0
    print(f'Base {base_pct}% = {base_cap:.0f}: {n} stocks, cost={cumsum:.0f}, max_price={max_price:.2f}')
