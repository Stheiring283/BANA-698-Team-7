import pandas as pd
import numpy as np

# ── Load raw data ──────────────────────────────────────────────────────────
df = pd.read_csv('bana698rawdata.csv')

# ── Definition : Sustained Engagement ───────────────────────────────────
# A post is "sustained" (1) if:
#   1. total_likes >= 1.5x early_likes  (likes at least 1.5x'd after the early window)
#   2. total_likes >= 60th percentile of all posts' total likes
#
# Posts where early_likes == 0 are excluded (flagged as 0) to avoid divide-by-zero.

total_likes_60th = np.percentile(df['total_likes'], 60)
likes_growth_ratio = df['total_likes'] / df['early_likes'].replace(0, np.nan)

df['sustained_engagement'] = (
    (likes_growth_ratio >= 1.5) &
    (df['total_likes'] >= total_likes_60th)
).fillna(0).astype(int)

# ── Summary ────────────────────────────────────────────────────────────────
n_total  = len(df)
n_sus    = df['sustained_engagement'].sum()
pct_sus  = n_sus / n_total * 100

he_0 = df.loc[df['sustained_engagement'] == 0, 'high_effort_engagement']
he_1 = df.loc[df['sustained_engagement'] == 1, 'high_effort_engagement']

print(f"60th percentile threshold for total_likes : {total_likes_60th:.0f}")
print(f"Posts classified as sustained             : {n_sus:,} / {n_total:,} ({pct_sus:.1f}%)")
print()
print("High-effort engagement by group:")
print(f"  Not sustained (n={len(he_0):,}) — mean: {he_0.mean():.2f}, median: {he_0.median():.1f}")
print(f"  Sustained     (n={len(he_1):,}) — mean: {he_1.mean():.2f}, median: {he_1.median():.1f}")
print(f"  HE lift: {(he_1.mean() / he_0.mean() - 1) * 100:.1f}%")

# ── Save updated CSV ───────────────────────────────────────────────────────
output_path = 'bana698rawdata_with_sustained_engagement.csv'
df.to_csv(output_path, index=False)
print(f"\nSaved to: {output_path}")