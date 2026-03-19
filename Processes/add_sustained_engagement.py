import pandas as pd
import numpy as np

# ── Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv('bana698data.csv', low_memory=False)

# ── Build total_likes ──────────────────────────────────────────────────────
df['total_likes'] = df['early_likes'] + df['post_early_likes']

# ── Definition #1: Sustained Engagement ───────────────────────────────────
# A post is "sustained" (1) if BOTH conditions are met:
#   1. post_early_likes / early_likes >= 1.5
#      (likes at least 1.5x'd after the early window)
#   2. post_early_total_engagement / early_total_engagement >= 1.5
#      (total engagement at least 1.5x'd after the early window)
#
# Posts where early_likes or early_total_engagement == 0 are coded as 0
# to avoid divide-by-zero.

ratio_like = df['post_early_likes'] / df['early_likes'].replace(0, np.nan)
ratio_eng  = df['post_early_total_engagement'] / df['early_total_engagement'].replace(0, np.nan)

df['sustained_engagement'] = (
    (ratio_like >= 1.5) & (ratio_eng >= 1.5)
).fillna(0).astype(int)

# ── Summary ────────────────────────────────────────────────────────────────
n_total = len(df)
n_sus   = df['sustained_engagement'].sum()

he_0 = df.loc[df['sustained_engagement'] == 0, 'high_effort_engagement']
he_1 = df.loc[df['sustained_engagement'] == 1, 'high_effort_engagement']

print(f"Posts classified as sustained : {n_sus:,} / {n_total:,} ({n_sus/n_total*100:.1f}%)")
print(f"Overall mean HE               : {df['high_effort_engagement'].mean():.4f}")
print(f"Mean HE — not sustained (0)   : {he_0.mean():.4f}")
print(f"Mean HE — sustained (1)       : {he_1.mean():.4f}")
print(f"HE lift                       : +{(he_1.mean()/he_0.mean()-1)*100:.1f}%")

# ── Save ───────────────────────────────────────────────────────────────────
df.to_csv('bana698data_v1.csv', index=False)
print(f"\nSaved to: bana698data_v1.csv")
