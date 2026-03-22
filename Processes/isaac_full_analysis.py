import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy.stats import zscore
import warnings
warnings.filterwarnings('ignore')

# ─── LOAD DATA ───────────────────────────────────────────────────────────────
xl = pd.ExcelFile(r'/Users/eyz/Desktop/Grad Project/dataset_algorithmic_persuasion_10000 (1).xlsx')
df = xl.parse('data')
df = df[df['post_format'].isin(['image', 'video', 'carousel'])]

# ─── PREP ────────────────────────────────────────────────────────────────────
df['verified']        = df['verified'].fillna(0).astype(int)
df['content_type']    = df['content_type'].astype('category')
df['post_format']     = df['post_format'].astype('category')
df['log_high_effort'] = np.log1p(df['high_effort_engagement'])
df['log_low_effort']  = np.log1p(df['low_effort_engagement'])

# ─── BUILD CUSTOM AAI ────────────────────────────────────────────────────────
# Four components, each z-scored then averaged equally
# Breadth:     reach / follower_count (algorithmic lift beyond organic base)
# Depth:       impressions_per_reach  (repeated surfacing per viewer)
# Momentum:    reach_growth_rate      (proxy for reach momentum 0-1)
# Persistence: exposure_persistence_hours (how long post stayed in circulation)
df['reach_to_follower'] = df['reach'] / df['follower_count']
df['z_breadth']         = zscore(df['reach_to_follower'],          nan_policy='omit')
df['z_depth']           = zscore(df['impressions_per_reach'],      nan_policy='omit')
df['z_momentum']        = zscore(df['reach_growth_rate'],          nan_policy='omit')
df['z_persistence']     = zscore(df['exposure_persistence_hours'], nan_policy='omit')
df['custom_aai']        = df[['z_breadth','z_depth','z_momentum','z_persistence']].mean(axis=1)

# Mean center for interaction terms
df['aai_c']       = df['custom_aai'] - df['custom_aai'].mean()
df['authority_c'] = df['authority_log'] - df['authority_log'].mean()

# ─── BUILD DTAI ──────────────────────────────────────────────────────────────
# Dual-Threshold Amplification Indicator (DTAI)
# DTAI = 1 if BOTH conditions met:
#   Condition 1 (Breadth):  reach_to_follower >= median (above avg algorithmic lift)
#   Condition 2 (Depth):    impressions_per_reach >= median (above avg repeated surfacing)
median_ipr = df['impressions_per_reach'].median()
median_rtf = df['reach_to_follower'].median()

df['dtai'] = (
    (df['reach_to_follower'] >= median_rtf) &
    (df['impressions_per_reach'] >= median_ipr)
).astype(int)

print("=== DATA LOADED ===")
print(f"Rows: {len(df)}")
print()

# ─── DESCRIPTIVE STATISTICS ──────────────────────────────────────────────────
print("=" * 60)
print("DESCRIPTIVE STATISTICS")
print("=" * 60)
print(df[['high_effort_engagement', 'log_high_effort', 'custom_aai',
          'authority_log', 'verified']].describe().round(4))
print()
zeros = (df['high_effort_engagement'] == 0).sum()
print(f"DV zeros:                  {zeros} ({zeros/len(df)*100:.1f}%)")
print(f"DV mean:                   {df['high_effort_engagement'].mean():.4f}")
print(f"DV variance:               {df['high_effort_engagement'].var():.4f}")
print(f"Overdispersion (var/mean): {df['high_effort_engagement'].var()/df['high_effort_engagement'].mean():.4f}")
print()

# ─── DTAI SUMMARY ────────────────────────────────────────────────────────────
print("=" * 60)
print("DUAL-THRESHOLD AMPLIFICATION INDICATOR (DTAI) — SUMMARY")
print("=" * 60)
n_dtai   = df['dtai'].sum()
pct_dtai = n_dtai / len(df) * 100
he_0 = df.loc[df['dtai'] == 0, 'high_effort_engagement']
he_1 = df.loc[df['dtai'] == 1, 'high_effort_engagement']
le_0 = df.loc[df['dtai'] == 0, 'low_effort_engagement']
le_1 = df.loc[df['dtai'] == 1, 'low_effort_engagement']

print(f"Median reach_to_follower threshold:     {median_rtf:.4f}")
print(f"Median impressions_per_reach threshold: {median_ipr:.4f}")
print(f"reach_to_follower >= median:            {(df['reach_to_follower'] >= median_rtf).sum():,} posts")
print(f"impressions_per_reach >= median:        {(df['impressions_per_reach'] >= median_ipr).sum():,} posts")
print(f"Both conditions met (DTAI = 1):         {n_dtai:,} / {len(df):,} ({pct_dtai:.1f}%)")
print()
print("High-effort engagement by group:")
print(f"  DTAI = 0 (n={len(he_0):,}) — mean: {he_0.mean():.2f}, median: {he_0.median():.1f}")
print(f"  DTAI = 1 (n={len(he_1):,}) — mean: {he_1.mean():.2f}, median: {he_1.median():.1f}")
print(f"  HE lift: {(he_1.mean() / he_0.mean() - 1) * 100:.1f}%")
print()
print("Low-effort engagement by group:")
print(f"  DTAI = 0 (n={len(le_0):,}) — mean: {le_0.mean():.2f}, median: {le_0.median():.1f}")
print(f"  DTAI = 1 (n={len(le_1):,}) — mean: {le_1.mean():.2f}, median: {le_1.median():.1f}")
print(f"  LE lift: {(le_1.mean() / le_0.mean() - 1) * 100:.1f}%")
print()

# ─── CORRELATIONS ────────────────────────────────────────────────────────────
print("=" * 60)
print("CORRELATION TABLE")
print("=" * 60)
print("AAI vs both DVs:")
print(f"  custom_aai x log_high_effort:  r = {df['custom_aai'].corr(df['log_high_effort']):.4f}")
print(f"  custom_aai x log_low_effort:   r = {df['custom_aai'].corr(df['log_low_effort']):.4f}")
print()
print("DTAI with key variables:")
for col, label in [
    ('log_high_effort', 'log_high_effort'),
    ('log_low_effort',  'log_low_effort'),
    ('custom_aai',      'custom_aai'),
    ('authority_log',   'authority_log'),
    ('verified',        'verified'),
]:
    r = df['dtai'].corr(df[col])
    print(f"  dtai x {label:<20} r = {r:.4f}")
print()
print("AAI component correlations with log_high_effort:")
for col, label in [
    ('reach_to_follower',        'Breadth (reach_to_follower)'),
    ('impressions_per_reach',    'Depth   (impressions_per_reach)'),
    ('reach_growth_rate',        'Momentum (reach_growth_rate)'),
    ('exposure_persistence_hours','Persistence (exposure_persistence_hours)'),
]:
    r = df[col].corr(df['log_high_effort'])
    print(f"  {label:<45} r = {r:.4f}")
print()

# ─── MODEL PREP ──────────────────────────────────────────────────────────────
model_cols = [
    'log_high_effort', 'log_low_effort',
    'custom_aai', 'aai_c', 'dtai',
    'authority_log', 'authority_c', 'verified',
    'content_type', 'post_format', 'influencer_id'
]
df_model = df[model_cols].dropna()
print(f"Model sample size: {len(df_model)}")
print()

# ─── SECTION 1: AAI → HE vs LE DIRECT COMPARISON ─────────────────────────────
print("=" * 60)
print("SECTION 1: AAI → HE vs LE DIRECT COMPARISON")
print("=" * 60)

m_aai_he = smf.ols('log_high_effort ~ custom_aai',
    data=df_model).fit(cov_type='cluster', cov_kwds={'groups': df_model['influencer_id']})
m_aai_le = smf.ols('log_low_effort ~ custom_aai',
    data=df_model).fit(cov_type='cluster', cov_kwds={'groups': df_model['influencer_id']})
m_aai_he_c = smf.ols(
    'log_high_effort ~ custom_aai + authority_log + verified + '
    'C(content_type, Treatment(reference="informational")) + '
    'C(post_format, Treatment(reference="image"))',
    data=df_model).fit(cov_type='cluster', cov_kwds={'groups': df_model['influencer_id']})
m_aai_le_c = smf.ols(
    'log_low_effort ~ custom_aai + authority_log + verified + '
    'C(content_type, Treatment(reference="informational")) + '
    'C(post_format, Treatment(reference="image"))',
    data=df_model).fit(cov_type='cluster', cov_kwds={'groups': df_model['influencer_id']})

print("No controls:")
for label, model in [('AAI → HE', m_aai_he), ('AAI → LE', m_aai_le)]:
    coef = model.params['custom_aai']; pval = model.pvalues['custom_aai']
    ci_lo = model.conf_int().loc['custom_aai', 0]; ci_hi = model.conf_int().loc['custom_aai', 1]
    sig = '***' if pval<0.001 else ('**' if pval<0.01 else ('*' if pval<0.05 else 'ns'))
    print(f"  {label:<12} coef={coef:.4f} | CI [{ci_lo:.4f}, {ci_hi:.4f}] | p={pval:.4f} {sig} | R²={model.rsquared:.4f} | {'positive' if coef>0 else 'negative'}")
print()
print("With controls:")
for label, model in [('AAI → HE', m_aai_he_c), ('AAI → LE', m_aai_le_c)]:
    coef = model.params['custom_aai']; pval = model.pvalues['custom_aai']
    ci_lo = model.conf_int().loc['custom_aai', 0]; ci_hi = model.conf_int().loc['custom_aai', 1]
    sig = '***' if pval<0.001 else ('**' if pval<0.01 else ('*' if pval<0.05 else 'ns'))
    print(f"  {label:<12} coef={coef:.4f} | CI [{ci_lo:.4f}, {ci_hi:.4f}] | p={pval:.4f} {sig} | R²={model.rsquared:.4f} | {'positive' if coef>0 else 'negative'}")
print()

# ─── SECTION 2: DTAI → HE AND LE ─────────────────────────────────────────────
print("=" * 60)
print("SECTION 2: DTAI → HIGH-EFFORT ENGAGEMENT")
print("=" * 60)
m_dtai_he = smf.ols('log_high_effort ~ dtai',
    data=df_model).fit(cov_type='cluster', cov_kwds={'groups': df_model['influencer_id']})
m_dtai_he_c = smf.ols(
    'log_high_effort ~ dtai + authority_log + verified + '
    'C(content_type, Treatment(reference="informational")) + '
    'C(post_format, Treatment(reference="image"))',
    data=df_model).fit(cov_type='cluster', cov_kwds={'groups': df_model['influencer_id']})

for label, model in [('No controls', m_dtai_he), ('With controls', m_dtai_he_c)]:
    coef = model.params['dtai']; pval = model.pvalues['dtai']
    ci_lo = model.conf_int().loc['dtai', 0]; ci_hi = model.conf_int().loc['dtai', 1]
    sig = '***' if pval<0.001 else ('**' if pval<0.01 else ('*' if pval<0.05 else 'ns'))
    print(f"{label}: coef={coef:.4f} | CI [{ci_lo:.4f}, {ci_hi:.4f}] | p={pval:.4f} {sig} | R²={model.rsquared:.4f}")
print()

print("=" * 60)
print("SECTION 2B: DTAI → LOW-EFFORT ENGAGEMENT")
print("=" * 60)
m_dtai_le = smf.ols('log_low_effort ~ dtai',
    data=df_model).fit(cov_type='cluster', cov_kwds={'groups': df_model['influencer_id']})
m_dtai_le_c = smf.ols(
    'log_low_effort ~ dtai + authority_log + verified + '
    'C(content_type, Treatment(reference="informational")) + '
    'C(post_format, Treatment(reference="image"))',
    data=df_model).fit(cov_type='cluster', cov_kwds={'groups': df_model['influencer_id']})

for label, model in [('No controls', m_dtai_le), ('With controls', m_dtai_le_c)]:
    coef = model.params['dtai']; pval = model.pvalues['dtai']
    ci_lo = model.conf_int().loc['dtai', 0]; ci_hi = model.conf_int().loc['dtai', 1]
    sig = '***' if pval<0.001 else ('**' if pval<0.01 else ('*' if pval<0.05 else 'ns'))
    print(f"{label}: coef={coef:.4f} | CI [{ci_lo:.4f}, {ci_hi:.4f}] | p={pval:.4f} {sig} | R²={model.rsquared:.4f}")
print()

print("=" * 60)
print("SECTION 2C: DTAI + AAI → HE TOGETHER")
print("=" * 60)
m_both_simple = smf.ols('log_high_effort ~ dtai + custom_aai',
    data=df_model).fit(cov_type='cluster', cov_kwds={'groups': df_model['influencer_id']})
m_both_controlled = smf.ols(
    'log_high_effort ~ dtai + custom_aai + authority_log + verified + '
    'C(content_type, Treatment(reference="informational")) + '
    'C(post_format, Treatment(reference="image"))',
    data=df_model).fit(cov_type='cluster', cov_kwds={'groups': df_model['influencer_id']})

for label, model in [('No controls', m_both_simple), ('With controls', m_both_controlled)]:
    print(f"{label}:")
    for var in ['dtai', 'custom_aai']:
        coef = model.params[var]; pval = model.pvalues[var]
        ci_lo = model.conf_int().loc[var, 0]; ci_hi = model.conf_int().loc[var, 1]
        sig = '***' if pval<0.001 else ('**' if pval<0.01 else ('*' if pval<0.05 else 'ns'))
        print(f"  {var:<20} coef={coef:.4f} | CI [{ci_lo:.4f}, {ci_hi:.4f}] | p={pval:.4f} {sig} | {'positive' if coef>0 else 'negative'}")
    print(f"  R² = {model.rsquared:.4f}")
    print()

# ─── SECTION 3: NODE ANALYSIS ─────────────────────────────────────────────────
print("=" * 60)
print("SECTION 3: NODE 1 — INFLUENCER CHARACTERISTICS → HE")
print("=" * 60)
print("Individual correlations:")
for col, label in [('authority_log', 'Authority Log'), ('verified', 'Verified')]:
    r = df[col].corr(df['log_high_effort'])
    print(f"  {label:<20} r = {r:.4f}")
print()
m_inf = smf.ols('log_high_effort ~ authority_log + verified',
    data=df_model).fit(cov_type='cluster', cov_kwds={'groups': df_model['influencer_id']})
print(m_inf.summary2().tables[1].round(4))
print(f"R² = {m_inf.rsquared:.4f}")
print()

print("=" * 60)
print("SECTION 3B: NODE 2 — CONTENT CHARACTERISTICS → HE")
print("=" * 60)
m_con = smf.ols(
    'log_high_effort ~ '
    'C(content_type, Treatment(reference="informational")) + '
    'C(post_format, Treatment(reference="image"))',
    data=df_model).fit(cov_type='cluster', cov_kwds={'groups': df_model['influencer_id']})
print(m_con.summary2().tables[1].round(4))
print(f"R² = {m_con.rsquared:.4f}")
print()

# ─── SECTION 4: MODERATION MODELS ─────────────────────────────────────────────
print("=" * 60)
print("SECTION 4: MODEL 1 — AAI × INFLUENCER → HE")
print("=" * 60)
m1 = smf.ols(
    'log_high_effort ~ aai_c * authority_c + aai_c * verified + '
    'C(content_type, Treatment(reference="informational")) + '
    'C(post_format, Treatment(reference="image"))',
    data=df_model).fit(cov_type='cluster', cov_kwds={'groups': df_model['influencer_id']})
print(m1.summary2().tables[1].round(4))
print(f"R² = {m1.rsquared:.4f}")
print()

print("=" * 60)
print("SECTION 4B: MODEL 2 — AAI × INFLUENCER → LE")
print("=" * 60)
m2 = smf.ols(
    'log_low_effort ~ aai_c * authority_c + aai_c * verified + '
    'C(content_type, Treatment(reference="informational")) + '
    'C(post_format, Treatment(reference="image"))',
    data=df_model).fit(cov_type='cluster', cov_kwds={'groups': df_model['influencer_id']})
print(m2.summary2().tables[1].round(4))
print(f"R² = {m2.rsquared:.4f}")
print()

print("=" * 60)
print("SECTION 4C: MODEL 3 — AAI × CONTENT → HE")
print("=" * 60)
m3 = smf.ols(
    'log_high_effort ~ '
    'aai_c * C(content_type, Treatment(reference="informational")) + '
    'aai_c * C(post_format, Treatment(reference="image")) + '
    'authority_log + verified',
    data=df_model).fit(cov_type='cluster', cov_kwds={'groups': df_model['influencer_id']})
print(m3.summary2().tables[1].round(4))
print(f"R² = {m3.rsquared:.4f}")
print()

print("=" * 60)
print("SECTION 4D: MODEL 4 — AAI × CONTENT → LE")
print("=" * 60)
m4 = smf.ols(
    'log_low_effort ~ '
    'aai_c * C(content_type, Treatment(reference="informational")) + '
    'aai_c * C(post_format, Treatment(reference="image")) + '
    'authority_log + verified',
    data=df_model).fit(cov_type='cluster', cov_kwds={'groups': df_model['influencer_id']})
print(m4.summary2().tables[1].round(4))
print(f"R² = {m4.rsquared:.4f}")
print()

# ─── SECTION 5: INTERACTION TERMS SUMMARY ────────────────────────────────────
print("=" * 60)
print("SECTION 5: INTERACTION TERMS SUMMARY")
print("=" * 60)
for label, model in [
    ('M1 AAI x Influencer → HE', m1),
    ('M2 AAI x Influencer → LE', m2),
    ('M3 AAI x Content    → HE', m3),
    ('M4 AAI x Content    → LE', m4),
]:
    print(f"{label}:")
    for param in model.params.index:
        if ':' in param:
            coef = model.params[param]; pval = model.pvalues[param]
            sig  = '***' if pval<0.001 else ('**' if pval<0.01 else ('*' if pval<0.05 else 'ns'))
            print(f"  {param[-55:]:<55} coef={coef:.4f} p={pval:.4f} {sig}")
    print()

# ─── SECTION 6: INFLUENCER → CONTENT INDEPENDENCE ────────────────────────────
print("=" * 60)
print("SECTION 6: INFLUENCER vs CONTENT CHARACTERISTICS — INDEPENDENCE CHECK")
print("=" * 60)
df['content_type_str'] = df['content_type'].astype(str)
df['post_format_str']  = df['post_format'].astype(str)
for ct in ['experiential', 'promotional']:
    df[f'is_{ct}'] = (df['content_type_str'] == ct).astype(int)
for pf in ['video', 'carousel']:
    df[f'is_{pf}'] = (df['post_format_str'] == pf).astype(int)

print("Correlations — Influencer vars vs Content vars:")
for outcome, label in [
    ('is_experiential', 'content=experiential'),
    ('is_promotional',  'content=promotional'),
    ('is_video',        'format=video'),
    ('is_carousel',     'format=carousel'),
]:
    r_auth = df['authority_log'].corr(df[outcome])
    r_ver  = df['verified'].corr(df[outcome])
    print(f"  {label:<25} r(authority)={r_auth:.4f}  r(verified)={r_ver:.4f}")
print()

print("Logistic regressions — does influencer predict content type/format?")
print()
for outcome, label in [
    ('is_experiential', 'content_type = experiential (vs informational)'),
    ('is_promotional',  'content_type = promotional (vs informational)'),
    ('is_video',        'post_format = video (vs image)'),
    ('is_carousel',     'post_format = carousel (vs image)'),
]:
    try:
        m = smf.logit(f'{outcome} ~ authority_log + verified',
            data=df).fit(cov_type='cluster',
                         cov_kwds={'groups': df['influencer_id']}, disp=False)
        print(f"  {label}")
        for var in ['authority_log', 'verified']:
            coef = m.params[var]; pval = m.pvalues[var]
            sig  = '***' if pval<0.001 else ('**' if pval<0.01 else ('*' if pval<0.05 else 'ns'))
            print(f"    {var:<15} coef={coef:.4f} | p={pval:.4f} {sig} | {'positive' if coef>0 else 'negative'}")
        print(f"    Pseudo R² = {m.prsquared:.4f}")
        print()
    except Exception as e:
        print(f"  {label}: ERROR — {e}")
        print()

# ─── SECTION 7: R² FULL SUMMARY ──────────────────────────────────────────────
print("=" * 60)
print("SECTION 7: R² FULL SUMMARY")
print("=" * 60)
print(f"  Influencer characteristics alone → HE:   R² = {m_inf.rsquared:.4f}")
print(f"  Content characteristics alone → HE:      R² = {m_con.rsquared:.4f}")
print(f"  AAI alone → HE (no controls):            R² = {m_aai_he.rsquared:.4f}")
print(f"  AAI alone → LE (no controls):            R² = {m_aai_le.rsquared:.4f}")
print(f"  AAI → HE (controlled):                   R² = {m_aai_he_c.rsquared:.4f}")
print(f"  AAI → LE (controlled):                   R² = {m_aai_le_c.rsquared:.4f}")
print(f"  DTAI → HE (no controls):                 R² = {m_dtai_he.rsquared:.4f}")
print(f"  DTAI → HE (controlled):                  R² = {m_dtai_he_c.rsquared:.4f}")
print(f"  DTAI → LE (no controls):                 R² = {m_dtai_le.rsquared:.4f}")
print(f"  DTAI → LE (controlled):                  R² = {m_dtai_le_c.rsquared:.4f}")
print(f"  DTAI + AAI → HE (no controls):           R² = {m_both_simple.rsquared:.4f}")
print(f"  DTAI + AAI → HE (controlled):            R² = {m_both_controlled.rsquared:.4f}")
print(f"  M1 AAI x Influencer → HE:                R² = {m1.rsquared:.4f}")
print(f"  M2 AAI x Influencer → LE:                R² = {m2.rsquared:.4f}")
print(f"  M3 AAI x Content    → HE:                R² = {m3.rsquared:.4f}")
print(f"  M4 AAI x Content    → LE:                R² = {m4.rsquared:.4f}")
