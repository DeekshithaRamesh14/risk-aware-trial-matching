import os
import ast
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle

RAW_DIR = '/dgxa_home/se25mbds002/trial_match/data/raw'
OUT_DIR = '/dgxa_home/se25mbds002/trial_match/data/processed'
os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. Load all splits ────────────────────────────────────────────────────────
dfs = {}
for phase in ['I', 'II', 'III']:
    for split in ['train', 'valid', 'test']:
        key = f'{phase}_{split}'
        dfs[key] = pd.read_csv(f'{RAW_DIR}/phase_{phase}_{split}.csv')
        dfs[key]['phase_label'] = phase
        dfs[key]['split'] = split

all_data = pd.concat(dfs.values(), ignore_index=True)
print(f'Total rows: {len(all_data)}')

# ── 2. Safe list parser ───────────────────────────────────────────────────────
def safe_parse(val):
    try:
        return ast.literal_eval(val)
    except:
        return []

# ── 3. Phase encoding ─────────────────────────────────────────────────────────
phase_map = {'I': 1, 'II': 2, 'III': 3}
all_data['phase_ordinal'] = all_data['phase_label'].map(phase_map)

# ── 4. Status encoding ────────────────────────────────────────────────────────
# terminated/withdrawn/suspended = high failure signal
all_data['is_terminated'] = all_data['status'].isin(
    ['terminated', 'withdrawn', 'suspended']).astype(int)

# ── 5. Drug count ─────────────────────────────────────────────────────────────
all_data['drug_count'] = all_data['drugs'].apply(
    lambda x: len(safe_parse(x)))

# ── 6. Disease count ──────────────────────────────────────────────────────────
all_data['disease_count'] = all_data['diseases'].apply(
    lambda x: len(safe_parse(x)))

# ── 7. ICD code count ─────────────────────────────────────────────────────────
def count_icds(val):
    try:
        outer = ast.literal_eval(val)
        total = 0
        for item in outer:
            inner = ast.literal_eval(item) if isinstance(item, str) else item
            total += len(inner) if isinstance(inner, list) else 1
        return total
    except:
        return 0

all_data['icd_count'] = all_data['icdcodes'].apply(count_icds)

# ── 8. Criteria length ────────────────────────────────────────────────────────
all_data['criteria_length'] = all_data['criteria'].fillna('').apply(len)

# ── 9. Has SMILES (drug chemical info available) ──────────────────────────────
all_data['has_smiles'] = all_data['smiless'].apply(
    lambda x: int(len(safe_parse(x)) > 0))

# ── 10. Study year ────────────────────────────────────────────────────────────
all_data['study_year'] = pd.to_datetime(
    all_data['study_first_submitted_date'], errors='coerce').dt.year
all_data['study_year'] = all_data['study_year'].fillna(
    all_data['study_year'].median())

# ── 11. Top disease encoding (target encode by failure rate) ──────────────────
def get_first_disease(val):
    parsed = safe_parse(val)
    return parsed[0].lower().strip() if parsed else 'unknown'

all_data['primary_disease'] = all_data['diseases'].apply(get_first_disease)

# Target encode: mean label per disease (on train only to avoid leakage)
train_mask = all_data['split'] == 'train'
disease_failure_rate = (
    all_data[train_mask]
    .groupby('primary_disease')['label']
    .mean()
    .rename('disease_target_enc')
)
all_data = all_data.merge(
    disease_failure_rate, on='primary_disease', how='left')
global_mean = all_data[train_mask]['label'].mean()
all_data['disease_target_enc'] = all_data['disease_target_enc'].fillna(global_mean)

# ── 12. Final feature matrix ──────────────────────────────────────────────────
FEATURE_COLS = [
    'phase_ordinal',
    'is_terminated',
    'drug_count',
    'disease_count',
    'icd_count',
    'criteria_length',
    'study_year',
    'disease_target_enc',
]

print('\n=== Feature matrix sample ===')
print(all_data[FEATURE_COLS].head())
print('\n=== Null check ===')
print(all_data[FEATURE_COLS].isnull().sum())
print('\n=== Label distribution ===')
print(all_data['label'].value_counts(normalize=True).round(3))

# ── 13. Save train/val/test splits ────────────────────────────────────────────
for split in ['train', 'valid', 'test']:
    mask = all_data['split'] == split
    X = all_data[mask][FEATURE_COLS].values
    y = all_data[mask]['label'].values
    nctids = all_data[mask]['nctid'].values
    np.save(f'{OUT_DIR}/X_{split}.npy', X)
    np.save(f'{OUT_DIR}/y_{split}.npy', y)
    np.save(f'{OUT_DIR}/nctids_{split}.npy', nctids)
    print(f'Saved {split}: X={X.shape}, y={y.shape}')

# Save feature names and disease encoder for API use
with open(f'{OUT_DIR}/feature_cols.pkl', 'wb') as f:
    pickle.dump(FEATURE_COLS, f)
with open(f'{OUT_DIR}/disease_target_enc.pkl', 'wb') as f:
    pickle.dump(disease_failure_rate, f)

# Save full dataframe with criteria text for retrieval stage
all_data[['nctid', 'criteria', 'title', 'split']].drop_duplicates(
    subset='nctid', keep='first').to_csv(
    f'{OUT_DIR}/trials_for_retrieval.csv', index=False)

print('\nFeature engineering complete.')
