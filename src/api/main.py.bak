import os
import pickle
import numpy as np
import pandas as pd
import faiss
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder
import shap

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = '/dgxa_home/se25mbds002/trial_match'
PROCESSED = f'{BASE}/data/processed'
MODELS = f'{BASE}/models'

# ── Load all artifacts at startup ─────────────────────────────────────────────
print('Loading artifacts...')

# Trials dataframe
trials_df = pd.read_csv(f'{PROCESSED}/trials_for_retrieval.csv')
print(f'Trials loaded: {len(trials_df)}')

# FAISS index
index = faiss.read_index(f'{PROCESSED}/trial_index.faiss')
index.nprobe = 10
print(f'FAISS index loaded: {index.ntotal} vectors')

# Bi-encoder
biencoder = SentenceTransformer(
    'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract')
print('Bi-encoder loaded.')

# Cross-encoder
crossencoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
print('Cross-encoder loaded.')

# XGBoost pipeline
with open(f'{MODELS}/xgb_pipeline.pkl', 'rb') as f:
    xgb_pipeline = pickle.load(f)
print('XGBoost pipeline loaded.')

# Feature columns
with open(f'{PROCESSED}/feature_cols.pkl', 'rb') as f:
    feature_cols = pickle.load(f)

# Disease target encoding map
with open(f'{PROCESSED}/disease_target_enc.pkl', 'rb') as f:
    disease_enc = pickle.load(f)

# SHAP explainer
with open(f'{MODELS}/shap_explainer.pkl', 'rb') as f:
    explainer = pickle.load(f)

global_mean = 0.573  # from training label distribution

print('All artifacts loaded. API ready.')

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title='Risk-Aware Trial Matching API')

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class PatientInput(BaseModel):
    patient_id: str
    age: int
    sex: str
    diagnosis_codes: list[str]
    medications: list[str]
    free_text_summary: str

# ── Helper: build XGBoost features for a trial row ───────────────────────────
def build_features(trial_row, phase_ordinal):
    primary_disease = trial_row.get('primary_disease', 'unknown')
    disease_enc_val = disease_enc.get(primary_disease, global_mean)
    criteria_text = str(trial_row.get('criteria', ''))

    features = {
        'phase_ordinal': phase_ordinal,
        'is_terminated': 0,  # unknown at match time
        'drug_count': 1,
        'disease_count': len(trial_row.get('diagnosis_codes', [])),
        'icd_count': 0,
        'criteria_length': len(criteria_text),
        'study_year': 2020,
        'disease_target_enc': disease_enc_val,
    }
    return [features[col] for col in feature_cols]

# ── Helper: top SHAP features ─────────────────────────────────────────────────
def get_top_shap(shap_row, n=3):
    abs_vals = np.abs(shap_row)
    top_idx = abs_vals.argsort()[-n:][::-1]
    return [
        {'feature': feature_cols[i], 'value': round(float(shap_row[i]), 4)}
        for i in top_idx
    ]

# ── Main endpoint ─────────────────────────────────────────────────────────────
@app.post('/match')
def match(patient: PatientInput):
    # Stage 1: Dense retrieval — top 100
    query_emb = biencoder.encode(
        [patient.free_text_summary],
        normalize_embeddings=True
    )
    scores, indices = index.search(query_emb, 100)

    candidates = []
    for idx, score in zip(indices[0], scores[0]):
        if idx < len(trials_df):
            row = trials_df.iloc[idx]
            candidates.append({
                'nctid': row['nctid'],
                'title': str(row['title']),
                'criteria': str(row['criteria'])[:512],
                'retrieval_score': float(score),
                'idx': int(idx)
            })

    # Stage 2: Cross-encoder reranking — top 20
    pairs = [(patient.free_text_summary, c['criteria']) for c in candidates]
    ce_scores = crossencoder.predict(pairs)
    ranked_idx = np.argsort(ce_scores)[::-1][:20]
    reranked = [(candidates[i], float(ce_scores[i])) for i in ranked_idx]

    # Normalise relevance scores to [0, 1]
    rel_scores = np.array([s for _, s in reranked])
    min_r, max_r = rel_scores.min(), rel_scores.max()
    norm_rel = (rel_scores - min_r) / (max_r - min_r + 1e-9)

    # Stage 3: SAE risk scoring + final ranking
    results = []
    for i, (candidate, rel_score) in enumerate(reranked):
        trial_row = trials_df.iloc[candidate['idx']]
        features = build_features(trial_row, phase_ordinal=2)
        X = np.array([features])

        sae_risk = float(xgb_pipeline.predict_proba(X)[0][1])
        final_score = float(norm_rel[i] * (1 - sae_risk))
        shap_vals = explainer.shap_values(X)[0]
        top_shap = get_top_shap(shap_vals)

        results.append({
            'nctid': candidate['nctid'],
            'title': candidate['title'],
            'relevance_score': round(float(rel_score), 4),
            'sae_risk': round(sae_risk, 4),
            'final_score': round(final_score, 4),
            'shap_top_features': top_shap
        })

    # Sort by final score descending
    results.sort(key=lambda x: x['final_score'], reverse=True)
    return results

@app.get('/health')
def health():
    return {'status': 'ok', 'trials_indexed': index.ntotal}
