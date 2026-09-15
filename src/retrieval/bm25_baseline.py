import os
import ast
import pandas as pd
import numpy as np
import pickle
import json
from rank_bm25 import BM25Okapi
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

PROCESSED_DIR = '/dgxa_home/se25mbds002/trial_match/data/processed'
MODEL_DIR = '/dgxa_home/se25mbds002/trial_match/models'
os.makedirs(MODEL_DIR, exist_ok=True)

# Load trials with criteria text
trials_df = pd.read_csv(f'{PROCESSED_DIR}/trials_for_retrieval.csv')
trials_df = trials_df.dropna(subset=['criteria']).reset_index(drop=True)
print(f'Total trials for retrieval index: {len(trials_df)}')

# Tokenize
stop_words = set(stopwords.words('english'))

def tokenize(text):
    tokens = word_tokenize(str(text).lower())
    return [t for t in tokens if t.isalnum() and t not in stop_words]

print('Tokenizing corpus...')
tokenized_corpus = [tokenize(text) for text in trials_df['criteria']]
print(f'Tokenization done. Sample: {tokenized_corpus[0][:10]}')

# Build BM25 index
print('Building BM25 index...')
bm25 = BM25Okapi(tokenized_corpus)

# Save index and mapping
with open(f'{MODEL_DIR}/bm25_index.pkl', 'wb') as f:
    pickle.dump(bm25, f)

trials_df[['nctid', 'title']].to_csv(
    f'{PROCESSED_DIR}/retrieval_index_map.csv', index=True)

print(f'BM25 index saved.')
print(f'Index size: {len(tokenized_corpus)} trials')

# Quick sanity check - query with a sample
sample_query = "lung cancer chemotherapy eligibility adult patients"
tokens = tokenize(sample_query)
scores = bm25.get_scores(tokens)
top5_idx = scores.argsort()[-5:][::-1]
print(f'\nSanity check - top 5 results for: "{sample_query}"')
for i, idx in enumerate(top5_idx):
    print(f'  {i+1}. [{trials_df.iloc[idx]["nctid"]}] '
          f'{str(trials_df.iloc[idx]["title"])[:70]} '
          f'(score: {scores[idx]:.3f})')

print('\nBM25 baseline ready.')
