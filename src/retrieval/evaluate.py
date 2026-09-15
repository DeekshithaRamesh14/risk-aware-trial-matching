import numpy as np
import pandas as pd
import pickle
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

PROCESSED_DIR = '/dgxa_home/se25mbds002/trial_match/data/processed'
MODEL_DIR = '/dgxa_home/se25mbds002/trial_match/models'

stop_words = set(stopwords.words('english'))

def tokenize(text):
    tokens = word_tokenize(str(text).lower())
    return [t for t in tokens if t.isalnum() and t not in stop_words]

def dcg_at_k(relevances, k):
    relevances = np.array(relevances[:k], dtype=float)
    if len(relevances) == 0:
        return 0.0
    discounts = np.log2(np.arange(2, len(relevances) + 2))
    return np.sum(relevances / discounts)

def ndcg_at_k(retrieved_ids, relevant_ids, k=10):
    relevant_set = set(relevant_ids)
    relevances = [1 if nid in relevant_set else 0 for nid in retrieved_ids[:k]]
    ideal = sorted(relevances, reverse=True)
    dcg = dcg_at_k(relevances, k)
    idcg = dcg_at_k(ideal, k)
    return dcg / idcg if idcg > 0 else 0.0

def mrr_at_k(retrieved_ids, relevant_ids, k=10):
    relevant_set = set(relevant_ids)
    for i, nid in enumerate(retrieved_ids[:k]):
        if nid in relevant_set:
            return 1.0 / (i + 1)
    return 0.0

def evaluate_bm25(k=10):
    # Load BM25 index
    with open(f'{MODEL_DIR}/bm25_index.pkl', 'rb') as f:
        bm25 = pickle.load(f)

    index_map = pd.read_csv(f'{PROCESSED_DIR}/retrieval_index_map.csv')

    # Load test trials as queries
    # For each test trial, use its criteria as query
    # Relevant = same nctid (self-retrieval sanity check)
    # Real evaluation needs patient-trial relevance labels
    test_nctids = np.load(f'{PROCESSED_DIR}/nctids_test.npy', allow_pickle=True)

    # Build nctid -> index mapping
    nctid_to_idx = {row['nctid']: idx for idx, row in index_map.iterrows()}

    # Load full trials for criteria text
    trials_df = pd.read_csv(f'{PROCESSED_DIR}/trials_for_retrieval.csv')
    nctid_to_criteria = dict(zip(trials_df['nctid'], trials_df['criteria']))

    ndcg_scores, mrr_scores = [], []
    missing = 0

    for nctid in test_nctids[:200]:  # sample 200 for speed
        if nctid not in nctid_to_criteria or nctid not in nctid_to_idx:
            missing += 1
            continue

        query_text = nctid_to_criteria[nctid]
        tokens = tokenize(str(query_text))
        scores = bm25.get_scores(tokens)
        top_k_idx = scores.argsort()[-k:][::-1]
        retrieved_ids = index_map.iloc[top_k_idx]['nctid'].tolist()

        # Ground truth: the trial itself (self-retrieval)
        relevant_ids = [nctid]

        ndcg_scores.append(ndcg_at_k(retrieved_ids, relevant_ids, k))
        mrr_scores.append(mrr_at_k(retrieved_ids, relevant_ids, k))

    print(f'Evaluated {len(ndcg_scores)} queries ({missing} skipped)')
    print(f'BM25 NDCG@{k}: {np.mean(ndcg_scores):.4f}')
    print(f'BM25  MRR@{k}: {np.mean(mrr_scores):.4f}')
    return np.mean(ndcg_scores), np.mean(mrr_scores)

if __name__ == '__main__':
    print('=== BM25 Baseline Evaluation ===')
    ndcg, mrr = evaluate_bm25(k=10)
    print(f'\nFinal: NDCG@10={ndcg:.4f}, MRR@10={mrr:.4f}')
    print('\nNote: self-retrieval evaluation — real evaluation')
    print('requires patient-trial relevance labels.')
